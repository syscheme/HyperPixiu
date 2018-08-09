# encoding: UTF-8

from __future__ import division

import os
import shelve
import logging
from collections import OrderedDict
from datetime import datetime
from time import sleep
from copy import copy
from abc import ABCMeta, abstractmethod
import traceback

from vnApp.EventChannel import Event, EventLoop, EventChannel

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from .MarketData import *

from vnpy.event import Event
from vnpy.trader.vtGlobal import globalSetting
from vnpy.trader.vtEvent import *
from vnpy.trader.vtGateway import *
from vnpy.trader.language import text
from vnpy.trader.vtFunction import getTempPath

# 日志级别
LOGLEVEL_DEBUG    = logging.DEBUG
LOGLEVEL_INFO     = logging.INFO
LOGLEVEL_WARN     = logging.WARN
LOGLEVEL_ERROR    = logging.ERROR
LOGLEVEL_CRITICAL = logging.CRITICAL

import jsoncfg # pip install json-cfg

########################################################################
class BaseApplication(object):

    __lastId__ =100

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        self._engine = mainRoutine
        self._settings = settings

        self._active = False                     # 工作状态

        # the app instance Id
        self._id = settings.id("")
        if len(self._id)<=0 :
            BaseApplication.__lastId__ +=1
            self._id = 'P%d' % BaseApplication.__lastId__

        # 日志级别函数映射
        self._loglevelFunctionDict = {
            LOGLEVEL_DEBUG:    self.debug,
            LOGLEVEL_INFO:     self.info,
            LOGLEVEL_WARN:     self.warn,
            LOGLEVEL_ERROR:    self.error,
            LOGLEVEL_CRITICAL: self.critical,
        }

    #----------------------------------------------------------------------
    @property
    def ident(self) :
        return self.__class__.__name__ +":" + self._id

    @property
    def app(self) : # for the thread wrapper
        return self

    @property
    def mainRoutine(self) :
        return self._engine

    @property
    def settings(self) :
        return self._settings

    @property
    def isActive(self) :
        return self._active
    
    @property
    def _dbConn(self) :
        return self._engine.dbConn

    #----------------------------------------------------------------------
    @abstractmethod
    def subscribeEvent(self, event, funcCallback) :
        if self._engine and self._engine._eventChannel:
            self._engine._eventChannel.register(event, funcCallback)

    #----------------------------------------------------------------------
    @abstractmethod
    def postEvent(self, eventType, edata):
        """发出事件"""
        event = Event(type_= eventType)
        event.dict_['data'] = edata
        self._engine._eventChannel.put(event)

    #---logging -----------------------
    def log(self, level, msg):
        if not level in self._loglevelFunctionDict : 
            return
        
        function = self._loglevelFunctionDict[level] # 获取日志级别对应的处理函数
        function(msg)

    def debug(self, msg):
        self._engine.debug('APP['+self.ident +'] ' + msg)
        
    def info(self, msg):
        """正常输出"""
        self._engine.info('APP['+self.ident +'] ' + msg)

    def warn(self, msg):
        """警告信息"""
        self._engine.warn('APP['+self.ident +'] ' + msg)
        
    def error(self, msg):
        """报错输出"""
        self._engine.error('APP['+self.ident +'] ' + msg)
        
    def critical(self, msg):
        """影响程序运行的严重错误"""
        self._engine.critical('APP['+self.ident +'] ' + msg)

    def logexception(self, ex):
        """报错输出+记录异常信息"""
        self._engine.logexception('APP['+self.ident +'] %s: %s' % (ex, traceback.format_exc()))

    #----------------------------------------------------------------------
    @abstractmethod
    def start(self):
        # TODO:
        self._active = True

    #----------------------------------------------------------------------
    @abstractmethod
    def stop(self):
        # TODO:
        self._active = False

    #----------------------------------------------------------------------
    @abstractmethod
    def step(self):
        # TODO:
        pass

    #----------------------------------------------------------------------
    @abstractmethod
    def dbInsert(self, dbName, collectionName, d):
        """向MongoDB中插入数据，d是具体数据"""
        if self._engine:
            self._engine.dbInsert(dbName, collectionName, d)
    
    #----------------------------------------------------------------------
    @abstractmethod
    def dbQuery(self, dbName, collectionName, d, sortKey='', sortDirection=ASCENDING):
        """从MongoDB中读取数据，d是查询要求，返回的是数据库查询的指针"""
        if not self._dbConn:
            self.writeLog(text.DATA_QUERY_FAILED)   
            return []

        db = self._dbConn[dbName]
        collection = db[collectionName]
            
        if sortKey:
            cursor = collection.find(d).sort(sortKey, sortDirection)    # 对查询出来的数据进行排序
        else:
            cursor = collection.find(d)

        if cursor:
            return list(cursor)

        return []
        
    #----------------------------------------------------------------------
    @abstractmethod
    def dbUpdate(self, dbName, collectionName, d, flt, upsert=False):
        """向MongoDB中更新数据，d是具体数据，flt是过滤条件，upsert代表若无是否要插入"""
        if not self._dbConn:
            self.writeLog(text.DATA_UPDATE_FAILED)        
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.replace_one(flt, d, upsert)
            
    #----------------------------------------------------------------------
    @abstractmethod
    def logEvent(self, eventType, content):
        """快速发出日志事件"""
        log = VtLogData()
        log.dsName = self.ident
        log.logContent = content
        self.postEvent(eventType, edata)

    #----------------------------------------------------------------------
    @abstractmethod
    def logError(self, eventType, content):
        """处理错误事件"""
    # TODO   error = event.dict_['data']
    #    self._lstErrors.append(error)

########################################################################
class ThreadedApplication(object):
    #----------------------------------------------------------------------
    def __init__(self, app):
        """Constructor"""
        self._app = app
        self.thread = Thread(target=self._run)

    #----------------------------------------------------------------------
    def _run(self):
        """执行连接 and receive"""
        while self._app._active:
            try :
                nextSleep = - self._app.step()
                if nextSleep >0:
                    sleep(min(2, nextSleep))
            except Exception as ex:
                self._app.error('ThreadedApplication::step() excepton: %s' % ex)
        self._app.info('ThreadedApplication exit')

    #----------------------------------------------------------------------
    @abstractmethod
    def start(self):
        ret = self._app.start()
        self.thread.start()
        self._app.debug('ThreadedApplication starts')
        return ret

    #----------------------------------------------------------------------
    @abstractmethod
    def stop(self):
        self._app.stop()
        self.thread.join()
        self._app.info('ThreadedApplication stopped')
    
########################################################################
class MainRoutine(object):
    """主引擎"""

    # 单例模式
    __metaclass__ = VtSingleton
    
    FINISHED_STATUS = [STATUS_ALLTRADED, STATUS_REJECTED, STATUS_CANCELLED]

    #----------------------------------------------------------------------
    def __init__(self, settings):
        """Constructor"""
        
        self._settings = settings
        self._threadless = True

        # 记录今日日期
        self.todayDate = datetime.now().strftime('%Y%m%d')
        
        # 创建EventChannel
        if self.threadless :
            self._eventChannel = EventLoop()
        else :
            self._eventChannel = EventChannel()
        # self._eventChannel.start()

        # 日志引擎实例
        self._logger = None
        self.initLogger()
        
        #----------------------------------------------------------------------
        # from old 数据引擎
        # 保存数据的字典和列表
        self._dictLatestTick = {}    # the latest tick of each symbol
        self._dictLatestContract = {}
        self._dictLatestOrder = {}
        self._dictWorkingOrder = {}  # 可撤销委托
        self._dictTrade = {}
        self._dictAccounts = {}
        self._dictPositions= {}
        self._lstLogs = []
        self._lstErrors = []
        
        # MongoDB数据库相关
        self._dbConn = None    # MongoDB客户端对象
        
        # 接口实例
        self._dictMarketDatas = OrderedDict()
        self._dlstMarketDatas = []
        
        # 应用模块实例
        self._dictApps = OrderedDict()
        self._dlstApps = []
        
        # 风控引擎实例（特殊独立对象）
        self._riskMgm = None
    
    @property
    def threadless(self) :
        return self._threadless

    @property
    def dbConn(self) :
        return self._dbConn

    #----------------------------------------------------------------------
    def addMarketData(self, dsModule, settings):
        """添加底层接口"""

        # 创建接口实例
        clsName = dsModule.className
        md = dsModule(self, settings)
        # 保存接口详细信息
        d = {
            'id': md.id,
            'dsDisplayName': settings.displayName(id),
            'dsType': clsName,
        }

        self._dictMarketDatas[md.exchange] = md
        self._dlstMarketDatas.append(d)
        self.info('md[%s] added: %s' %(md.exchange, d))
    #----------------------------------------------------------------------
    def addApp(self, appModule, settings):
        """添加上层应用"""

        app = appModule(self, settings)

        clsName = app.__class__.__name__
        id = app.ident
        
        # 创建应用实例
        self._dictApps[id] = app
        
        # 将应用引擎实例添加到主引擎的属性中
        self.__dict__[id] = self._dictApps[id]
        
        # 保存应用信息
        d = {
            'appName': id,
            'displayName': settings.displayName(id),
            'appWidget': settings.widget(id),
#            'appIco': appModule.appIco
        }
        self._dlstApps.append(d)
        self.info('app[%s] added: %s' %(id, d))
        
    #----------------------------------------------------------------------
    def getMarketData(self, dsName, exchange=None):
        """获取接口"""
        if dsName in self._dictMarketDatas:
            return self._dictMarketDatas[dsName]
        else:
            self.error('getMarketData() %s not exist' % dsName)
            return None
        
    #----------------------------------------------------------------------
    def start(self):

        self.debug('starting event channel')
        if self._eventChannel:
            self._eventChannel.start()
            self.info('started event channel')

        self.dbConnect()
        
        self.debug('starting market data subscribers')
        for (k, ds) in self._dictMarketDatas.items():
            if ds == None:
                continue

            if self.threadless :
                ds.connect()
                self.debug('market subscriber[%s] connected' % k)
            else :
                ds.start()
                self.debug('market subscriber[%s] started' % k)

        self.debug('starting applications')
        for (k, app) in self._dictApps.items() :
            if app == None:
                continue
            
            self.debug('staring app[%s]' % k)
            app.start()
            self.info('started app[%s]' % k)

        self.info('main-routine started')

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # 安全关闭所有接口
        for ds in self._dictMarketDatas.values():        
            ds.close()
        
        # 停止事件引擎
        self._eventChannel.stop()
        
        # 停止上层应用引擎
        for app in self._dictApps.values():
            app.stop()
        
        # # 保存数据引擎里的合约数据到硬盘
        # self.dataEngine.saveContracts()

    #----------------------------------------------------------------------
    def loop(self):

        self.info(u'MainRoutine start looping')
        c=0
        
        busy = True
        while True:
            if not self.threadless :
                try :
                    sleep(1)
                    self.debug(u'MainThread heartbeat')
                except KeyboardInterrupt as ki:
                    break

                continue

            # loop mode as below
            if not busy:
                sleep(0.5)

            busy = False
            for (k, ds) in self._dictMarketDatas.items():
                try :
                    if ds == None:
                        continue
                    ds.step()
                except Exception as ex:
                    print("eventCH exception %s %s" % (ex, traceback.format_exc()))

            pending = self._eventChannel.pendingSize
            busy =  pending >0

            pending = min(20, pending)
            for i in range(0, pending) :
                try :
                    self._eventChannel.step()
                    c+=1
                except KeyboardInterrupt as ki:
                    exit(-1)
                except Exception, ex:
                    print("eventCH exception %s %s" % (ex, traceback.format_exc()))

            # if c % 10 ==0:
            #     self.debug(u'MainThread heartbeat')

        self.info(u'MainRoutine finish looping')

    #----------------------------------------------------------------------
    def subscribeMarketData(self, subscribeReq, dsName):
        """订阅特定接口的行情"""
        ds = self.getMarketData(dsName)
        
        if ds:
            ds.subscribe(subscribeReq)
  
    #----------------------------------------------------------------------
    # methods about logging
    # ----------------------------------------------------------------------
    def initLogger(self):
        """初始化日志引擎"""
        if not jsoncfg.node_exists(self._settings.logger):
            return
        
        # 创建引擎

        # abbout the logger
        # ----------------------------------------------------------------------
        self._logger   = logging.getLogger()        
        self._logfmtr  = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
        self._loglevel = LOGLEVEL_CRITICAL
        
        self._hdlrConsole = None
        self._hdlrFile = None
        
        # 日志级别函数映射
        self._loglevelFunctionDict = {
            LOGLEVEL_DEBUG:    self.debug,
            LOGLEVEL_INFO:     self.info,
            LOGLEVEL_WARN:     self.warn,
            LOGLEVEL_ERROR:    self.error,
            LOGLEVEL_CRITICAL: self.critical,
        }

        # 设置日志级别
        self.setLogLevel(self._settings.logger.level(LOGLEVEL_DEBUG)) # LOGLEVEL_INFO))
        
        BOOL_TRUE = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

        # 设置输出
        tmpval = self._settings.logger.console('True').lower()
        if tmpval in BOOL_TRUE and not self._hdlrConsole:
            """添加终端输出"""
            self._hdlrConsole = logging.StreamHandler()
            self._hdlrConsole.setLevel(self._loglevel)
            self._hdlrConsole.setFormatter(self._logfmtr)
            self._logger.addHandler(self._hdlrConsole)
        else :
            # 添加NullHandler防止无handler的错误输出
            nullHandler = logging.NullHandler()
            self._logger.addHandler(nullHandler)    

        tmpval = self._settings.logger.file('True').lower()
        if tmpval in BOOL_TRUE and not self._hdlrFile:
            filepath = getTempPath('vnApp' + datetime.now().strftime('%Y%m%d') + '.log')
            self._hdlrFile = logging.FileHandler(filepath)
            self._hdlrFile.setLevel(self._loglevel)
            self._hdlrFile.setFormatter(self._logfmtr)
            self._logger.addHandler(self._hdlrFile)
            
        # 注册事件监听
        tmpval = self._settings.logger.event.log('True').lower()
        if tmpval in BOOL_TRUE :
            self._eventChannel.register(EventChannel.EVENT_LOG, self.eventHdlr_Log)

        tmpval = self._settings.logger.event.error('True').lower()
        if tmpval in BOOL_TRUE :
            self._eventChannel.register(EventChannel.EVENT_ERROR, self.eventHdlr_Error)

    def setLogLevel(self, level):
        """设置日志级别"""
        if self._logger ==None:
            return

        self._logger.setLevel(level)
        self._loglevel = level
    
    def log(self, level, msg):
        if not level in self._loglevelFunctionDict : 
            return
        
        function = self._loglevelFunctionDict[level] # 获取日志级别对应的处理函数
        function(msg)

    def debug(self, msg):
        """开发时用"""
        if self._logger ==None:
            return

        self._logger.debug(msg)
        
    def info(self, msg):
        """正常输出"""
        if self._logger ==None:
            return

        self._logger.info(msg)

    def warn(self, msg):
        """警告信息"""
        if self._logger ==None:
            return

        self._logger.warn(msg)
        
    def error(self, msg):
        """报错输出"""
        if self._logger ==None:
            return

        self._logger.error(msg)
        
    def critical(self, msg):
        """影响程序运行的严重错误"""
        if self._logger ==None:
            return

        self._logger.critical(msg)

    def logexception(self, ex):
        """报错输出+记录异常信息"""
        if self._logger ==None:
            return

        traceback.format_exc()
        #s elf._logger.exception(msg)

    def eventHdlr_Log(self, event):
        """处理日志事件"""
        if self._logger ==None:
            return

        log = event.dict_['data']
        function = self._loglevelFunctionDict[log.logLevel]     # 获取日志级别对应的处理函数
        msg = '\t'.join([log.dsName, log.logContent])
        function(msg)

    def eventHdlr_Error(self, event):
        """
        处理错误事件
        """
        if self._logger ==None:
            return

        error = event.dict_['data']
        self.error(u'错误代码：%s，错误信息：%s' %(error.errorID, error.errorMsg))

    # #----------------------------------------------------------------------
    # def sendOrder(self, orderReq, dsName):
    #     """对特定接口发单"""
    #     # 如果创建了风控引擎，且风控检查失败则不发单
    #     if self._riskMgm and not self._riskMgm.checkRisk(orderReq, dsName):
    #         return ''

    #     ds = self.getMarketData(dsName)
        
    #     if ds:
    #         vtOrderID = ds.sendOrder(orderReq)
    #         self.dataEngine.updateOrderReq(orderReq, vtOrderID)     # 更新发出的委托请求到数据引擎中
    #         return vtOrderID
    #     else:
    #         return ''
        
    # #----------------------------------------------------------------------
    # def cancelOrder(self, cancelOrderReq, dsName):
    #     """对特定接口撤单"""
    #     ds = self.getMarketData(dsName)
        
    #     if ds:
    #         ds.cancelOrder(cancelOrderReq)   
  
    # #----------------------------------------------------------------------
    # def qryAccount(self, dsName):
    #     """查询特定接口的账户"""
    #     ds = self.getMarketData(dsName)
        
    #     if ds:
    #         ds.qryAccount()      
        
    # #----------------------------------------------------------------------
    # def qryPosition(self, dsName):
    #     """查询特定接口的持仓"""
    #     ds = self.getMarketData(dsName)
        
    #     if ds:
    #         ds.qryPosition()
            
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    def writeLog(self, content):
        """快速发出日志事件"""
        log = VtLogData()
        log.logContent = content
        log.dsName = 'MAIN_ENGINE'
        event = Event(type_=EVENT_LOG)
        event.dict_['data'] = log
        self._eventChannel.put(event)        
    
    #----------------------------------------------------------------------
    def dbConnect(self):
        """连接MongoDB数据库"""
        if not self._dbConn:
            # 读取MongoDB的设置
            dbhost = self._settings.database.host('localhost')
            dbport = self._settings.database.port(27017)
            self.debug('connecting DB[%s :%s]'%(dbhost, dbport))
            
            try:
                # 设置MongoDB操作的超时时间为0.5秒
                self._dbConn = MongoClient(dbhost, dbport, connectTimeoutMS=500)
                
                # 调用server_info查询服务器状态，防止服务器异常并未连接成功
                self._dbConn.server_info()

                # 如果启动日志记录，则注册日志事件监听函数
                if self._settings.database.logging("") in ['True']:
                    self._eventChannel.register(EVENT_LOG, self.dbLogging)
                    
                self.info('connecting DB[%s :%s] %s'%(dbhost, dbport, text.DATABASE_CONNECTING_COMPLETED))
            except ConnectionFailure:
                self.error('failed to connect DB[%s :%s] %s' %(dbhost, dbport, text.DATABASE_CONNECTING_FAILED))
            except:
                self.error('failed to connect DB[%s :%s]' %(dbhost, dbport))
    
    #----------------------------------------------------------------------
    @abstractmethod
    def dbInsert(self, dbName, collectionName, d):
        """向MongoDB中插入数据，d是具体数据"""
        if not self._dbConn:
            self.writeLog(text.DATA_INSERT_FAILED)
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.insert_one(d)

    #----------------------------------------------------------------------
    def dbLogging(self, event):
        """向MongoDB中插入日志"""
        log = event.dict_['data']
        d = {
            'content': log.logContent,
            'time': log.logTime,
            'ds': log.dsName
        }
        self.dbInsert(LOG_DB_NAME, self.todayDate, d)
    
    #----------------------------------------------------------------------
    def getAllGatewayDetails(self):
        """查询引擎中所有底层接口的信息"""
        return self._dlstMarketDatas
    
    #----------------------------------------------------------------------
    def getAllAppDetails(self):
        """查询引擎中所有上层应用的信息"""
        return self._dlstApps
    
    #----------------------------------------------------------------------
    def getApp(self, appName):
        """获取APP引擎对象"""
        return self._dictApps[appName]
    
    #----------------------------------------------------------------------
    
    # #----------------------------------------------------------------------
    # def convertOrderReq(self, req):
    #     """转换委托请求"""
    #     return self.dataEngine.convertOrderReq(req)

    # #----------------------------------------------------------------------
    # def getLog(self):
    #     """查询日志"""
    #     return self.dataEngine.getLog()
    
    # #----------------------------------------------------------------------
    # def getError(self):
    #     """查询错误"""
    #     return self.dataEngine.getError()
    
    # #----------------------------------------------------------------------
    # def processTickEvent(self, event):
    #     """处理成交事件"""
    #     tick = event.dict_['data']
    #     self._dictLatestTick[tick.vtSymbol] = tick    
    
    # #----------------------------------------------------------------------
    # def processContractEvent(self, event):
    #     """处理合约事件"""
    #     contract = event.dict_['data']
    #     self._dictLatestContract[contract.vtSymbol] = contract
    #     self._dictLatestContract[contract.symbol] = contract       # 使用常规代码（不包括交易所）可能导致重复
    
    # #----------------------------------------------------------------------
    # def processOrderEvent(self, event):
    #     """处理委托事件"""
    #     order = event.dict_['data']        
    #     self._dictLatestOrder[order.vtOrderID] = order
        
    #     # 如果订单的状态是全部成交或者撤销，则需要从workingOrderDict中移除
    #     if order.status in self.FINISHED_STATUS:
    #         if order.vtOrderID in self._dictWorkingOrder:
    #             del self._dictWorkingOrder[order.vtOrderID]
    #     # 否则则更新字典中的数据        
    #     else:
    #         self._dictWorkingOrder[order.vtOrderID] = order
            
    #     # 更新到持仓细节中
    #     detail = self.getPositionDetail(order.vtSymbol)
    #     detail.updateOrder(order)            
            
    # #----------------------------------------------------------------------
    # def processTradeEvent(self, event):
    #     """处理成交事件"""
    #     trade = event.dict_['data']
        
    #     self._dictTrade[trade.vtTradeID] = trade
    
    #     # 更新到持仓细节中
    #     detail = self.getPositionDetail(trade.vtSymbol)
    #     detail.updateTrade(trade)        

    # #----------------------------------------------------------------------
    # def processPositionEvent(self, event):
    #     """处理持仓事件"""
    #     pos = event.dict_['data']
        
    #     self._dictPositions[pos.vtPositionName] = pos
    
    #     # 更新到持仓细节中
    #     detail = self.getPositionDetail(pos.vtSymbol)
    #     detail.updatePosition(pos)                
        
    # #----------------------------------------------------------------------
    # def processAccountEvent(self, event):
    #     """处理账户事件"""
    #     account = event.dict_['data']
    #     self._dictAccounts[account.vtAccountID] = account
    
    # #----------------------------------------------------------------------
    # def eventHdlr_Log(self, event):
    #     """处理日志事件"""
    #     log = event.dict_['data']
    #     self._lstLogs.append(log)
    
    # #----------------------------------------------------------------------
    # def processErrorEvent(self, event):
    #     """处理错误事件"""
    #     error = event.dict_['data']
    #     self._lstErrors.append(error)
        

########################################################################
class PositionDetail(object):
    """本地维护的持仓信息"""
    WORKING_STATUS = [STATUS_UNKNOWN, STATUS_NOTTRADED, STATUS_PARTTRADED]
    
    MODE_NORMAL = 'normal'          # 普通模式
    MODE_SHFE = 'shfe'              # 上期所今昨分别平仓
    MODE_TDPENALTY = 'tdpenalty'    # 平今惩罚

    #----------------------------------------------------------------------
    def __init__(self, vtSymbol, contract=None):
        """Constructor"""
        self.vtSymbol = vtSymbol
        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.name = EMPTY_UNICODE    
        self.size = 1
        
        if contract:
            self.symbol = contract.symbol
            self.exchange = contract.exchange
            self.name = contract.name
            self.size = contract.size
        
        self.longPos = EMPTY_INT
        self.longYd = EMPTY_INT
        self.longTd = EMPTY_INT
        self.longPosFrozen = EMPTY_INT
        self.longYdFrozen = EMPTY_INT
        self.longTdFrozen = EMPTY_INT
        self.longPnl = EMPTY_FLOAT
        self.longPrice = EMPTY_FLOAT
        
        self.shortPos = EMPTY_INT
        self.shortYd = EMPTY_INT
        self.shortTd = EMPTY_INT
        self.shortPosFrozen = EMPTY_INT
        self.shortYdFrozen = EMPTY_INT
        self.shortTdFrozen = EMPTY_INT
        self.shortPnl = EMPTY_FLOAT
        self.shortPrice = EMPTY_FLOAT
        
        self.lastPrice = EMPTY_FLOAT
        
        self.mode = self.MODE_NORMAL
        self.exchange = EMPTY_STRING
        
        self._dictWorkingOrder = {}
        
    #----------------------------------------------------------------------
    def updateTrade(self, trade):
        """成交更新"""
        # 多头
        if trade.direction is DIRECTION_LONG:
            # 开仓
            if trade.offset is OFFSET_OPEN:
                self.longTd += trade.volume
            # 平今
            elif trade.offset is OFFSET_CLOSETODAY:
                self.shortTd -= trade.volume
            # 平昨
            elif trade.offset is OFFSET_CLOSEYESTERDAY:
                self.shortYd -= trade.volume
            # 平仓
            elif trade.offset is OFFSET_CLOSE:
                # 上期所等同于平昨
                if self.exchange is EXCHANGE_SHFE:
                    self.shortYd -= trade.volume
                # 非上期所，优先平今
                else:
                    self.shortTd -= trade.volume
                    
                    if self.shortTd < 0:
                        self.shortYd += self.shortTd
                        self.shortTd = 0    
        # 空头
        elif trade.direction is DIRECTION_SHORT:
            # 开仓
            if trade.offset is OFFSET_OPEN:
                self.shortTd += trade.volume
            # 平今
            elif trade.offset is OFFSET_CLOSETODAY:
                self.longTd -= trade.volume
            # 平昨
            elif trade.offset is OFFSET_CLOSEYESTERDAY:
                self.longYd -= trade.volume
            # 平仓
            elif trade.offset is OFFSET_CLOSE:
                # 上期所等同于平昨
                if self.exchange is EXCHANGE_SHFE:
                    self.longYd -= trade.volume
                # 非上期所，优先平今
                else:
                    self.longTd -= trade.volume
                    
                    if self.longTd < 0:
                        self.longYd += self.longTd
                        self.longTd = 0
                    
        # 汇总
        self.calculatePrice(trade)
        self.calculatePosition()
        self.calculatePnl()
    
    #----------------------------------------------------------------------
    def updateOrder(self, order):
        """委托更新"""
        # 将活动委托缓存下来
        if order.status in self.WORKING_STATUS:
            self._dictWorkingOrder[order.vtOrderID] = order
            
        # 移除缓存中已经完成的委托
        else:
            if order.vtOrderID in self._dictWorkingOrder:
                del self._dictWorkingOrder[order.vtOrderID]
                
        # 计算冻结
        self.calculateFrozen()
    
    #----------------------------------------------------------------------
    def updatePosition(self, pos):
        """持仓更新"""
        if pos.direction is DIRECTION_LONG:
            self.longPos = pos.position
            self.longYd = pos.ydPosition
            self.longTd = self.longPos - self.longYd
            self.longPnl = pos.positionProfit
            self.longPrice = pos.price
        elif pos.direction is DIRECTION_SHORT:
            self.shortPos = pos.position
            self.shortYd = pos.ydPosition
            self.shortTd = self.shortPos - self.shortYd
            self.shortPnl = pos.positionProfit
            self.shortPrice = pos.price
            
        #self.output()
    
    #----------------------------------------------------------------------
    def updateOrderReq(self, req, vtOrderID):
        """发单更新"""
        vtSymbol = req.vtSymbol        
            
        # 基于请求生成委托对象
        order = VtOrderData()
        order.vtSymbol = vtSymbol
        order.symbol = req.symbol
        order.exchange = req.exchange
        order.offset = req.offset
        order.direction = req.direction
        order.totalVolume = req.volume
        order.status = STATUS_UNKNOWN
        
        # 缓存到字典中
        self._dictWorkingOrder[vtOrderID] = order
        
        # 计算冻结量
        self.calculateFrozen()
        
    #----------------------------------------------------------------------
    def updateTick(self, tick):
        """行情更新"""
        self.lastPrice = tick.lastPrice
        self.calculatePnl()
        
    #----------------------------------------------------------------------
    def calculatePnl(self):
        """计算持仓盈亏"""
        self.longPnl = self.longPos * (self.lastPrice - self.longPrice) * self.size
        self.shortPnl = self.shortPos * (self.shortPrice - self.lastPrice) * self.size
        
    #----------------------------------------------------------------------
    def calculatePrice(self, trade):
        """计算持仓均价（基于成交数据）"""
        # 只有开仓会影响持仓均价
        if trade.offset == OFFSET_OPEN:
            if trade.direction == DIRECTION_LONG:
                cost = self.longPrice * self.longPos
                cost += trade.volume * trade.price
                newPos = self.longPos + trade.volume
                if newPos:
                    self.longPrice = cost / newPos
                else:
                    self.longPrice = 0
            else:
                cost = self.shortPrice * self.shortPos
                cost += trade.volume * trade.price
                newPos = self.shortPos + trade.volume
                if newPos:
                    self.shortPrice = cost / newPos
                else:
                    self.shortPrice = 0
    
    #----------------------------------------------------------------------
    def calculatePosition(self):
        """计算持仓情况"""
        self.longPos = self.longTd + self.longYd
        self.shortPos = self.shortTd + self.shortYd      
        
    #----------------------------------------------------------------------
    def calculateFrozen(self):
        """计算冻结情况"""
        # 清空冻结数据
        self.longPosFrozen = EMPTY_INT
        self.longYdFrozen = EMPTY_INT
        self.longTdFrozen = EMPTY_INT
        self.shortPosFrozen = EMPTY_INT
        self.shortYdFrozen = EMPTY_INT
        self.shortTdFrozen = EMPTY_INT     
        
        # 遍历统计
        for order in self._dictWorkingOrder.values():
            # 计算剩余冻结量
            frozenVolume = order.totalVolume - order.tradedVolume
            
            # 多头委托
            if order.direction is DIRECTION_LONG:
                # 平今
                if order.offset is OFFSET_CLOSETODAY:
                    self.shortTdFrozen += frozenVolume
                # 平昨
                elif order.offset is OFFSET_CLOSEYESTERDAY:
                    self.shortYdFrozen += frozenVolume
                # 平仓
                elif order.offset is OFFSET_CLOSE:
                    self.shortTdFrozen += frozenVolume
                    
                    if self.shortTdFrozen > self.shortTd:
                        self.shortYdFrozen += (self.shortTdFrozen - self.shortTd)
                        self.shortTdFrozen = self.shortTd
            # 空头委托
            elif order.direction is DIRECTION_SHORT:
                # 平今
                if order.offset is OFFSET_CLOSETODAY:
                    self.longTdFrozen += frozenVolume
                # 平昨
                elif order.offset is OFFSET_CLOSEYESTERDAY:
                    self.longYdFrozen += frozenVolume
                # 平仓
                elif order.offset is OFFSET_CLOSE:
                    self.longTdFrozen += frozenVolume
                    
                    if self.longTdFrozen > self.longTd:
                        self.longYdFrozen += (self.longTdFrozen - self.longTd)
                        self.longTdFrozen = self.longTd
                        
            # 汇总今昨冻结
            self.longPosFrozen = self.longYdFrozen + self.longTdFrozen
            self.shortPosFrozen = self.shortYdFrozen + self.shortTdFrozen
            
    #----------------------------------------------------------------------
    def output(self):
        """"""
        print self.vtSymbol, '-'*30
        print 'long, total:%s, td:%s, yd:%s' %(self.longPos, self.longTd, self.longYd)
        print 'long frozen, total:%s, td:%s, yd:%s' %(self.longPosFrozen, self.longTdFrozen, self.longYdFrozen)
        print 'short, total:%s, td:%s, yd:%s' %(self.shortPos, self.shortTd, self.shortYd)
        print 'short frozen, total:%s, td:%s, yd:%s' %(self.shortPosFrozen, self.shortTdFrozen, self.shortYdFrozen)        
    
    #----------------------------------------------------------------------
    def convertOrderReq(self, req):
        """转换委托请求"""
        # 普通模式无需转换
        if self.mode is self.MODE_NORMAL:
            return [req]
        
        # 上期所模式拆分今昨，优先平今
        elif self.mode is self.MODE_SHFE:
            # 开仓无需转换
            if req.offset is OFFSET_OPEN:
                return [req]
            
            # 多头
            if req.direction is DIRECTION_LONG:
                posAvailable = self.shortPos - self.shortPosFrozen
                tdAvailable = self.shortTd- self.shortTdFrozen
                ydAvailable = self.shortYd - self.shortYdFrozen            
            # 空头
            else:
                posAvailable = self.longPos - self.longPosFrozen
                tdAvailable = self.longTd - self.longTdFrozen
                ydAvailable = self.longYd - self.longYdFrozen
                
            # 平仓量超过总可用，拒绝，返回空列表
            if req.volume > posAvailable:
                return []
            # 平仓量小于今可用，全部平今
            elif req.volume <= tdAvailable:
                req.offset = OFFSET_CLOSETODAY
                return [req]
            # 平仓量大于今可用，平今再平昨
            else:
                l = []
                
                if tdAvailable > 0:
                    reqTd = copy(req)
                    reqTd.offset = OFFSET_CLOSETODAY
                    reqTd.volume = tdAvailable
                    l.append(reqTd)
                    
                reqYd = copy(req)
                reqYd.offset = OFFSET_CLOSEYESTERDAY
                reqYd.volume = req.volume - tdAvailable
                l.append(reqYd)
                
                return l
            
        # 平今惩罚模式，没有今仓则平昨，否则锁仓
        elif self.mode is self.MODE_TDPENALTY:
            # 多头
            if req.direction is DIRECTION_LONG:
                td = self.shortTd
                ydAvailable = self.shortYd - self.shortYdFrozen
            # 空头
            else:
                td = self.longTd
                ydAvailable = self.longYd - self.longYdFrozen
                
            # 这里针对开仓和平仓委托均使用一套逻辑
            
            # 如果有今仓，则只能开仓（或锁仓）
            if td:
                req.offset = OFFSET_OPEN
                return [req]
            # 如果平仓量小于昨可用，全部平昨
            elif req.volume <= ydAvailable:
                if self.exchange is EXCHANGE_SHFE:
                    req.offset = OFFSET_CLOSEYESTERDAY
                else:
                    req.offset = OFFSET_CLOSE
                return [req]
            # 平仓量大于昨可用，平仓再反向开仓
            else:
                l = []
                
                if ydAvailable > 0:
                    reqClose = copy(req)
                    if self.exchange is EXCHANGE_SHFE:
                        reqClose.offset = OFFSET_CLOSEYESTERDAY
                    else:
                        reqClose.offset = OFFSET_CLOSE
                    reqClose.volume = ydAvailable
                    
                    l.append(reqClose)
                    
                reqOpen = copy(req)
                reqOpen.offset = OFFSET_OPEN
                reqOpen.volume = req.volume - ydAvailable
                l.append(reqOpen)
                
                return l
        
        # 其他情况则直接返回空
        return []


