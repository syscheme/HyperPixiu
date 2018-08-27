# encoding: UTF-8

from __future__ import division

import os
import logging
from collections import OrderedDict
from threading import Thread
from datetime import datetime
import time
from copy import copy
from abc import ABCMeta, abstractmethod
import traceback

from .EventChannel import Event, EventLoop, EventChannel, EventData
from .language import text

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

########################################################################
# 常量定义
########################################################################

# 数据库名称
SETTING_DB_NAME = 'vnDB_Setting'
POSITION_DB_NAME = 'vnDB_Position'

TICK_DB_NAME   = 'vnDB_Tick'
DAILY_DB_NAME  = 'vnDB_Daily'
MINUTE_DB_NAME = 'vnDB_1Min'

# 日志级别
LOGLEVEL_DEBUG    = logging.DEBUG
LOGLEVEL_INFO     = logging.INFO
LOGLEVEL_WARN     = logging.WARN
LOGLEVEL_ERROR    = logging.ERROR
LOGLEVEL_CRITICAL = logging.CRITICAL

# 数据库
LOG_DB_NAME = 'vnDB_Log'

import jsoncfg # pip install json-cfg

# def loadSettings(filepath):
#     """读取配置"""
#     try :
#         return jsoncfg.load_config(filepath)
#     except Exception as e :
#         print('failed to load configure[%s] :%s' % (filepath, e))
#         return None

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
        return self.__class__.__name__ + self._id

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
    def init(self): # return True if succ
        return True

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
    def dbEnsureIndex(self, collectionName, definition, unique=False, dbName=None):
        """向MongoDB中插入数据，d是具体数据"""
        if not dbName or len(dbName) <=0:
            dbName = self._id

        if self._engine:
            self._engine.dbEnsureIndex(dbName, collectionName, definition, unique)
                    # db = self._dbConn['Account']
                    # collection = db[tblName]
                    # collection.ensure_index([('vtTradeID', ASCENDING)], unique=True) #TODO this should init ONCE
                    # collection = db[tblName]
                    # collection.update({'vtTradeID':t.vtTradeID}, t.__dict__, True)
    @abstractmethod
    def dbInsert(self, collectionName, d, dbName =None):
        """向MongoDB中插入数据，d是具体数据"""
        if not dbName or len(dbName) <=0:
            dbName = self._id
        if self._engine:
            self._engine.dbInsert(dbName, collectionName, d)
    
    @abstractmethod
    def dbQuery(self, collectionName, d, sortKey='', sortDirection=ASCENDING, dbName =None):
        """从MongoDB中读取数据，d是查询要求，返回的是数据库查询的指针"""
        if not dbName or len(dbName) <=0:
            dbName = self._id
        if self._engine:
            return self._engine.dbQuery(dbName, collectionName, d, sortKey, sortDirection)

        return []
        
    @abstractmethod
    def dbUpdate(self, collectionName, d, flt, upsert=True,  dbName =None):
        """向MongoDB中更新数据，d是具体数据，flt是过滤条件，upsert代表若无是否要插入"""
        if not dbName or len(dbName) <=0:
            dbName = self._id
        if self._engine:
            self._engine.dbUpdate(dbName, collectionName, d, flt, upsert)
            
    #----------------------------------------------------------------------
    @abstractmethod
    def logEvent(self, eventType, content):
        """快速发出日志事件"""
        log = LogData()
        log.dsName = self.ident
        log.logContent = content
        self.postEvent(eventType, log)

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
                    time.sleep(min(2, nextSleep))
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
class Singleton(type):
    """
    单例，应用方式:静态变量 __metaclass__ = Singleton
    """
    
    _instances = {}

    #----------------------------------------------------------------------
    def __call__(cls, *args, **kwargs):
        """调用"""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            
        return cls._instances[cls]

########################################################################
class MainRoutine(object):
    """主引擎"""

    # 单例模式
    __metaclass__ = Singleton

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

        self._bRun = True
        
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
        clsName = dsModule.__class__.__name__
        md = dsModule(self, settings)
        # 保存接口详细信息
        d = {
            'id': md.id,
            'dsDisplayName': settings.displayName(md.id),
            'dsType': clsName,
        }

        self._dictMarketDatas[md.exchange] = md
        self._dlstMarketDatas.append(d)
        self.info('md[%s] added: %s' %(md.exchange, d))

        return md

    #----------------------------------------------------------------------
    def addApp(self, appModule, settings):
        """添加上层应用"""

        app = appModule(self, settings)
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

        return app
        
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
            self.info('started app[%shows]' % k)

        self.info('main-routine started')

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        self._bRun = False

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
        while self._bRun:
            if not self.threadless :
                try :
                    time.sleep(1)
                    self.debug(u'MainThread heartbeat')
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    break

                continue

            # loop mode as below
            if not busy:
                time.sleep(0.5)

            busy = False
            for (k, ds) in self._dictMarketDatas.items():
                try :
                    if ds == None:
                        continue
                    ds.step()
                except Exception as ex:
                    self.error("marketdata step exception %s %s" % (ex, traceback.format_exc()))

            for app in self._dictApps.values() :
                try :
                    app.step()
                except Exception as ex:
                    self.error("app step exception %s %s" % (ex, traceback.format_exc()))

            pending = self._eventChannel.pendingSize
            busy =  pending >0

            pending = min(20, pending)
            for i in range(0, pending) :
                try :
                    self._eventChannel.step()
                    c+=1
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    exit(-1)
                except Exception, ex:
                    self.error("eventCH step exception %s %s" % (ex, traceback.format_exc()))

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
    
    @abstractmethod
    def log(self, level, msg):
        if not level in self._loglevelFunctionDict : 
            return
        
        function = self._loglevelFunctionDict[level] # 获取日志级别对应的处理函数
        function(msg)

    @abstractmethod
    def debug(self, msg):
        """开发时用"""
        if self._logger ==None:
            return

        self._logger.debug(msg)
        
    @abstractmethod
    def info(self, msg):
        """正常输出"""
        if self._logger ==None:
            return

        self._logger.info(msg)

    @abstractmethod
    def warn(self, msg):
        """警告信息"""
        if self._logger ==None:
            return

        self._logger.warn(msg)
        
    @abstractmethod
    def error(self, msg):
        """报错输出"""
        if self._logger ==None:
            return

        self._logger.error(msg)
        
    @abstractmethod
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

    #----------------------------------------------------------------------
    def dbConnect(self):
        """连接MongoDB数据库"""
        if not self._dbConn:
            # 读取MongoDB的设置
            dbhost = self._settings.database.host('localhost')
            dbport = self._settings.database.port(27017)
            if len(dbhost) <=0:
                return

            self.debug('connecting DB[%s :%s]'%(dbhost, dbport))

            try:
                # 设置MongoDB操作的超时时间为0.5秒
                self._dbConn = MongoClient(dbhost, dbport, connectTimeoutMS=500)
                
                # 调用server_info查询服务器状态，防止服务器异常并未连接成功
                self._dbConn.server_info()

                # 如果启动日志记录，则注册日志事件监听函数
                if self._settings.database.logging("") in ['True']:
                    self._eventChannel.register(LogData.EVENT_TAG, self.dbLogging)
                    
                self.info('connecting DB[%s :%s] %s'%(dbhost, dbport, text.DATABASE_CONNECTING_COMPLETED))
            except ConnectionFailure:
                self.error('failed to connect to DB[%s :%s] %s' %(dbhost, dbport, text.DATABASE_CONNECTING_FAILED))
            except:
                self.error('failed to connect to DB[%s :%s]' %(dbhost, dbport))
    
    #----------------------------------------------------------------------
    @abstractmethod
    def dbEnsureIndex(self, dbName, collectionName, definition, unique=False):
        """向MongoDB中定义index"""
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.ensure_index(definition, unique)
        self.debug('dbEnsureIndex() %s.%s added: %s' % (dbName, collectionName, definition))

    @abstractmethod
    def dbInsert(self, dbName, collectionName, d):
        """向MongoDB中插入数据，d是具体数据"""
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        self.debug('dbInsert() %s[%s] adding: %s' % (dbName, collectionName, d))
        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.insert_one(d)
        self.debug('dbInsert() %s[%s] added: %s' % (dbName, collectionName, d))

    @abstractmethod
    def dbQuery(self, dbName, collectionName, d, sortKey='', sortDirection=ASCENDING):
        """从MongoDB中读取数据，d是查询要求，返回的是数据库查询的指针"""
        if not self._dbConn:
            self.error(text.DATA_QUERY_FAILED)   
            return []

        self.debug('dbQuery() %s[%s] flt: %s' % (dbName, collectionName, d))
        db = self._dbConn[dbName]
        collection = db[collectionName]
            
        if sortKey:
            cursor = collection.find(d).sort(sortKey, sortDirection)    # 对查询出来的数据进行排序
        else:
            cursor = collection.find(d)

        if cursor:
            return list(cursor)

        return []
        
    @abstractmethod
    def dbUpdate(self, dbName, collectionName, d, flt, upsert=True):
        """向MongoDB中更新数据，d是具体数据，flt是过滤条件，upsert代表若无是否要插入"""
        if not self._dbConn:
            self.error(text.DATA_UPDATE_FAILED)        
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.replace_one(flt, d, upsert)

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


########################################################################
class ErrorData(EventData):
    """错误数据类"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(ErrorData, self).__init__()
        
        self.errorID = EventData.EMPTY_STRING             # 错误代码
        self.errorMsg = EventData.EMPTY_UNICODE           # 错误信息
        self.additionalInfo = EventData.EMPTY_UNICODE     # 补充信息
        
        self.errorTime = time.strftime('%X', time.localtime())    # 错误生成时间


########################################################################
class LogData(EventData):
    """日志数据类"""
    EVENT_TAG = 'eLog'                      # 日志事件，全局通用

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(LogData, self).__init__()
        
        self.logTime = time.strftime('%X', time.localtime())    # 日志生成时间
        self.logContent = EventData.EMPTY_UNICODE                         # 日志信息
        self.logLevel = LOGLEVEL_INFO                                    # 日志级别

#----------------------------------------------------------------------
def getTempPath(name):
    """获取存放临时文件的路径"""
    tempPath = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)
        
    path = os.path.join(tempPath, name)
    return path
