# encoding: UTF-8

'''
This module represent a basic account
'''
from __future__ import division

from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import multiprocessing
import copy

import jsoncfg # pip install json-cfg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from vnpy.trader.vtGlobal import globalSetting
from vnpy.trader.vtObject import VtTickData, VtBarData
from vnpy.trader.vtConstant import *
from vnpy.trader.vtGateway import VtOrderData, VtTradeData

########################################################################
# 常量定义
########################################################################
from vnpy.trader.app.ctaStrategy import ctaBase

# 涉及到的交易方向类型
ORDER_BUY   = ctaBase.CTAORDER_BUY    # u'买开' 是指投资者对未来价格趋势看涨而采取的交易手段，买进持有看涨合约，意味着帐户资金买进合约而冻结
ORDER_SELL  = ctaBase.CTAORDER_SELL   # u'卖平' 是指投资者对未来价格趋势不看好而采取的交易手段，而将原来买进的看涨合约卖出，投资者资金帐户解冻
ORDER_SHORT = ctaBase.CTAORDER_SHORT  # u'卖开' 是指投资者对未来价格趋势看跌而采取的交易手段，卖出看跌合约。卖出开仓，帐户资金冻结
ORDER_COVER = ctaBase.CTAORDER_COVER  # u'买平' 是指投资者将持有的卖出合约对未来行情不再看跌而补回以前卖出合约，与原来的卖出合约对冲抵消退出市场，帐户资金解冻

# 本地停止单状态
STOPORDER_WAITING   = u'WAITING'   #u'等待中'
STOPORDER_CANCELLED = u'CANCELLED' #u'已撤销'
STOPORDER_TRIGGERED = u'TRIGGERED' #u'已触发'

# 本地停止单前缀
STOPORDERPREFIX = 'vnStopOrder.'

# 数据库名称
SETTING_DB_NAME = 'vnDB_Setting'
POSITION_DB_NAME = 'vnDB_Position'

TICK_DB_NAME   = 'vnDB_Tick'
DAILY_DB_NAME  = 'vnDB_Daily'
MINUTE_DB_NAME = 'vnDB_1Min'

# 引擎类型，用于区分当前策略的运行环境
ENGINETYPE_BACKTESTING = 'backtesting'  # 回测
ENGINETYPE_TRADING = 'trading'          # 实盘

# CTA模块事件
EVENT_LOG      = 'eVNLog'          # 相关的日志事件
EVENT_STRATEGY = 'eVNStrategy.'    # 策略状态变化事件

########################################################################
def loadSettings(filepath):
    """读取配置"""
    try :
        return jsoncfg.load_config(filepath)
    except Exception as e :
        print('failed to load configure[%s] :%s' % (filepath, e))
        return None

    # with open(filepath) as f:
    #     l = json.load(f)
            
    #     for setting in l:
    #         self.loadStrategy(setting)
    # return settings

########################################################################
from abc import ABCMeta, abstractmethod
class Account(object):
    """
    Basic Account
    """
    __lastId__ =10000

    # state of Account
    STATE_OPEN  = 'open'   # during trading hours
    STATE_CLOSE = 'close'  # during market close

    #----------------------------------------------------------------------
    def __init__(self, trader, dvrBrokerClass, settings):
        """Constructor"""

        # the app instance Id
        self._id = settings.id("")
        if len(self._id)<=0 :
            Account.__lastId__ +=1
            self._id = 'A%d' % Account.__lastId__

        self._settings     = settings
        self._trader       = trader
        self._dbConn       = None # TO CLEAN
        self._state        = Account.STATE_CLOSE

        # trader executer
        self._dvrBroker = dvrBrokerClass(self, self._settings.broker)

        self._dateToday      = None # date of previous close
        self._datePrevClose  = None # date of previous close
        self._prevPositions = {} # dict from symbol to previous VtPositionData
        self._dictPositions = {} # dict from symbol to latest VtPositionData
        self._dictTrades = {} # dict from tradeId to trade confirmed during today

        self.capital = 0        # 起始本金（默认10万）
        self._cashAvail =0
        
        # 保存策略实例的字典
        # key为策略名称，value为策略实例，注意策略名称不允许重复
        self._strategyDict = {}

        self.slippage  = self._settings.slippage(0)           # 假设的滑点
        self.rate      = self._settings.ratePer10K(30)/10000  # 假设的佣金比例（适用于百分比佣金）
        self.size      = self._settings.size(1)               # 合约大小，默认为1    
        self._priceTick = self._settings.priceTick(0)      # 价格最小变动 
        
        self.initData = []          # 初始化用的数据
        
        # # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # # value为包含所有相关strategy对象的list
        # self.tickStrategyDict = {}
        
        # # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # # key为vtOrderID，value为strategy对象
        # self.orderStrategyDict = {}     
        
        # # 保存策略名称和委托号列表的字典
        # # key为name，value为保存orderID（限价+本地停止）的集合
        # self.strategyOrderDict = {}

        self._lstLogs = []               # 日志记录

    #----------------------------------------------------------------------
    #  properties
    #----------------------------------------------------------------------
    @property
    def priceTick(self):
        return self._priceTick

    @property
    def dbName(self):
        if not self._trader:
            return 'dbAccount'
        return self._trader.dbName

    #----------------------------------------------------------------------
    @abstractmethod
    def loadSettings(filename) :
        self._settings = jsoncfg.load_config(filename)

    @property
    def ident(self) :
        if not self._dvrBroker:
            return self.__class__.__name__ +"." + self._id
        return self._dvrBroker.__class__.__name__ +"." + self._id

    @abstractmethod
    def cashAmount(self): raise NotImplementedError # returns (avail, total)

    @abstractmethod
    def positionOf(self, vtSymbol): raise NotImplementedError # returns (availVol, totalVol)

    @abstractmethod
    def cancelAll(self, name):
        if self._dvrBroker == None:
            raise NotImplementedError
        self._dvrBroker.cancelAll(name)
        
    @abstractmethod
    def cancelOrder(self, vtOrderID):
        if self._dvrBroker == None:
            raise NotImplementedError
        self._dvrBroker.cancelOrder(vtOrderID)

    @abstractmethod
    def cancelStopOrder(self, stopOrderID): raise NotImplementedError

    @abstractmethod
    def insertData(self, dbName, collectionName, data): raise NotImplementedError

    @abstractmethod
    def putStrategyEvent(self, name):
        """发送策略更新事件，回测中忽略"""
        pass

    @abstractmethod
    def saveSyncData(self, strategy):
        """保存同步数据"""
        pass

    #----------------------------------------------------------------------
    @abstractmethod
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy):
        """发单"""
        if self._dvrBroker == None:
            raise NotImplementedError

        source = 'ACCOUNT'
        if strategy:
            source = strategy.name

        self._dvrBroker.placeOrder(volume, vtSymbol, orderType, price, source)

    #----------------------------------------------------------------------
    @abstractmethod
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy):
        if self._dvrBroker == None:
            raise NotImplementedError

        source = 'ACCOUNT'
        if strategy:
            source = strategy.name

        self._dvrBroker.placeStopOrder(volume, vtSymbol, orderType, price, source)

    @abstractmethod
    def calcAmountOfTrade(self, symbol, price, volume): raise NotImplementedError

    # return volume, commission, slippage
    @abstractmethod
    def maxBuyVolume(self, vtSymbol, price): raise NotImplementedError

    def roundToPriceTick(self, price):
        """取整价格到合约最小价格变动"""
        if not self._priceTick:
            return price
        
        newPrice = round(price/self._priceTick, 0) * self._priceTick
        return newPrice    

    #----------------------------------------------------------------------
    # callbacks about timing
    #----------------------------------------------------------------------
    @abstractmethod
    def onDayClose(self):
        self.saveDB() # save the account data into DB
        
        self._datePrevClose = self._dateToday
        self._dateToday = None
        
        self._state = Account.STATE_CLOSE

    @abstractmethod
    def onDayOpen(self, newDate):
        if Account.STATE_OPEN == self._state:
            if newDate == self._dateToday :
                return
            self.onDayClose()

        self._dateToday = newDate
        self._dictTrades.clear() # clean the trade list
        self._prevPositions = self._dictPositions
        self._state = Account.STATE_OPEN
    
    @abstractmethod
    def onTimer(self):
        # TODO refresh from BrokerDriver
        pass

    #----------------------------------------------------------------------
    # method to access Account DB
    #----------------------------------------------------------------------
    @abstractmethod
    def saveDB(self):
        ''' save the account data into DB 
        1) trades that confirmed
        2) 
        '''

        # TO CLEARN _dbConn
        if not self._trader and not self._dbConn:
            # 设置MongoDB操作的超时时间为0.5秒
            self._dbConn = MongoClient('mongo-vnpy', 27017, connectTimeoutMS=500)
                
            # 调用server_info查询服务器状态，防止服务器异常并未连接成功
            self._dbConn.server_info()

        # part 1. the confirmed trades
        tblName = "trades." + self.ident
        for t in self._dictTrades.values():
            if self._trader:
                self._trader.insertData(self._trader.dbName, tblName, t)
            elif self._dbConn :
                db = self._dbConn['Account']
                collection = db[tblName]
                collection.ensure_index([('vtTradeID', ASCENDING)], unique=True) #TODO this should init ONCE
                collection = db[tblName]
                collection.update({'vtTradeID':t.vtTradeID}, t.__dict__, True)

        # part 2. the daily position
        tblName = "dPos." + self.ident

        result, _ = self.calculateStat()
        for l in result:
            if self._trader:
                self._trader.insertData(self._trader.dbName, tblName, l)
            elif self._dbConn :
                db = self._dbConn['Account']
                collection = db[tblName]
                collection.ensure_index([('date', ASCENDING), ('symbol', ASCENDING)], unique=True) #TODO this should init ONCE
                collection.update({'date':l['date'], 'symbol':l['symbol']}, l, True)

    @abstractmethod
    def loadDB(self, since =None):
        ''' load the account data from DB ''' 
        if not since :
            since = self._datePrevClose
        pass

    @abstractmethod
    def calculateStat(self):
        """今日交易的结果"""

        tradesOfSymbol = {}        # 成交列表
        for t in self._dictTrades.values():
            if len(t.vtSymbol) <=0:
                t.vtSymbol='$' #default symbol

            if not t.vtSymbol in tradesOfSymbol:
                tradesOfSymbol[t.vtSymbol] = []
            tradesOfSymbol[t.vtSymbol].append(t)

        result = []

        for s in tradesOfSymbol.keys():
            if not s in self._dictPositions:
                continue

            currentPos = self._dictPositions[s]

            if s in self._prevPositions:
                prevPos = self._prevPositions[s]
            else:
                prevPos = VtPositionData()
            
            tcBuy      = 0
            tcSell     = 0
            tradingPnl = 0             # 交易盈亏
            totalPnl   = 0               # 总盈亏
            
            turnover   = 0               # 成交量
            commission = 0             # 手续费
            slippage   = 0               # 滑点
            netPnl     = 0                 # 净盈亏
            txnHist    = ""

            # 持仓部分
            # TODO get from currentPos:  openPosition = openPosition
            positionPnl = round(prevPos.position * (currentPos.price - prevPos.price) * self.size, 3)
            """
            计算盈亏
            size: 合约乘数
            rate：手续费率
            slippage：滑点点数
            """
            posChange = 0
            closePosition =0
            for trade in tradesOfSymbol[s]:
                if t.direction == DIRECTION_LONG:
                    posChange = trade.volume
                    tcBuy += 1
                else:
                    posChange = -trade.volume
                    tcSell += 1

                txnHist += "%+dx%s" % (posChange, trade.price)

                tradingPnl += round(posChange * (currentPos.price - trade.price) * self.size, 2)
                closePosition += posChange
                tover, comis, slpfee = self.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
                turnover += tover
                commission += comis
                slippage += slpfee

            # 汇总
            totalPnl = round(tradingPnl + positionPnl, 2)
            netPnl = round(totalPnl - commission - slippage, 2)
            
            dstat = {
                'symbol'      : s,
                'date'        : self._dateToday,   # 日期
                'recentPrice' : currentPos.price,  # 当日收盘
                'prevClose'   : prevPos.price,     # 昨日收盘
                
                'calcPos'     : closePosition,
                'cBuy'        : tcBuy,             # 成交数量
                'cSell'       : tcSell,            # 成交数量
                
                'tradingPnl'  : tradingPnl,        # 交易盈亏
                'positionPnl' : positionPnl,       # 持仓盈亏
                'totalPnl'    : round(tradingPnl + positionPnl, 2), # 总盈亏
                
                'turnover'    : turnover,          # 成交量
                'commission'  : commission,        # 手续费
                'slippage'    : slippage,          # 滑点
                'netPnl'      : netPnl,            # 净盈亏
                
                'txnHist'     : txnHist
                }

            result.append(dstat)

        return result, tradesOfSymbol

    #----------------------------------------------------------------------
    # callbacks from BrokerDriver
    #----------------------------------------------------------------------
    @abstractmethod
    def onCancelOrder(self, data, reqid):
        """撤单回调"""
        pass

    @abstractmethod
    def onBatchCancel(self, data, reqid):
        """批量撤单回调"""
        pass

    @abstractmethod
    def onTrade(self, trade):
        """交易成功回调"""
        self._dictTrades[trade.vtTradeID] = trade

        # update the current postion, this may overwrite during the sync by BrokerDriver
        s = trade.symbol
        if not s in self._dictPositions :
            self._dictPositions[s] = VtPositionData()
            pos = self._dictPositions[s]
            pos.symbol = s
            pos.vtSymbol = trade.vtSymbol
            pos.exchange = trade.exchange

        pos = self._dictPositions[s]
        # 持仓相关
        pos.price      = trade.price
        pos.direction  = trade.direction      # 持仓方向
        # pos.frozen =  # 冻结数量

        if trade.direction == DIRECTION_LONG:
            pos.position += trade.volume
        else:
            pos.position -= trade.volume

    @abstractmethod
    def onTradeError(self, msg, reqid):
        """错误回调"""
        pass

    @abstractmethod
    def onGetAccountBalance(self, data, reqid):
        """查询余额回调"""
        pass
        
    @abstractmethod
    def onGetOrder(self, data, reqid):
        """查询单一委托回调"""
        pass

    def onGetOrders(self, data, reqid):
        """查询委托回调"""
        pass
        
    @abstractmethod
    def onGetMatchResults(self, data, reqid):
        """查询成交回调"""
        pass
        
    @abstractmethod
    def onGetMatchResult(self, data, reqid):
        """查询单一成交回调"""
        pass

    @abstractmethod
    def onPlaceOrder(self, data, reqid):
        """委托回调"""
        pass

    @abstractmethod
    def onStopOrder(self, data, reqid):
        """委托回调"""
        pass

    @abstractmethod
    def onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        pass

    #----------------------------------------------------------------------
    @abstractmethod
    def loadBar(self, dbName, collectionName, startDate):
        """直接返回初始化数据列表中的Bar"""
        return self.initData
    
    @abstractmethod
    def loadTick(self, dbName, collectionName, startDate):
        """直接返回初始化数据列表中的Tick"""
        return self.initData

    #----------------------------------------------------------------------
    @abstractmethod
    def log(self, message):
        """记录日志"""
        self._lstLogs.append(message)
        self.stdout(message)

    @abstractmethod
    def stdout(self, message):
        """输出内容"""
        print str(datetime.now()) + " ACC[" + self.ident + "] " + message

    #----------------------------------------------------------------------
    def loadStrategy(self, setting):
        """载入策略"""
        try:
            name = setting['name']
            className = setting['className']
        except Exception:
            msg = traceback.format_exc()
            self.log(u'错误策略配置：%s' %msg)
            return
        
        # 获取策略类
        strategyClass = STRATEGY_CLASS.get(className, None)
        if not strategyClass:
            self.log(u'找不到策略类：%s' %className)
            return
        
        # 防止策略重名
        if name in self._strategyDict:
            self.log(u'策略实例重名：%s' %name)
        else:
            # 创建策略实例
            strategy = strategyClass(self, setting)  
            self._strategyDict[name] = strategy
            
            # 创建委托号列表
            self.strategyOrderDict[name] = set()
            
            # 保存Tick映射关系
            if strategy.vtSymbol in self.tickStrategyDict:
                l = self.tickStrategyDict[strategy.vtSymbol]
            else:
                l = []
                self.tickStrategyDict[strategy.vtSymbol] = l
            l.append(strategy)
            
    #----------------------------------------------------------------------
    def getStrategyNames(self):
        """查询所有策略名称"""
        return self._strategyDict.keys()        
        
    #----------------------------------------------------------------------
    def getStrategyVar(self, name):
        """获取策略当前的变量字典"""
        if name in self._strategyDict:
            strategy = self._strategyDict[name]
            varDict = OrderedDict()
            
            for key in strategy.varList:
                varDict[key] = strategy.__getattribute__(key)
            
            return varDict
        else:
            self.log(u'策略实例不存在：' + name)    
            return None
    
    #----------------------------------------------------------------------
    def getStrategyParam(self, name):
        """获取策略的参数字典"""
        if name in self._strategyDict:
            strategy = self._strategyDict[name]
            paramDict = OrderedDict()
            
            for key in strategy.paramList:  
                paramDict[key] = strategy.__getattribute__(key)
            
            return paramDict
        else:
            self.log(u'策略实例不存在：' + name)    
            return None
    
    #----------------------------------------------------------------------
    def initStrategy(self, name):
        """初始化策略"""
        if not name in self._strategyDict:
            strategy = self._strategyDict[name]
            self.log(u'策略实例不存在：%s' %name)
            return
            
        if strategy and strategy.inited:
            self.log(u'请勿重复初始化策略实例：%s' %name)
            return

        strategy.inited = True
        self.callStrategyFunc(strategy, strategy.onInit)
        self.loadSyncData(strategy)                             # 初始化完成后加载同步数据
        self.subscribeMarketData(strategy)                      # 加载同步数据后再订阅行情

    #---------------------------------------------------------------------
    def startStrategy(self, name):
        """启动策略"""
        if name in self._strategyDict:
            strategy = self._strategyDict[name]
            
            if strategy.inited and not strategy.trading:
                strategy.trading = True
                self.callStrategyFunc(strategy, strategy.onStart)
        else:
            self.log(u'策略实例不存在：%s' %name)
    
    #----------------------------------------------------------------------
    def stopStrategy(self, name):
        """停止策略"""
        if name in self._strategyDict:
            strategy = self._strategyDict[name]
            
            if strategy.trading:
                strategy.trading = False
                self.callStrategyFunc(strategy, strategy.onStop)
                
                # 对该策略发出的所有限价单进行撤单
                for vtOrderID, s in self.orderStrategyDict.items():
                    if s is strategy:
                        self.cancelOrder(vtOrderID)
                
                # 对该策略发出的所有本地停止单撤单
                for stopOrderID, so in self.workingStopOrderDict.items():
                    if so.strategy is strategy:
                        self.cancelStopOrder(stopOrderID)   
        else:
            self.log(u'策略实例不存在：%s' %name)    
            
    #----------------------------------------------------------------------
    def callStrategyFunc(self, strategy, func, params=None):
        """调用策略的函数，若触发异常则捕捉"""
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            # 停止策略，修改状态为未初始化
            strategy.trading = False
            strategy.inited = False
            
            # 发出日志
            content = '\n'.join([u'策略%s触发异常已停止' %strategy.name,
                                traceback.format_exc()])
            self.log(content)
            
    #----------------------------------------------------------------------
    def initAll(self):
        """全部初始化"""
        for name in self._strategyDict.keys():
            self.initStrategy(name)    
            
    #----------------------------------------------------------------------
    def startAll(self):
        """全部启动"""
        for name in self._strategyDict.keys():
            self.startStrategy(name)
            
    #----------------------------------------------------------------------
    def stop(self):
        """停止"""
        pass

    #----------------------------------------------------------------------
    def stopAll(self):
        """全部停止"""
        for name in self._strategyDict.keys():
            self.stopStrategy(name)    
    
    #----------------------------------------------------------------------
    def saveSetting(self):
        """保存策略配置"""
        with open(self.settingfilePath, 'w') as f:
            l = []
            
            for strategy in self._strategyDict.values():
                setting = {}
                for param in strategy.paramList:
                    setting[param] = strategy.__getattribute__(param)
                l.append(setting)
            
            jsonL = json.dumps(l, indent=4)
            f.write(jsonL)
    
    #----------------------------------------------------------------------
    def roundToPriceTick(self, price):
        """取整价格到合约最小价格变动"""
        if not self._priceTick:
            return price
        
        newPrice = round(price/self._priceTick, 0) * self._priceTick
        return newPrice

    #----------------------------------------------------------------------
    #  account daily statistics methods
    #----------------------------------------------------------------------
    def updateDailyStat(self, dt, price):
        """更新每日收盘价"""
        date = dt.date()
        self._statDaily = DailyResult(date, price)

        # 将成交添加到每日交易结果中
        for trade in self._dvrBroker.tradeDict.values():
            self._statDaily.addTrade(trade)
            
    def evaluateDailyStat(self, startdate, enddate):
        previousClose = 0
        openPosition = 0
        # TODO: read the statDaily from the DB
        # for dailyResult in self.dailyResultDict.values():
        #     dailyResult.previousClose = previousClose
        #     previousClose = dailyResult.closePrice
            
        #     dailyResult.calculatePnl(self._account, openPosition)
        #     openPosition = dailyResult.closePosition
            
        # 生成DataFrame
        resultDict ={}
        for k in dailyResult.__dict__.keys() :
            if k == 'tradeList' : # to exclude some columns
                continue
            resultDict[k] =[]

        for dailyResult in self.dailyResultDict.values():
            for k, v in dailyResult.__dict__.items() :
                if k in resultDict :
                    resultDict[k].append(v)
                
        resultDf = pd.DataFrame.from_dict(resultDict)
        
        # 计算衍生数据
        resultDf = resultDf.set_index('date')

        return resultDf







########################################################################
class StopOrder(object):
    """本地停止单"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.vtSymbol = EMPTY_STRING
        self.orderType = EMPTY_UNICODE
        self.direction = EMPTY_UNICODE
        self.offset = EMPTY_UNICODE
        self.price = EMPTY_FLOAT
        self.volume = EMPTY_INT
        
        self.strategy = None             # 下停止单的策略对象
        self.stopOrderID = EMPTY_STRING  # 停止单的本地编号 
        self.status = EMPTY_STRING       # 停止单状态

########################################################################
class Account_AShare(Account):
    """
    回测Account
    函数接口和策略引擎保持一样，
    从而实现同一套代码从回测到实盘。
    """

    #----------------------------------------------------------------------
    def __init__(self, trader, dvrBrokerClass, settings=None):
        """Constructor"""
        super(Account_AShare, self).__init__(trader, dvrBrokerClass, settings)

    #----------------------------------------------------------------------
    def cashAmount(self): # returns (avail, total)
        return (self._cashAvail, 0)

    def positionOf(self, vtSymbol): # returns (availVol, totalVol)
        return (0, 0)

    #----------------------------------------------------------------------
    def calcAmountOfTrade(self, symbol, price, volume):
        # 交易手续费=印花税+过户费+券商交易佣金
        volumeX1 = abs(volume) * self.size
        turnOver = price * volumeX1

        # 印花税: 成交金额的1‰ 。目前向卖方单边征收
        tax = 0
        if volumeX1 <0:
            tax = turnOver /1000
            
        #过户费（仅上海收取，也就是买卖上海股票时才有）：每1000股收取1元，不足1000股按1元收取
        transfer =0
        if len(symbol)>2 and (symbol[1]=='6' or symbol[1]=='7'):
            transfer = int((volumeX1+999)/1000)
            
        #3.券商交易佣金 最高为成交金额的3‰，最低5元起，单笔交易佣金不满5元按5元收取。
        commission = max(turnOver * self.rate, 5)

        return turnOver, tax + transfer + commission, volumeX1 * self.slippage

    #----------------------------------------------------------------------
    # determine buy ability according to the available cash
    # return volume, commission, slippage
    def maxBuyVolume(self, vtSymbol, price):
        if price <=0 :
            return 0, 0, 0

        cash, _  = self.cashAmount()
        volume   = int(cash / price / self.size)
        turnOver, commission, slippage = self.calcAmountOfTrade(vtSymbol, price, volume)
        if cash >= (turnOver + commission + slippage) :
            return volume, commission, slippage

        volume -= int((commission + slippage) / price / self.size) +1
        if volume <=0:
            return 0, 0, 0

        turnOver, commission, slippage = self.calcAmountOfTrade(vtSymbol, price, volume)
        return volume, commission, slippage


########################################################################
class VtPositionData(object): # (VtBaseData):
    """持仓数据类"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(VtPositionData, self).__init__()
        
        # 代码编号相关
        self.symbol   = EMPTY_STRING            # 合约代码
        self.exchange = EMPTY_STRING            # 交易所代码
        self.vtSymbol = EMPTY_STRING            # 合约在vt系统中的唯一代码，合约代码.交易所代码  
        
        # 持仓相关
        self.direction      = EMPTY_STRING      # 持仓方向
        self.position       = EMPTY_INT         # 持仓量
        self.frozen         = EMPTY_INT         # 冻结数量
        self.price          = EMPTY_FLOAT       # 持仓均价
        self.vtPositionName = EMPTY_STRING      # 持仓在vt系统中的唯一代码，通常是vtSymbol.方向
        # self.ydPosition     = EMPTY_INT         # 昨持仓
        # self.positionProfit = EMPTY_FLOAT       # 持仓盈亏

