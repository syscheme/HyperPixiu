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

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

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
from abc import ABCMeta, abstractmethod
class Account(object):
    """
    Basic Account
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        self._accountId = ""
        self._thisTradeDate = None
        self._lastTradeDate = None

        self.capital = 0        # 起始本金（默认10万）
        self.slippage = 0       # 假设的滑点
        self.rate = 30/10000    # 假设的佣金比例（适用于百分比佣金）
        self.size = 1           # 合约大小，默认为1    
        self.priceTick = 0      # 价格最小变动 
        
        self.limitOrderCount = 0                    # 限价单编号
        self.limitOrderDict = OrderedDict()         # 限价单字典
        self.workingLimitOrderDict = OrderedDict()  # 活动限价单字典，用于进行撮合用
        
        # 本地停止单字典, key为stopOrderID，value为stopOrder对象
        self.stopOrderDict = {}             # 停止单撤销后不会从本字典中删除
        self.workingStopOrderDict = {}      # 停止单撤销后会从本字典中删除
        # 本地停止单编号计数
        self.stopOrderCount = 0
        # stopOrderID = STOPORDERPREFIX + str(stopOrderCount)

        self.tradeCount = 0             # 成交编号
        self.tradeDict = OrderedDict()  # 成交字典
        
        # 保存策略实例的字典
        # key为策略名称，value为策略实例，注意策略名称不允许重复
        self.strategyDict = {}
        
        # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # value为包含所有相关strategy对象的list
        self.tickStrategyDict = {}
        
        # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # key为vtOrderID，value为strategy对象
        self.orderStrategyDict = {}     
        
        # 保存策略名称和委托号列表的字典
        # key为name，value为保存orderID（限价+本地停止）的集合
        self.strategyOrderDict = {}

        self.logList = []               # 日志记录

    @abstractmethod
    def cashAmount(self): raise NotImplementedError # returns (avail, total)

    @abstractmethod
    def positionOf(self, vtSymbol): raise NotImplementedError # returns (availVol, totalVol)

    @abstractmethod
    def cancelAll(self, name): raise NotImplementedError
        
    @abstractmethod
    def cancelOrder(self, vtOrderID): raise NotImplementedError

    @abstractmethod
    def cancelStopOrder(self, stopOrderID): raise NotImplementedError

    @abstractmethod
    def getPriceTick(self, strategy): raise NotImplementedError

    @abstractmethod
    def insertData(self, dbName, collectionName, data): raise NotImplementedError

    @abstractmethod
    def putStrategyEvent(self, name): raise NotImplementedError

    @abstractmethod
    def saveSyncData(self, strategy): raise NotImplementedError

    @abstractmethod
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy): raise NotImplementedError

    @abstractmethod
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy): raise NotImplementedError

    @abstractmethod
    def calcAmountOfTrade(self, symbol, price, volume): raise NotImplementedError

    @abstractmethod
    def onDayOpen(self, newDate): raise NotImplementedError

    @abstractmethod
    def log(self, message):
        """记录日志"""
        log = str(self.dt) + ' ' + message 
        self.logList.append(log)
        print str(datetime.now()) + " ACC[" + self._accountId + "] " + message

    #----------------------------------------------------------------------
    def stdout(self, message):
        """输出内容"""
        print str(datetime.now()) + " ACC[" + self._accountId + "] " + message

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
        if name in self.strategyDict:
            self.log(u'策略实例重名：%s' %name)
        else:
            # 创建策略实例
            strategy = strategyClass(self, setting)  
            self.strategyDict[name] = strategy
            
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
        return self.strategyDict.keys()        
        
    #----------------------------------------------------------------------
    def getStrategyVar(self, name):
        """获取策略当前的变量字典"""
        if name in self.strategyDict:
            strategy = self.strategyDict[name]
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
        if name in self.strategyDict:
            strategy = self.strategyDict[name]
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
        if not name in self.strategyDict:
            strategy = self.strategyDict[name]
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
        if name in self.strategyDict:
            strategy = self.strategyDict[name]
            
            if strategy.inited and not strategy.trading:
                strategy.trading = True
                self.callStrategyFunc(strategy, strategy.onStart)
        else:
            self.log(u'策略实例不存在：%s' %name)
    
    #----------------------------------------------------------------------
    def stopStrategy(self, name):
        """停止策略"""
        if name in self.strategyDict:
            strategy = self.strategyDict[name]
            
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
        for name in self.strategyDict.keys():
            self.initStrategy(name)    
            
    #----------------------------------------------------------------------
    def startAll(self):
        """全部启动"""
        for name in self.strategyDict.keys():
            self.startStrategy(name)
            
    #----------------------------------------------------------------------
    def stop(self):
        """停止"""
        pass

    #----------------------------------------------------------------------
    def stopAll(self):
        """全部停止"""
        for name in self.strategyDict.keys():
            self.stopStrategy(name)    
    
    #----------------------------------------------------------------------
    def saveSetting(self):
        """保存策略配置"""
        with open(self.settingfilePath, 'w') as f:
            l = []
            
            for strategy in self.strategyDict.values():
                setting = {}
                for param in strategy.paramList:
                    setting[param] = strategy.__getattribute__(param)
                l.append(setting)
            
            jsonL = json.dumps(l, indent=4)
            f.write(jsonL)
    
    #----------------------------------------------------------------------
    def loadSetting(self):
        """读取策略配置"""
        with open(self.settingfilePath) as f:
            l = json.load(f)
            
            for setting in l:
                self.loadStrategy(setting)
    

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
    def __init__(self):
        """Constructor"""
        super(Account_AShare, self).__init__()

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
        volume   = cash / price / self.size
        turnOver, commission, slippage = self.calcAmountOfTrade(vtSymbol, price, volume)
        if cash >= (turnOver + commission + slippage) :
            return volume, commission, slippage

        volume -= round((commission + slippage) / price / self.size +1)
        if volume <=0:
            return 0, 0, 0

        turnOver, commission, slippage = self.calcAmountOfTrade(vtSymbol, price, volume)
        return volume, commission, slippage

    #----------------------------------------------------------------------
    @abstractmethod
    def onDayOpen(self, newDate):
        self._lastTradeDate =self._thisTradeDate
        self._thisTradeDate =newDate
