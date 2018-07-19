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
        
        self.logList = []               # 日志记录
        
        # 当前最新数据，用于模拟成交用
        self.tick = None
        self.bar  = None
        self.dt   = None      # 最新的时间
        self._accountId = ""

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
    def log(self, message):
        """记录日志"""
        log = str(self.dt) + ' ' + message 
        self.logList.append(log)
        print str(datetime.now()) + " ACC[" + self._accountId + "] " + message

    #----------------------------------------------------------------------
    def stdout(self, message):
        """输出内容"""
        print str(datetime.now()) + " ACC[" + self._accountId + "] " + message


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