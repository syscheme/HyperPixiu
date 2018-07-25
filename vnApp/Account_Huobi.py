# encoding: UTF-8

'''
本文件中包含的是vnApp模块的回测引擎，回测引擎的API和CTA引擎一致，
可以使用和实盘相同的代码进行回测。
'''
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

from vnApp.Account import *

########################################################################
class Account_Huobi(Account):
    """
    回测Account
    函数接口和策略引擎保持一样，
    从而实现同一套代码从回测到实盘。
    """
    
    TICK_MODE = 'tick'
    BAR_MODE = 'bar'

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

        super(BTAccount_AShare, self).__init__()

        self.clearBackTesting()
        self.clearResult()

        # 回测相关属性
        # -----------------------------------------
        self.engineType = ENGINETYPE_BACKTESTING    # 引擎类型为回测
        self.mode = self.BAR_MODE   # 回测模式，默认为K线
        
        self.strategyBT = ""       # name of 回测策略
        
        self.startDate = ''
        self.initDays = 0        
        self.endDate = ''

        self.dbClient = None        # 数据库客户端
        self.dbCursor = None        # 数据库指针
        
        self.initData = []          # 初始化用的数据
        self.dbName = ''            # 回测数据库名
        self.symbol = ''            # 回测集合名
        
        self.dataStartDate = None       # 回测数据开始日期，datetime对象
        self.dataEndDate = None         # 回测数据结束日期，datetime对象
        self.strategyStartDate = None   # 策略启动日期（即前面的数据用于初始化），datetime对象
        
        # 当前最新数据，用于模拟成交用
        self.tick = None
        self.bar  = None
        self.dt   = None      # 最新的时间

        # 日线回测结果计算用
        self.dailyResultDict = OrderedDict()
    
