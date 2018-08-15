# encoding: UTF-8

'''
本文件中包含的是vnApp模块的回测引擎，回测引擎的API和CTA引擎一致，
可以使用和实盘相同的代码进行回测。
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
from vnpy.trader.vtObject import VtTickData, KLineData
from vnpy.trader.vtConstant import *
from vnpy.trader.vtGateway import VtOrderData, VtTradeData

from vnApp.Account import *

########################################################################
class BTAccount_AShare(Account_AShare):
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

        self._dbConn = None        # 数据库客户端
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
    
    #------------------------------------------------
    # 通用功能
    #------------------------------------------------    
    @property
    def strategy(self):
        return self._strategyDict[self.strategyName]

    #----------------------------------------------------------------------
    def clearResult(self):
        self.resultList = []             # 交易结果列表
        self.posList = [0]               # 每笔成交后的持仓情况        
        self.tradeTimeList = []          # 每笔成交时间戳
    
    #----------------------------------------------------------------------
    def roundToPriceTick(self, price):
        """取整价格到合约最小价格变动"""
        if not self.priceTick:
            return price
        
        newPrice = round(price/self.priceTick, 0) * self.priceTick
        return newPrice

    #------------------------------------------------
    # 参数设置相关
    #------------------------------------------------
    
    #----------------------------------------------------------------------
    def setStartDate(self, startDate='20100416', initDays=10):
        """设置回测的启动日期"""
        self.startDate = startDate
        self.initDays = initDays
        
        self.dataStartDate = datetime.strptime(startDate, '%Y%m%d')
        
        initTimeDelta = timedelta(initDays)
        self.strategyStartDate = self.dataStartDate + initTimeDelta
        
    #----------------------------------------------------------------------
    def setEndDate(self, endDate=''):
        """设置回测的结束日期"""
        self.endDate = endDate
        
        if endDate:
            self.dataEndDate = datetime.strptime(endDate, '%Y%m%d')
            
            # 若不修改时间则会导致不包含dataEndDate当天数据
            self.dataEndDate = self.dataEndDate.replace(hour=23, minute=59)    
        
    #----------------------------------------------------------------------
    def setBacktestingMode(self, mode):
        """设置回测模式"""
        self.mode = mode
    
    #----------------------------------------------------------------------
    def setDatabase(self, dbName, symbol):
        """设置历史数据所用的数据库"""
        self.dbName = dbName
        self.symbol = symbol
    
    #----------------------------------------------------------------------
    def setSlippage(self, slippage):
        """设置滑点点数"""
        self.slippage = slippage
        
    #----------------------------------------------------------------------
    def setSize(self, size):
        """设置合约大小"""
        self.size = size
        
    #----------------------------------------------------------------------
    def setRate(self, rate):
        """设置佣金比例"""
        self.rate = rate
        
    #----------------------------------------------------------------------
    def setPriceTick(self, priceTick):
        """设置价格最小变动"""
        self.priceTick = priceTick
    
    #------------------------------------------------
    # 数据回放相关
    #------------------------------------------------    
    #----------------------------------------------------------------------
    def stdout(self, message):
        """输出内容"""
        self._account.stdout(message)
    
    #----------------------------------------------------------------------
    def loadHistoryData(self):
        """载入历史数据"""
        self._dbConn = pymongo.MongoClient(globalSetting['mongoHost'], globalSetting['mongoPort'])
        collection = self._dbConn[self.dbName][self.symbol]          

        self.stdout(u'开始载入数据')
      
        # 首先根据回测模式，确认要使用的数据类
        if self.mode == self.BAR_MODE:
            dataClass = KLineData
            func = self.OnNewBar
        else:
            dataClass = VtTickData
            func = self.OnNewTick

        # 载入初始化需要用的数据
        flt = {'datetime':{'$gte':self.dataStartDate,
                           '$lt':self.strategyStartDate}}        
        initCursor = collection.find(flt).sort('datetime')
        
        # 将数据从查询指针中读取出，并生成列表
        self.initData = []              # 清空initData列表
        for d in initCursor:
            data = dataClass()
            data.__dict__ = d
            self.initData.append(data)      
        
        # 载入回测数据
        if not self.dataEndDate:
            flt = {'datetime':{'$gte':self.strategyStartDate}}   # 数据过滤条件
        else:
            flt = {'datetime':{'$gte':self.strategyStartDate,
                               '$lte':self.dataEndDate}}  
        self.dbCursor = collection.find(flt).sort('datetime')
        
        cRows = initCursor.count() + self.dbCursor.count()
        self.stdout(u'载入完成，数据量：%s' %cRows)
        return cRows
        
    #----------------------------------------------------------------------
    def runBacktesting(self):
        """运行回测"""

        self._id = "BT_%s.%s" % (self.symbol, self.strategyName)

        # 载入历史数据
        if self.loadHistoryData() <=0 :
            self.stdout(u'Quit testing due to empty data')
            return
        
        # 首先根据回测模式，确认要使用的数据类
        if self.mode == self.BAR_MODE:
            dataClass = KLineData
            func = self.OnNewBar
        else:
            dataClass = VtTickData
            func = self.OnNewTick

        self.setCapital(self.capital, True)

        self.stdout(u'开始回测')
        
        self.strategy.inited = True
        self.strategy.onInit()
        self.stdout(u'策略初始化完成')
        
        self.strategy.trading = True
        self.strategy.onStart()
        self.stdout(u'策略启动完成')
        
        self.stdout(u'开始回放数据')

        for d in self.dbCursor:
            data = dataClass()
            data.__dict__ = d
            func(data)     
            
        if self.mode == self.BAR_MODE:
            self._execEndClose = self.bar.close
            self._execEnd      = self.bar.date
        else:
            self._execEndClose = self.tick.priceTick
            self._execEnd      = self.tick.date

        self.stdout(u'数据回放结束')
        
    #----------------------------------------------------------------------
    def onDayOpen(self, newDate):
        super(BTAccount_AShare, self).onDayOpen(newDate)

        self.strategy._posAvail = self.strategy.pos
        if self.strategy._posAvail > 400:
            exit
        self.strategy.onDayOpen(newDate)

    #----------------------------------------------------------------------
    def OnNewBar(self, bar):
        """新的K线"""

        # shift the trade date and notify dayOpen if date changes
        if self._dateToday != bar.date :
            self.onDayOpen(bar.date)

        if self.bar ==None:
            self._execStartClose = bar.close
            self._execStart = bar.date

        self.bar = bar
        self.dt  = bar.datetime
        
        self.crossLimitOrder()      # 先撮合限价单
        self.crossStopOrder()       # 再撮合停止单
        self.strategy.onBar(bar)    # 推送K线到策略中
        
        self.updateDailyClose(self.dt, bar.close)
    
    #----------------------------------------------------------------------
    def OnNewTick(self, tick):
        """新的Tick"""

        # shift the trade date and notify dayOpen if date changes
        if self._account._dateToday != tick.date :
            self._account._datePrevClose =self._account._dateToday
            self._account._dateToday =tick.date
            self.strategy.onDayOpen(tick.date)

        if self.tick ==None:
            self._execStartClose = tick.priceTick
            self._execStart = tick.date

        self.tick = tick
        self.dt = tick.datetime
        
        self.crossLimitOrder()
        self.crossStopOrder()
        self.strategy.onTick(tick)
        
        self.updateDailyClose(self.dt, tick.lastPrice)
        
    #----------------------------------------------------------------------
    def initStrategy(self, strategyClass, setting=None):
        """
        初始化策略
        setting是策略的参数设置，如果使用类中写好的默认设置则可以不传该参数
        """
        strategy = strategyClass(self, setting)  
        self.strategyName = strategy.className
        self._strategyDict[self.strategyName] = strategy
    
    #----------------------------------------------------------------------
    def bestBarCrossPrice(self, bar): 
        return round(((bar.open + bar.close) *4 + bar.high + bar.low) /10, 2)

    #----------------------------------------------------------------------
    def crossLimitOrder(self):
        """基于最新数据撮合限价单
        A limit order is an order placed with a brokerage to execute a buy or 
        sell transaction at a set number of shares and at a specified limit
        price or better. It is a take-profit order placed with a bank or
        brokerage to buy or sell a set amount of a financial instrument at a
        specified price or better; because a limit order is not a market order,
        it may not be executed if the price set by the investor cannot be met
        during the period of time in which the order is left open.
        """
        # 先确定会撮合成交的价格
        if self.mode == self.BAR_MODE:
            buyCrossPrice = self.bar.low        # 若买入方向限价单价格高于该价格，则会成交
            sellCrossPrice = self.bar.high      # 若卖出方向限价单价格低于该价格，则会成交
            buyBestCrossPrice = self.bestBarCrossPrice(self.bar)   # 在当前时间点前发出的买入委托可能的最优成交价
            sellBestCrossPrice = self.bestBarCrossPrice(self.bar)  # 在当前时间点前发出的卖出委托可能的最优成交价
            
            # 张跌停封板
            if buyCrossPrice <=self.bar.open*0.9 :
                buyCrossPrice =0
            if sellCrossPrice >= self.bar.open*1.1 :
                sellCrossPrice =0
        else:
            buyCrossPrice = self.tick.askPrice1
            sellCrossPrice = self.tick.bidPrice1
            buyBestCrossPrice = self.tick.askPrice1
            sellBestCrossPrice = self.tick.bidPrice1
        
        # 遍历限价单字典中的所有限价单
        for orderID, order in self.workingLimitOrderDict.items():
            # 推送委托进入队列（未成交）的状态更新
            if not order.status:
                order.status = OrderData.STATUS_NOTTRADED
                self.strategy.onOrder(order)

            # 判断是否会成交
            buyCross = OrderData.DIRECTION_LONG and 
                        order.price>=buyCrossPrice and
                        buyCrossPrice > 0)      # 国内的tick行情在涨停时askPrice1为0，此时买无法成交
            
            sellCross = OrderData.DIRECTION_SHORT and 
                         order.price<=sellCrossPrice and
                         sellCrossPrice > 0)    # 国内的tick行情在跌停时bidPrice1为0，此时卖无法成交
            
            # 如果发生了成交
            if not buyCross and not sellCross:
                continue

            # 推送成交数据
            self.tradeCount += 1            # 成交编号自增1
            tradeID = str(self.tradeCount)
            trade = VtTradeData()
            trade.vtSymbol = order.vtSymbol
            trade.tradeID = tradeID
            trade.vtTradeID = tradeID
            trade.orderID = order.orderID
            trade.vtOrderID = order.orderID
            trade.direction = order.direction
            trade.offset = order.offset
            trade.symbol = order.symbol
            
            # 以买入为例：
            # 1. 假设当根K线的OHLC分别为：100, 125, 90, 110
            # 2. 假设在上一根K线结束(也是当前K线开始)的时刻，策略发出的委托为限价105
            # 3. 则在实际中的成交价会是100而不是105，因为委托发出时市场的最优价格是100
            trade.volume = 0
            tradeAmount =0
            (turnoverO, commissionO, slippageO) =(0,0,0)
            (turnoverT, commissionT, slippageT) =(0,0,0)

            cashAval, _= self.cashAmount()
            if buyCross:
                turnoverO, commissionO, slippageO = self.calcAmountOfTrade(order.symbol, order.price, order.totalVolume)
                trade.volume = order.totalVolume
                if (turnoverO + commissionO + slippageO) > cashAval : # the volume should depends on available cache
                    trade.volume =0

                trade.price = min(order.price, buyBestCrossPrice)
                self.strategy.pos += trade.volume
            elif self.strategy.pos >0:
                turnoverO, commissionO, slippageO = self.calcAmountOfTrade(order.symbol, order.price, -order.totalVolume)
                orderVolume = -order.totalVolume
                trade.volume = min(self.strategy.pos, order.totalVolume)
                trade.price = max(order.price, sellBestCrossPrice)
                self.strategy.pos -= trade.volume
            
            if trade.volume >0:
                tvolume = trade.volume
                if not buyCross:
                    tvolume = -trade.volume
                
                turnoverT, commissionT, slippageT = self.calcAmountOfTrade(trade.symbol, trade.price, tvolume)
                if buyCross:
                    tradeAmount = turnoverT + commissionT + slippageT
                else :
                    tradeAmount = turnoverT - commissionT - slippageT

                trade.tradeTime = self.dt.strftime('%H:%M:%S')
                trade.dt = self.dt
                self.strategy.onTrade(trade)
            
                self.tradeDict[tradeID] = trade
                self.log('limCrossed:pos(%s) %s[%s,%s,%s=%dx%s] per order:%s[%sx%s], cash[%s/%s]' %
                    (self.strategy.pos, tradeID, trade.direction, trade.vtSymbol, tradeAmount, trade.volume, trade.price, orderID, order.totalVolume, order.price, self._cashAvail, 'na'))
            
                # 推送委托数据
                order.tradedVolume = trade.tradeTime
                order.status = OrderData.STATUS_ALLTRADED
                if order.tradedVolume < order.totalVolume :
                    order.status = OrderData.STATUS_PARTTRADED
                self.strategy.onOrder(order)
            else :
                # 推送委托数据
                order.tradedVolume = 0
                order.status = OrderData.STATUS_CANCELLED
                self.strategy.onOrder(order)

            # update avail cache
            if buyCross: #this was a buy
                # self._cashAvail += turnoverO + commissionO + slippageO
                # self._cashAvail -= tradeAmount
                dCacheAval = (turnoverO + commissionO + slippageO) -tradeAmount
                self.cashChange(dCacheAval)
            else :
                # self._cashAvail += tradeAmount
                self.cashChange(tradeAmount)
            
            # 从字典中删除该限价单
            if orderID in self.workingLimitOrderDict:
                del self.workingLimitOrderDict[orderID]
                
    #----------------------------------------------------------------------
    def crossStopOrder(self): 
        """基于最新数据撮合停止单
            A stop order is an order to buy or sell a security when its price moves past
            a particular point, ensuring a higher probability of achieving a predetermined 
            entry or exit price, limiting the investor's loss or locking in a profit. Once 
            the price crosses the predefined entry/exit point, the stop order becomes a
            market order.
        """
        # 先确定会撮合成交的价格，这里和限价单规则相反
        if self.mode == self.BAR_MODE:
            buyCrossPrice  = self.bar.high   # 若买入方向停止单价格低于该价格，则会成交
            sellCrossPrice = self.bar.low    # 若卖出方向限价单价格高于该价格，则会成交
            bestCrossPrice = self.bestBarCrossPrice(self.bar)  # 最优成交价，买入停止单不能低于，卖出停止单不能高于
            maxVolumeCross = self.bar.volume
        else:
            buyCrossPrice  = self.tick.lastPrice
            sellCrossPrice = self.tick.lastPrice
            bestCrossPrice = self.tick.lastPrice
            topVolumeCross = self.tick.volume
        
        # 遍历停止单字典中的所有停止单
        for stopOrderID, so in self.workingStopOrderDict.items():
            # 判断是否会成交
            buyCross  = OrderData.DIRECTION_LONG)  and so.price<=buyCrossPrice
            sellCross = OrderData.DIRECTION_SHORT) and so.price>=sellCrossPrice
            
            # 忽略未发生成交
            if not buyCross and not sellCross : # and (so.volume < maxVolumeCross):
                continue;

            # 更新停止单状态，并从字典中删除该停止单
            so.status = OrderData.ORDER_TRIGGERED
            if stopOrderID in self.workingStopOrderDict:
                del self.workingStopOrderDict[stopOrderID]                        

            # 推送成交数据
            self.tradeCount += 1            # 成交编号自增1
            tradeID = str(self.tradeCount)
            trade = VtTradeData()
            trade.vtSymbol = so.vtSymbol
            trade.tradeID = tradeID
            trade.vtTradeID = tradeID
                
            if buyCross:
                self.strategy.pos += so.volume
                trade.price = max(bestCrossPrice, so.price)
            else:
                self.strategy.pos -= so.volume
                trade.price = min(bestCrossPrice, so.price)                
                
            self.limitOrderCount += 1
            orderID = str(self.limitOrderCount)
            trade.orderID = orderID
            trade.vtOrderID = orderID
            trade.direction = so.direction
            trade.offset = so.offset
            trade.volume = so.volume
            trade.tradeTime = self.dt.strftime('%H:%M:%S')
            trade.dt = self.dt
                
            self.tradeDict[tradeID] = trade
                
            # 推送委托数据
            order = VtOrderData()
            order.vtSymbol = so.vtSymbol
            order.symbol = so.vtSymbol
            order.orderID = orderID
            order.vtOrderID = orderID
            order.direction = so.direction
            order.offset = so.offset
            order.price = so.price
            order.totalVolume = so.volume
            order.tradedVolume = so.volume
            order.status = OrderData.STATUS_ALLTRADED
            order.orderTime = trade.tradeTime
                
            self.limitOrderDict[orderID] = order
                
            # 按照顺序推送数据
            self.strategy.onStopOrder(so)
            self.strategy.onOrder(order)
            self.strategy.onTrade(trade)
    
    #------------------------------------------------
    # 策略接口相关
    #------------------------------------------------      

    #----------------------------------------------------------------------
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy):
        """发单"""
        self.limitOrderCount += 1
        orderID = str(self.limitOrderCount)
        
        order = VtOrderData()
        order.vtSymbol = vtSymbol
        order.price = self.roundToPriceTick(price)
        order.totalVolume = volume
        order.orderID = orderID
        order.vtOrderID = orderID
        order.orderTime = self.dt.strftime('%H:%M:%S')
        
        # 委托类型映射
        if orderType = OrderData.ORDER_BUY:
            order.direction = OrderData.DIRECTION_LONG
            order.offset = OrderData.OFFSET_OPEN
        elif orderType = OrderData.ORDER_SELL:
            order.direction = OrderData.DIRECTION_SHORT
            order.offset = OrderData.OFFSET_CLOSE
        elif orderType = OrderData.ORDER_SHORT:
            order.direction = OrderData.DIRECTION_SHORT
            order.offset = OrderData.OFFSET_OPEN
        elif orderType = OrderData.ORDER_COVER:
            order.direction = OrderData.DIRECTION_LONG
            order.offset = OrderData.OFFSET_CLOSE     
        
        # 保存到限价单字典中
        self.workingLimitOrderDict[orderID] = order
        self.limitOrderDict[orderID] = order

        # reduce available cash
        if order.direction = OrderData.DIRECTION_LONG :
            turnoverO, commissionO, slippageO = self.calcAmountOfTrade(order.symbol, order.price, order.totalVolume)
            # self._cashAvail -= turnoverO + commissionO + slippageO
            self.cashChange(-(turnoverO + commissionO + slippageO))

        return [orderID]
    
    #----------------------------------------------------------------------
    def cancelOrder(self, vtOrderID):
        """撤单"""
        if vtOrderID in self.workingLimitOrderDict:
            order = self.workingLimitOrderDict[vtOrderID]
            
            order.status = OrderData.STATUS_CANCELLED
            order.cancelTime = self.dt.strftime('%H:%M:%S')
            
            # restore available cash
            if order.direction = OrderData.DIRECTION_LONG :
                # self._cashAvail += order.price * order.totalVolume * self.size # TODO: I have ignored the commission here
                self.cashChange(order.price * order.totalVolume)

            self.strategy.onOrder(order)
            
            del self.workingLimitOrderDict[vtOrderID]
        
    #----------------------------------------------------------------------
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy):
        """发停止单（本地实现）"""
        self.stopOrderCount += 1
        stopOrderID = STOPORDERPREFIX + str(self.stopOrderCount)
        
        so = StopOrder()
        so.vtSymbol = vtSymbol
        so.price = self.roundToPriceTick(price)
        so.volume = volume
        so.strategy = strategy
        so.status = OrderData.ORDER_WAITING
        so.stopOrderID = stopOrderID
        
        if orderType = OrderData.ORDER_BUY:
            so.direction = OrderData.DIRECTION_LONG
            so.offset = OrderData.OFFSET_OPEN
        elif orderType = OrderData.ORDER_SELL:
            so.direction = OrderData.DIRECTION_SHORT
            so.offset = OrderData.OFFSET_CLOSE
        elif orderType = OrderData.ORDER_SHORT:
            so.direction = OrderData.DIRECTION_SHORT
            so.offset = OrderData.OFFSET_OPEN
        elif orderType = OrderData.ORDER_COVER:
            so.direction = OrderData.DIRECTION_LONG
            so.offset = OrderData.OFFSET_CLOSE           
        
        # 保存stopOrder对象到字典中
        self.stopOrderDict[stopOrderID] = so
        self.workingStopOrderDict[stopOrderID] = so
        
        # 推送停止单初始更新
        self.strategy.onStopOrder(so)        
        
        return [stopOrderID]
    
    #----------------------------------------------------------------------
    def cancelStopOrder(self, stopOrderID):
        """撤销停止单"""
        # 检查停止单是否存在
        if stopOrderID in self.workingStopOrderDict:
            so = self.workingStopOrderDict[stopOrderID]
            so.status = OrderData.ORDER_CANCELLED
            del self.workingStopOrderDict[stopOrderID]
            self.strategy.onStopOrder(so)
    
    #----------------------------------------------------------------------
    def putStrategyEvent(self, name):
        """发送策略更新事件，回测中忽略"""
        pass
     
    #----------------------------------------------------------------------
    def insertData(self, dbName, collectionName, data):
        """考虑到回测中不允许向数据库插入数据，防止实盘交易中的一些代码出错"""
        pass
    
    #----------------------------------------------------------------------
    def loadBar(self, dbName, collectionName, startDate):
        """直接返回初始化数据列表中的Bar"""
        return self.initData
    
    #----------------------------------------------------------------------
    def loadTick(self, dbName, collectionName, startDate):
        """直接返回初始化数据列表中的Tick"""
        return self.initData
    
    #----------------------------------------------------------------------
    def cancelAll(self, name):
        """全部撤单"""
        # 撤销限价单
        for orderID in self.workingLimitOrderDict.keys():
            self.cancelOrder(orderID)
        
        # 撤销停止单
        for stopOrderID in self.workingStopOrderDict.keys():
            self.cancelStopOrder(stopOrderID)

    #----------------------------------------------------------------------
    def saveSyncData(self, strategy):
        """保存同步数据（无效）"""
        pass
    
    #----------------------------------------------------------------------
    def getPriceTick(self, strategy):
        """获取最小价格变动"""
        return self.priceTick
    
    #------------------------------------------------
    # 结果计算相关
    #------------------------------------------------      
    
    #----------------------------------------------------------------------
    def calculateTransactions(self):
        """
        计算回测结果
        """
        self.stdout(u'计算回测结果')
        
        # 首先基于回测后的成交记录，计算每笔交易的盈亏
        self.clearResult()

        buyTrades = []              # 未平仓的多头交易
        sellTrades = []             # 未平仓的空头交易

        # ---------------------------
        # scan all 交易
        # ---------------------------
        # convert the trade records into result records then put them into resultList
        for trade in self.tradeDict.values():
            # 复制成交对象，因为下面的开平仓交易配对涉及到对成交数量的修改
            # 若不进行复制直接操作，则计算完后所有成交的数量会变成0
            trade = copy.copy(trade)
            
            # buy交易
            # ---------------------------
            if trade.direction = OrderData.DIRECTION_LONG:

                if not sellTrades:
                    # 如果尚无空头交易
                    buyTrades.append(trade)
                    continue

                # 当前多头交易为平空
                while True:
                    entryTrade = sellTrades[0]
                    exitTrade = trade
                    
                    # 清算开平仓交易
                    closedVolume = min(exitTrade.volume, entryTrade.volume)
                    result = TradingResult(entryTrade.price, entryTrade.dt, 
                                           exitTrade.price, exitTrade.dt,
                                           -closedVolume, self.rate, self.slippage, self.size)

                    self.resultList.append(result)
                    
                    self.posList.extend([-1,0])
                    self.tradeTimeList.extend([result.entryDt, result.exitDt])
                    
                    # 计算未清算部分
                    entryTrade.volume -= closedVolume
                    exitTrade.volume -= closedVolume
                    
                    # 如果开仓交易已经全部清算，则从列表中移除
                    if not entryTrade.volume:
                        sellTrades.pop(0)
                    
                    # 如果平仓交易已经全部清算，则退出循环
                    if not exitTrade.volume:
                        break
                    
                    # 如果平仓交易未全部清算，
                    if exitTrade.volume:
                        # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                        # 等于新的反向开仓交易，添加到队列中
                        if not sellTrades:
                            buyTrades.append(exitTrade)
                            break
                        # 如果开仓交易还有剩余，则进入下一轮循环
                        else:
                            pass

                continue 
                # end of # 多头交易

            # 空头交易        
            # ---------------------------
            if not buyTrades:
                # 如果尚无多头交易
                sellTrades.append(trade)
                continue

            # 当前空头交易为平多
            while True:
                entryTrade = buyTrades[0]
                exitTrade = trade
                
                # 清算开平仓交易
                closedVolume = min(exitTrade.volume, entryTrade.volume)
                result = TradingResult(entryTrade.price, entryTrade.dt, 
                                       exitTrade.price, exitTrade.dt,
                                       closedVolume, self.rate, self.slippage, self.size)

                self.resultList.append(result)
                self.posList.extend([1,0])
                self.tradeTimeList.extend([result.entryDt, result.exitDt])

                # 计算未清算部分
                entryTrade.volume -= closedVolume
                exitTrade.volume -= closedVolume
                
                # 如果开仓交易已经全部清算，则从列表中移除
                if not entryTrade.volume:
                    buyTrades.pop(0)
                
                # 如果平仓交易已经全部清算，则退出循环
                if not exitTrade.volume:
                    break
                
                # 如果平仓交易未全部清算，
                if exitTrade.volume:
                    # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                    # 等于新的反向开仓交易，添加到队列中
                    if not buyTrades:
                        sellTrades.append(exitTrade)
                        break
                    # 如果开仓交易还有剩余，则进入下一轮循环
                    else:
                        pass                    

                continue 
                # end of 空头交易

        # end of scanning tradeDict
        
        # ---------------------------
        # 结算日
        # ---------------------------
        # 到最后交易日尚未平仓的交易，则以最后价格平仓
        for trade in buyTrades:
            result = TradingResult(trade.price, trade.dt, self._execEndClose, self.dt, 
                                   trade.volume, self.rate, self.slippage, self.size)
            self.resultList.append(result)
            
        for trade in sellTrades:
            result = TradingResult(trade.price, trade.dt, self._execEndClose, self.dt, 
                                   -trade.volume, self.rate, self.slippage, self.size)
            self.resultList.append(result)

        # return resultList;
        return self.settleResult()
        
    #----------------------------------------------------------------------
    def settleResult(self):
        # 检查是否有交易
        if not self.resultList:
            self.stdout(u'无交易结果')
            return {}
        
        # 然后基于每笔交易的结果，我们可以计算具体的盈亏曲线和最大回撤等        
        capital = 0             # 资金
        maxCapital = 0          # 资金最高净值
        drawdown = 0            # 回撤
        
        totalResult = 0         # 总成交数量
        totalTurnover = 0       # 总成交金额（合约面值）
        totalCommission = 0     # 总手续费
        totalSlippage = 0       # 总滑点
        
        timeList = []           # 时间序列
        pnlList = []            # 每笔盈亏序列
        capitalList = []        # 盈亏汇总的时间序列
        drawdownList = []       # 回撤的时间序列
        
        winningResult = 0       # 盈利次数
        losingResult = 0        # 亏损次数		
        totalWinning = 0        # 总盈利金额		
        totalLosing = 0         # 总亏损金额        
        
        for result in self.resultList:
            capital += result.pnl
            maxCapital = max(capital, maxCapital)
            drawdown = capital - maxCapital
            
            pnlList.append(result.pnl)
            timeList.append(result.exitDt)      # 交易的时间戳使用平仓时间
            capitalList.append(capital)
            drawdownList.append(drawdown)
            
            totalResult += 1
            totalTurnover += result.turnover
            totalCommission += result.commission
            totalSlippage += result.slippage
            
            if result.pnl >= 0:
                winningResult += 1
                totalWinning += result.pnl
            else:
                losingResult += 1
                totalLosing += result.pnl
                
        # 计算盈亏相关数据
        winningRate = winningResult/totalResult*100         # 胜率
        
        averageWinning = 0                                  # 这里把数据都初始化为0
        averageLosing = 0
        profitLossRatio = 0
        
        if winningResult:
            averageWinning = totalWinning/winningResult     # 平均每笔盈利
        if losingResult:
            averageLosing = totalLosing/losingResult        # 平均每笔亏损
        if averageLosing:
            profitLossRatio = -averageWinning/averageLosing # 盈亏比

        # 返回回测结果
        d = {}
        d['capital'] = capital
        d['maxCapital'] = maxCapital
        d['drawdown'] = drawdown
        d['totalResult'] = totalResult
        d['totalTurnover'] = totalTurnover
        d['totalCommission'] = totalCommission
        d['totalSlippage'] = totalSlippage
        d['timeList'] = timeList
        d['pnlList'] = pnlList
        d['capitalList'] = capitalList
        d['drawdownList'] = drawdownList
        d['winningRate'] = winningRate
        d['averageWinning'] = averageWinning
        d['averageLosing'] = averageLosing
        d['profitLossRatio'] = profitLossRatio
        d['posList'] = self.posList
        d['tradeTimeList'] = self.tradeTimeList
        d['resultList'] = self.resultList
        
        return d
        
    #----------------------------------------------------------------------
    def plotBacktestingResult(self, d):
        # 绘图
        plt.rcParams['agg.path.chunksize'] =10000
        fig = plt.figure(figsize=(10, 16))
        
        pCapital = plt.subplot(4, 1, 1)
        pCapital.set_ylabel("capital")
        pCapital.plot(d['capitalList'], color='r', lw=0.8)
        
        pDD = plt.subplot(4, 1, 2)
        pDD.set_ylabel("DD")
        pDD.bar(range(len(d['drawdownList'])), d['drawdownList'], color='g')
        
        pPnl = plt.subplot(4, 1, 3)
        pPnl.set_ylabel("pnl")
        pPnl.hist(d['pnlList'], bins=50, color='c')

        pPos = plt.subplot(4, 1, 4)
        pPos.set_ylabel("Position")
        if d['posList'][-1] == 0:
            del d['posList'][-1]
        tradeTimeIndex = [item.strftime("%m/%d %H:%M:%S") for item in d['tradeTimeList']]
        xindex = np.arange(0, len(tradeTimeIndex), np.int(len(tradeTimeIndex)/10))
        tradeTimeIndex = map(lambda i: tradeTimeIndex[i], xindex)
        pPos.plot(d['posList'], color='k', drawstyle='steps-pre')
        pPos.set_ylim(-1.2, 1.2)
        plt.sca(pPos)
        plt.tight_layout()
        plt.xticks(xindex, tradeTimeIndex, rotation=30)  # 旋转15
        
        plt.savefig('BT-%s.png' % self._id, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()

    #----------------------------------------------------------------------
    def showBacktestingResult(self):
        """显示回测结果"""

        d = self.calculateTransactions()
        originGain = 0.0
        if self._execStartClose >0 :
            originGain = (self._execEndClose - self._execStartClose)*100/self._execStartClose

        # 输出
        self.stdout('-' * 30)
        self.stdout(u'回放日期：\t%s(C:%.2f)~%s(C:%.2f): %s%%'  %(self._execStart, self._execStartClose, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.stdout(u'交易日期：\t%s(C:%.2f)~%s(C:%.2f)' % (d['timeList'][0], self._execStartClose, d['timeList'][-1], self._execEndClose))
        
        self.stdout(u'总交易次数：\t%s' % formatNumber(d['totalResult'],0))        
        self.stdout(u'总盈亏：\t%s' % formatNumber(d['capital']))
        self.stdout(u'最大回撤: \t%s' % formatNumber(min(d['drawdownList'])))                
        
        self.stdout(u'平均每笔盈利：\t%s' %formatNumber(d['capital']/d['totalResult']))
        self.stdout(u'平均每笔滑点：\t%s' %formatNumber(d['totalSlippage']/d['totalResult']))
        self.stdout(u'平均每笔佣金：\t%s' %formatNumber(d['totalCommission']/d['totalResult']))
        
        self.stdout(u'胜率\t\t%s%%' %formatNumber(d['winningRate']))
        self.stdout(u'盈利交易平均值\t%s' %formatNumber(d['averageWinning']))
        self.stdout(u'亏损交易平均值\t%s' %formatNumber(d['averageLosing']))
        self.stdout(u'盈亏比：\t%s' %formatNumber(d['profitLossRatio']))

        # self.plotBacktestingResult(d)
    
    
    #----------------------------------------------------------------------
    def clearBackTesting(self):
        """清空之前回测的结果"""

        # 清空限价单相关
        self.limitOrderCount = 0
        self.limitOrderDict.clear()
        self.workingLimitOrderDict.clear()        
        
        # 清空停止单相关
        self.stopOrderCount = 0
        self.stopOrderDict.clear()
        self.workingStopOrderDict.clear()
        
        # 清空成交相关
        self.tradeCount = 0
        self.tradeDict.clear()
        self.dailyResultDict ={}

        self.capital = 100000       # 回测时的起始本金（默认10万）
        self._casheAvail = 0        # 起始cache = capital
        self.slippage = 0           # 回测时假设的滑点
        self.rate = 20/10000        # 回测时假设的佣金比例（适用于百分比佣金）
        self.size = 100             # 合约大小，默认为1    
        self.priceTick = 0          # 价格最小变动 

        self._execStart = ''
        self._execEnd = ''
        self._execStartClose = 0.0
        self._execEndClose = 0.0

    #----------------------------------------------------------------------
    def batchBacktesting(self, strategyList, d):
        """批量回测结果"""

        # self.loadHistoryData()

        for strategy in strategyList:
            if strategy ==None :
                continue

            self.clearBackTesting()
            self.initStrategy(strategy, d)
            self.runBacktesting()
            # self.showBacktestingResult()
            self.showDailyResult()
        
    #----------------------------------------------------------------------
    def runOptimization(self, strategyClass, optimizationSetting):
        """优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.stdout(u'优化设置有问题，请检查')
        
        # 遍历优化
        self.resultList =[]
        for setting in settingList:
            self.clearBackTesting() # must not clear resultList
            self.stdout('-' * 30)
            self.stdout('setting: %s' %str(setting))
            self.initStrategy(strategyClass, setting)
            self.runBacktesting()
            df = self.calculateDailyResult()
            df, d = self.calculateDailyStatistics(df)            

            df.to_csv(self._id+'.OPT' + datetime.now().strftime('%H%M%S') +'.csv')

            try:
                targetValue = d[targetName]
            except KeyError:
                targetValue = 0
            self.resultList.append(([str(setting)], targetValue, d))
        
        # 显示结果
        self.resultList.sort(reverse=True, key=lambda result:result[1])
        self.stdout('-' * 30)
        self.stdout(u'优化结果：')
        for result in self.resultList:
            self.stdout(u'参数：%s，目标[%s]：%s' %(result[0], targetName, result[1]))    
        return self.resultList
            
    #----------------------------------------------------------------------
    def runParallelOptimization(self, strategyClass, optimizationSetting):
        """并行优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.stdout(u'优化设置有问题，请检查')
        
        # 多进程优化，启动一个对应CPU核心数量的进程池
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        l = []

        for setting in settingList:
            l.append(pool.apply_async(optimize, (strategyClass, setting,
                                                 targetName, self.mode, 
                                                 self.startDate, self.initDays, self.endDate,
                                                 self.slippage, self.rate, self.size, self.priceTick,
                                                 self.dbName, self.symbol)))
        pool.close()
        pool.join()
        
        # 显示结果
        resultList = [res.get() for res in l]
        resultList.sort(reverse=True, key=lambda result:result[1])
        self.stdout('-' * 30)
        self.stdout(u'优化结果：')
        for result in resultList:
            self.stdout(u'参数：%s，目标：%s' %(result[0], result[1]))    
            
        return resultList

    #----------------------------------------------------------------------
    def updateDailyClose(self, dt, price):
        """更新每日收盘价"""
        date = dt.date()

        if date not in self.dailyResultDict:
            self.dailyResultDict[date] = DailyResult(date, price)
        else:
            self.dailyResultDict[date].closePrice = price
            
    #----------------------------------------------------------------------
    def calculateDailyResult(self):
        """计算按日统计的交易结果"""
        self.stdout(u'计算按日统计结果')

        if self.tradeDict ==None or len(self.tradeDict) <=0:
            return None
        
        # 将成交添加到每日交易结果中
        for trade in self.tradeDict.values():
            date = trade.dt.date()
            dailyResult = self.dailyResultDict[date]
            dailyResult.addTrade(trade)
            
        # 遍历计算每日结果
        previousClose = 0
        openPosition = 0
        for dailyResult in self.dailyResultDict.values():
            dailyResult.previousClose = previousClose
            previousClose = dailyResult.closePrice
            
            dailyResult.calculatePnl(self, openPosition)
            openPosition = dailyResult.closePosition
            
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
    
    #----------------------------------------------------------------------
    def calculateDailyStatistics(self, df):
        """计算按日统计的结果"""
        df['balance'] = df['netPnl'].cumsum() + self.capital
        df['return'] = (np.log(df['balance']) - np.log(df['balance'].shift(1))).fillna(0)
        df['highlevel'] = df['balance'].rolling(min_periods=1,window=len(df),center=False).max()
        df['drawdown'] = df['balance'] - df['highlevel']        
        df['ddPercent'] = df['drawdown'] / df['highlevel'] * 100
        
        # 计算统计结果
        startDate = df.index[0]
        endDate = df.index[-1]

        totalDays = len(df)
        profitDays = len(df[df['netPnl']>0])
        lossDays = len(df[df['netPnl']<0])
        
        endBalance = df['balance'].iloc[-1]
        maxDrawdown = df['drawdown'].min()
        maxDdPercent = df['ddPercent'].min()
        
        totalNetPnl = df['netPnl'].sum()
        dailyNetPnl = totalNetPnl / totalDays
        
        totalCommission = df['commission'].sum()
        dailyCommission = totalCommission / totalDays
        
        totalSlippage = df['slippage'].sum()
        dailySlippage = totalSlippage / totalDays
        
        totalTurnover = df['turnover'].sum()
        dailyTurnover = totalTurnover / totalDays
        
        totalTradeCount = df['tradeCountBuy'].sum() + df['tradeCountSell'].sum()
        dailyTradeCount = totalTradeCount / totalDays
        
        totalReturn = (endBalance/self.capital - 1) * 100
        annualizedReturn = totalReturn / totalDays * 240
        dailyReturn = df['return'].mean() * 100
        returnStd = df['return'].std() * 100
        
        if returnStd:
            sharpeRatio = dailyReturn / returnStd * np.sqrt(240)
        else:
            sharpeRatio = 0
            
        # 返回结果
        result = {
            'startDate': startDate,
            'endDate': endDate,
            'totalDays': totalDays,
            'profitDays': profitDays,
            'lossDays': lossDays,
            'endBalance': endBalance,
            'maxDrawdown': maxDrawdown,
            'maxDdPercent': maxDdPercent,
            'totalNetPnl': totalNetPnl,
            'dailyNetPnl': dailyNetPnl,
            'totalCommission': totalCommission,
            'dailyCommission': dailyCommission,
            'totalSlippage': totalSlippage,
            'dailySlippage': dailySlippage,
            'totalTurnover': totalTurnover,
            'dailyTurnover': dailyTurnover,
            'totalTradeCount': totalTradeCount,
            'dailyTradeCount': dailyTradeCount,
            'totalReturn': totalReturn,
            'annualizedReturn': annualizedReturn,
            'dailyReturn': dailyReturn,
            'returnStd': returnStd,
            'sharpeRatio': sharpeRatio
        }
        
        return df, result
    
    #----------------------------------------------------------------------
    def showDailyResult(self, df=None, result=None):
        """显示按日统计的交易结果"""
        if df is None:
            df = self.calculateDailyResult()
            df, result = self.calculateDailyStatistics(df)

        # df.to_csv(self._id+'.csv',columns=""+DailyResult.EXPORT_COLUMNS)
        df.to_csv(self._id+'.csv')
        originGain = 0.0
        if self._execStartClose >0 :
            originGain = (self._execEndClose - self._execStartClose)*100/self._execStartClose

        # 输出统计结果
        self.stdout('-' * 30)
        self.stdout(u'回放日期：\t%s(close:%.2f)~%s(close:%.2f): %s%%'  %(self._execStart, self._execStartClose, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.stdout(u'交易日期：\t%s(close:%.2f)~%s(close:%.2f)' % (result['startDate'], self._execStartClose, result['endDate'], self._execEndClose))
        
        self.stdout(u'交易日数：\t%s (盈利%s,亏损%s)' % (result['totalDays'], result['profitDays'], result['lossDays']))
        
        self.stdout(u'起始资金：\t%s' % formatNumber(self.capital))
        self.stdout(u'结束资金：\t%s' % formatNumber(result['endBalance']))
    
        self.stdout(u'总收益率：\t%s%%' % formatNumber(result['totalReturn']))
        self.stdout(u'年化收益：\t%s%%' % formatNumber(result['annualizedReturn']))
        self.stdout(u'总盈亏：\t%s' % formatNumber(result['totalNetPnl']))
        self.stdout(u'最大回撤: \t%s' % formatNumber(result['maxDrawdown']))   
        self.stdout(u'百分比最大回撤: %s%%' % formatNumber(result['maxDdPercent']))   
        
        self.stdout(u'总手续费：\t%s' % formatNumber(result['totalCommission']))
        self.stdout(u'总滑点：\t%s' % formatNumber(result['totalSlippage']))
        self.stdout(u'总成交金额：\t%s' % formatNumber(result['totalTurnover']))
        self.stdout(u'总成交笔数：\t%s' % formatNumber(result['totalTradeCount'],0))
        
        self.stdout(u'日均盈亏：\t%s' % formatNumber(result['dailyNetPnl']))
        self.stdout(u'日均手续费：\t%s' % formatNumber(result['dailyCommission']))
        self.stdout(u'日均滑点：\t%s' % formatNumber(result['dailySlippage']))
        self.stdout(u'日均成交金额：\t%s' % formatNumber(result['dailyTurnover']))
        self.stdout(u'日均成交笔数：\t%s' % formatNumber(result['dailyTradeCount']))
        
        self.stdout(u'日均收益率：\t%s%%' % formatNumber(result['dailyReturn']))
        self.stdout(u'收益标准差：\t%s%%' % formatNumber(result['returnStd']))
        self.stdout(u'Sharpe Ratio：\t%s' % formatNumber(result['sharpeRatio']))
        
        # self.plotDailyResult(df)

    #----------------------------------------------------------------------
    def plotDailyResult(self, df):
        # 绘图
        plt.rcParams['agg.path.chunksize'] =10000

        fig = plt.figure(figsize=(10, 16))
        
        pBalance = plt.subplot(4, 1, 1)
        pBalance.set_title(self._id + ' Balance')
        df['balance'].plot(legend=True)
        
        pDrawdown = plt.subplot(4, 1, 2)
        pDrawdown.set_title('Drawdown')
        pDrawdown.fill_between(range(len(df)), df['drawdown'].values)
        
        pPnl = plt.subplot(4, 1, 3)
        pPnl.set_title('Daily Pnl') 
        df['netPnl'].plot(kind='bar', legend=False, grid=False, xticks=[])

        pKDE = plt.subplot(4, 1, 4)
        pKDE.set_title('Daily Pnl Distribution')
        df['netPnl'].hist(bins=50)
        
        plt.savefig('DR-%s.png' % self._id, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()
       
        
########################################################################
class TradingResult(object):
    """每笔交易的结果"""

   #----------------------------------------------------------------------
    def __init__(self, entryPrice, entryDt, exitPrice, 
                 exitDt, volume, rate, slippage, size):
        """Constructor"""
        self.entryPrice = entryPrice    # 开仓价格
        self.exitPrice = exitPrice      # 平仓价格
        
        self.entryDt = entryDt          # 开仓时间datetime    
        self.exitDt = exitDt            # 平仓时间
        
        self.volume = volume    # 交易数量（+/-代表方向）
        
        self.turnover   = (self.entryPrice + self.exitPrice) *size*abs(volume)   # 成交金额
        entryCommission = self.entryPrice *size*abs(volume) *rate
        if entryCommission < 2.0:
            entryCommission =2.0

        exitCommission = self.exitPrice *size*abs(volume) *rate
        if exitCommission < 2.0:
            exitCommission =2.0

        self.commission = entryCommission + exitCommission
        self.slippage   = slippage*2*size*abs(volume)                            # 滑点成本
        self.pnl        = ((self.exitPrice - self.entryPrice) * volume * size 
                            - self.commission - self.slippage)                   # 净盈亏


########################################################################
class DailyResult(object):
    """每日交易的结果"""

    EXPORT_COLUMNS = "date,openPosition,closePosition,previousClose,closePrice,tradeCountBuy,tradeCountSell,turnover,commission,tradingPnl,positionPnl,totalPnl,netPnl,txnHist"

    #----------------------------------------------------------------------
    def __init__(self, date, closePrice):
        """Constructor"""
        self.date = date                # 日期
        self.closePrice = closePrice    # 当日收盘价
        self.previousClose = 0          # 昨日收盘价
        
        self.tradeList = []             # 成交列表
        self.tradeCountBuy = 0             # 成交数量
        self.tradeCountSell = 0             # 成交数量
        
        self.openPosition = 0           # 开盘时的持仓
        self.closePosition = 0          # 收盘时的持仓
        
        self.tradingPnl = 0             # 交易盈亏
        self.positionPnl = 0            # 持仓盈亏
        self.totalPnl = 0               # 总盈亏
        
        self.turnover = 0               # 成交量
        self.commission = 0             # 手续费
        self.slippage = 0               # 滑点
        self.netPnl = 0                 # 净盈亏
        self.txnHist = ""
        
    #----------------------------------------------------------------------
    def addTrade(self, trade):
        """添加交易"""
        self.tradeList.append(trade)

    #----------------------------------------------------------------------
    def calculatePnl(self, account, openPosition=0):
        """
        计算盈亏
        size: 合约乘数
        rate：手续费率
        slippage：滑点点数
        """
        # 持仓部分
        self.openPosition = openPosition
        self.positionPnl = self.openPosition * (self.closePrice - self.previousClose) * account.size
        self.closePosition = self.openPosition
        
        # 交易部分
        self.tradeCountBuy = 0
        self.tradeCountSell = 0
        
        for trade in self.tradeList:
            if trade.direction = OrderData.DIRECTION_LONG:
                posChange = trade.volume
                self.tradeCountBuy += 1
            else:
                posChange = -trade.volume
                self.tradeCountSell += 1
                
            self.tradingPnl += posChange * (self.closePrice - trade.price) * account.size
            self.txnHist += "%+dx%s" % (posChange, trade.price)
            self.closePosition += posChange
            turnover, commission, slippagefee = account.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
            self.turnover += turnover
            self.commission += commission
            self.slippage += slippagefee
        
        # 汇总
        self.totalPnl = round(self.tradingPnl + self.positionPnl, 2)
        self.netPnl = round(self.totalPnl - self.commission - self.slippage, 2)


########################################################################
class OptimizationSetting(object):
    """优化设置"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.paramDict = OrderedDict()
        
        self.optimizeTarget = ''        # 优化目标字段
        
    #----------------------------------------------------------------------
    def addParameter(self, name, start, end=None, step=None):
        """增加优化参数"""
        if end is None and step is None:
            self.paramDict[name] = [start]
            return 
        
        if end < start:
            print u'参数起始点必须不大于终止点'
            return
        
        if step <= 0:
            print u'参数布进必须大于0'
            return
        
        l = []
        param = start
        
        while param <= end:
            l.append(param)
            param += step
        
        self.paramDict[name] = l
        
    #----------------------------------------------------------------------
    def generateSetting(self):
        """生成优化参数组合"""
        # 参数名的列表
        nameList = self.paramDict.keys()
        paramList = self.paramDict.values()
        
        # 使用迭代工具生产参数对组合
        productList = list(product(*paramList))
        
        # 把参数对组合打包到一个个字典组成的列表中
        settingList = []
        for p in productList:
            d = dict(zip(nameList, p))
            settingList.append(d)
    
        return settingList
    
    #----------------------------------------------------------------------
    def setOptimizeTarget(self, target):
        """设置优化目标字段"""
        self.optimizeTarget = target


#----------------------------------------------------------------------
def formatNumber(n, dec=2):
    """格式化数字到字符串"""
    rn = round(n, dec)      # 保留两位小数
    return format(rn, ',')  # 加上千分符
    

#----------------------------------------------------------------------
def optimize(strategyClass, setting, targetName,
             mode, startDate, initDays, endDate,
             slippage, rate, size, priceTick,
             dbName, symbol):

    """多进程优化时跑在每个进程中运行的函数"""
    account = BTAccount_AShare()
    account.setBacktestingMode(mode)
    account.setStartDate(startDate, initDays)
    account.setEndDate(endDate)
    account.setSlippage(slippage)
    account.setRate(rate)
    account.setSize(size)
    account.setPriceTick(priceTick)
    account.setDatabase(dbName, symbol)
    
    account.initStrategy(strategyClass, setting)
    account.runBacktesting()
    
    df = account.calculateDailyResult()
    df, d = account.calculateDailyStatistics(df)
    try:
        targetValue = d[targetName]
    except KeyError:
        targetValue = 0
                    
    return (str(setting), targetValue, d)    
    
hs300s= [
        "600000","600008","600009","600010","600011","600015","600016","600018","600019","600023",
        "600025","600028","600029","600030","600031","600036","600038","600048","600050","600061",
        "600066","600068","600085","600089","600100","600104","600109","600111","600115","600118",
        "600153","600157","600170","600176","600177","600188","600196","600208","600219","600221",
        "600233","600271","600276","600297","600309","600332","600339","600340","600346","600352",
        "600362","600369","600372","600373","600376","600383","600390","600398","600406","600415",
        "600436","600438","600482","600487","600489","600498","600516","600518","600519","600522",
        "600535","600547","600549","600570","600583","600585","600588","600606","600637","600660",
        "600663","600674","600682","600688","600690","600703","600704","600705","600739","600741",
        "600795","600804","600809","600816","600820","600837","600867","600886","600887","600893",
        "600900","600909","600919","600926","600958","600959","600977","600999","601006","601009",
        "601012","601018","601021","601088","601099","601108","601111","601117","601155","601166",
        "601169","601186","601198","601211","601212","601216","601225","601228","601229","601238",
        "601288","601318","601328","601333","601336","601360","601377","601390","601398","601555",
        "601600","601601","601607","601611","601618","601628","601633","601668","601669","601688",
        "601718","601727","601766","601788","601800","601808","601818","601828","601838","601857",
        "601866","601877","601878","601881","601888","601898","601899","601901","601919","601933",
        "601939","601958","601985","601988","601989","601991","601992","601997","601998","603160",
        "603260","603288","603799","603833","603858","603993","000001","000002","000060","000063",
        "000069","000100","000157","000166","000333","000338","000402","000413","000415","000423",
        "000425","000503","000538","000540","000559","000568","000623","000625","000627","000630",
        "000651","000671","000709","000723","000725","000728","000768","000776","000783","000786",
        "000792","000826","000839","000858","000876","000895","000898","000938","000959","000961",
        "000963","000983","001965","001979","002007","002008","002024","002027","002044","002050",
        "002065","002074","002081","002085","002142","002146","002153","002202","002230","002236",
        "002241","002252","002294","002304","002310","002352","002385","002411","002415","002450",
        "002456","002460","002466","002468","002470","002475","002493","002500","002508","002555",
        "002558","002572","002594","002601","002602","002608","002624","002625","002673","002714",
        "002736","002739","002797","002925","300003","300015","300017","300024","300027","300033",
        "300059","300070","300072","300122","300124","300136","300144","300251","300408","300433"
        ]
