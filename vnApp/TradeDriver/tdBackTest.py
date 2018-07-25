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
from vnpy.trader.vtObject import VtTickData, VtBarData
from vnpy.trader.vtConstant import *
from vnpy.trader.vtGateway import VtOrderData, VtTradeData

from ..Account import *
from .tdBase import *

########################################################################
class tdBackTest(tdBase):
    """
    回测TradeDriver
    函数接口和TradeDriver保持一样，
    从而实现同一套代码从回测到实盘。
    """
    
    TICK_MODE = 'tick'
    BAR_MODE  = 'bar'

    #----------------------------------------------------------------------
    def __init__(self, account, settings, mode=None):
        """Constructor"""

        super(tdBase, self).__init__(account, settings, mode)

    #------------------------------------------------
    # Impl of TraderDriver
    #------------------------------------------------    
    def placeOrder(self, amount, symbol, type_, price=None, source=None):
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
        if type_ == ORDER_BUY:
            order.direction = DIRECTION_LONG
            order.offset = OFFSET_OPEN
        elif type_ == ORDER_SELL:
            order.direction = DIRECTION_SHORT
            order.offset = OFFSET_CLOSE
        elif ortype_derType == ORDER_SHORT:
            order.direction = DIRECTION_SHORT
            order.offset = OFFSET_OPEN
        elif type_ == ORDER_COVER:
            order.direction = DIRECTION_LONG
            order.offset = OFFSET_CLOSE     
        
        # 保存到限价单字典中
        self.workingLimitOrderDict[orderID] = order
        self.limitOrderDict[orderID] = order

        # reduce available cash
        if order.direction == DIRECTION_LONG :
            turnoverO, commissionO, slippageO = amountOfTrade(order.symbol, order.price, order.totalVolume, self.size, self.slippage, self.rate)
            self._cashAvail -= turnoverO + commissionO + slippageO
        
        return [orderID]
    
    #----------------------------------------------------------------------
    def cancelOrder(self, vtOrderID):
        """撤单"""
        if vtOrderID in self.workingLimitOrderDict:
            order = self.workingLimitOrderDict[vtOrderID]
            
            order.status = STATUS_CANCELLED
            order.cancelTime = self.dt.strftime('%H:%M:%S')
            
            # restore available cash
            if order.direction == DIRECTION_LONG :
                self._cashAvail += order.price * order.totalVolume * self.size # TODO: I have ignored the commission here

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
        so.status = STOPORDER_WAITING
        so.stopOrderID = stopOrderID
        
        if orderType == ORDER_BUY:
            so.direction = DIRECTION_LONG
            so.offset = OFFSET_OPEN
        elif orderType == ORDER_SELL:
            so.direction = DIRECTION_SHORT
            so.offset = OFFSET_CLOSE
        elif orderType == ORDER_SHORT:
            so.direction = DIRECTION_SHORT
            so.offset = OFFSET_OPEN
        elif orderType == ORDER_COVER:
            so.direction = DIRECTION_LONG
            so.offset = OFFSET_CLOSE           
        
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
            so.status = STOPORDER_CANCELLED
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
    def cancelAll(self, name):
        """全部撤单"""
        # 撤销限价单
        for orderID in self.workingLimitOrderDict.keys():
            self.cancelOrder(orderID)
        
        # 撤销停止单
        for stopOrderID in self.workingStopOrderDict.keys():
            self.cancelStopOrder(stopOrderID)

    #----------------------------------------------------------------------
    def calcAmountOfTrade(vtSymbol, price, volume):
    # def amountOfTrade(symbol, price, volume, size, slippage=0, rate=3/1000) :
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
    def saveSyncData(self, strategy):
        """保存同步数据（无效）"""
        pass
    
    #----------------------------------------------------------------------
    def getPriceTick(self, strategy):
        """获取最小价格变动"""
        return self.priceTick
