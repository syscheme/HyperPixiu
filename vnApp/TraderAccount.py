# encoding: UTF-8

'''
本文件中实现了CTA策略引擎，针对CTA类型的策略，抽象简化了部分底层接口的功能。
'''

from __future__ import division

import json
import os
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from copy import copy

from vnpy.event import Event
from vnpy.trader.vtEvent import *
from vnpy.trader.vtConstant import *
from vnpy.trader.vtObject import VtTickData, KLineData
from vnpy.trader.vtGateway import VtSubscribeReq, VtOrderReq, VtCancelOrderReq, VtLogData
from vnpy.trader.vtFunction import todayDate, getJsonPath

from vnApp.strategies import STRATEGY_CLASS
from vnApp.Account import *

########################################################################
class TraderAccount(Account):
    """CTA策略引擎"""
    settingFileName = 'CTA_setting.json'
    settingfilePath = getJsonPath(settingFileName, __file__)
    
    STATUS_FINISHED = set([STATUS_REJECTED, STATUS_CANCELLED, STATUS_ALLTRADED])

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, eventChannel):
        """Constructor"""

        super(TraderAccount, self).__init__()

        self._mainRoutine  = mainRoutine
        self._eventChannel = eventChannel
        
        # 当前日期
        self.today = todayDate()
        
        # 成交号集合，用来过滤已经收到过的成交推送
        self.tradeSet = set()
        
        # 引擎类型为实盘
        self.engineType = ENGINETYPE_TRADING
        
        # 注册日式事件类型
        self._mainRoutine.registerLogEvent(EVENT_LOG)
        
        # 注册事件监听
        self.registerEvent()
 
    #----------------------------------------------------------------------
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy):
        """发单"""
        contract = self._mainRoutine.getContract(vtSymbol)
        
        req = VtOrderReq()
        req.symbol = contract.symbol
        req.exchange = contract.exchange
        req.vtSymbol = contract.vtSymbol
        req.price = self.roundToPriceTick(contract.priceTick, price)
        req.volume = volume
        
        req.productClass = strategy.productClass
        req.currency = strategy.currency        
        
        # 设计为CTA引擎发出的委托只允许使用限价单
        req.priceType = PRICETYPE_LIMITPRICE    
        
        # CTA委托类型映射
        if orderType == OrderData.ORDER_BUY:
            req.direction = OrderData.DIRECTION_LONG
            req.offset = OrderData.OFFSET_OPEN
            
        elif orderType = OrderData.ORDER_SELL:
            req.direction = OrderData.DIRECTION_SHORT
            req.offset = OrderData.OFFSET_CLOSE
                
        elif orderType = OrderData.ORDER_SHORT:
            req.direction = OrderData.DIRECTION_SHORT
            req.offset = OrderData.OFFSET_OPEN
            
        elif orderType = OrderData.ORDER_COVER:
            req.direction = OrderData.DIRECTION_LONG
            req.offset = OrderData.OFFSET_CLOSE
            
        # 委托转换
        reqList = self._mainRoutine.convertOrderReq(req)
        vtOrderIDList = []
        
        if not reqList:
            return vtOrderIDList
        
        for convertedReq in reqList:
            vtOrderID = self._mainRoutine.sendOrder(convertedReq, contract.gatewayName)    # 发单
            self.orderStrategyDict[vtOrderID] = strategy                                 # 保存vtOrderID和策略的映射关系
            self.strategyOrderDict[strategy.name].add(vtOrderID)                         # 添加到策略委托号集合中
            vtOrderIDList.append(vtOrderID)
            
        self.log(u'策略%s发送委托，%s，%s，%s@%s' 
                         %(strategy.name, vtSymbol, req.direction, volume, price))
        
        return vtOrderIDList
    
    #----------------------------------------------------------------------
    def cancelOrder(self, vtOrderID):
        """撤单"""
        # 查询报单对象
        order = self._mainRoutine.getOrder(vtOrderID)
        
        # 如果查询成功
        if order:
            # 检查是否报单还有效，只有有效时才发出撤单指令
            orderFinished = (order.status==STATUS_ALLTRADED or order.status==STATUS_CANCELLED)
            if not orderFinished:
                req = VtCancelOrderReq()
                req.symbol = order.symbol
                req.exchange = order.exchange
                req.frontID = order.frontID
                req.sessionID = order.sessionID
                req.orderID = order.orderID
                self._mainRoutine.cancelOrder(req, order.gatewayName)    

    #----------------------------------------------------------------------
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy):
        """发停止单（本地实现）"""
        self.stopOrderCount += 1
        stopOrderID = STOPORDERPREFIX + str(self.stopOrderCount)
        
        so = StopOrder()
        so.vtSymbol = vtSymbol
        so.orderType = orderType
        so.price = price
        so.volume = volume
        so.strategy = strategy
        so.stopOrderID = stopOrderID
        so.status = OrderData.ORDER_WAITING
        
        if orderType == OrderData.ORDER_BUY:
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
        
        # 保存stopOrderID到策略委托号集合中
        self.strategyOrderDict[strategy.name].add(stopOrderID)
        
        # 推送停止单状态
        strategy.onStopOrder(so)
        
        return [stopOrderID]
    
    #----------------------------------------------------------------------
    def cancelStopOrder(self, stopOrderID):
        """撤销停止单"""
        # 检查停止单是否存在
        if stopOrderID in self.workingStopOrderDict:
            so = self.workingStopOrderDict[stopOrderID]
            strategy = so.strategy
            
            # 更改停止单状态为已撤销
            so.status = OrderData.ORDER_CANCELLED
            
            # 从活动停止单字典中移除
            del self.workingStopOrderDict[stopOrderID]
            
            # 从策略委托号集合中移除
            s = self.strategyOrderDict[strategy.name]
            if stopOrderID in s:
                s.remove(stopOrderID)
            
            # 通知策略
            strategy.onStopOrder(so)

    #----------------------------------------------------------------------
    def processStopOrder(self, tick):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        vtSymbol = tick.vtSymbol
        
        # 首先检查是否有策略交易该合约
        if vtSymbol in self._idxTickToStrategy:
            # 遍历等待中的停止单，检查是否会被触发
            for so in self.workingStopOrderDict.values():
                if so.vtSymbol == vtSymbol:
                    longTriggered = OrderData.DIRECTION_LONG and tick.lastPrice>=so.price        # 多头停止单被触发
                    shortTriggered = OrderData.DIRECTION_SHORT and tick.lastPrice<=so.price     # 空头停止单被触发
                    
                    if longTriggered or shortTriggered:
                        # 买入和卖出分别以涨停跌停价发单（模拟市价单）
                        if so.direction= OrderData.DIRECTION_LONG:
                            price = tick.upperLimit
                        else:
                            price = tick.lowerLimit
                        
                        # 发出市价委托
                        self.sendOrder(so.vtSymbol, so.orderType, price, so.volume, so.strategy)
                        
                        # 从活动停止单字典中移除该停止单
                        del self.workingStopOrderDict[so.stopOrderID]
                        
                        # 从策略委托号集合中移除
                        s = self.strategyOrderDict[so.strategy.name]
                        if so.stopOrderID in s:
                            s.remove(so.stopOrderID)
                        
                        # 更新停止单状态，并通知策略
                        so.status = OrderData.ORDER_TRIGGERED
                        so.strategy.onStopOrder(so)

    #----------------------------------------------------------------------
    def processTickEvent(self, event):
        """处理行情推送"""
        tick = event.dict_['data']
        
        tick = copy(tick)
        
        # 收到tick行情后，先处理本地停止单（检查是否要立即发出）
        self.processStopOrder(tick)
        
        # 推送tick到对应的策略实例进行处理
        if tick.vtSymbol in self._idxTickToStrategy:
            # tick时间可能出现异常数据，使用try...except实现捕捉和过滤
            try:
                # 添加datetime字段
                if not tick.datetime:
                    tick.datetime = datetime.strptime(' '.join([tick.date, tick.time]), '%Y%m%d %H:%M:%S.%f')
            except ValueError:
                self.log(traceback.format_exc())
                return
                
            # 逐个推送到策略实例中
            l = self._idxTickToStrategy[tick.vtSymbol]
            for strategy in l:
                self.callStrategyFunc(strategy, strategy.onTick, tick)
    
    #----------------------------------------------------------------------
    def processOrderEvent(self, event):
        """处理委托推送"""
        order = event.dict_['data']
        
        vtOrderID = order.vtOrderID
        
        if vtOrderID in self.orderStrategyDict:
            strategy = self.orderStrategyDict[vtOrderID]            
            
            # 如果委托已经完成（拒单、撤销、全成），则从活动委托集合中移除
            if order.status in self.STATUS_FINISHED:
                s = self.strategyOrderDict[strategy.name]
                if vtOrderID in s:
                    s.remove(vtOrderID)
            
            self.callStrategyFunc(strategy, strategy.onOrder, order)
    
    #----------------------------------------------------------------------
    def processTradeEvent(self, event):
        """处理成交推送"""
        trade = event.dict_['data']
        
        # 过滤已经收到过的成交回报
        if trade.vtTradeID in self.tradeSet:
            return
        self.tradeSet.add(trade.vtTradeID)
        
        # 将成交推送到策略对象中
        if trade.vtOrderID in self.orderStrategyDict:
            strategy = self.orderStrategyDict[trade.vtOrderID]
            
            # 计算策略持仓
            if trade.direction = OrderData.DIRECTION_LONG:
                strategy.pos += trade.volume
            else:
                strategy.pos -= trade.volume
            
            self.callStrategyFunc(strategy, strategy.onTrade, trade)
            
            # 保存策略持仓到数据库
            self.saveSyncData(strategy)              
    
    #----------------------------------------------------------------------
    def registerEvent(self):
        """注册事件监听"""
        self._eventChannel.register(EVENT_TICK, self.processTickEvent)
        self._eventChannel.register(EVENT_ORDER, self.processOrderEvent)
        self._eventChannel.register(EVENT_TRADE, self.processTradeEvent)
 
    #----------------------------------------------------------------------
    def insertData(self, dbName, collectionName, data):
        """插入数据到数据库（这里的data可以是VtTickData或者VtBarData）"""
        self._mainRoutine.dbInsert(dbName, collectionName, data)
    
    #----------------------------------------------------------------------
    def loadBar(self, dbName, collectionName, days):
        """从数据库中读取Bar数据，startDate是datetime对象"""
        startDate = self.today - timedelta(days)
        
        d = {'datetime':{'$gte':startDate}}
        barData = self._mainRoutine.dbQuery(dbName, collectionName, d, 'datetime')
        
        l = []
        for d in barData:
            bar = KLineData()
            bar.__dict__ = d
            l.append(bar)
        return l
    
    #----------------------------------------------------------------------
    def loadTick(self, dbName, collectionName, days):
        """从数据库中读取Tick数据，startDate是datetime对象"""
        startDate = self.today - timedelta(days)
        
        d = {'datetime':{'$gte':startDate}}
        tickData = self._mainRoutine.dbQuery(dbName, collectionName, d, 'datetime')
        
        l = []
        for d in tickData:
            tick = VtTickData()
            tick.__dict__ = d
            l.append(tick)
        return l    
    
    #----------------------------------------------------------------------
    def log(self, content):
        """快速发出CTA模块日志事件"""
        log = VtLogData()
        log.logContent = content
        log.gatewayName = 'CTA_STRATEGY'
        event = Event(type_=EVENT_LOG)
        event.dict_['data'] = log
        self._eventChannel.put(event)   
    
    #----------------------------------------------------------------------
    def subscribeMarketData(self, strategy):
        """订阅行情"""
        # 订阅合约
        contract = self._mainRoutine.getContract(strategy.vtSymbol)
        if contract:
            req = VtSubscribeReq()
            req.symbol = contract.symbol
            req.exchange = contract.exchange
            
            # 对于IB接口订阅行情时所需的货币和产品类型，从策略属性中获取
            req.currency = strategy.currency
            req.productClass = strategy.productClass
            
            self._mainRoutine.subscribe(req, contract.gatewayName)
        else:
            self.log(u'%s的交易合约%s无法找到' %(strategy.name, strategy.vtSymbol))

    #----------------------------------------------------------------------
    def putStrategyEvent(self, name):
        """触发策略状态变化事件（通常用于通知GUI更新）"""
        strategy = self._strategyDict[name]
        d = {k:strategy.__getattribute__(k) for k in strategy.varList}
        
        event = Event(EVENT_STRATEGY+name)
        event.dict_['data'] = d
        self._eventChannel.put(event)
        
        d2 = {k:str(v) for k,v in d.items()}
        d2['name'] = name
        event2 = Event(EVENT_STRATEGY)
        event2.dict_['data'] = d2
        self._eventChannel.put(event2)        
        
    #----------------------------------------------------------------------
    def saveSyncData(self, strategy):
        """保存策略的持仓情况到数据库"""
        flt = {'name': strategy.name,
               'vtSymbol': strategy.vtSymbol}
        
        d = copy(flt)
        for key in strategy.syncList:
            d[key] = strategy.__getattribute__(key)
        
        self._mainRoutine.dbUpdate(POSITION_DB_NAME, strategy.className,
                                 d, flt, True)
        
        content = u'策略%s同步数据保存成功，当前持仓%s' %(strategy.name, strategy.pos)
        self.log(content)
    
    #----------------------------------------------------------------------
    def loadSyncData(self, strategy):
        """从数据库载入策略的持仓情况"""
        flt = {'name': strategy.name,
               'vtSymbol': strategy.vtSymbol}
        syncData = self._mainRoutine.dbQuery(POSITION_DB_NAME, strategy.className, flt)
        
        if not syncData:
            return
        
        d = syncData[0]
        
        for key in strategy.syncList:
            if key in d:
                strategy.__setattr__(key, d[key])
                
    #----------------------------------------------------------------------
    def roundToPriceTick(self, priceTick, price):
        """取整价格到合约最小价格变动"""
        if not priceTick:
            return price
        
        newPrice = round(price/priceTick, 0) * priceTick
        return newPrice    
    
    #----------------------------------------------------------------------
    def cancelAll(self, name):
        """全部撤单"""
        s = self.strategyOrderDict[name]
        
        # 遍历列表，全部撤单
        # 这里不能直接遍历集合s，因为撤单时会修改s中的内容，导致出错
        for orderID in list(s):
            if STOPORDERPREFIX in orderID:
                self.cancelStopOrder(orderID)
            else:
                self.cancelOrder(orderID)

    #----------------------------------------------------------------------
    def getPriceTick(self, strategy):
        """获取最小价格变动"""
        contract = self._mainRoutine.getContract(strategy.vtSymbol)
        if contract:
            return contract.priceTick
        return 0
        
        
