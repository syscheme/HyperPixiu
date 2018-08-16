# encoding: UTF-8

'''
This module represent a basic account
'''
from __future__ import division

from datetime import datetime, timedelta
from collections import OrderedDict
# from itertools import product
import multiprocessing
import copy
import threading
import traceback

import jsoncfg # pip install json-cfg

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

from .EventChannel import EventChannel, EventData, datetime2float

# 如果安装了seaborn则设置为白色风格
# try:
#     import seaborn as sns       
#     sns.set_style('whitegrid')  
# except ImportError:
#     pass

from pymongo import ASCENDING

########################################################################
# 常量定义
########################################################################

# 数据库名称
SETTING_DB_NAME = 'vnDB_Setting'
POSITION_DB_NAME = 'vnDB_Position'

TICK_DB_NAME   = 'vnDB_Tick'
DAILY_DB_NAME  = 'vnDB_Daily'
MINUTE_DB_NAME = 'vnDB_1Min'

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

    EVENT_ORDER = 'eOrder.'                 # 报单回报事件
    EVENT_TRADE = 'eTrade.'                 # 报单回报事件

    # state of Account
    STATE_OPEN  = 'open'   # during trading hours
    STATE_CLOSE = 'close'  # during market close

    SYMBOL_CASH = '.RMB.' # the dummy symbol in order to represent cache in _dictPositions

    #----------------------------------------------------------------------
    def __init__(self, trader, settings):
        """Constructor"""

        self._lock = threading.Lock()

        self._settings     = settings
        self._trader       = trader

        # the app instance Id
        self._id = settings.id("")
        if len(self._id)<=0 :
            Account.__lastId__ +=1
            self._id = 'ACNT%d' % Account.__lastId__

        self._orderId = int(datetime2float(datetime.now())) %100000000 # start with a big number
        self._exchange = settings.exchange("")

        self._state        = Account.STATE_CLOSE

        # trader executer
        # self._dvrBroker = dvrBrokerClass(self, self._settings)

        self._dateToday      = None # date of previous close
        self._datePrevClose  = None # date of previous close
        self._prevPositions = {} # dict from symbol to previous PositionData

        self._dictOutgoingOrders = {} # the outgoing orders dict from reqId to OrderData that has not been confirmed with broker's orderId
        self._lstOrdersToCancel = []
        # cached data from broker
        self._dictPositions = { # dict from symbol to latest PositionData
            Account.SYMBOL_CASH : PositionData()
        }
        self._dictTrades = {} # dict from tradeId to trade confirmed during today
        self._dictStopOrders = {} # dict from broker's orderId to OrderData that has been submitted but not yet traded
        self._dictLimitOrders = {} # dict from broker's orderId to OrderData that has been submitted but not yet traded


        # self.capital = 0        # 起始本金（默认10万）
        # self._cashAvail =0
        
        self._dbName   = self._settings.dbName(self._id)           # 假设的滑点
        self.slippage  = self._settings.slippage(0)           # 假设的滑点
        self.rate      = self._settings.ratePer10K(30)/10000  # 假设的佣金比例（适用于百分比佣金）
        self.size      = self._settings.size(1)               # 合约大小，默认为1    
        self._priceTick = self._settings.priceTick(0)      # 价格最小变动 
        
        # self.initData = []          # 初始化用的数据
        
        # # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # # value为包含所有相关strategy对象的list
        # self._idxTickToStrategy = {}
        
        # # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # # key为vtOrderID，value为strategy对象
        # self.orderStrategyDict = {}     
        
        # # 保存策略名称和委托号列表的字典
        # # key为name，value为保存orderID（限价+本地停止）的集合
        # self.strategyOrderDict = {}

        # self._lstLogs = []               # 日志记录

    #----------------------------------------------------------------------
    #  properties
    #----------------------------------------------------------------------
    @property
    def priceTick(self):
        return self._priceTick

    @property
    def dbName(self):
        return self._dbName

    @property
    def ident(self) :
        return self.__class__.__name__ +"." + self._id

    @property
    def nextOrderReqId(self):
        with self._lock :
            self._orderId +=1
            return '%s@%s' % (self._orderId, self.ident)

    @property
    def collectionName_dpos(self):   return "dPos." + self.ident
    @property
    def collectionName_trade(self): return "trade." + self.ident

    #----------------------------------------------------------------------
    @abstractmethod
    def getPosition(self, symbol): # returns PositionData
        with self._lock :
            if not symbol in self._dictPositions:
                return PositionData()
            return copy.copy(self._dictPositions[symbol])

    def getAllPositions(self): # returns PositionData
        with self._lock :
            for pos in self._dictPositions.values() :
                price = self._trader.latestPrice(pos.symbol)
                if price >0:
                    pos.price = price
            return copy.deepcopy(self._dictPositions)

    @abstractmethod
    def cashAmount(self): # returns (avail, total)
        with self._lock :
            pos = self._dictPositions[Account.SYMBOL_CASH]
            volprice = pos.price * self.size
            return (pos.posAvail * volprice), (pos.position * volprice)

    def cashChange(self, dAvail=0, dTotal=0):
        with self._lock :
            return self._cashChange(dAvail, dTotal)

    @abstractmethod
    def insertData(self, collectionName, data) :
        self._trader.dbInsert(self._dbName, collectionName, data)

    def postEvent_Order(self, orderData):
        self._trader.postEvent(Account.EVENT_ORDER, copy.copy(orderData))
        self.debug('posted %s[%s]' % (Account.EVENT_ORDER, orderData.brokerOrderId))

    #----------------------------------------------------------------------
    # Account operations
    @abstractmethod
    def sendOrder(self, symbol, orderType, price, volume, strategy):
        """发单"""
        source = 'ACCOUNT'
        if strategy:
            source = strategy.id

        orderData = OrderData(self)
        # 代码编号相关
        orderData.symbol      = symbol
        orderData.exchange    = self._exchange
        orderData.price       = self.roundToPriceTick(price) # 报单价格
        orderData.totalVolume = volume    # 报单总数量
        orderData.source      = source

        # 报单方向
        if orderType == OrderData.ORDER_BUY:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_SELL:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_CLOSE
        elif orderType == OrderData.ORDER_SHORT:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_COVER:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_CLOSE     

        with self._lock :
            self._dictOutgoingOrders[orderData.reqId] = orderData
            self.debug('enqueued order[%s]' % orderData.desc)

        return orderData.reqId

    @abstractmethod
    def cancelOrder(self, brokerOrderId):
        with self._lock :
            self._lstOrdersToCancel.append(brokerOrderId)

        self.debug('enqueued order[%s]' % brokerOrderId)
        # self._broker_cancelOrder(brokerOrderId)

    @abstractmethod
    def sendStopOrder(self, symbol, orderType, price, volume, strategy):

        source = 'ACCOUNT'
        if strategy:
            source = strategy.name

        orderData = OrderData(self, True)
        # 代码编号相关
        orderData.symbol      = symbol
        orderData.exchange    = self._exchange
        orderData.price       = self.roundToPriceTick(price) # 报单价格
        orderData.totalVolume = volume    # 报单总数量
        orderData.source      = source
        
        # 报单方向
        if orderType == OrderData.ORDER_BUY:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_SELL:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_CLOSE
        elif orderType == OrderData.ORDER_SHORT:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_COVER:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_CLOSE     

        with self._lock :
            self._dictOutgoingOrders[orderData.reqId] = orderData
            self.debug('enqueued stopOrder[%s]' % orderData.desc)

        return orderData.reqId

    @abstractmethod
    def batchCancel(self, brokerOrderIds):
        for o in brokerOrderIds:
            self.cancelOrder(o)
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Interactions with BrokerDriver
    @abstractmethod
    def _broker_placeOrder(self, orderData):
        """发单"""
        raise NotImplementedError

    @abstractmethod
    def _broker_onOrderPlaced(self, orderData):
        """委托回调"""
        # order placed, move it from _dictOutgoingOrders to _dictLimitOrders
        with self._lock :
            del self._dictOutgoingOrders[orderData.reqId]
            if OrderData.STOPORDERPREFIX in orderData.reqId :
                self._dictStopOrders[orderData.brokerOrderId] = orderData
            else :
                self._dictLimitOrders[orderData.brokerOrderId] = orderData

        self.info('order[%s] has been placed, brokerOrderId[%s]' % (orderData.desc, orderData.brokerOrderId))

        if orderData.direction == OrderData.DIRECTION_LONG:
            turnover, commission, slippage = self.calcAmountOfTrade(orderData.symbol, orderData.price, orderData.totalVolume)
            self._cashChange(-(turnover + commission + slippage))

        self.postEvent_Order(orderData)

    @abstractmethod
    def _broker_cancelOrder(self, brokerOrderId):
        """撤单"""
        raise NotImplementedError

    @abstractmethod
    def _broker_onCancelled(self, orderData):
        """撤单回调"""
        orderData.status = OrderData.STATUS_CANCELLED
        if len(orderData.cancelTime) <=0:
            orderData.cancelTime = self._broker_datetimeAsOf().strftime('%H:%M:%S.%f')[:3]

        with self._lock :
            try :
                if not OrderData.STOPORDERPREFIX in orderData.reqId :
                    del self._dictLimitOrders[orderData.brokerOrderId]
                else :
                    del self._dictStopOrders[orderData.brokerOrderId]
            except:
                pass

            try :
                del self._dictOutgoingOrders[orderData.reqId]
            except:
                pass

            if orderData.direction == OrderData.DIRECTION_LONG:
                turnover, commission, slippage = self.calcAmountOfTrade(orderData.symbol, orderData.price, orderData.totalVolume)
                self._cashChange(turnover + commission + slippage)

        self.info('order.brokerOrderId[%s] canceled' % orderData.brokerOrderId)
        self.postEvent_Order(orderData)

    def findOrdersOfStrategy(self, strategyId, symbol=None):
        ret = []
        with self._lock :
            for o in self._dictLimitOrders.values():
                if o.source == strategyId:
                    ret.append(o)
            for o in self._dictStopOrders.values():
                if o.source == strategyId:
                    ret.append(o)
            for o in self._dictOutgoingOrders.values():
                if o.source == strategyId:
                    ret.append(o)

        return ret

    @abstractmethod
    def _broker_onOrderDone(self, orderData):
        """委托被执行"""
        with self._lock :
            try :
                if not OrderData.STOPORDERPREFIX in orderData.reqId :
                    del self._dictLimitOrders[orderData.brokerOrderId]
                else :
                    del self._dictStopOrders[orderData.brokerOrderId]
            except:
                pass

            if orderData.direction == OrderData.DIRECTION_LONG:
                turnover, commission, slippage = self.calcAmountOfTrade(orderData.symbol, orderData.price, orderData.totalVolume)
                self._cashChange(turnover + commission + slippage)

        self.postEvent_Order(orderData)

    @abstractmethod
    def _broker_onTrade(self, trade):
        """交易成功回调"""
        if trade.brokerTradeId in self._dictTrades:
            return

        trade.tradeID = "T" +trade.brokerTradeId +"@" + self.ident # to make the tradeID global unique
        with self._lock :
            self._dictTrades[trade.brokerTradeId] = trade

            # update the current postion, this may overwrite during the sync by BrokerDriver
            s = trade.symbol
            if not s in self._dictPositions :
                self._dictPositions[s] = PositionData()
                pos = self._dictPositions[s]
                pos.symbol = s
#                pos.vtSymbol = trade.vtSymbol
                pos.exchange = trade.exchange
            else:
                pos = self._dictPositions[s]
            
            # 持仓相关
            pos.price      = trade.price
            pos.direction  = trade.direction      # 持仓方向
            # pos.frozen =  # 冻结数量

            # update the position of symbol and its average cost
            if trade.direction != OrderData.DIRECTION_LONG:
                turnover, commission, slippage = self.calcAmountOfTrade(s, trade.price, -trade.volume)
                tradeAmount = turnover - commission - slippage
                # sold, increase both cash aval/total
                self._cashChange(tradeAmount, tradeAmount)

                pos.position -= trade.volume
                pos.posAvail  -= trade.volume
            else :
                turnover, commission, slippage = self.calcAmountOfTrade(s, trade.price, trade.volume)
                tradeAmount = turnover + commission + slippage
                self._cashChange(-tradeAmount, -tradeAmount)
                # calclulate pos.avgPrice
                if self.size <=0:
                    self.size =1
                cost = pos.position * pos.avgPrice *self.size
                cost += tradeAmount
                pos.position += trade.volume
                if pos.position >0:
                    pos.avgPrice = cost / pos.position /self.size
                else: pos.avgPrice =0

                # TODO: T+0 also need to increase pos.avalPos
                
            pos.stampByTrader = trade.dt  # the current position is calculated based on trade
            self.info('broker_onTrade() processed: %s=>pos' % (trade.desc))#, pos.desc))

        self._trader.postEvent(Account.EVENT_TRADE, copy.copy(trade))

    @abstractmethod
    def _broker_datetimeAsOf(self):
        return datetime.now()

    @abstractmethod
    def _broker_onGetAccountBalance(self, data, reqid):
        """查询余额回调"""
        pass
        
    @abstractmethod
    def _broker_onGetOrder(self, data, reqid):
        """查询单一委托回调"""
        pass

    def _broker_onGetOrders(self, data, reqid):
        """查询委托回调"""
        pass
        
    @abstractmethod
    def _broker_onGetMatchResults(self, data, reqid):
        """查询成交回调"""
        pass
        
    @abstractmethod
    def _broker_onGetMatchResult(self, data, reqid):
        """查询单一成交回调"""
        pass

    @abstractmethod
    def _broker_onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        pass

    # end with BrokerDriver
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Application routine
    def step(self) :
        pass

    # end of App routine
    #----------------------------------------------------------------------

    @abstractmethod
    def _cashChange(self, dAvail=0, dTotal=0): # thread unsafe
        pos = self._dictPositions[Account.SYMBOL_CASH]
        volprice = pos.price * self.size
        if pos.price <=0 :   # if cache.price not initialized
            volprice = pos.price =1
            if self.size >0:
                pos.price /=self.size

        dAvail /= volprice
        dTotal /= volprice
        
        self.debug('cashChange() avail[%s%+.3f] total[%s%+.3f]' % (pos.posAvail, dAvail, pos.position, dTotal))#, pos.desc))
        # double check if the cash account goes to negative
        newAvail, newTotal = pos.posAvail + dAvail, pos.position + dTotal
        if newAvail<0 or newTotal <0 or newAvail >(newTotal*1.05):
            self.error('cashChange() something wrong: newAvail[%s] newTotal[%s]' % (newAvail, newTotal)) #, pos.desc))
            exit(-1)

        pos.posAvail = newAvail
        pos.position = newTotal
        pos.stampByTrader = self._broker_datetimeAsOf()
        return True

    #----------------------------------------------------------------------

    @abstractmethod
    def calcAmountOfTrade(self, symbol, price, volume): raise NotImplementedError

    # return volume, commission, slippage
    @abstractmethod
    #----------------------------------------------------------------------
    # determine buy ability according to the available cash
    # return buy-volume-capabitilty, sell-volumes
    def maxOrderVolume(self, symbol, price):
        # calculate max buy volumes
        volume =0
        if price > 0 :
            cash, _  = self.cashAmount()
            volume   = round(cash / price / self.size -0.999,0)
            turnOver, commission, slippage = self.calcAmountOfTrade(symbol, price, volume)
            if cash < (turnOver + commission + slippage) :
                volume -= int((commission + slippage) / price / self.size) +1
            if volume <=0:
                volume =0
        
        with self._lock :
            if symbol in  self._dictPositions :
                return volume, self._dictPositions[symbol].posAvail

        return volume, 0

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
    def onStart(self):
        # ensure the DB collection has the index applied
        self._trader.dbEnsureIndex(self.collectionName_trade, [('brokerTradeId', ASCENDING)], True)
        self._trader.dbEnsureIndex(self.collectionName_dpos,  [('date', ASCENDING), ('symbol', ASCENDING)], True)

    @abstractmethod
    def onDayClose(self):
        self.dbSaveDataOfDay() # save the account data into DB
        
        self._datePrevClose = self._dateToday
        self._dateToday = None
        
        self._state = Account.STATE_CLOSE
        self.debug('onDayClose() saved positions, updated state')

    @abstractmethod
    def onDayOpen(self, newDate):
        if Account.STATE_OPEN == self._state:
            if newDate == self._dateToday :
                return
            self.onDayClose()

        self._dateToday = newDate
        self._dictTrades.clear() # clean the trade list
        # shift the positions, must do copy each PositionData
        self._prevPositions = self.getAllPositions()
        self._state = Account.STATE_OPEN
        self.debug('onDayOpen() shift pos to dict _prevPositions, updated state')
    
    @abstractmethod
    def onTimer(self, dt):
        # TODO refresh from BrokerDriver
        pass

    #----------------------------------------------------------------------
    # method to access Account DB
    #----------------------------------------------------------------------
    @abstractmethod
    def dbSaveDataOfDay(self):
        ''' save the account data into DB 
        1) trades that confirmed
        2) 
        '''
        if not self._trader:
            return

        with self._lock :
            # part 1. the confirmed trades
            tl = self._dictTrades.values()
            for t in tl:
                self._trader.dbUpdate(self.collectionName_trade, t.__dict__, {'brokerTradeId': t.brokerTradeId})
            self.info('saveDataOfDay() saved %s trades' % len(tl))

        # part 2. the daily position
        result, _ = self.calcDailyPositions()
        for l in result:
            self._trader.dbUpdate(self.collectionName_dpos, l, {'date':l['date'], 'symbol':l['symbol']})
        self.info('saveDataOfDay() saved positions: %s' % result)

    @abstractmethod
    def loadDB(self, since =None):
        ''' load the account data from DB ''' 
        if not since :
            since = self._datePrevClose
        pass

    #----------------------------------------------------------------------
    @abstractmethod
    def stdout(self, message):
        """输出内容"""
        print str(self._broker_datetimeAsOf()) + " ACC[" + self.ident + "] " + message

    def debug(self, msg):
        self._trader._engine.debug('ACC[%s,%s] %s' % (self.ident, self._broker_datetimeAsOf().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], msg))
        
    def info(self, msg):
        """正常输出"""
        self._trader._engine.info('ACC[%s,%s] %s' % (self.ident, self._broker_datetimeAsOf().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], msg))

    def warn(self, msg):
        """警告信息"""
        self._trader._engine.warn('ACC[%s,%s] %s' % (self.ident, self._broker_datetimeAsOf().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], msg))
        
    def error(self, msg):
        """报错输出"""
        self._trader._engine.error('ACC[%s,%s] %s' % (self.ident, self._broker_datetimeAsOf().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], msg))

    #----------------------------------------------------------------------
    def stop(self):
        """停止"""
        pass

    #----------------------------------------------------------------------
    #  account daily statistics methods
    #----------------------------------------------------------------------
    @abstractmethod
    def calcDailyPositions(self):
        """今日交易的结果"""

        tradesOfSymbol = { Account.SYMBOL_CASH:[] }        # 成交列表
        with self._lock :
            for t in self._dictTrades.values():
                if len(t.symbol) <=0:
                    t.symbol='$' #default symbol

                if not t.symbol in tradesOfSymbol:
                    tradesOfSymbol[t.symbol] = []
                tradesOfSymbol[t.symbol].append(t)

        result = []
        currentPositions = self.getAllPositions() # this is a duplicated copy, so thread-safe
        for s in currentPositions.keys():
            if not s in tradesOfSymbol.keys():
                tradesOfSymbol[s] = [] # fill a dummy trade list

        for s in tradesOfSymbol.keys():

            currentPos = currentPositions[s]

            if s in self._prevPositions:
                with self._lock :
                    prevPos = self._prevPositions[s]
            else:
                prevPos = PositionData()
            
            tcBuy      = 0
            tcSell     = 0
            tradingPnl = 0             # 交易盈亏
            totalPnl   = 0               # 总盈亏
            
            turnover   = 0               # 成交量
            commission = 0             # 手续费
            slippage   = 0               # 滑点
            netPnl     = 0                 # 净盈亏
            txnHist    = ""

            # DayX brought some:
            # {'recentPos': 200.0, 'cBuy': 2, 'recentPrice': 3.35, 'prevPos': 160.0, 'symbol': 'A601005', 'posAvail': 160.0, 'calcPos': 200.0, 
            # 'commission': 43.96, 'netPnl': -1472.74, 'avgPrice': 3.444, 'prevClose': 3.48, 'calcMValue': 67000.0, 'positionPnl': -1508.78, 
            # 'dailyPnl': -1428.78, 'cSell': 0, 'slippage': 0.0, 'date': u'20121219', 'tradingPnl': 80.0, 'asof': [datetime.datetime(2012, 12, 19, 14, 48), 0],
            # 'txnHist': '+20x3.31+20x3.35', 'turnover': 13320.0}
            # DayY no trade：
            # {'recentPos': 292.0, 'cBuy': 0, 'recentPrice': 3.26, 'prevPos': 292.0, 'symbol': 'A601005', 'posAvail': 292.0, 'calcPos': 292.0, 
            # 'commission': 0.0, 'netPnl': -4383.74, 'avgPrice': 3.41, 'prevClose': 3.27, 'calcMValue': 95192.0, 'positionPnl': -4383.74, 
            # 'dailyPnl': -4383.74, 'cSell': 0, 'slippage': 0.0, 'date': u'20121226', 'tradingPnl': 0.0, 'asof': [datetime.datetime(2012, 12, 24, 10, 20), 0],
            # 'txnHist': '', 'turnover': 0.0}
            # sold-all at last day
            # {'recentPos': 0.0, 'cBuy': 0, 'recentPrice': 3.94, 'prevPos': 292.0, 'symbol': 'A601005', 'posAvail': 0.0, 'calcPos': 0.0, 
            # 'commission': 375.14, 'netPnl': 15097.11, 'avgPrice': 3.41, 'prevClose': 3.59, 'calcMValue': 0.0, 'positionPnl': 15472.26, 
            # 'dailyPnl': 15472.26, 'cSell': 1, 'slippage': 0.0, 'date': u'20121228', 'tradingPnl': 0.0, 'asof': [datetime.datetime(2012, 12, 28, 15, 0), 0],
            #  'txnHist': '-292x3.94', 'turnover': 115048.0}]            

            # 持仓部分
            # TODO get from currentPos:  openPosition = openPosition
            positionPnl = round(prevPos.position * (currentPos.price - currentPos.avgPrice) * self.size, 3)
            """
            计算盈亏
            size: 合约乘数
            rate：手续费率
            slippage：滑点点数
            """
            posChange = 0
            calcPosition = prevPos.position

            for trade in tradesOfSymbol[s]:
                if trade.direction == OrderData.DIRECTION_LONG:
                    posChange = trade.volume
                    tcBuy += 1
                else:
                    posChange = -trade.volume
                    tcSell += 1

                txnHist += "%+dx%s" % (posChange, trade.price)

                tradingPnl += posChange * (currentPos.price - trade.price) * self.size
                calcPosition += posChange
                tover, comis, slpfee = self.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
                turnover += tover
                commission += comis
                slippage += slpfee

            if Account.SYMBOL_CASH == s: # cash dosn't need to sum trades
                calcPosition = currentPos.position
            elif calcPosition != currentPos.position:
                self.stdout("%s WARN: %s.calcDailyPositions() calcPos[%s] mismatch currentPos[%s]" %(self._dateToday, s, calcPosition, currentPos.position))

            # 汇总
            totalPnl = tradingPnl + positionPnl
            netPnl   = totalPnl - commission - slippage
            # stampstr = ''
            # if currentPos.stampByTrader :
            #     stampstr += currentPos.stampByTrader
            # [ currentPos.stampByTrader, currentPos.stampByBroker ]
            
            dstat = {
                'symbol'      : s,
                'date'        : self._dateToday,   # 日期
                'recentPrice' : round(currentPos.price, 3),     # 当日收盘
                'avgPrice'    : round(currentPos.avgPrice, 3),  # 持仓均价
                'recentPos'   : round(currentPos.position, 3),  # 当日收盘
                'posAvail'    : round(currentPos.posAvail, 3),  # the rest avail pos
                'calcPos'     : round(calcPosition, 3),     # MarketValue
                'calcMValue'  : round(calcPosition*currentPos.price*self.size , 2),     # 昨日收盘
                'prevClose'   : round(prevPos.price, 3),     # 昨日收盘
                'prevPos'     : round(prevPos.position, 3),     # 昨日收盘
                
                'turnover'    : round(turnover, 2),          # 成交量
                'commission'  : round(commission, 2),        # 手续费
                'slippage'    : round(slippage, 2),          # 滑点

                'tradingPnl'  : round(tradingPnl, 2),        # 交易盈亏
                'positionPnl' : round(positionPnl, 2),      # 持仓盈亏
                'dailyPnl'    : round(tradingPnl + positionPnl, 2), # 总盈亏
                'netPnl'      : round(netPnl, 2),           # 净盈亏
                
                'cBuy'        : tcBuy,             # 成交数量
                'cSell'       : tcSell,            # 成交数量
                'txnHist'     : txnHist,
                'asof'        : [ currentPos.stampByTrader, currentPos.stampByBroker ]
                }

            result.append(dstat)

        return result, tradesOfSymbol

    # def updateDailyStat(self, dt, price):
    #     """更新每日收盘价"""
    #     date = dt.date()
    #     self._statDaily = DailyResult(date, price)

    #     # 将成交添加到每日交易结果中
    #     for trade in self._dictTrades.values():
    #         self._statDaily.addTrade(trade)
            
    # def evaluateDailyStat(self, startdate, enddate):
    #     previousClose = 0
    #     openPosition = 0
    #     # TODO: read the statDaily from the DB
    #     # for dailyResult in self.dailyResultDict.values():
    #     #     dailyResult.previousClose = previousClose
    #     #     previousClose = dailyResult.closePrice
            
    #     #     dailyResult.calculatePnl(self._account, openPosition)
    #     #     openPosition = dailyResult.closePosition
            
    #     # 生成DataFrame
    #     resultDict ={}
    #     for k in dailyResult.__dict__.keys() :
    #         if k == 'tradeList' : # to exclude some columns
    #             continue
    #         resultDict[k] =[]

    #     for dailyResult in self.dailyResultDict.values():
    #         for k, v in dailyResult.__dict__.items() :
    #             if k in resultDict :
    #                 resultDict[k].append(v)
                
    #     resultDf = pd.DataFrame.from_dict(resultDict)
        
    #     # 计算衍生数据
    #     resultDf = resultDf.set_index('date')

    #     return resultDf


########################################################################
class Account_AShare(Account):
    """
    A股帐号，主要实现交易费和T+1
    """

    #----------------------------------------------------------------------
    def __init__(self, trader, settings=None):
        """Constructor"""
        super(Account_AShare, self).__init__(trader, settings)

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

    @abstractmethod
    def onDayOpen(self, newDate):
        super(Account_AShare, self).onDayOpen(newDate)
        with self._lock :
            # A-share will not keep yesterday's order alive
            # all the out-standing order will be cancelled
            self._dictOutgoingOrders.clear()
            
            self._lstOrdersToCancel =[]
            self._dictLimitOrders.clear()
            self._dictStopOrders.clear()

            # shift yesterday's position as available
            for pos in self._dictPositions.values():
                if Account.SYMBOL_CASH == pos.symbol:
                    continue
                self.debug('onDayOpen() shifting pos[%s] %s to avail %s' % (pos.symbol, pos.position, pos.posAvail))
                pos.posAvail = pos.position

        #TODO: sync with broker

########################################################################
class OrderData(EventData):
    """订单数据类"""
    # 本地停止单前缀
    STOPORDERPREFIX = 'SO#'

    # 涉及到的交易方向类型
    ORDER_BUY   = u'BUY'   # u'买开' 是指投资者对未来价格趋势看涨而采取的交易手段，买进持有看涨合约，意味着帐户资金买进合约而冻结
    ORDER_SELL  = u'SELL'  # u'卖平' 是指投资者对未来价格趋势不看好而采取的交易手段，而将原来买进的看涨合约卖出，投资者资金帐户解冻
    ORDER_SHORT = u'SHORT' # u'卖开' 是指投资者对未来价格趋势看跌而采取的交易手段，卖出看跌合约。卖出开仓，帐户资金冻结
    ORDER_COVER = u'COVER' # u'买平' 是指投资者将持有的卖出合约对未来行情不再看跌而补回以前卖出合约，与原来的卖出合约对冲抵消退出市场，帐户资金解冻

    # 本地停止单状态
    STOPORDER_WAITING   = u'WAITING'   #u'等待中'
    STOPORDER_CANCELLED = u'CANCELLED' #u'已撤销'
    STOPORDER_TRIGGERED = u'TRIGGERED' #u'已触发'

    # 方向常量
    DIRECTION_NONE = u'none'
    DIRECTION_LONG = u'long'
    DIRECTION_SHORT = u'short'
    DIRECTION_UNKNOWN = u'unknown'
    DIRECTION_NET = u'net'
    DIRECTION_SELL = u'sell'      # IB接口
    DIRECTION_COVEREDSHORT = u'covered short'    # 证券期权

    # 开平常量
    OFFSET_NONE = u'none'
    OFFSET_OPEN = u'open'
    OFFSET_CLOSE = u'close'
    OFFSET_CLOSETODAY = u'close today'
    OFFSET_CLOSEYESTERDAY = u'close yesterday'
    OFFSET_UNKNOWN = u'unknown'

    # 状态常量
    STATUS_NOTTRADED = u'pending'
    STATUS_PARTTRADED = u'partial filled'
    STATUS_ALLTRADED = u'filled'
    STATUS_CANCELLED = u'cancelled'
    STATUS_REJECTED = u'rejected'
    STATUS_UNKNOWN = u'unknown'

    #----------------------------------------------------------------------
    def __init__(self, account, stopOrder=False):
        """Constructor"""
        super(OrderData, self).__init__()
        
        # 代码编号相关
        self.reqId   = account.nextOrderReqId # 订单编号
        if stopOrder:
            self.reqId  = OrderData.STOPORDERPREFIX +self.reqId

        self.brokerOrderId = EventData.EMPTY_STRING   # 订单在vt系统中的唯一编号，通常是 Gateway名.订单编号
        self.accountId = account.ident          # 成交归属的帐号
        self.exchange = account._exchange    # 交易所代码

        self.symbol   = EventData.EMPTY_STRING         # 合约代码
        
        # 报单相关
        self.direction = EventData.EMPTY_UNICODE  # 报单方向
        self.offset = EventData.EMPTY_UNICODE     # 报单开平仓
        self.price = EventData.EMPTY_FLOAT        # 报单价格
        self.totalVolume = EventData.EMPTY_INT    # 报单总数量
        self.tradedVolume = EventData.EMPTY_INT   # 报单成交数量
        self.status = EventData.EMPTY_UNICODE     # 报单状态
        
        self.orderTime  = account._broker_datetimeAsOf().strftime('%H:%M:%S.%f')[:-3] # 发单时间
        self.cancelTime = EventData.EMPTY_STRING  # 撤单时间
        self.source     = EventData.EMPTY_STRING  # trigger source

    @property
    def desc(self) :
        return '%s(%s)-%s: %dx%s' % (self.direction, self.symbol, self.reqId, self.totalVolume, self.price)

# ########################################################################
# take the same as OrderData
# class StopOrder(object):
#     """本地停止单"""

#     #----------------------------------------------------------------------
#     def __init__(self):
#         """Constructor"""
#         self.vtSymbol = EventData.EMPTY_STRING
#         self.orderType = EventData.EMPTY_UNICODE
#         self.direction = EventData.EMPTY_UNICODE
#         self.offset = EventData.EMPTY_UNICODE
#         self.price = EventData.EMPTY_FLOAT
#         self.volume = EventData.EMPTY_INT
        
#         self.strategy = None             # 下停止单的策略对象
#         self.stopOrderID = EventData.EMPTY_STRING  # 停止单的本地编号 
#         self.status = EventData.EMPTY_STRING       # 停止单状态

########################################################################
class TradeData(EventData):
    """成交数据类"""
    # # 状态常量
    # STATUS_NOTTRADED  = u'NOTTRADED' # u'未成交'
    # STATUS_PARTTRADED = u'PARTTRADED' # u'部分成交'
    # STATUS_ALLTRADED  = u'ALLTRADED' # u'全部成交'
    # STATUS_CANCELLED  = u'CANCELLED' # u'已撤销'
    # STATUS_REJECTED   = u'REJECTED' # u'拒单'
    # STATUS_UNKNOWN    = u'UNKNOWN' # u'未知'

    #----------------------------------------------------------------------
    def __init__(self, account):
        """Constructor"""
        super(TradeData, self).__init__()
        
        self.tradeID   = EventData.EMPTY_STRING           # 成交编号
        self.brokerTradeId = EventData.EMPTY_STRING           # 成交在vt系统中的唯一编号，通常是 Gateway名.成交编号
        self.accountId = account.ident                    # 成交归属的帐号
        self.exchange = account._exchange                 # 交易所代码

        # 代码编号相关
        self.symbol = EventData.EMPTY_STRING              # 合约代码
        
        self.orderID = EventData.EMPTY_STRING             # 订单编号
        # self.vtOrderID = EventData.EMPTY_STRING           # 订单在vt系统中的唯一编号，通常是 Gateway名.订单编号
        
        # 成交相关
        self.direction = EventData.EMPTY_UNICODE          # 成交方向
        self.offset = EventData.EMPTY_UNICODE             # 成交开平仓
        self.price = EventData.EMPTY_FLOAT                # 成交价格
        self.volume = EventData.EMPTY_INT                 # 成交数量
        self.dt     = None                      # 成交时间 datetime
   
    @property
    def desc(self) :
        return '%s(%s)-%s: %dx%s' % (self.direction, self.symbol, self.brokerTradeId, self.volume, self.price)

########################################################################
class PositionData(EventData):
    """持仓数据类"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(PositionData, self).__init__()
        
        # 代码编号相关
        self.symbol   = EventData.EMPTY_STRING            # 合约代码
        self.exchange = EventData.EMPTY_STRING            # 交易所代码
        self.vtSymbol = EventData.EMPTY_STRING            # 合约在vt系统中的唯一代码，合约代码.交易所代码  
        
        # 持仓相关
        self.direction      = EventData.EMPTY_STRING      # 持仓方向
        self.position       = EventData.EMPTY_INT         # 持仓量
        self.posAvail       = EventData.EMPTY_INT         # 冻结数量
        self.price          = EventData.EMPTY_FLOAT       # 持仓最新交易价
        self.avgPrice       = EventData.EMPTY_FLOAT       # 持仓均价
        self.vtPositionName = EventData.EMPTY_STRING      # 持仓在vt系统中的唯一代码，通常是vtSymbol.方向
        # self.ydPosition     = EventData.EMPTY_INT         # 昨持仓
        # self.positionProfit = EventData.EMPTY_FLOAT       # 持仓盈亏
        self.stampByTrader   = EventData.EMPTY_INT         # 该持仓数是基于Trader的计算
        self.stampByBroker = EventData.EMPTY_INT        # 该持仓数是基于与broker的数据同步


########################################################################
class PositionDetail(object):
    """本地维护的持仓信息"""
    WORKING_STATUS = [OrderData.STATUS_UNKNOWN, OrderData.STATUS_NOTTRADED, OrderData.STATUS_PARTTRADED]
    
    MODE_NORMAL = 'normal'          # 普通模式
    MODE_SHFE = 'shfe'              # 上期所今昨分别平仓
    MODE_TDPENALTY = 'tdpenalty'    # 平今惩罚

    #----------------------------------------------------------------------
    def __init__(self, vtSymbol, contract=None):
        """Constructor"""
        self.vtSymbol = vtSymbol
        self.symbol = EventData.EMPTY_STRING
        self.exchange = EventData.EMPTY_STRING
        self.name = EventData.EMPTY_UNICODE    
        self.size = 1
        
        if contract:
            self.symbol = contract.symbol
            self.exchange = contract.exchange
            self.name = contract.name
            self.size = contract.size
        
        self.longPos = EventData.EMPTY_INT
        self.longYd = EventData.EMPTY_INT
        self.longTd = EventData.EMPTY_INT
        self.longPosFrozen = EventData.EMPTY_INT
        self.longYdFrozen = EventData.EMPTY_INT
        self.longTdFrozen = EventData.EMPTY_INT
        self.longPnl = EventData.EMPTY_FLOAT
        self.longPrice = EventData.EMPTY_FLOAT
        
        self.shortPos = EventData.EMPTY_INT
        self.shortYd = EventData.EMPTY_INT
        self.shortTd = EventData.EMPTY_INT
        self.shortPosFrozen = EventData.EMPTY_INT
        self.shortYdFrozen = EventData.EMPTY_INT
        self.shortTdFrozen = EventData.EMPTY_INT
        self.shortPnl = EventData.EMPTY_FLOAT
        self.shortPrice = EventData.EMPTY_FLOAT
        
        self.lastPrice = EventData.EMPTY_FLOAT
        
        self.mode = self.MODE_NORMAL
        self.exchange = EventData.EMPTY_STRING
        
        self._dictWorkingOrder = {}

