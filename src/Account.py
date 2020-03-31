# encoding: UTF-8

'''
This module defines a basic account
'''
from __future__ import division

from EventData    import EventData, EVENT_NAME_PREFIX, datetime2float
from Application  import BaseApplication
from MarketData  import MarketState
import HistoryData  as hist

from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

# from itertools import product
import threading # for locker
import copy
import traceback
# from pymongo import ASCENDING

#----------------------------------------------------------------------
def formatNumber(n, dec=2):
    """格式化数字到字符串"""
    rn = round(n, dec)      # 保留两位小数
    return format(rn, ',')  # 加上千分符

########################################################################
class MetaAccount(BaseApplication):
    ''' to make sure the child impl don't miss neccessary methods
    '''

    INDEX_ASCENDING = 'ASC' # ASCENDING

    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs):
        """Constructor"""
        super(MetaAccount, self).__init__(program, **kwargs)

        self._id = self.getConfig('accountId', self._id, True)
        self._exchange = self.getConfig('exchange', 'Unknown', True)
        self.__trader = None

    @property
    def exchange(self) : return self._exchange
    @property
    def trader(self): return self.__trader
    @property
    def account(self): return self
    def hostTrader(self, trader): self.__trader = trader

    @property
    def executable(self): return True

    @abstractmethod 
    def getPosition(self, symbol): raise NotImplementedError
    @abstractmethod 
    def positionState(self) :
        ''' get the account capitial including cash and positions
        @return tuple:
            cashAvail
            cashTotal
            positions [PositionData]
        '''
        raise NotImplementedError

    @abstractmethod 
    def cashAmount(self): raise NotImplementedError
    @abstractmethod 
    def summrizeBalance(self, positions=None, cashTotal=0) :
        ''' sum up the account capitial including cash and positions
        '''
        raise NotImplementedError
    @abstractmethod 
    def cashChange(self, dAvail=0, dTotal=0): raise NotImplementedError
    @abstractmethod 
    def record(self, category, row): raise NotImplementedError
    @abstractmethod 
    def postEvent_Order(self, orderData): raise NotImplementedError
    @abstractmethod 
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy): raise NotImplementedError
    @abstractmethod 
    def cancelOrder(self, brokerOrderId): raise NotImplementedError
    @abstractmethod 
    def batchCancel(self, brokerOrderIds): raise NotImplementedError
    @abstractmethod 
    def cancelAllOrders(self): raise NotImplementedError
    @abstractmethod 
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy): raise NotImplementedError
    @abstractmethod 
    def findOrdersOfStrategy(self, strategyId, symbol=None): raise NotImplementedError
    @abstractmethod 
    def datetimeAsOfMarket(self): raise NotImplementedError
    
    # @abstractmethod 
    # def _broker_placeOrder(self, orderData): raise NotImplementedError
    # @abstractmethod 
    # def _broker_cancelOrder(self, orderData): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onOrderPlaced(self, orderData): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onCancelled(self, orderData): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onOrderDone(self, orderData): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onTrade(self, trade): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetAccountBalance(self, data, reqid): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetOrder(self, data, reqid): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetOrders(self, data, reqid): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetMatchResults(self, data, reqid): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetMatchResult(self, data, reqid): raise NotImplementedError
    # @abstractmethod 
    # def _broker_onGetTimestamp(self, data, reqid): raise NotImplementedError

    @abstractmethod 
    def calcAmountOfTrade(self, symbol, price, volume): raise NotImplementedError
    @abstractmethod 
    def maxOrderVolume(self, symbol, price): raise NotImplementedError
    @abstractmethod 
    def roundToPriceTick(self, price): raise NotImplementedError
    # @abstractmethod 
    # def onStart(self): raise NotImplementedError
    # must be duplicated other than forwarding to _nest def doAppStep(self) : return self._nest.doAppStep()
    @abstractmethod 
    def onDayOpen(self, newDate): raise NotImplementedError
    @abstractmethod 
    def onDayClose(self): raise NotImplementedError
    @abstractmethod 
    def onTimer(self, dt): raise NotImplementedError
    # @abstractmethod 
    # def saveDB(self): raise NotImplementedError
    # @abstractmethod 
    # def loadDB(self, since =None): raise NotImplementedError
    @abstractmethod 
    def calcDailyPositions(self): raise NotImplementedError
    # @abstractmethod 
    # def saveSetting(self): raise NotImplementedError
    # @abstractmethod 
    # def updateDailyStat(self, dt, price): raise NotImplementedError
    # @abstractmethod 
    # def evaluateDailyStat(self, startdate, enddate): raise NotImplementedError

########################################################################
class Account(MetaAccount):
    """
    Basic Account
    """
    __lastId__ =10000

    RECCATE_ORDER       = 'Order'    # 报单历史记录, data=OrderData
    RECCATE_TRADE       = 'Trade'    # 报单回报历史记录, data=TradeData
    RECCATE_DAILYPOS    = 'DPos'     # 每日持仓历史记录, data=DailyPosition
    RECCATE_DAILYRESULT = 'DRes'     # 每日结果历史记录, data=DailyPosition

    EVENT_PREFIX   = EVENT_NAME_PREFIX + 'acc'
    EVENT_ORDER    = EVENT_PREFIX + RECCATE_ORDER    # 报单事件, data=OrderData
    EVENT_TRADE    = EVENT_PREFIX + RECCATE_TRADE    # 报单回报事件, data=TradeData
    EVENT_DAILYPOS = EVENT_PREFIX + RECCATE_DAILYPOS     # 每日持仓事件, data=DailyPosition

    # state of Account
    STATE_OPEN  = 'open'   # during trading hours
    STATE_CLOSE = 'close'  # during market close

    SYMBOL_CASH = '.RMB.'  # the dummy symbol in order to represent cache in _dictPositions

    BROKER_API_SYNC  = 'brocker.sync'  # sync API to call broker
    BROKER_API_ASYNC = 'brocker.async' # async API to call broker

    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs): # accountId, exchange, ratePer10K =30, contractSize=1, slippage =0.0, priceTick=0.0, jsettings =None):
        """Constructor
        """
        super(Account, self).__init__(program, **kwargs)

        self._slippage      = self.getConfig('slippage', 0.0)
        self._ratePer10K    = self.getConfig('ratePer10K', 30)
        self._contractSize  = self.getConfig('contractSize', 0.0)
        self._priceTick     = self.getConfig('priceTick', 0.0)
        self._dbName        = self.getConfig('dbName', self._id) 

        self._pctReservedCash = self.getConfig('pctReservedCash', 1.0) # cash at percent of total cap, in order not to buy securities at full
        if self._pctReservedCash < 0.5: self._pctReservedCash =0.5

        self._lock = threading.Lock()
        # the app instance Id
        if not self._id or len(self._id)<=0 :
            Account.__lastId__ +=1
            self._id = 'ACNT%d' % Account.__lastId__

        self._orderId = int(datetime2float(datetime.now())) % 100000000 # start with a big number

        self._state        = Account.STATE_CLOSE
        self._mode         = Account.BROKER_API_ASYNC

        self._recorder = None
        if self._contractSize <=0:
            self._contractSize =1

        # trader executer
        # self._dvrBroker = dvrBrokerClass(self, self._settings)

        self._dateToday      = None # date of previous close
        self._datePrevClose  = None # date of previous close
        self._prevPositions = {} # dict from symbol to previous PositionData
        self._todayResult = None

        self._dictOutgoingOrders = {} # the outgoing orders dict from reqId to OrderData that has not been confirmed with broker's orderId
        self._lstOrdersToCancel = []

        # cached data from broker
        cashpos = PositionData()
        cashpos.symbol = self.cashSymbol
        self._dictPositions = { # dict from symbol to latest PositionData
            self.cashSymbol : cashpos
        }
        self._dictTrades = {} # dict from tradeId to trade confirmed during today
        self._dictStopOrders = {} # dict from broker's orderId to OrderData that has been submitted but not yet traded
        self._dictLimitOrders = {} # dict from broker's orderId to OrderData that has been submitted but not yet traded

        # self.capital = 0        # 起始本金（默认10万）
        # self._cashAvail =0
        self._stampLastSync =0
        self._syncInterval  = 10

    def save(self):
        objId = '%s/%s' % (self.__class__.__name__, self._id)
        
        state = ( \
        self._dateToday, \
        self._datePrevClose, \
        self._prevPositions, \
        self._todayResult)

        self.program.saveObject(state, objId +'/s1')
        self.program.saveObject(self._dictOutgoingOrders, objId +'/outOrders')
        self.program.saveObject(self._lstOrdersToCancel, objId +'/ordersToCancel')
        self.program.saveObject(self._dictPositions, objId +'/positions')
        self.program.saveObject(self._dictTrades, objId +'/trades')
        self.program.saveObject((self._dictStopOrders,self._dictLimitOrders), objId +'/orders')

    def restore(self):  
        objId = '%s/%s' % (self.__class__.__name__, self._id)

        try:
            s1 = self.program.loadObject(objId +'/s1')
            if not s1:
                return False

            self._dateToday, \
            self._datePrevClose, \
            self._prevPositions, \
            self._todayResult \
            = s1

            self._dictOutgoingOrders = self.program.loadObject(objId +'/outOrders')
            self._lstOrdersToCancel = self.program.loadObject(objId +'/ordersToCancel')
            self._dictPositions = self.program.loadObject(objId +'/positions')
            self._dictTrades = self.program.loadObject(objId +'/trades')
            self._dictStopOrders,self._dictLimitOrders = self.program.loadObject(objId +'/orders')
            return True
        except Exception as ex:
            self.logexception(ex)
            
        return False

    #----------------------------------------------------------------------
    #  properties
    #----------------------------------------------------------------------
    @property
    def recorder(self):
        return self._recorder

    @property
    def cashSymbol(self):
        return self.SYMBOL_CASH # the dummy symbol in order to represent cache in _dictPositions

    @property
    def priceTick(self):
        return self._priceTick

    @property
    def contractSize(self):
        return self._contractSize

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

    def positionState(self) :
        ''' get the account capitial including cash and positions
        @return tuple:
            cashAvail
            cashTotal
            positions {symbol:PositionData}
        '''
        cashAvail, cashTotal = self.cashAmount()

        allpos ={}
        with self._lock :
            for s, pos in self._dictPositions.items() :
                if self.cashSymbol != s:
                    price = self.trader.marketState.latestPrice(pos.symbol)
                    if price >0:
                        pos.price = price
            allpos = copy.deepcopy(self._dictPositions)
            del allpos[self.cashSymbol]
        return cashAvail, cashTotal, allpos

    def summrizeBalance(self, positions=None, cashTotal=0) :
        ''' sum up the account capitial including cash and positions
        '''
        if positions is None:
            _, cashTotal, positions = self.positionState()

        posValueSubtotal =0
        for s, pos in positions.items():
            posValueSubtotal += pos.position * pos.price * self.contractSize

        return round(cashTotal,2), round(posValueSubtotal,2)

    def tradeBeginOfDay(dt = None):
        return (dt if dt else datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)

    def tradeEndOfDay(dt = None):
        return (dt if dt else datetime.now()).replace(hour=23, minute=59, second=59, microsecond=999999)

    def duringTradeHours(dt =None) : # to test if the time is trading hours
        if not dt:
            dt = datetime.now()

        return (dt >= Account.tradeBeginOfDay(dt) and dt <= Account.tradeEndOfDay(dt))

    #----------------------------------------------------------------------
    def datetimeAsOfMarket(self):
        return self.trader.marketState.getAsOf() if self.trader.marketState else datetime.now()

    def getPosition(self, symbol): # returns PositionData
        with self._lock :
            if not symbol in self._dictPositions:
                return PositionData()
            return copy.copy(self._dictPositions[symbol])

    def cashAmount(self): # returns (avail, total)
        with self._lock :
            pos = self._dictPositions[self.cashSymbol]
            volprice = pos.price * self._contractSize
            return (pos.posAvail * volprice), (pos.position * volprice)

    def cashChange(self, dAvail=0, dTotal=0):
        with self._lock :
            return self.__changePos(self.cashSymbol, dAvail, dTotal)

    def record(self, category, row) :
        if not self._recorder and self.trader :
            self._recorder = self.trader.recorder
        
        if self._recorder :
            return self._recorder.pushRow(category, row)

    def postEvent_Order(self, orderData):
        self.postEventData(Account.EVENT_ORDER, copy.copy(orderData))

    #----------------------------------------------------------------------
    # Account operations
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

        if self._mode == Account.BROKER_API_ASYNC :
            with self._lock :
                self._dictOutgoingOrders[orderData.reqId] = orderData
                self.debug('enqueued order[%s]' % orderData.desc)
        else :
            self._broker_placeOrder(orderData)

        return orderData.reqId

    def cancelOrder(self, brokerOrderId):

        if self._mode == Account.BROKER_API_ASYNC :
            with self._lock:
                if self._mode == Account.BROKER_API_ASYNC :
                    self._lstOrdersToCancel.append(brokerOrderId)
                    self.debug('enqueued order[%s]' % brokerOrderId)
                    return

            # self._mode == Account.BROKER_API_SYNC
        orderData = None
        with self._lock:
            try :
                dict = self._dictStopOrders if OrderData.STOPORDERPREFIX in brokerOrderId else self._dictLimitOrders
                orderData = dict[brokerOrderId]
            except KeyError:
                pass

        if orderData :
            self._broker_cancelOrder(orderData)

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

    def batchCancel(self, brokerOrderIds):
        for o in brokerOrderIds:
            self.cancelOrder(o)

    def cancelAllOrders(self):
        batch = []
        with self._lock :
            batch = self._dictOutgoingOrders.keys()
        self.batchCancel(batch)

    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Interactions with BrokerDriver
    def _broker_placeOrder(self, orderData):
        """发单"""
        raise NotImplementedError

    def _broker_onOrderPlaced(self, orderData):
        """委托回调"""

        if len(orderData.stampSubmitted) <=0:
            orderData.stampSubmitted = self.datetimeAsOfMarket().strftime('%H:%M:%S.%f')[:3]

        # order placed, move it from _dictOutgoingOrders to _dictLimitOrders
        with self._lock :
            try :
                del self._dictOutgoingOrders[orderData.reqId]
            except: pass

            if OrderData.STOPORDERPREFIX in orderData.reqId :
                self._dictStopOrders[orderData.brokerOrderId] = orderData
            else :
                self._dictLimitOrders[orderData.brokerOrderId] = orderData

        self.info('order[%s] has been placed, got brokerOrderId[%s]' % (orderData.desc, orderData.brokerOrderId))

        if orderData.direction == OrderData.DIRECTION_LONG:
            turnover, commission, slippage = self.calcAmountOfTrade(orderData.symbol, orderData.price, orderData.totalVolume)
            self.__changePos(self.cashSymbol, -(turnover + commission + slippage))
        elif orderData.direction == OrderData.DIRECTION_SHORT:
            with self._lock :
                pos = self._dictPositions[orderData.symbol]
                if pos:
                    pos.posAvail = round(pos.posAvail - orderData.totalVolume, 2)

        self.postEvent_Order(orderData)
        self.record(Account.RECCATE_ORDER, orderData)

    def _broker_cancelOrder(self, brokerOrderId):
        """撤单"""

    def _broker_onCancelled(self, orderData):
        """撤单回调"""
        orderData.status = OrderData.STATUS_CANCELLED
        if len(orderData.stampCanceled) <=0:
            orderData.stampCanceled = self.datetimeAsOfMarket().strftime('%H:%M:%S.%f')[:3]

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
                self.__changePos(self.cashSymbol, turnover + commission + slippage)
            elif orderData.direction == OrderData.DIRECTION_SHORT:
                pos = self._dictPositions[orderData.symbol]
                if pos:
                    pos.posAvail = round(pos.posAvail + orderData.totalVolume, 2)

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

    def _broker_onOrderDone(self, orderData):
        """委托被执行"""
        if len(orderData.stampFinished) <=0:
            orderData.stampFinished = self.datetimeAsOfMarket().strftime('%H:%M:%S.%f')[:3]

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
                self.__changePos(self.cashSymbol, turnover + commission + slippage)

        self.postEvent_Order(orderData)

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
            strPrevPos = '%s/%s' % (pos.posAvail, pos.position)

            # update the position of symbol and its average cost
            if trade.direction != OrderData.DIRECTION_LONG:
                turnover, commission, slippage = self.calcAmountOfTrade(s, trade.price, -trade.volume)
                tradeAmount = turnover - commission - slippage
                # sold, increase both cash aval/total
                self.__changePos(self.cashSymbol, tradeAmount, tradeAmount)

                pos.position -= trade.volume
                # posAvail was sustracted when pos.posAvail -= trade.volume
            else :
                turnover, commission, slippage = self.calcAmountOfTrade(s, trade.price, trade.volume)
                tradeAmount = turnover + commission + slippage
                self.__changePos(self.cashSymbol, -tradeAmount, -tradeAmount)
                # calclulate pos.avgPrice
                if self._contractSize <=0:
                    self._contractSize =1
                cost = pos.position * pos.avgPrice *self._contractSize
                cost += tradeAmount
                pos.position += trade.volume
                if pos.position >0:
                    pos.avgPrice = cost / pos.position /self._contractSize
                else: pos.avgPrice =0

                # TODO: T+0 also need to increase pos.avalPos
                
            pos.stampByTrader = trade.datetime  # the current position is calculated based on trade
        
        cashAvail, cashTotal = self.cashAmount()
        self.info('broker_onTrade() trade[%s] processed, pos[%s->%s/%s] cash[%.2f/%.2f]' % (trade.desc, strPrevPos, pos.posAvail, pos.position, cashAvail, cashTotal))#, pos.desc))
        self.postEventData(Account.EVENT_TRADE, copy.copy(trade))
        self.record(Account.RECCATE_TRADE, trade)

    def _broker_onOpenOrders(self, dictOrders):
        """枚举订单回调"""
        orderIds = dictOrders.keys()
        newlist = []
        gonelist = []
        with self._lock :
            for o in orderIds :
                if o in self._dictLimitOrders.keys() :
                    self._dictLimitOrders[o].status = o.status
                elif o in self._dictStopOrders.keys() :
                    self._dictStopOrders[o].status = o.status
                else: # new
                    newlist.append(o)
            
            for o in self._dictLimitOrders.keys() :
                if not o in orderIds:
                    gonelist.append(self._dictLimitOrders[o])
                
            for o in self._dictStopOrders.keys() :
                if not o in orderIds:
                    gonelist.append(self._dictStopOrders[o])

    def _broker_onTradedOrders(self, dictOrders):
        """枚举订单回调"""
        orderIds = dictOrders.keys()
        newlist = []
        gonelist = []
        with self._lock :
            for o in orderIds :
                if o in self._dictLimitOrders.keys() :
                    self._dictLimitOrders[o].status = o.status
                elif o in self._dictStopOrders.keys() :
                    self._dictStopOrders[o].status = o.status
                else: # new
                    newlist.append(o)
            
            for o in self._dictLimitOrders.keys() :
                if not o in orderIds:
                    gonelist.append(self._dictLimitOrders[o])
                
            for o in self._dictStopOrders.keys() :
                if not o in orderIds:
                    gonelist.append(self._dictStopOrders[o])

        self.warn('unrecognized open orders, force to cancel: %s' % newlist)
        self.batchCancel(newlist)

        for o in gonelist:
            self.warn('gone orders, force to perform onCancel: %s' % newlist)
            self._broker_cancelOrder(o)

    def _broker_onGetAccountBalance(self, data, reqid):
        """查询余额回调"""
        pass
        
    def _broker_onGetOrder(self, data, reqid):
        """查询单一委托回调"""
        pass

    def _broker_onGetOrders(self, data, reqid):
        """查询委托回调"""
        pass
        
    def _broker_onGetMatchResults(self, data, reqid):
        """查询成交回调"""
        pass
        
    def _broker_onGetMatchResult(self, data, reqid):
        """查询单一成交回调"""
        pass

    def _broker_onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        pass

    def _brocker_procSyncData(self):
        pass

    def _brocker_triggerSync(self):
        pass

    # end with BrokerDriver
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # impl of BaseApplication

    def doAppInit(self): # return True if succ
        if not super(Account, self).doAppInit() :
            return False

        if not self._recorder :
            # find the recorder from the program
            searchKey = '.%s' % self._id
            for recorderId in self._program.listApps(hist.Recorder) :
                pos = recorderId.find(searchKey)
                if pos >0 and recorderId[pos:] == searchKey:
                    self._recorder = self._program.getApp(recorderId)
                    if self._recorder : break

        if self._recorder :
            self.info('taking recoder[%s]' % self._recorder.ident)
            self._recorder.registerCategory(Account.RECCATE_ORDER,       params= {'columns' : OrderData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_TRADE,       params= {'columns' : TradeData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYPOS,    params= {'columns' : DailyPosition.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYRESULT, params= {'columns' : DailyResult.COLUMNS})

            # # ensure the DB collection has the index applied
            # self._recorder.configIndex(self.collectionName_trade, [('brokerTradeId', INDEX_ASCENDING)], True)
            # self._recorder.configIndex(self.collectionName_dpos,  [('date', INDEX_ASCENDING), ('symbol', INDEX_ASCENDING)], True)

        # # find the marketstate
        # if not self.trader.marketState :
        #     for obsId in self._program.listByType(MarketState) :
        #         marketstate = self._program.getObj(obsId)
        #         if marketstate and marketstate.exchange == self.exchange:
        #             self._marketState = marketstate
        #             break

        return True

    def doAppStep(self):

        # step 1. flush out-going orders and cancels to the broker
        outgoingOrders = []
        ordersToCancel = []

        cStep =0

        with self._lock:
            outgoingOrders = copy.deepcopy(self._dictOutgoingOrders.values()) if len(self._dictOutgoingOrders) >0 else []

            # find out he orderData by brokerOrderId
            for odid in self._lstOrdersToCancel :
                orderData = None
                cStep +=1
                try :
                    if OrderData.STOPORDERPREFIX in odid :
                        orderData = self._dictStopOrders[odid]
                    else :
                        orderData = self._dictLimitOrders[odid]
                except KeyError:
                    pass

                if orderData :
                    ordersToCancel.append(copy.copy(orderData))

            self._lstOrdersToCancel = []

        for co in ordersToCancel:
            self._broker_cancelOrder(co)
            cStep +=1

        for no in outgoingOrders:
            self._broker_placeOrder(no)
            cStep +=1

        if (len(ordersToCancel) + len(outgoingOrders)) >0:
            self.debug('step() cancelled %d orders, placed %d orders'% (len(ordersToCancel), len(outgoingOrders)))

        # step 2. sync positions and order with the broker
        self._brocker_procSyncData()
        stampNow = datetime2float(datetime.now())
        if self._stampLastSync + self._syncInterval < stampNow :
            self._stampLastSync = stampNow
            self._brocker_triggerSync()
            cStep +=1

        return cStep

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __changePos(self, symbol, dAvail=0, dTotal=0): # thread unsafe
        pos = self._dictPositions[symbol]
        volprice = pos.price * self._contractSize
        if pos.price <=0 and symbol == self.cashSymbol:   # if cache.price not initialized
            volprice = pos.price =1
            if self._contractSize >0:
                pos.price /=self._contractSize

        dAvail /= volprice
        dTotal /= volprice
        strTxn = '%s:avail[%s%+.3f],total[%s%+.3f]' % (symbol, pos.posAvail, dAvail, pos.position, dTotal)
        
        self.debug('__changePos() %s' % strTxn)

        newAvail, newTotal = pos.posAvail + dAvail, pos.position + dTotal

        # double check if the cash account goes to negative
        allowedError = pos.position * 0.0001 # because some err at float calculating
        if newAvail< -allowedError :
            newAvail =0.0
        
        if newTotal <-allowedError or newAvail >(newTotal*1.05): # because some err at float calculating
            raise ValueError('__changePos(%s) something wrong: newAvail[%s] newTotal[%s] with allowed err[%s]' % (strTxn, newAvail, newTotal, allowedError))

        pos.posAvail = round(newAvail, 3)
        pos.position = round(newTotal, 3)
        pos.stampByTrader = self.datetimeAsOfMarket()
        return True

    #----------------------------------------------------------------------

    @abstractmethod
    def calcAmountOfTrade(self, symbol, price, volume):
        raise NotImplementedError
    # return volume, commission, slippage

    #----------------------------------------------------------------------
    # determine buy ability according to the available cash
    # return buy-volume-capabitilty, sell-volumes
    def maxOrderVolume(self, symbol, price):
        # calculate max buy volumes
        volume =0
        if price > 0 :
            cashAvail, cashTotal, positions = self.positionState()
            _, posvalue = self.summrizeBalance(positions, cashTotal)
            reservedCash = (cashTotal + posvalue) * self._pctReservedCash /100
            cashAvail -= reservedCash

            volume   = round(cashAvail / (price + self._priceTick) / self._contractSize -0.999, 0)
            turnOver, commission, slippage = self.calcAmountOfTrade(symbol, price, volume)
            if cashAvail < (turnOver + commission + slippage) :
                volume -= int((commission + slippage) / price / self._contractSize) +1
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
        
        newPrice = round(round(price/self._priceTick, 0) * self._priceTick, 4)
        return newPrice

    #----------------------------------------------------------------------
    # callbacks about timing
    def OnEvent(self, ev):
        '''
        process the event
        '''
        pass

    def onDayClose(self):

        if Account.STATE_CLOSE == self._state:
            return

        self.debug('onDayClose() calculating daily result')
        cTrades =0
        positions, _ = self.calcDailyPositions()
        self._todayResult.cash, self._todayResult.posValue = self.summrizeBalance()
        self._todayResult.endBalance = round(self._todayResult.cash + self._todayResult.posValue, 3)

        # part 1. 汇总 the confirmed trades, and save
        with self._lock :
            for trade in self._dictTrades.values():
                if trade.datetime and self._dateToday != trade.datetime.strftime('%Y-%m-%d') :
                    self.warn('onDayClose(%s) found a trade not as of today: %s @%s' % (self._dateToday, trade.desc, trade.datetime) )
                    continue

                cTrades +=1

                if trade.direction == OrderData.DIRECTION_LONG:
                    posChange = trade.volume
                    self._todayResult.tcBuy += 1
                else:
                    posChange = -trade.volume
                    self._todayResult.tcSell += 1
                    
                self._todayResult.txnHist += "%+dx%s@%s" % (posChange, trade.symbol, formatNumber(trade.price,3))

                self._todayResult.tradingPnl += round(posChange * (self._marketState.latestPrice(trade.symbol) - trade.price) * self._contractSize, 2)
                turnover, commission, slippagefee = self.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
                self._todayResult.turnover += turnover
                self._todayResult.commission += commission
                self._todayResult.slippage += slippagefee

            self._todayResult.totalPnl = round(self._todayResult.tradingPnl + self._todayResult.positionPnl, 2)
            self._todayResult.netPnl = round(self._todayResult.totalPnl - self._todayResult.commission - self._todayResult.slippage, 2)

            self._todayResult.turnover   = round(self._todayResult.turnover, 3)
            self._todayResult.commission = round(self._todayResult.commission, 3)
            self._todayResult.slippage   = round(self._todayResult.slippage, 3)

            cTds = len(self._dictTrades)
            if cTds >0 :
                self.info('onDayClose(%s) summed %s/%s trades: %s' % (self._dateToday, cTrades, cTds, self._todayResult.txnHist))
            else:
                self.debug('onDayClose(%s) no trades' % (self._dateToday))

        #TODO # part 2. summarize the positions
        # self._todayResult.positionPnl=0
        # if len(positions) >0:
        #     for dpos in positions:
        #         # self.trader.dbUpdate(self.collectionName_dpos, dpos, {'date':dpos['date'], 'symbol':dpos['symbol']})
        #         line = self.record(Account.RECCATE_DAILYPOS, dpos)
        #         self._todayResult.positionPnl += round(self.openPosition * (self.closePrice - self.previousClose) * account.size, 3)
        #         self.closePosition = self.openPosition
        #     self.debug('saveDataOfDay() saved %d positions into DB' % len(positions))

        # part 3. record the daily result and positions
        self.record(Account.RECCATE_DAILYRESULT, self._todayResult)

        self._datePrevClose = self._dateToday
        self._dateToday = None
        
        self._state = Account.STATE_CLOSE
        self.debug('onDayClose(%s) saved positions, updated state' % self._datePrevClose)

    def onDayOpen(self, newDate):
        if Account.STATE_OPEN == self._state:
            if self._dateToday and newDate <= self._dateToday :
                return
            self.onDayClose()

        # shift the positions, must do copy each PositionData
        cashAvail, cashTotal, self._prevPositions = self.positionState()
        posCap =0

        if self._todayResult :
            prevBalance = self._todayResult.endBalance
            posCap = self._todayResult.posValue
        else :
            _, posCap = self.summrizeBalance(positions=self._prevPositions, cashTotal=cashTotal)
            prevBalance =  cashTotal + posCap

        self._dateToday = newDate

        self._todayResult = DailyResult(self._dateToday, startBalance=prevBalance)
        self._todayResult.cash, self._todayResult.posValue = cashTotal, posCap

        self._dictTrades.clear() # clean the trade list
        self._state = Account.STATE_OPEN
        self.debug('onDayOpen(%s) updated todayResult: %s+%s' % (self._dateToday, self._todayResult.cash, self._todayResult.posValue))
    
    def onTimer(self, dt):
        # TODO refresh from BrokerDriver
        pass

    # #----------------------------------------------------------------------
    # # method to access Account DB
    def loadDB(self, since =None):
        ''' load the account data from DB ''' 
        if not since :
            since = self._datePrevClose
        pass

    # #----------------------------------------------------------------------
    # @property
    # def __logtag(self):
    #     # asof = self.datetimeAsOfMarket()
    #     # asof = asof.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] if asof else ''
    #     # return 'ACC[%s,%s] ' % (self.ident, asof)
    #     return 'ACC[%s] ' % self.ident

    # def debug(self, msg):
    #     super(Account, self).debug(self.__logtag + msg)
        
    # def info(self, msg):
    #     super(Account, self).info(self.__logtag + msg)

    # def warn(self, msg):
    #     super(Account, self).warn(self.__logtag + msg)
        
    # def error(self, msg):
    #     super(Account, self).error(self.__logtag + msg)

    #----------------------------------------------------------------------
    #  account daily statistics methods

    def calcDailyPositions(self):
        """今日交易的结果"""

        tradesOfSymbol = {} # { self.cashSymbol: [] }        # 成交列表
        if not self.trader:
            return [], tradesOfSymbol

        with self._lock :
            for t in self._dictTrades.values():
                if len(t.symbol) <=0:
                    t.symbol='$' #default symbol

                if not t.symbol in tradesOfSymbol:
                    tradesOfSymbol[t.symbol] = []
                tradesOfSymbol[t.symbol].append(t)

        result = []
        cashAvail, cashTotal, currentPositions = self.positionState() # this is a duplicated copy, so thread-safe
        for s in currentPositions.keys():
            if not s in tradesOfSymbol.keys():
                tradesOfSymbol[s] = [] # fill a dummy trade list

        for s in tradesOfSymbol.keys():
            currentPos = currentPositions[s]
            ohlc =  self.trader.marketState.dailyOHLC_sofar(s)

            if s in self._prevPositions:
                with self._lock :
                    prevPos = copy.copy(self._prevPositions[s])
            else:
                prevPos = PositionData()
            
            dpos = DailyPosition()
            dpos.initPositions(self, s, currentPos, prevPos, ohlc)
            
            for trade in tradesOfSymbol[s]:
                dpos.pushTrade(self, trade)

            dpos.close()
            self.record(Account.RECCATE_DAILYPOS, dpos)

            self.postEventData(Account.EVENT_DAILYPOS, dpos)
            result.append(dpos.__dict__)

        return result, tradesOfSymbol


########################################################################
class Account_AShare(Account):
    """
    A股帐号，主要实现交易费和T+1
    """

    ANNUAL_TRADE_DAYS = 244 # AShare has about 244 trade days every year

    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs):
        """Constructor"""
        # some default settings known for AShare
        if not 'exchange' in kwargs.keys() :
            kwargs['exchange'] ='AShare'
        if not 'contractSize' in kwargs.keys() :
            kwargs['contractSize'] =100
        if not 'priceTick' in kwargs.keys() :
            kwargs['priceTick'] = 0.01

        super(Account_AShare, self).__init__(program, **kwargs) # accountId, exchange ='AShare', ratePer10K =ratePer10K, contractSize=100, slippage =0.0, priceTick=0.01, jsettings =jsettings)

    def tradeBeginOfDay(dt = None):
        return (dt if dt else datetime.now()).replace(hour=9, minute=30, second=0, microsecond=0)

    def tradeEndOfDay(dt = None):
        return (dt if dt else datetime.now()).replace(hour=15, minute=0, second=0, microsecond=0)

    def duringTradeHours(dt =None) : # to test if the time is in 9:28 ~11:32 and 13:00 ~15:00
        if not dt:
            dt = datetime.now()
        
        if dt.weekday() in [5,6] : # datetime.now().weekday() map 0-Mon,1-Tue,2-Wed,3-Thu,4-Fri,5-Sat,6-Sun
            return False
        
        if not dt.hour in [9, 10, 11, 13, 14] :
            return False
        
        if 9 == dt.hour and dt.minute < 29 :
            return False
        
        if 11 == dt.hour and dt.minute >31 :
            return False

        return True

    #----------------------------------------------------------------------
    def calcAmountOfTrade(self, symbol, price, volume):
        # 交易手续费=印花税+过户费+券商交易佣金
        volumeX1 = abs(volume) * self._contractSize
        turnOver = price * volumeX1

        # 印花税: 成交金额的1‰, 目前向卖方单边征收
        tax = 0
        if volumeX1 <0:
            tax = turnOver /1000
            
        #过户费（仅上海收取，也就是买卖上海股票时才有）：每1000股收取1元，不足1000股按1元收取
        transfer =0
        if len(symbol)>2 and (symbol[1]=='6' or symbol[1]=='7'):
            transfer = int((volumeX1+999)/1000)
            
        #3.券商交易佣金 最高为成交金额的3‰，最低5元起，单笔交易佣金不满5元按5元收取。
        commission = max(turnOver * self._ratePer10K /10000, 5)

        return turnOver, tax + transfer + commission, volumeX1 * self._slippage

    def onDayOpen(self, newDate):
        super(Account_AShare, self).onDayOpen(newDate)
        with self._lock :
            # A-share will not keep yesterday's order alive
            # all the out-standing order will be cancelled
            cOutgoingOrders, cOrdersToCancel, cLimitOrders, cStopOrders = len(self._dictOutgoingOrders), len(self._lstOrdersToCancel), len(self._dictLimitOrders), len(self._dictStopOrders)

            self._dictOutgoingOrders.clear()
            self._lstOrdersToCancel =[]
            self._dictLimitOrders.clear()
            self._dictStopOrders.clear()

            # shift yesterday's position as available
            strshift=''
            for pos in self._dictPositions.values():
                # cash also need shift as above _dictOutgoingOrders.clear() may lead to avalCash != total
                # if self.cashSymbol == pos.symbol:
                #     continues
                if pos.position != pos.posAvail :
                    strshift += '%s[%sov%s], ' % (pos.symbol, pos.position, pos.posAvail)
                pos.posAvail = pos.position

            if sum([cOutgoingOrders, cOrdersToCancel, cLimitOrders, cStopOrders])>0 or len(strshift) >0:
                self.info('onDayOpen(%s) cleared %d,%d,%d,%d orders and shifted avail-positions: %s' % (self._dateToday, cOutgoingOrders, cOrdersToCancel, cLimitOrders, cStopOrders, strshift))

        #TODO: sync with broker

########################################################################
class OrderData(EventData):
    """订单数据类"""
    # 本地停止单前缀
    STOPORDERPREFIX = 'SO#'

    # 涉及到的交易方向类型
    ORDER_BUY   = 'BUY'   # u'买开' 是指投资者对未来价格趋势看涨而采取的交易手段，买进持有看涨合约，意味着帐户资金买进合约而冻结
    ORDER_SELL  = 'SELL'  # u'卖平' 是指投资者对未来价格趋势不看好而采取的交易手段，而将原来买进的看涨合约卖出，投资者资金帐户解冻
    ORDER_SHORT = 'SHORT' # u'卖开' 是指投资者对未来价格趋势看跌而采取的交易手段，卖出看跌合约。卖出开仓，帐户资金冻结
    ORDER_COVER = 'COVER' # u'买平' 是指投资者将持有的卖出合约对未来行情不再看跌而补回以前卖出合约，与原来的卖出合约对冲抵消退出市场，帐户资金解冻

    # 本地停止单状态
    STOPORDER_WAITING   = 'WAITING'   #u'等待中'
    STOPORDER_CANCELLED = 'CANCELLED' #u'已撤销'
    STOPORDER_TRIGGERED = 'TRIGGERED' #u'已触发'

    # 方向常量
    DIRECTION_NONE = 'NONE'
    DIRECTION_LONG = 'LONG'
    DIRECTION_SHORT = 'SHORT'
    DIRECTION_UNKNOWN = 'UNKNOWN'
    DIRECTION_NET =  'NET'
    DIRECTION_SELL = 'SELL'  # IB接口
    DIRECTION_COVEREDSHORT = 'COVERED_SHORT'    # 证券期权

    # 开平常量
    OFFSET_NONE = 'none'
    OFFSET_OPEN = 'open'
    OFFSET_CLOSE = 'close'
    OFFSET_CLOSETODAY = 'close today'
    OFFSET_CLOSEYESTERDAY = 'close yesterday'
    OFFSET_UNKNOWN = 'unknown'

    # 状态常量
    STATUS_CREATED    = 'created'
    STATUS_SUBMITTED  = 'submitted'
    STATUS_PARTTRADED = 'partial-filled'
    STATUS_ALLTRADED  = 'filled'
    STATUS_PARTCANCEL = 'partial-canceled'
    STATUS_CANCELLED  = 'canceled'
    STATUS_REJECTED   = 'rejected'
    STATUS_UNKNOWN    = 'unknown'

    STATUS_OPENNING   = [STATUS_SUBMITTED, STATUS_PARTTRADED]
    STATUS_FINISHED   = [STATUS_ALLTRADED, STATUS_PARTCANCEL]
    STATUS_CLOSED     = STATUS_FINISHED + [STATUS_CANCELLED]

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime,symbol,reqId,brokerOrderId,direction,price,totalVolume,tradedVolume,offset,status,source' # ,stampSubmitted,stampCanceled,stampFinished'

    #----------------------------------------------------------------------
    def __init__(self, account, stopOrder=False, reqId = None):
        """Constructor"""
        super(OrderData, self).__init__()
        
        # 代码编号相关
        self.reqId   = reqId if reqId else account.nextOrderReqId # 订单编号
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
        self.status = OrderData.STATUS_CREATED     # 报单状态
        self.source   = EventData.EMPTY_STRING  # trigger source
        
        self.stampSubmitted  = EventData.EMPTY_STRING # 发单时间
        self.stampCanceled   = EventData.EMPTY_STRING  # 撤单时间
        self.stampFinished   = EventData.EMPTY_STRING  # 撤单时间

    @property
    def desc(self) :
        return 'O%s:%s(%s) %dx%s' % (self.reqId, self.direction, self.symbol, self.totalVolume, self.price)

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
    # STATUS_SUBMITTED  = u'NOTTRADED' # u'未成交'
    # STATUS_PARTTRADED = u'PARTTRADED' # u'部分成交'
    # STATUS_ALLTRADED  = u'ALLTRADED' # u'全部成交'
    # STATUS_CANCELLED  = u'CANCELLED' # u'已撤销'
    # STATUS_REJECTED   = u'REJECTED' # u'拒单'
    # STATUS_UNKNOWN    = u'UNKNOWN' # u'未知'

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime,symbol,tradeID,brokerTradeId,direction,orderID,price,volume,offset'

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
   
    @property
    def desc(self) :
        return 'T%s:%s(%s) %dx%s' % (self.brokerTradeId, self.direction, self.symbol, self.volume, round(self.price, 3))

########################################################################
class PositionData(EventData):
    '''持仓数据类'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,direction,position,posAvail,price,avgPrice,stampByTrader,stampByBroker'

    #----------------------------------------------------------------------
    def __init__(self):
        '''Constructor'''

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
        # self.vtPositionName = EventData.EMPTY_STRING      # 持仓在vt系统中的唯一代码，通常是vtSymbol.方向
        # self.ydPosition     = EventData.EMPTY_INT         # 昨持仓
        # self.positionProfit = EventData.EMPTY_FLOAT       # 持仓盈亏
        self.stampByTrader   = EventData.EMPTY_INT         # 该持仓数是基于Trader的计算
        self.stampByBroker   = EventData.EMPTY_INT        # 该持仓数是基于与broker的数据同步

########################################################################
class DailyPosition(EventData):
    '''每日交易的结果'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'date,symbol,recentPrice,avgPrice,recentPos,posAvail,calcPos,calcMValue,prevClose,prevPos,execOpen,execHigh,execLow,turnover,commission,_slippage,tradingPnl,positionPnl,dailyPnl,netPnl,cBuy,cSell,txns'

    def __init__(self):
        """Constructor"""
        self.symbol      = EventData.EMPTY_STRING
        self.date        = EventData.EMPTY_STRING   # 日期
        self.recentPrice = EventData.EMPTY_FLOAT     # 当日收盘
        self.avgPrice    = EventData.EMPTY_FLOAT  # 持仓均价
        self.recentPos   = EventData.EMPTY_FLOAT  # 当日收盘
        self.posAvail    = EventData.EMPTY_FLOAT  # the rest avail pos
        self.calcPos     = EventData.EMPTY_FLOAT     # 
        self.calcMValue  = EventData.EMPTY_FLOAT # MarketValue
        self.prevClose   = EventData.EMPTY_FLOAT     # 昨日收盘
        self.prevPos     = EventData.EMPTY_FLOAT    # 昨日收盘
        self.execOpen    = EventData.EMPTY_FLOAT
        self.execHigh    = EventData.EMPTY_FLOAT 
        self.execLow     = EventData.EMPTY_FLOAT
                
        self.turnover    = EventData.EMPTY_FLOAT        # 成交量
        self.commission  = EventData.EMPTY_FLOAT       # 手续费
        self._slippage   = EventData.EMPTY_FLOAT         # 滑点

        self.tradingPnl  = EventData.EMPTY_FLOAT    # 交易盈亏
        self.positionPnl = EventData.EMPTY_FLOAT     # 持仓盈亏
        self.dailyPnl    = EventData.EMPTY_FLOAT # 总盈亏
        self.netPnl      = EventData.EMPTY_FLOAT          # 净盈亏
                
        self.cBuy        = EventData.EMPTY_INT   # 成交数量
        self.cSell       = EventData.EMPTY_INT  # 成交数量
        self.txns        = EventData.EMPTY_STRING
        self._asofList   = []

    def initPositions(self, account, symbol, currentPos, prevPos, ohlc =None):
        """Constructor"""

        self.symbol      = symbol
        self.date        = account._dateToday             # 日期
        self.recentPrice = round(currentPos.price, 3)     # 当日收盘
        self.avgPrice    = round(currentPos.avgPrice, 3)  # 持仓均价
        self.recentPos   = round(currentPos.position, 3)  # 当日收盘
        self.posAvail    = round(currentPos.posAvail, 3)  # the rest avail pos
        self.calcPos     = round(prevPos.position, 3)     # 
        self.prevClose   = round(prevPos.price, 3)     # 昨日收盘
        self.prevPos     = round(prevPos.position, 3)     # 昨日收盘
        self._asofList   =[ currentPos.stampByTrader, currentPos.stampByBroker ]
        # 持仓部分
        self.positionPnl = round(prevPos.position * (currentPos.price - currentPos.avgPrice) * account._contractSize, 3)

        # TODO if ohlc:
        #     _, self.execOpen, self.execHigh, self.execLow, _ = ohlc

        # self.calcMValue  = round(calcPosition*currentPos.price*self._contractSize , 2),     # 昨日收盘
    def pushTrade(self, account, trade) :
        '''
        DayX bought some:
        {'recentPos': 200.0, 'cBuy': 2, 'recentPrice': 3.35, 'prevPos': 160.0, 'symbol': 'A601005', 'posAvail': 160.0, 'calcPos': 200.0, 
        'commission': 43.96, 'netPnl': -1472.74, 'avgPrice': 3.444, 'prevClose': 3.48, 'calcMValue': 67000.0, 'positionPnl': -1508.78, 
        'dailyPnl': -1428.78, 'cSell': 0, 'slippage': 0.0, 'date': u'20121219', 'tradingPnl': 80.0, 'asof': [datetime(2012, 12, 19, 14, 48), 0],
        'txns': '+20x3.31+20x3.35', 'turnover': 13320.0}

        DayY no trade：
        {'recentPos': 292.0, 'cBuy': 0, 'recentPrice': 3.26, 'prevPos': 292.0, 'symbol': 'A601005', 'posAvail': 292.0, 'calcPos': 292.0, 
        'commission': 0.0, 'netPnl': -4383.74, 'avgPrice': 3.41, 'prevClose': 3.27, 'calcMValue': 95192.0, 'positionPnl': -4383.74, 
        'dailyPnl': -4383.74, 'cSell': 0, 'slippage': 0.0, 'date': u'20121226', 'tradingPnl': 0.0, 'asof': [datetime(2012, 12, 24, 10, 20), 0],
        'txns': '', 'turnover': 0.0}

        sold-all at last day
        {'recentPos': 0.0, 'cBuy': 0, 'recentPrice': 3.94, 'prevPos': 292.0, 'symbol': 'A601005', 'posAvail': 0.0, 'calcPos': 0.0, 
        'commission': 375.14, 'netPnl': 15097.11, 'avgPrice': 3.41, 'prevClose': 3.59, 'calcMValue': 0.0, 'positionPnl': 15472.26, 
        'dailyPnl': 15472.26, 'cSell': 1, 'slippage': 0.0, 'date': u'20121228', 'tradingPnl': 0.0, 'asof': [datetime(2012, 12, 28, 15, 0), 0],
         'txns': '-292x3.94', 'turnover': 115048.0}]            
        '''

        posChange =0
        if trade.direction == OrderData.DIRECTION_LONG:
            posChange = trade.volume
            self.cBuy += 1
        else:
            posChange = -trade.volume
            self.cSell += 1

        self.txns += "%+dx%s" % (posChange, trade.price)

        self.tradingPnl += posChange * (self.recentPrice - trade.price) * account._contractSize
        self.calcPos += posChange

        tover, comis, slpfee = account.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
        self.turnover += round(tover, 3)
        self.commission += round(comis, 3)
        self._slippage += round(slpfee, 3)

    def close(self) :
        # 汇总
        self.dailyPnl = round(self.dailyPnl, 3)
        self.tradingPnl = round(self.tradingPnl, 3)

        self.totalPnl = self.tradingPnl + self.positionPnl
        self.netPnl   = self.totalPnl - self.commission - self._slippage
        # stampstr = ''
        # if currentPos.stampByTrader :
        #     stampstr += currentPos.stampByTrader
        # [ currentPos.stampByTrader, currentPos.stampByBroker ]

########################################################################
class DailyResult(object):
    '''每日交易的结果'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'date,startBalance,tcBuy,tcSell,tradingPnl,positionPnl,totalPnl,turnover,commission,slippage,netPnl,posValue,cash,endBalance,txnHist'

    #----------------------------------------------------------------------
    def __init__(self, date, startBalance=0):
        """Constructor"""

        self.date = date                # 日期
        self.startBalance = round(startBalance,2)  # 开始资产
        self.tcBuy = 0                  # 成交数量
        self.tcSell = 0                 # 成交数量
        self.tradeList = []             # 成交列表
        self.tradingPnl = 0             # 交易盈亏
        self.positionPnl = 0            # 持仓盈亏
        self.totalPnl = 0               # 总盈亏
        self.posValue = 0               # 总持仓市值
        self.cash     = 0               # 总现金
        self.endBalance = self.startBalance  # 结束资产
        
        self.turnover = 0               # 成交量
        self.commission = 0             # 手续费
        self.slippage = 0               # 滑点
        self.netPnl = 0                 # 净盈亏
        
        self.txnHist = ""

    # #----------------------------------------------------------------------
    # def addTrade(self, trade):
    #     """添加交易"""
    #     self.tradeList.append(trade)
    # #----------------------------------------------------------------------
    # def calculatePnl(self, account, openPosition=0):
    #     """
    #     计算盈亏
    #     size: 合约乘数
    #     rate：手续费率
    #     slippage：滑点点数
    #     """
    #     # 持仓部分
    #     self.openPosition = openPosition
    #     self.positionPnl = round(self.openPosition * (self.closePrice - self.previousClose) * account._contractSize, 3)
    #     self.closePosition = self.openPosition
        
    #     # 交易部分
    #     self.tcBuy = 0
    #     self.tcSell = 0
        
    #     for trade in self.tradeList:
    #         if trade.direction == OrderData.DIRECTION_LONG:
    #             posChange = trade.volume
    #             self.tcBuy += 1
    #         else:
    #             posChange = -trade.volume
    #             self.tcSell += 1
                
    #         self.txnHist += "%+dx%s" % (posChange, trade.price)

    #         self.tradingPnl += round(posChange * (self.closePrice - trade.price) * account._contractSize, 2)
    #         self.closePosition += posChange
    #         turnover, commission, slippagefee = account.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
    #         self.turnover += turnover
    #         self.commission += commission
    #         self.slippage += slippagefee
        
    #     # 汇总
    #     self.totalPnl = round(self.tradingPnl + self.positionPnl, 2)
    #     self.netPnl = round(self.totalPnl - self.commission - self.slippage, 2)
