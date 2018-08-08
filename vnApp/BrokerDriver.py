# encoding: UTF-8

from .Account import *
import threading

########################################################################
from abc import ABCMeta, abstractmethod
class BrokerDriver(object):
    """交易API
    BrokerDriver appears as an API and it interacts with the broker
    BrokerDriver most likely will have a background thread the keep sync-ed with
    the broker, so that its member datat should be thread-safe
    """
    SYNC_MODE = 'sync'
    ASYNC_MODE = 'async'

    EVENT_TRADE = 'eTrade.'                 # 成交回报事件
    EVENT_ORDER = 'eOrder.'                 # 报单回报事件
    EVENT_POSITION = 'ePosition.'           # 持仓回报事件
    EVENT_ACCOUNT = 'eAccount.'             # 账户回报事件
    EVENT_CONTRACT = 'eContract.'           # 合约基础信息回报事件

    #----------------------------------------------------------------------
    def __init__(self, account, settings, mode=None):

        """Constructor"""
        self._lock = threading.Lock()

        self._account  = account
        self._settings = settings

        self._mode = self.ASYNC_MODE
        if mode:
            self._mode = mode

        self._active = False         # API工作状态   
        self._reqid = 0              # 请求编号

        self._dictPositions = { # dict from symbol to latest VtPositionData
            Account.SYMBOL_CASH : VtPositionData()
        }

    @property
    def className(self) :
        return self.__class__.__name__

    @property
    def size(self) : return self._account.size

    @property
    def slippage(self) : return self._account.slippage

    @property
    def rate(self) : return self._account.rate

    #----------------------------------------------------------------------
    @abstractmethod
    def cancelOrder(self, orderid):
        """撤单"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onCancelOrder(self, data, reqid):
        """撤单回调"""
        _account.onCancelOrder(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def batchCancel(self, orderids):
        """批量撤单"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onBatchCancel(self, data, reqid):
        """批量撤单回调"""
        _account.onBatchCancel(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def getAllPositions(self):
        with self._lock :
            return copy.deepcopy(self._dictPositions)

    @abstractmethod
    def getPositions(self, symbol):
        with self._lock :
            if not symbol in self._dictPositions:
                return VtPositionData()
            return copy(self._dictPositions[symbol])

    @abstractmethod
    def getAccountBalance(self, accountid):
        """查询余额"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onGetAccountBalance(self, data, reqid):
        """查询余额回调"""
        _account.onGetAccountBalance(self, data, reqid)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def getOrder(self, orderid):
        """查询某一委托"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onGetOrder(self, data, reqid):
        """查询单一委托回调"""
        _account.onGetOrder(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def getOrders(self, symbol, states, types=None, startDate=None, 
                  endDate=None, from_=None, direct=None, size=None):
        """查询委托"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onGetOrders(self, data, reqid):
        """查询委托回调"""
        _account.onGetOrders(self, data, reqid)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def getMatchResults(self, symbol, types=None, startDate=None, 
                  endDate=None, from_=None, direct=None, size=None):
        """查询委托"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onGetMatchResults(self, data, reqid):
        """查询成交回调"""
        _account.onGetMatchResults(self, data, reqid)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def getMatchResult(self, orderid):
        """查询某一委托"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onGetMatchResult(self, data, reqid):
        """查询单一成交回调"""
        _account.onGetMatchResult(self, data, reqid)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def placeOrder(vtOrder):
        """下单"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onPlaceOrder(self, data, reqid):
        """委托回调"""
        self._account.onPlaceOrder(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def placeStopOrder(self, volume, vtSymbol, orderType, price, source) :
        """下单"""
        raise NotImplementedError
    
    #----------------------------------------------------------------------
    @abstractmethod
    def onStopOrder(self, data, reqid):
        """委托回调"""
        _account.onStopOrder(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def getTimestamp(self):
        """查询系统时间"""
        raise NotImplementedError
    
    #----------------------------------------------------------------------
    def onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        self._account.onGetTimestamp(self, data, reqid)

    ##############################################################
    # moved from Account
    def onTraded(self, trade):
        """交易成功回调"""
        if trade.vtTradeID in self._dictTrades:
            return

        self._dictTrades[trade.vtTradeID] = trade

        # update the current postion, this may overwrite during the sync by BrokerDriver
        s = trade.symbol
        if not s in currentPositions :
            self._dictPositions[s] = VtPositionData()
            pos = self._dictPositions[s]
            pos.symbol = s
            pos.vtSymbol = trade.vtSymbol
            pos.exchange = trade.exchange
        else:
            pos = self._dictPositions[s]

        # 持仓相关
        pos.price      = trade.price
        pos.direction  = trade.direction      # 持仓方向
        # pos.frozen =  # 冻结数量

        # update the position of symbol and its average cost
        tdvol = trade.volume
        if trade.direction != DIRECTION_LONG:
            tdvol = -tdvol 
        else: 
            # pos.price
            cost = pos.position * pos.avgPrice
            cost += trade.volume * trade.price
            newPos = pos.position + trade.volume
            if newPos:
                pos.avgPrice = cost / newPos

        pos.position += tdvol
        pos.stampByTrader = trade.dt

    @abstractmethod # from Account_AShare
    def onTrade(self, trade):
        super(Account_AShare, self).onTrade(trade)

        if trade.direction != DIRECTION_LONG:
            pos = self._dictPositions[trade.symbol]
            pos.posAvail -= trade.volume
