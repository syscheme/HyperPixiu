from ..Account import *

########################################################################
from abc import ABCMeta, abstractmethod
class tdBase(object):
    """交易API"""
    SYNC_MODE = 'sync'
    ASYNC_MODE = 'async'

    #----------------------------------------------------------------------
    def __init__(self, account, settings, mode=None):

        """Constructor"""
        self._account  = account
        self._settings = settings

        self._mode = self.ASYNC_MODE
        if mode:
            self._mode = mode

        self._active = False         # API工作状态   
        self._reqid = 0              # 请求编号
        self.queue = Queue()        # 请求队列
        self.pool = None            # 线程池
        
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
    def onError(self, msg, reqid):
        """错误回调"""
        _account.onTradeError(self, msg, reqid)

    #----------------------------------------------------------------------
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
    def placeOrder(self, volume, symbol, type_, price=None, source=None):
        """下单"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onPlaceOrder(self, data, reqid):
        """委托回调"""
        _account.onPlaceOrder(self, data, reqid)

    #----------------------------------------------------------------------
    @abstractmethod
    def placeStopOrder(self, volume, vtSymbol, orderType, price, source)
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
    @abstractmethod
    def onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        _account.onGetTimestamp(self, data, reqid)
        
