# encoding: UTF-8

from __future__ import division

from Application import BaseApplication
from MarketData import *
import traceback
from datetime import datetime

from abc import ABCMeta, abstractmethod

########################################################################
class MarketObserver(BaseApplication):
    ''' abstract application to observe market
    '''
    TAG_BACKTEST = '$BT'

    DUMMY_DT_EOS = datetime(2999, 12, 31, 23,59,59)
    DUMMY_DATE_EOS = DUMMY_DT_EOS.strftime('%Y%m%d')
    DUMMY_TIME_EOS = DUMMY_DT_EOS.strftime('%H%M%S')

    __lastId__ =100

    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        '''Constructor
        '''
        super(MarketObserver, self).__init__(program, settings)

        # the MarketData instance Id
        # self._id = settings.id("")
        # if len(self._id)<=0 :
        #     MarketData.__lastId__ +=1
        #     self._id = 'MD%d' % MarketData.__lastId__

        # self._mr = program
        # self._eventCh  = program._eventLoop
        # self._exchange = settings.exchange(self._id)

        self.subDict = {}
        self.proxies = {}
    
    #----------------------------------------------------------------------
    @property
    def exchange(self) :
        return self._exchange

    @property
    def subscriptions(self):
        return self.subDict
        
    #----------------------------------------------------------------------
    # inqueries to some market data
    # https://www.cnblogs.com/bradleon/p/6106595.html
    def query(self, symbol, eventType, since, cortResp=None):
        """查询请求""" 
        '''
        will call cortResp.send(csvline) when the result comes
        '''
        scale =1200
        if (eventType == EVENT_TICK) :
            scale=0
        elif (eventType == EVENT_TICK) :
            scale = 1
        
        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?scale=5&datalen=1000" % (scale1min, linesExpected)

        req = (uri, httpParams, reqData, funcMethod, callbackResp)

        self.enqueue(reqId, req)
        return reqId
   
    #----------------------------------------------------------------------
    # if the MarketData has background thread, connect() will not start the thread
    # but start() will
    @abstractmethod
    def connect(self):
        """连接"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass

    #--- impll of BaseApplication -----------------------
    def start(self):
        """连接"""
        if not self.connect() :
            return False
        return super(MarketObserver,self).start()
        
    @abstractmethod
    def step(self):
        """连接"""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """停止"""
        if self.isActive:
            super(MarketObserver,self).stop()
            self.close()
        
    #----------------------------------------------------------------------
    def fmtSubscribeKey(self, symbol, eventType):
        key = '%s>%s' %(eventType, symbol)
        return key

    def chopSubscribeKey(self, key):
        ''' chop the pair (eventType, symbol) out of a given key
        '''
        pos = key.find('>')
        return key[:pos], key[pos+1:]

    #----------------------------------------------------------------------
    def onError(self, msg):
        """错误推送"""
        self.error(msg)
        
'''
    #----------------------------------------------------------------------
    @abstractmethod
    def subscribe(self, symbol, eventType =EVENT_TICK):
        """订阅成交细节"""
        pass

    #----------------------------------------------------------------------
    def unsubscribe(self, symbol, eventType):
        """取消订阅主题"""
        key = self.fmtSubscribeKey(symbol, eventType)
        if key not in self.subDict:
            return

        self.doUnsubscribe(key)
        del self.subDict[key]

    #----------------------------------------------------------------------
    @abstractmethod
    def doUnsubscribe(self, key):
        """取消订阅主题"""
        raise NotImplementedError
'''
