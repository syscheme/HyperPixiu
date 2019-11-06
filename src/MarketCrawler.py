# encoding: UTF-8

from __future__ import division

from Application import BaseApplication, datetime2float
from MarketData import *

import traceback
from datetime import datetime

from abc import ABCMeta, abstractmethod

########################################################################
class MarketCrawler(BaseApplication):
    ''' abstract application to observe market
    '''
    __lastId__ =100

    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        '''Constructor
        '''
        super(MarketCrawler, self).__init__(program, settings)

        self._symbolsToPoll = []
        # the MarketData instance Id
        # self._id = settings.id("")
        # if len(self._id)<=0 :
        #     MarketData.__lastId__ +=1
        #     self._id = 'MD%d' % MarketData.__lastId__

        # self._mr = program
        # self._eventCh  = program._eventLoop
        # self._exchange = settings.exchange(self._id)

        self._steps = []
        self.__genSteps={}
    
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
   
    def subscribe(self, symbols):
        c =0
        for s in symbols:
            if s in self._symbolsToPoll:
                continue

            self._symbolsToPoll.append(s)
            c +=1
        
        if c <=0:
            return c
        
        self._symbolsToPoll.sort()
        return c

    def unsubscribe(self, symbols):
        c = len(self._symbolsToPoll)
        for s in symbols:
            self._symbolsToPoll.remove(s)
        
        if c ==len(self._symbolsToPoll):
            return c
        
        self._symbolsToPoll.sort()
        return len(self._symbolsToPoll)

    #--- new methods defined in MarketCrawler ---------
    # if the MarketData has background thread, connect() will not start the thread
    # but start() will
    def connect(self):
        '''
        return True if connected 
        '''
        return True

    def close(self):
        pass

    #--- impl/overwrite of BaseApplication -----------------------
    def doAppInit(self): # return True if succ
        if not super(MarketCrawler, self).doAppInit() :
            return False

        return self.connect()

    def OnEvent(self, event):
        '''
        process the event
        '''
        pass

    def doAppStep(self):
        '''
        @return True if busy at this step
        '''
        self._stepAsOf = datetime2float(datetime.now())
        busy = False

        for s in self._steps:
            if not s in self.__genSteps.keys() or not self.__genSteps[s] :
                self.__genSteps[s] = s()
            try :
                if next(self.__genSteps[s]) :
                    busy = True
            except StopIteration:
                self.__genSteps[s] = None

        return busy

    def stop(self):
        super(MarketCrawler, self).stop()
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
