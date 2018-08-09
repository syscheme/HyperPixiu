# encoding: UTF-8

from __future__ import division

from vnpy.trader.vtConstant import *
from vnpy.trader.vtObject import VtBarData, VtTickData

import traceback
from abc import ABCMeta, abstractmethod

########################################################################
class MarketData(object):
    # Market相关events
    EVENT_TICK = 'eTick.'                   # TICK行情事件，可后接具体的vtSymbol
    EVENT_MARKET_DEPTH0 = 'eMD0.'           # Market depth0
    EVENT_MARKET_DEPTH2 = 'eMD2.'           # Market depth2
    EVENT_KLINE_1MIN    = 'eKL1m.'
    EVENT_KLINE_5MIN    = 'eKL5m.'
    EVENT_KLINE_15MIN   = 'eKL15m.'
    EVENT_KLINE_30MIN   = 'eKL30m.'
    EVENT_KLINE_1HOUR   = 'eKL1h.'
    EVENT_KLINE_4HOUR   = 'eKL4h.'
    EVENT_KLINE_1DAY    = 'eKL1d.'

    DATA_SRCTYPE_REALTIME   = 'market'
    DATA_SRCTYPE_IMPORT     = 'import'
    DATA_SRCTYPE_BACKTEST   = 'backtest'

    __lastId__ =100

    from abc import ABCMeta, abstractmethod

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings, srcType=DATA_SRCTYPE_REALTIME):
        """Constructor"""

        # the MarketData instance Id
        self._id = settings.id("")
        if len(self._id)<=0 :
            MarketData.__lastId__ +=1
            self._id = 'MD%d' % MarketData.__lastId__

        self._mr = mainRoutine
        self._eventCh  = mainRoutine._eventChannel
        self._exchange = settings.exchange(self._id)

        self._active = False
        self.subDict = {}
        
        self.proxies = {}
    
    #----------------------------------------------------------------------
    @property
    def id(self) :
        return self._id

    @property
    def exchange(self) :
        return self._exchange

    @property
    def active(self):
        return self._active

    @property
    def subscriptions(self):
        return self.subDict
        
    #----------------------------------------------------------------------
    # if the MarketData has background thread, connect() will not start the thread
    # but start() will
    @abstractmethod
    def connect(self):
        """连接"""
        raise NotImplementedError
        return self.active

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def start(self):
        """连接"""
        self.connect()
        
    @abstractmethod
    def step(self):
        """连接"""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """停止"""
        if self._active:
            self._active = False
            self.close()
        
    #----------------------------------------------------------------------
    def subscribeKey(self, symbol, eventType):
        key = '%s>%s' %(eventType, symbol)
        return key

    #----------------------------------------------------------------------
    # return eventType, symbol
    def chopSubscribeKey(self, key):
        pos = key.find('>')
        return key[:pos], key[pos+1:]

    #----------------------------------------------------------------------
    @abstractmethod
    def subscribe(self, symbol, eventType =EVENT_TICK):
        """订阅成交细节"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    def unsubscribe(self, symbol, eventType):
        """取消订阅主题"""
        key = self.subscribeKey(symbol, eventType)
        if key not in self.subDict:
            return

        self.doUnsubscribe(key)
        del self.subDict[key]

    #----------------------------------------------------------------------
    @abstractmethod
    def doUnsubscribe(self, key):
        """取消订阅主题"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onError(self, msg):
        """错误推送"""
        print (msg)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def postMarketEvent(self, event):
        if self._eventCh ==None:
            return

        self._eventCh.put(event)
        self.info('posted %s%s' % (event.type_, event.dict_['data'].vtSymbol))

    #---logging -----------------------
    def debug(self, msg):
        self._mr.debug('MD['+self.id +'] ' + msg)
        
    def info(self, msg):
        """正常输出"""
        self._mr.info('MD['+self.id +'] ' + msg)

    def warn(self, msg):
        """警告信息"""
        self._mr.warn('MD['+self.id +'] ' + msg)
        
    def error(self, msg):
        """报错输出"""
        self._mr.error('MD['+self.id +'] ' + msg)
        
    def logexception(self, ex):
        """报错输出+记录异常信息"""
        self._mr.logexception('MD['+self.id +'] %s: %s' % (ex, traceback.format_exc()))
    
########################################################################
from threading import Thread
from time import sleep

class ThreadedMd(object):
    #----------------------------------------------------------------------
    def __init__(self, marketData):
        """Constructor"""
        self._md = marketData
        self.thread = Thread(target=self._run)

    #----------------------------------------------------------------------
    def _run(self):
        """执行连接 and receive"""
        while self._md._active:
            try :
                nextSleep = - self._md.step()
                if nextSleep >0:
                    sleep(nextSleep)
            except Exception as ex:
                self._md.error('ThreadedMd::step() excepton: %s' % ex)
        self._md.info('ThreadedMd exit')

    #----------------------------------------------------------------------
    @abstractmethod
    def start(self):
        self._md.start()
        self.thread.start()
        self._md.debug('ThreadedMd starts')
        return self._md._active

    #----------------------------------------------------------------------
    @abstractmethod
    def stop(self):
        self._md.stop()
        self.thread.join()
        self._md.info('ThreadedMd stopped')
    

########################################################################
class mdTickData(VtTickData):
    """Tick行情数据类"""

    #----------------------------------------------------------------------
    def __init__(self, md, symbol =None):
        """Constructor"""
        super(mdTickData, self).__init__()
        
        self.exchange   = md.exchange
        # self.sourceType = md._sourceType          # 数据来源类型
        if symbol:
            self.symbol = symbol
            self.vtSymbol = '.'.join([self.symbol, self.exchange])

########################################################################
class mdKLineData(VtBarData):
    """K线数据"""

    #----------------------------------------------------------------------
    def __init__(self, md, symbol =None):
        """Constructor"""
        super(mdKLineData, self).__init__()
        
        self.exchange   = md._exchange
        self.sourceType = md._sourceType          # 数据来源类型
        if symbol:
            self.symbol = symbol
            self.vtSymbol = '.'.join([self.symbol, self.exchange])

