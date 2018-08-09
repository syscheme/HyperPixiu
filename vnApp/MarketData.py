# encoding: UTF-8

from __future__ import division

from vnpy.trader.vtConstant import *
from .EventChannel import EventData, Event

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
        self.info('posted %s%s' % (event.type_, event.dict_['data'].desc))

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
class mdTickData(EventData):
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

        # 代码相关
        self.symbol = EMPTY_STRING              # 合约代码
        self.exchange = EMPTY_STRING            # 交易所代码
        self.vtSymbol = EMPTY_STRING            # 合约在vt系统中的唯一代码，通常是 合约代码.交易所代码
        
        # 成交数据
        self.lastPrice = EMPTY_FLOAT            # 最新成交价
        self.lastVolume = EMPTY_INT             # 最新成交量
        self.volume = EMPTY_INT                 # 今天总成交量
        self.openInterest = EMPTY_INT           # 持仓量
        self.time = EMPTY_STRING                # 时间 11:20:56.5
        self.date = EMPTY_STRING                # 日期 20151009
        self.datetime = None                    # python的datetime时间对象
        
        # 常规行情
        self.openPrice = EMPTY_FLOAT            # 今日开盘价
        self.highPrice = EMPTY_FLOAT            # 今日最高价
        self.lowPrice = EMPTY_FLOAT             # 今日最低价
        self.preClosePrice = EMPTY_FLOAT
        
        self.upperLimit = EMPTY_FLOAT           # 涨停价
        self.lowerLimit = EMPTY_FLOAT           # 跌停价
        
        # 五档行情
        self.bidPrice1 = EMPTY_FLOAT
        self.bidPrice2 = EMPTY_FLOAT
        self.bidPrice3 = EMPTY_FLOAT
        self.bidPrice4 = EMPTY_FLOAT
        self.bidPrice5 = EMPTY_FLOAT
        
        self.askPrice1 = EMPTY_FLOAT
        self.askPrice2 = EMPTY_FLOAT
        self.askPrice3 = EMPTY_FLOAT
        self.askPrice4 = EMPTY_FLOAT
        self.askPrice5 = EMPTY_FLOAT        
        
        self.bidVolume1 = EMPTY_INT
        self.bidVolume2 = EMPTY_INT
        self.bidVolume3 = EMPTY_INT
        self.bidVolume4 = EMPTY_INT
        self.bidVolume5 = EMPTY_INT
        
        self.askVolume1 = EMPTY_INT
        self.askVolume2 = EMPTY_INT
        self.askVolume3 = EMPTY_INT
        self.askVolume4 = EMPTY_INT
        self.askVolume5 = EMPTY_INT         

    @property
    def desc(self) :
        return 'tick[%s]@%s' % (self.vtSymbol, self.datetime.strftime('%Y%m%dT%H%M%S'))

########################################################################
class mdKLineData(EventData):
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

        self.vtSymbol = EMPTY_STRING        # vt系统代码
        self.symbol = EMPTY_STRING          # 代码
        self.exchange = EMPTY_STRING        # 交易所
    
        self.open = EMPTY_FLOAT             # OHLC
        self.high = EMPTY_FLOAT
        self.low = EMPTY_FLOAT
        self.close = EMPTY_FLOAT
        
        self.date = EMPTY_STRING            # bar开始的时间，日期
        self.time = EMPTY_STRING            # 时间
        self.datetime = None                # python的datetime时间对象
        
        self.volume = EMPTY_INT             # 成交量
        self.openInterest = EMPTY_INT       # 持仓量    

    @property
    def desc(self) :
        return 'kline[%s]@%s' % (self.vtSymbol, self.datetime.strftime('%Y%m%dT%H%M%S'))

