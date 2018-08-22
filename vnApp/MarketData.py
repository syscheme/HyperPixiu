# encoding: UTF-8

from __future__ import division

# from vnpy.trader.vtConstant import *
from .EventChannel import EventData, Event, EVENT_NAME_PREFIX

import traceback
from abc import ABCMeta, abstractmethod

########################################################################
class MarketData(object):
    # Market相关events
    EVENT_TICK          = EVENT_NAME_PREFIX + 'Tick'                   # TICK行情事件，可后接具体的vtSymbol
    EVENT_MARKET_DEPTH0 = EVENT_NAME_PREFIX + 'MD0'           # Market depth0
    EVENT_MARKET_DEPTH2 = EVENT_NAME_PREFIX + 'MD2'           # Market depth2
    EVENT_KLINE_1MIN    = EVENT_NAME_PREFIX + 'KL1m'
    EVENT_KLINE_5MIN    = EVENT_NAME_PREFIX + 'KL5m'
    EVENT_KLINE_15MIN   = EVENT_NAME_PREFIX + 'KL15m'
    EVENT_KLINE_30MIN   = EVENT_NAME_PREFIX + 'KL30m'
    EVENT_KLINE_1HOUR   = EVENT_NAME_PREFIX + 'KL1h'
    EVENT_KLINE_4HOUR   = EVENT_NAME_PREFIX + 'KL4h'
    EVENT_KLINE_1DAY    = EVENT_NAME_PREFIX + 'KL1d'

    EVENT_T2KLINE_1MIN  = EVENT_NAME_PREFIX + 'T2K1m'

    TAG_BACKTEST = '$BT'

    __lastId__ =100

    from abc import ABCMeta, abstractmethod

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
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
    def ident(self) :
        return self.__class__.__name__ +"." + self._id

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
#        return self.active

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
        self.debug('posted %s%s' % (event.type_, event.dict_['data'].desc))

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
class TickData(EventData):
    """Tick行情数据类"""

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        """Constructor"""
        super(TickData, self).__init__()
        
        # 代码相关
        self.symbol = EventData.EMPTY_STRING              # 合约代码
        self.vtSymbol = EventData.EMPTY_STRING            # 合约在vt系统中的唯一代码，通常是 合约代码.交易所代码

        self.exchange   = exchange
        # self.sourceType = md._sourceType          # 数据来源类型
        if symbol:
            self.symbol = symbol
            self.vtSymbol = '.'.join([self.symbol, self.exchange])
        
        # 成交数据
        self.lastPrice = EventData.EMPTY_FLOAT            # 最新成交价
        self.lastVolume = EventData.EMPTY_INT             # 最新成交量
        self.volume = EventData.EMPTY_INT                 # 今天总成交量
        self.openInterest = EventData.EMPTY_INT           # 持仓量
        self.time = EventData.EMPTY_STRING                # 时间 11:20:56.5
        self.date = EventData.EMPTY_STRING                # 日期 20151009
        self.datetime = None                    # python的datetime时间对象
        
        # 常规行情
        self.openPrice = EventData.EMPTY_FLOAT            # 今日开盘价
        self.highPrice = EventData.EMPTY_FLOAT            # 今日最高价
        self.lowPrice = EventData.EMPTY_FLOAT             # 今日最低价
        self.preClosePrice = EventData.EMPTY_FLOAT
        
        self.upperLimit = EventData.EMPTY_FLOAT           # 涨停价
        self.lowerLimit = EventData.EMPTY_FLOAT           # 跌停价
        
        # 五档行情
        self.bidPrice1 = EventData.EMPTY_FLOAT
        self.bidPrice2 = EventData.EMPTY_FLOAT
        self.bidPrice3 = EventData.EMPTY_FLOAT
        self.bidPrice4 = EventData.EMPTY_FLOAT
        self.bidPrice5 = EventData.EMPTY_FLOAT
        
        self.askPrice1 = EventData.EMPTY_FLOAT
        self.askPrice2 = EventData.EMPTY_FLOAT
        self.askPrice3 = EventData.EMPTY_FLOAT
        self.askPrice4 = EventData.EMPTY_FLOAT
        self.askPrice5 = EventData.EMPTY_FLOAT        
        
        self.bidVolume1 = EventData.EMPTY_INT
        self.bidVolume2 = EventData.EMPTY_INT
        self.bidVolume3 = EventData.EMPTY_INT
        self.bidVolume4 = EventData.EMPTY_INT
        self.bidVolume5 = EventData.EMPTY_INT
        
        self.askVolume1 = EventData.EMPTY_INT
        self.askVolume2 = EventData.EMPTY_INT
        self.askVolume3 = EventData.EMPTY_INT
        self.askVolume4 = EventData.EMPTY_INT
        self.askVolume5 = EventData.EMPTY_INT         

    @property
    def desc(self) :
        return 'tick.%s@%s_%dx%s' % (self.vtSymbol, self.datetime.strftime('%Y%m%dT%H%M%S'),self.volume,round(self.lastPrice,2))

########################################################################
class KLineData(EventData):
    """K线数据"""

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        """Constructor"""
        super(KLineData, self).__init__()
        
        self.symbol = EventData.EMPTY_STRING          # 代码
        self.vtSymbol = EventData.EMPTY_STRING 

        self.exchange   = exchange
        # self.sourceType = md._sourceType          # 数据来源类型
        if symbol:
            self.symbol = symbol
            self.vtSymbol = '.'.join([self.symbol, self.exchange])

    
        self.open = EventData.EMPTY_FLOAT             # OHLC
        self.high = EventData.EMPTY_FLOAT
        self.low = EventData.EMPTY_FLOAT
        self.close = EventData.EMPTY_FLOAT
        
        self.date = EventData.EMPTY_STRING            # bar开始的时间，日期
        self.time = EventData.EMPTY_STRING            # 时间
        self.datetime = None                # python的datetime时间对象
        
        self.volume = EventData.EMPTY_INT             # 成交量
        self.openInterest = EventData.EMPTY_INT       # 持仓量    

    @property
    def desc(self) :
        return 'kline[%s]@%s' % (self.symbol, self.datetime.strftime('%Y%m%dT%H%M%S') if self.datetime else '')


########################################################################
class TickToKLineMerger(object):
    """
    K线合成器，支持：
    1. 基于Tick合成1分钟K线
    """

    #----------------------------------------------------------------------
    def __init__(self, onKLine1min):
        """Constructor"""
        self._dictKline = {}             # 1分钟K线对象
        self._dickLastTick = {}          # 上一TICK缓存对象
        self.onKline1min = onKLine1min      # 1分钟K线回调函数
        
    #----------------------------------------------------------------------
    def pushTick(self, tick):
        """TICK更新"""

        kline = None # 尚未创建对象
        
        if tick.symbol in self._dictKline:
            kline = self._dictKline[tick.symbol]
            if kline.datetime.minute != tick.datetime.minute:
                # 生成上一分钟K线的时间戳
                kline.datetime = kline.datetime.replace(second=0, microsecond=0)  # 将秒和微秒设为0
                kline.date = kline.datetime.strftime('%Y%m%d')
                kline.time = kline.datetime.strftime('%H:%M:%S.%f')
            
                # 推送已经结束的上一分钟K线
                if self.onKline1min :
                    self.onKline1min(kline)
                
                kline = None # 创建新的K线对象
            
        # 初始化新一分钟的K线数据
        if not kline:
            # 创建新的K线对象
            kline = KLineData(tick.exchange + '_t2k', tick.exchasymbolnge)
            kline.open = tick.lastPrice
            kline.high = tick.lastPrice
            kline.low = tick.lastPrice
        # 累加更新老一分钟的K线数据
        else:                                   
            kline.high = max(kline.high, tick.lastPrice)
            kline.low = min(kline.low, tick.lastPrice)

        # 通用更新部分
        kline.close = tick.lastPrice        
        kline.datetime = tick.datetime  
        kline.openInterest = tick.openInterest
        self._dictKline[tick.symbol] = kline
   
        if tick.symbol in self._dickLastTick.keys():
            volumeChange = tick.volume - self._dickLastTick[tick.symbol].volume   # 当前K线内的成交量
            kline.volume += max(volumeChange, 0)             # 避免夜盘开盘lastTick.volume为昨日收盘数据，导致成交量变化为负的情况
            
        # 缓存Tick
        self._dickLastTick[tick.symbol] = tick

########################################################################
class KlineToXminMerger(object):
    """
    K线合成器，支持：
    2. 基于1分钟K线合成X分钟K线（X可以是2、3、5、10、15、30	）
    """

    #----------------------------------------------------------------------
    def __init__(self, onKLineXmin, xmin=15) :
        """Constructor"""
        self._dictKlineXmin = {}        # 1分钟K线对象
        self._dictKlineIn = {}          # 上一Input缓存对象
        self._xmin = xmin            # X的值
        self.onXminBar = onKLineXmin  # X分钟K线的回调函数
        
    #----------------------------------------------------------------------
    def pushKLine(self, kline):
        """1分钟K线更新"""
        klineXmin = None
        # 尚未创建对象
        if kline.symbol in self._dictKlineXmin:
            klineXmin = self._dictKlineXmin[kline.symbol]
            if not (kline.datetime.minute + 1) % self._xmin:   # 可以用X整除, X分钟已经走完

                klineXmin.datetime = klineXmin.datetime.replace(second=0, microsecond=0)  # 将秒和微秒设为0
                klineXmin.date = klineXmin.datetime.strftime('%Y%m%d')
                klineXmin.time = klineXmin.datetime.strftime('%H:%M:%S.%f')
                
                # 推送, X分钟策略计算和决策
                if self.onXminBar :
                    self.onXminBar(klineXmin)
                
                # 清空老K线缓存对象
                klineXmin = None

        # 初始化新一分钟的K线数据
        if not klineXmin:
            # 创建新的K线对象
            klineXmin = KLineData(kline.exchange + '_k2x', kline.symbol)
            klineXmin.open = kline.open
            klineXmin.high = kline.high
            klineXmin.low = kline.low
            klineXmin.datetime = kline.datetime    # 以第一根分钟K线的开始时间戳作为X分钟线的时间戳
        else:                                   
            # 累加更新老一分钟的K线数据
            klineXmin.high = max(klineXmin.high, kline.high)
            klineXmin.low = min(klineXmin.low, kline.low)

        # 通用部分
        klineXmin.close = kline.close        
        klineXmin.openInterest = kline.openInterest
        klineXmin.volume += int(kline.volume)                

        # 清空老K线缓存对象
        self._dictKlineXmin[kline.symbol] = klineXmin
        self._dictKlineIn[kline.symbol] = kline
