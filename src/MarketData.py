# encoding: UTF-8

from __future__ import division

from EventData import *
from Application import MetaObj
from datetime import datetime, timedelta

MARKETDATE_EVENT_PREFIX = EVENT_NAME_PREFIX + 'md'

# Market相关events
EVENT_TICK          = MARKETDATE_EVENT_PREFIX + 'Tick'                   # TICK行情事件，可后接具体的vtSymbol
EVENT_MARKET_DEPTH0 = MARKETDATE_EVENT_PREFIX + 'MD0'           # Market depth0
EVENT_MARKET_DEPTH2 = MARKETDATE_EVENT_PREFIX + 'MD2'           # Market depth2
EVENT_KLINE_PREFIX  = MARKETDATE_EVENT_PREFIX + 'KL'
EVENT_KLINE_1MIN    = EVENT_KLINE_PREFIX      + '1m'
EVENT_KLINE_5MIN    = EVENT_KLINE_PREFIX      + '5m'
EVENT_KLINE_15MIN   = EVENT_KLINE_PREFIX      + '15m'
EVENT_KLINE_30MIN   = EVENT_KLINE_PREFIX      + '30m'
EVENT_KLINE_1HOUR   = EVENT_KLINE_PREFIX      + '1h'
EVENT_KLINE_4HOUR   = EVENT_KLINE_PREFIX      + '4h'
EVENT_KLINE_1DAY    = EVENT_KLINE_PREFIX      + '1d'

EVENT_T2KLINE_1MIN  = MARKETDATE_EVENT_PREFIX + 'T2K1m'

EVENT_MARKET_HOUR   = MARKETDATE_EVENT_PREFIX + 'Hr'

########################################################################
class MarketData(EventData):
    """Tick行情数据类"""

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        """Constructor"""
        super(MarketData, self).__init__()
        
        # 代码相关
        self.symbol = EventData.EMPTY_STRING              # 合约代码
        self.vtSymbol = EventData.EMPTY_STRING            # 合约在vt系统中的唯一代码，通常是 合约代码.交易所代码

        self.exchange   = exchange
        # self.sourceType = md._sourceType          # 数据来源类型
        if symbol and len(symbol)>0:
            self.symbol = self.vtSymbol = symbol
            if  len(exchange)>0 :
                self.vtSymbol = '.'.join([self.symbol, self.exchange])
        
        self.datetime = None                    # python的datetime时间对象
        self.time = EventData.EMPTY_STRING                # 时间 11:20:56.5
        self.date = EventData.EMPTY_STRING                # 日期 20151009

    @property
    def asof(self) :
        if not self.datetime :
            try :
                self.datetime = datetime.strptime(self.date + ' ' + self.time, '%Y%m%d %H:%M:%S')
            except:
                self.datetime = DT_EPOCH
                
        return self.datetime

########################################################################
class TickData(MarketData):
    """Tick行情数据类"""

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        """Constructor"""
        super(TickData, self).__init__(exchange, symbol)
        
        # 成交数据
        self.price = EventData.EMPTY_FLOAT            # 最新成交价
        self.volume = EventData.EMPTY_INT             # 最新成交量
        self.openInterest = EventData.EMPTY_INT           # 持仓量
        
        # 常规行情
        self.open = EventData.EMPTY_FLOAT            # 今日开盘价
        self.high = EventData.EMPTY_FLOAT            # 今日最高价
        self.low = EventData.EMPTY_FLOAT             # 今日最低价
        self.prevClose = EventData.EMPTY_FLOAT
        
        self.upperLimit = EventData.EMPTY_FLOAT           # 涨停价
        self.lowerLimit = EventData.EMPTY_FLOAT           # 跌停价
        
        # 五档行情
        # bid to buy: price and volume
        self.b1P = EventData.EMPTY_FLOAT 
        self.b2P = EventData.EMPTY_FLOAT
        self.b3P = EventData.EMPTY_FLOAT
        self.b4P = EventData.EMPTY_FLOAT
        self.b5P = EventData.EMPTY_FLOAT
        self.b1V = EventData.EMPTY_INT
        self.b2V = EventData.EMPTY_INT
        self.b3V = EventData.EMPTY_INT
        self.b4V = EventData.EMPTY_INT
        self.b5V = EventData.EMPTY_INT
        
        # ask to sell: price and volume
        self.a1P = EventData.EMPTY_FLOAT
        self.a2P = EventData.EMPTY_FLOAT
        self.a3P = EventData.EMPTY_FLOAT
        self.a4P = EventData.EMPTY_FLOAT
        self.a5P = EventData.EMPTY_FLOAT        
        
        
        self.a1V = EventData.EMPTY_INT
        self.a2V = EventData.EMPTY_INT
        self.a3V = EventData.EMPTY_INT
        self.a4V = EventData.EMPTY_INT
        self.a5V = EventData.EMPTY_INT         

    @property
    def desc(self) :
        return 'tick.%s@%s_%dx%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S'), self.volume,round(self.price,2))

########################################################################
class KLineData(MarketData):
    """K线数据"""

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        """Constructor"""
        super(KLineData, self).__init__(exchange, symbol)
        
        self.open  = EventData.EMPTY_FLOAT             # OHLC
        self.high  = EventData.EMPTY_FLOAT
        self.low   = EventData.EMPTY_FLOAT
        self.close = EventData.EMPTY_FLOAT
        
        self.volume = EventData.EMPTY_INT             # 成交量
        self.openInterest = EventData.EMPTY_INT       # 持仓量    

    @property
    def desc(self) :
        return 'kline.%s@%s>%sx%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S') if self.datetime else '', self.volume, round(self.close,2))

    '''
    @property
    def OHLCV(self) :
        return self.open, self.high, self.low, self.close, self.volume
    '''

########################################################################
class DictToKLine(object):

    def __init__(self, eventType, symbol, exchange=None):
        """Constructor"""
        super(DictToKLine, self).__init__()
        self._type = eventType
        self._symbol = symbol
        self._exchange = exchange if exchange else 'na'

    @property
    def fields(self) :
        return 'date,time,open,high,low,close,volume,ammount'

    @abstractmethod
    def convert(self, row, exchange, symbol =None) :
        if not 'mdKL' in self._type :
            raise NotImplementedError

        kl = KLineData(self._exchange, self._symbol)
        kl.open = float(row['open'])
        kl.high = float(row['high'])
        kl.low = float(row['low'])
        kl.close = float(row['close'])
        kl.volume = float(row['volume'])
        kl.date = datetime.strptime(row['date'], '%Y/%m/%d').strftime('%Y%m%d')
        kl.time = row['time']+":00"
        
        kl.datetime = datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:%S')
        dataOf = kl.datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:00')
        ev = Event(type_=self._type)
        ev.setData(kl)
        return ev

########################################################################
class McCvsToKLine(DictToKLine):

    def __init__(self, eventType, symbol, exchange=None):
        """Constructor"""
        super(McCvsToKLine, self).__init__(eventType, symbol, exchange)

    @abstractmethod
    def convert(self, csvrow, eventType =None, symbol =None) :
        if not 'mdKL' in self._type :
            raise NotImplementedError

        kl = KLineData('', symbol)
        kl.open = float(csvrow['Open'])
        kl.high = float(csvrow['High'])
        kl.low = float(csvrow['Low'])
        kl.close = float(csvrow['Close'])
        kl.volume = float(csvrow['TotalVolume'])
        kl.date = datetime.strptime(csvrow['Date'], '%Y-%m-%d').strftime('%Y%m%d')
        kl.time = csvrow['Time']+":00"
        
        kl.datetime = datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:%S')
        dataOf = kl.datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:00')
        ev = Event(type_=self._type)
        ev.setData(kl)
        return ev

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
            kline.open = tick.price
            kline.high = tick.price
            kline.low = tick.price
        # 累加更新老一分钟的K线数据
        else:                                   
            kline.high = max(kline.high, tick.price)
            kline.low = min(kline.low, tick.price)

        # 通用更新部分
        kline.close = tick.price        
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
        self._klineWk = None
        self._klineIn = None        # 上一Input缓存对象
        self._xmin = xmin             # X的值
        self.onXminBar = onKLineXmin  # X分钟K线的回调函数

    #----------------------------------------------------------------------
    def pushKLineEvent(self, klineEv, dtAsOf=None):
        """1分钟K线更新"""
        d = klineEv.data
        self.pushKLineData(d, dtAsOf)

    def flush(self):
        klineOut = self._klineWk
        # 推送, X分钟策略计算和决策
        if self.onXminBar :
            self.onXminBar(klineOut)
        
        # 清空老K线缓存对象
        self._klineWk = None
        return klineOut

    def pushKLineData(self, kline, dtAsOf=None):
        """1分钟K线更新"""

        # 尚未创建对象
        if self._klineWk:
            if kline.datetime > self._klineWk.datetime :
                self._klineWk.date = self._klineWk.datetime.strftime('%Y%m%d')
                self._klineWk.time = self._klineWk.datetime.strftime('%H:%M:%S.%f')
                    
                # 推送, X分钟策略计算和决策
                return self.flush()

        # 初始化新一分钟的K线数据
        if not self._klineWk:
            # 创建新的K线对象
            self._klineWk = KLineData(kline.exchange + '_k2x', kline.symbol)
            self._klineWk.open = kline.open
            self._klineWk.high = kline.high
            self._klineWk.low = kline.low
            if dtAsOf :
                self._klineWk.datetime = dtAsOf
            else :
                timeTil = kline.datetime + timedelta(minutes=self._xmin)
                timeTil = timeTil.replace(minute=int(timeTil.minute/self._xmin)*self._xmin, second=0, microsecond=0)
                self._klineWk.datetime = timeTil    # 以x分钟K线末作为X分钟线的时间戳
            self._klineWk.date = self._klineWk.datetime.strftime('%Y%m%d')
            self._klineWk.time = self._klineWk.datetime.strftime('%H%M%S')
        else:                                   
            # 累加更新老一分钟的K线数据
            self._klineWk.high = max(self._klineWk.high, kline.high)
            self._klineWk.low = min(self._klineWk.low, kline.low)

        # 通用部分
        self._klineWk.close = kline.close        
        self._klineWk.openInterest = kline.openInterest
        self._klineWk.volume += int(kline.volume)                

        # 清空老K线缓存对象
        self._klineIn = kline
        return None

########################################################################
class DataToEvent(object):

    def __init__(self, sink):
        """Constructor"""
        super(DataToEvent, self).__init__()
        self._sink = sink
        self._dict = {}

    @property
    def fields(self) :
        return None

    @abstractmethod
    def push(self, csvrow, eventType =None, symbol =None) :
        raise NotImplementedError

    @abstractmethod
    def _cbMarketEvent(self, eventType, eventData, dataOf =None):
        if eventType in self._dict.keys() and self._dict[eventType]['dataOf'] < dataOf:
            if self._sink and self._dict[eventType]['data']:
                event = Event()
                event.type_ = eventType
                event.setData(self._dict[eventType]['data'])
                self._sink(event)

        d =  {
            'dataOf' : dataOf if dataOf else eventData.datetime,
            'data' : eventData
            }

        self._dict[eventType] = d

########################################################################
class MarketState(MetaObj):
    '''
    '''
    def __init__(self, exchange):
        """Constructor"""
        super(MarketState, self).__init__()
        self._exchange = exchange
    
    @property
    def exchange(self) : return self._exchange

    @abstractmethod
    def listOberserves(self) :
        ''' list the symbol list that is oberserving
        '''
        raise NotImplementedError

    @abstractmethod
    def addMonitor(self, symbol) :
        ''' add a symbol to monitor
        '''
        raise NotImplementedError

    @abstractmethod
    def latestPrice(self, symbol) :
        ''' query for latest price of the given symbol
        @return the price
        '''
        raise NotImplementedError

    @abstractmethod
    def getAsOf(self, symbol=None) :
        ''' 
        @return the datetime as of latest observing
        '''
        raise NotImplementedError

    @abstractmethod
    def dailyOHLC_sofar(self, symbol) :
        ''' 
        @return (datestr, open, high, low, close) as of today
        '''
        raise NotImplementedError

    @abstractmethod
    def updateByEvent(self, ev) :
        ''' 
        @ev  could be Event(Tick), Event(KLine), Event(Perspective)
        @return True if updated
        '''
        raise NotImplementedError

