# encoding: UTF-8

from __future__ import division

from EventData import *
from Application import MetaObj
from datetime import datetime, timedelta
from collections import OrderedDict

import math
import numpy as np
from PIL import Image

MARKETDATE_EVENT_PREFIX = EVENT_NAME_PREFIX + 'md'
EXPORT_FLOATS_DIMS = 4 # take the minimal dim=4
PRICE_DISPLAY_ROUND_DECIMALS = 3 

BASE_LOG10x2 = math.log(10) *2

def floatNormalize(v):
    if v <0: return 0.0
    return v if v<1.0 else 1.0

def floatNormalize_LOG10(v, base=1.0, scale=1.0):
    v = float(v/base)
    v = math.log(v) / BASE_LOG10x2 *scale +0.5 # 0.1x lead to 0 and 10x lead to 1
    return floatNormalize(v)

def floatNormalize_PriceChange(newPrice, basePrice=1.0):
    v = float(newPrice/basePrice)
    (v -1) *5.0 + 0.5 # -20% leads to 0, and +20% leads to 1

def floatNormalize_VolumeChange(newVolume, baseVolume=1.0):
    return floatNormalize_LOG10(newVolume, baseVolume)

def floatNormalize_M1X5(var, base=1.0):
    return (float(var/base) -1) *5.0 + 0.5

NORMALIZE_ID        = 'D%sM1X5' % EXPORT_FLOATS_DIMS
FUNC_floatNormalize = floatNormalize_M1X5

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

EVENT_MONEYFLOW_PREFIX  = MARKETDATE_EVENT_PREFIX + 'MF'
EVENT_MONEYFLOW_1MIN    = EVENT_MONEYFLOW_PREFIX + '1m'
EVENT_MONEYFLOW_5MIN    = EVENT_MONEYFLOW_PREFIX + '5m'
EVENT_MONEYFLOW_1DAY    = EVENT_MONEYFLOW_PREFIX + '1d'

########################################################################
class MarketData(EventData):
    '''Tick行情数据类'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,exchange,date,time'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
        super(MarketData, self).__init__()
        
        # 代码相关
        self.symbol = EventData.EMPTY_STRING              # 合约代码
        self.vtSymbol = EventData.EMPTY_STRING            # 合约在vt系统中的唯一代码，通常是 合约代码.交易所代码

        self.exchange   = exchange
        # self.sourceType = md._sourceType          # 数据来源类型
        if symbol and len(symbol)>0:
            self.symbol = self.vtSymbol = symbol
            if exchange and len(exchange)>0 :
                self.vtSymbol = '.'.join([self.symbol, self.exchange])
        
        self.datetime = None                    # python的datetime时间对象
        self.time = EventData.EMPTY_STRING                # 时间 11:20:56.5
        self.date = EventData.EMPTY_STRING                # 日期 20151009

    @property
    def asof(self) :
        if not self.datetime :
            try :
                self.datetime = datetime.strptime(self.date + 'T' + self.time, '%Y-%m-%dT%H:%M:%S')
            except:
                self.datetime = DT_EPOCH
                
        return self.datetime

    @abstractmethod
    def float4C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim = EXPORT_FLOATS_DIMS for neural network computing
        '''
        raise NotImplementedError

    @abstractmethod
    def float6C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim = 6 for neural network computing
        '''
        raise NotImplementedError

########################################################################
class TickData(MarketData):
    '''Tick行情数据类'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,exchange,date,time,price,volume,open,high,low,prevClose,total,b1P,b2P,b3P,b4P,b5P,b1V,b2V,b3V,b4V,b5V,a1P,a2P,a3P,a4P,a5P,a1V,a2V,a3V,a4V,a5V' # ',upperLimit,lowerLimit'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
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
        self.b1P, self.b1V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.b2P, self.b2V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.b3P, self.b3V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.b4P, self.b4V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.b5P, self.b5V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        # ask to sell: price and volume
        self.a1P, self.a1V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.a2P, self.a2V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.a3P, self.a3V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.a4P, self.a4V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT
        self.a5P, self.a5V = EventData.EMPTY_FLOAT, EventData.EMPTY_FLOAT

    @property
    def desc(self) :
        return 'tick.%s@%s_%dx%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S'), self.volume,round(self.price, PRICE_DISPLAY_ROUND_DECIMALS))

    def __calculateLean(self, X, Y):
        lenX = len(X)        
        if lenX >= len(Y):
            return 0

        sums= [Y[i] *X[i] for i in range(lenX)]
        xsqr= [x*x for x in range(lenX)]
        lean = sum([(sums[i]/xsqr[i]) if xsqr[i]>0 else 0.0 for i in range(lenX)])
        return lean

    @abstractmethod
    def hatch(symbol, exchange=None, **kwargs) :
        tk = TickData(exchange, symbol)
        # 成交数据
        tk.price, tk.volume = float(kwargs['price']), float(kwargs['volume'])
        
        # 常规行情
        tk.open, tk.high, tk.low, tk.prevClose = float(kwargs['open']), float(kwargs['high']), float(kwargs['low']), float(kwargs['prevClose'])
        
        # 五档行情
        # bid to buy: price and volume
        tk.b1P, tk.b1V = float(kwargs['b1P']), float(kwargs['b1V'])
        tk.b2P, tk.b2V = float(kwargs['b2P']), float(kwargs['b2V'])
        tk.b3P, tk.b3V = float(kwargs['b3P']), float(kwargs['b3V'])
        tk.b4P, tk.b4V = float(kwargs['b4P']), float(kwargs['b4V'])
        tk.b5P, tk.b5V = float(kwargs['b5P']), float(kwargs['b5V'])
        # ask to sell: price and volume
        tk.a1P, tk.a1V = float(kwargs['a1P']), float(kwargs['a1V'])
        tk.a2P, tk.a2V = float(kwargs['a2P']), float(kwargs['a2V'])
        tk.a3P, tk.a3V = float(kwargs['a3P']), float(kwargs['a3V'])
        tk.a4P, tk.a4V = float(kwargs['a4P']), float(kwargs['a4V'])
        tk.a5P, tk.a5V = float(kwargs['a5P']), float(kwargs['a5V'])

        if '/' in kwargs['date']:
            tk.date = datetime.strptime(kwargs['date'], '%Y/%m/%d').strftime('%Y-%m-%d')
        else:
            tk.date = kwargs['date']

        tk.time = kwargs['time']
        if tk.time.count(':') <2:
            tk.time += ":00"
        
        tk.datetime = datetime.strptime(tk.date + 'T' + tk.time[:8], '%Y-%m-%dT%H:%M:%S')
        ev = Event(EVENT_TICK)
        ev.setData(tk)
        return ev

    @abstractmethod
    def float4C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim = EXPORT_FLOATS_DIMS for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        leanAsks = self.__calculateLean(X=[(x- self.price) \
            for x in [self.a1P, self.a1P, self.a1P, self.a1P, self.a1P]],
            Y=[self.a1V, self.a1V, self.a1V, self.a1V, self.a1V])

        leanBids = self.__calculateLean(X=[(x- self.price) \
            for x in [self.b1P, self.b1P, self.b1P, self.b1P, self.b1P] ],
            Y=[self.b1V, self.b1V, self.b1V, self.b1V, self.b1V])

        # the basic dims=4
        ret = [
            FUNC_floatNormalize(self.price, baseline_Price), 
            FUNC_floatNormalize(self.volume, baseline_Volume),
            float(leanAsks), 
            float(leanBids)
        ] + [0.0] * ( EXPORT_FLOATS_DIMS -4)

        # the optional dims
        if EXPORT_FLOATS_DIMS > 4:
            ret[4] = FUNC_floatNormalize(self.high, baseline_Price)
        if EXPORT_FLOATS_DIMS > 5:
            ret[5] = FUNC_floatNormalize(self.low, baseline_Price)
        if EXPORT_FLOATS_DIMS > 6:
            ret[6] = FUNC_floatNormalize(self.open, baseline_Price)

        return ret

    @abstractmethod
    def float6C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim =6 for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        #TODO： has the folllowing be normalized into [0.0, 1.0] ???
        leanAsks = self.__calculateLean(X=[(x- self.price) \
            for x in [self.a1P, self.a1P, self.a1P, self.a1P, self.a1P]],
            Y=[self.a1V, self.a1V, self.a1V, self.a1V, self.a1V])

        leanBids = self.__calculateLean(X=[(x- self.price) \
            for x in [self.b1P, self.b1P, self.b1P, self.b1P, self.b1P] ],
            Y=[self.b1V, self.b1V, self.b1V, self.b1V, self.b1V])

        # the basic dims=6
        ret = [
            floatNormalize_LOG10(self.price, baseline_Price),
            floatNormalize_LOG10(self.volume, baseline_Volume),
            float(leanAsks), 
            float(leanBids),
            0.0, 0.0]

        return ret

########################################################################
class KLineData(MarketData):
    '''K线数据'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,exchange,date,time,open,high,low,close,volume' #,openInterest'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        super(KLineData, self).__init__(exchange, symbol)
        
        self.open  = EventData.EMPTY_FLOAT             # OHLC
        self.high  = EventData.EMPTY_FLOAT
        self.low   = EventData.EMPTY_FLOAT
        self.close = EventData.EMPTY_FLOAT
        
        self.volume = EventData.EMPTY_INT             # 成交量
        self.openInterest = EventData.EMPTY_INT       # 持仓量    

    @property
    def desc(self) :
        return 'kline.%s@%s>%sx%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S') if self.datetime else '', self.volume, round(self.close, PRICE_DISPLAY_ROUND_DECIMALS))

    '''
    @property
    def OHLCV(self) :
        return self.open, self.high, self.low, self.close, self.volume
    '''

    @abstractmethod
    def hatch(symbol, evType, exchange=None, **kwargs) :
        if not EVENT_KLINE_PREFIX in evType :
            raise NotImplementedError

        kl = KLineData(exchange, symbol)
        kl.open = float(kwargs['open'])
        kl.high = float(kwargs['high'])
        kl.low = float(kwargs['low'])
        kl.close = float(kwargs['close'])
        kl.volume = float(kwargs['volume'])
        if '/' in kwargs['date']:
            kl.date = datetime.strptime(kwargs['date'], '%Y/%m/%d').strftime('%Y-%m-%d')
        else:
            kl.date = kwargs['date']

        kl.time = kwargs['time']
        if kl.time.count(':') <2:
            kl.time += ":00"
        
        kl.datetime = datetime.strptime(kl.date + 'T' + kl.time[:8], '%Y-%m-%dT%H:%M:%S')
        ev = Event(evType)
        ev.setData(kl)
        return ev

    def hatchByMc(self, symbol, evType, exchange=None, **kwargs) :
        if not EVENT_KLINE_PREFIX in evType :
            raise NotImplementedError

        kl = KLineData(exchange, symbol)
        kl.open = float(kwargs['Open'])
        kl.high = float(kwargs['High'])
        kl.low = float(kwargs['Low'])
        kl.close = float(kwargs['Close'])
        kl.volume = float(kwargs['TotalVolume'])
        kl.date = datetime.strptime(kwargs['Date'], '%Y-%m-%d').strftime('%Y-%m-%d')
        kl.time = kwargs['Time']+":00"
        
        kl.datetime = datetime.strptime(kl.date + 'T' + kl.time, '%Y-%m-%dT%H:%M:%S')
        dataOf = kl.datetime.replace(second=0)
        ev = Event(evType)
        ev.setData(kl)
        return ev

    @abstractmethod
    def float4C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim = EXPORT_FLOATS_DIMS for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        # the basic dims, min=4
        ret = [
            FUNC_floatNormalize(self.close, baseline_Price), 
            FUNC_floatNormalize(self.volume, baseline_Volume),
            FUNC_floatNormalize(self.high, baseline_Price), 
            FUNC_floatNormalize(self.low, baseline_Price), 
        ] + [0.0] * ( EXPORT_FLOATS_DIMS -4)

        # the optional dims
        if EXPORT_FLOATS_DIMS > 4:
            ret[4] = FUNC_floatNormalize(self.open, baseline_Price)
        if EXPORT_FLOATS_DIMS > 5:
            ret[5] = FUNC_floatNormalize(self.openInterest, baseline_Price)

        return ret

    @abstractmethod
    def float6C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim =6 for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        # the 6-dims floats
        ret = [
            floatNormalize_LOG10(self.close, baseline_Price, 1.5),
            floatNormalize(20*(self.high / self.close -1)),
            floatNormalize(20*(self.close / self.low -1)),
            floatNormalize_LOG10(self.volume, baseline_Volume, 1.5),
            floatNormalize(20*(self.open / self.close -1) +0.5),
            0.0
        ]

        return ret

########################################################################
class MoneyflowData(MarketData):
    '''资金流数据'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,exchange,date,time,price,netamount,ratioNet,ratioR0,ratioR3cate'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
        super(MoneyflowData, self).__init__(exchange, symbol)
        
        self.price       = EventData.EMPTY_FLOAT   # 价格
        self.netamount   = EventData.EMPTY_FLOAT   # 净流入金额
        self.ratioNet    = EventData.EMPTY_FLOAT   # 净流入率
        self.ratioR0     = EventData.EMPTY_FLOAT   # 主力流入率
        self.ratioR3cate = EventData.EMPTY_FLOAT   # 散户流入率（分钟资金流时）或 行业净流入率（日资金流时）

    @property
    def desc(self) :
        return 'mf.%s@%s>%s/%.2f,%.2f' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S') if self.datetime else '', self.netamount, self.ratioR0, self.ratioR3cate)

    @abstractmethod
    def hatch(symbol, evType, exchange=None, **kwargs) :
        if not MARKETDATE_EVENT_PREFIX in evType :
            raise NotImplementedError
        
        md = MoneyflowData(exchange, symbol)
        md.price = float(kwargs['price'])
        md.netamount = float(kwargs['netamount'])
        md.ratioR0 = float(kwargs['ratioR0'] if 'ratioR0' in kwargs.keys() else kwargs['r0_ratio'])
        md.ratioR3cate = float(kwargs['ratioR3cate'] if 'ratioR3cate' in kwargs.keys() else kwargs['r3cate_ratio'])
        md.ratioNet = float(kwargs['ratioNet'])  if 'ratioNet' in kwargs.keys() else (md.ratioR0 +md.ratioR3cate)

        if '/' in kwargs['date']:
            md.date = datetime.strptime(kwargs['date'], '%Y/%m/%d').strftime('%Y-%m-%d')
        else:
            md.date = kwargs['date']

        md.time = kwargs['time']
        if md.time.count(':') <2:
            md.time += ":00"
        
        md.datetime = datetime.strptime(md.date + 'T' + md.time[:8], '%Y-%m-%dT%H:%M:%S')
        ev = Event(evType)
        ev.setData(md)
        return ev

    def toFloats(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        # the floats, prioirty first
        ret = [
            floatNormalize_LOG10(self.netamount, baseline_Price*baseline_Volume), # priority-H1, TODO: indeed the ratio of turnover would be more worthy here. It supposed can be calculated from netamount, ratioNet and netMarketCap
            FUNC_floatNormalize(self.ratioNet,    1.0),                          # priority-H2
            FUNC_floatNormalize(self.ratioR0,     1.0),                          # priority-H3
            FUNC_floatNormalize(self.ratioR3cate, 1.0),                          # likely r3=ratioNet-ratioR0
            floatNormalize_LOG10(self.price,     baseline_Price), # optional because usually this has been presented via KLine/Ticks
        ]

        #TODO: other optional dims
        return ret

    @abstractmethod
    def float4C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim =4 for neural network computing
        '''
        ret = self.toFloats(baseline_Price, baseline_Volume)
        return ret[:4] if len(ret) >= 4 else ret +[0.0]* (4-len(ret))

    @abstractmethod
    def float6C(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''
        @return float[] with dim =6 for neural network computing
        '''
        ret = self.toFloats(baseline_Price, baseline_Volume)
        return ret[:4] if len(ret) >= 4 else ret +[0.0]* (4-len(ret))

########################################################################
class TickToKLineMerger(object):
    '''
    K线合成器，支持：
    1. 基于Tick合成1分钟K线
    '''

    #----------------------------------------------------------------------
    def __init__(self, onKLine1min):
        '''Constructor'''
        self._dictKline = {}             # 1分钟K线对象
        self._dickLastTick = {}          # 上一TICK缓存对象
        self.onKline1min = onKLine1min      # 1分钟K线回调函数
        
    #----------------------------------------------------------------------
    def pushTick(self, tick):
        '''TICK更新'''

        kline = None # 尚未创建对象
        
        if tick.symbol in self._dictKline:
            kline = self._dictKline[tick.symbol]
            if kline.datetime.minute != tick.datetime.minute:
                # 生成上一分钟K线的时间戳
                kline.datetime = kline.datetime.replace(second=0, microsecond=0)  # 将秒和微秒设为0
                kline.date = kline.datetime.strftime('%Y-%m-%d')
                kline.time = kline.datetime.strftime('%H:%M:%S.%f')
            
                # 推送已经结束的上一分钟K线
                if self.onKline1min :
                    self.onKline1min(kline)
                
                kline = None # 创建新的K线对象
            
        # 初始化新一分钟的K线数据
        if not kline:
            # 创建新的K线对象
            kline = KLineData(tick.exchange + '_t2k', tick.symbol)
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
    '''
    K线合成器，支持：
    2. 基于1分钟K线合成X分钟K线（X可以是2、3、5、10、15、30	）
    '''

    #----------------------------------------------------------------------
    def __init__(self, onKLineXmin, xmin=15) :
        '''Constructor'''
        self._klineWk = None
        self._klineIn = None        # 上一Input缓存对象
        self._xmin = xmin             # X的值
        self.onXminBar = onKLineXmin  # X分钟K线的回调函数

    #----------------------------------------------------------------------
    def pushKLineEvent(self, klineEv, dtAsOf=None):
        '''1分钟K线更新'''
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
        '''1分钟K线更新'''

        # 尚未创建对象
        if self._klineWk:
            if kline.datetime > self._klineWk.datetime :
                self._klineWk.date = self._klineWk.datetime.strftime('%Y-%m-%d')
                self._klineWk.time = self._klineWk.datetime.strftime('%H:%M:%S.%f')
                    
                # 推送, X分钟策略计算和决策
                return self.flush()

        # 初始化新一分钟的K线数据
        if not self._klineWk:
            # 创建新的K线对象
            self._klineWk = KLineData(kline.exchange if '_k2x' in kline.exchange else kline.exchange + '_k2x', kline.symbol)
            self._klineWk.open = kline.open
            self._klineWk.high = kline.high
            self._klineWk.low = kline.low
            if dtAsOf :
                self._klineWk.datetime = dtAsOf
            else :
                timeTil = kline.datetime + timedelta(minutes=self._xmin)
                timeTil = timeTil.replace(minute=int(timeTil.minute/self._xmin)*self._xmin, second=0, microsecond=0)
                self._klineWk.datetime = timeTil    # 以x分钟K线末作为X分钟线的时间戳
            self._klineWk.date = self._klineWk.datetime.strftime('%Y-%m-%d')
            self._klineWk.time = self._klineWk.datetime.strftime('%H:%M:%S')
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
        '''Constructor'''
        super(DataToEvent, self).__init__()
        self._sink = sink
        self._dict = {}

    @property
    def fields(self) : return None

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
        '''Constructor'''
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
        @return the price, datetimeAsOf
        '''
        raise NotImplementedError

    @abstractmethod
    def getAsOf(self, symbol=None, evType =None) :
        ''' 
        @return the datetime as of latest observing
        '''
        raise NotImplementedError

    @abstractmethod
    def sizesOf(self, symbol, evType =None) :
        ''' 
        @return the size of specified symbol/evType
        '''
        raise NotImplementedError
        return 0, 0

    @abstractmethod
    def resize(self, symbol, evType, evictSize) :
        raise NotImplementedError

    @abstractmethod
    def descOf(self, symbol) :
        ''' 
        @return the desc of specified symbol
        '''
        return '%s' % symbol

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

    def exportF1548(self, symbol=None) :

        F4SECHMA_1548 = OrderedDict({
            'asof':              1,
            EVENT_KLINE_1MIN :  30,
            EVENT_KLINE_5MIN :  96,
            EVENT_KLINE_1DAY : 260,
        })

        ret = self.export4C(symbol, d4wished=F4SECHMA_1548)
        if not ret: return [0.0] * 1548

        if isinstance(ret, list) and 1548 ==len(ret):
            return ret

        raise ValueError('export4C() unexpected ret')

    '''
    import math
    for i in range(16*16):
        if i<=0:
            ret=[(0,0)]
            continue

        a = int(math.sqrt(i))
        b = i - a*a
        c = int(a/2)

        if a == c*2:
            x, y = -c, c
            if b >=a:
                y -= a
                x += (b -a)
            else: y -= b
        else:
            x, y = c+1, -c
            if b >=a:
                y += a
                x -= (b -a)
            else: y += b
        ret.append((x+7, y+7))
        print('%s=%d ^2 +%d@(%d, %d)' % (i, a, b, x, y))

    print('COORDS_Snail16x16=%s' % ret)
    '''
    COORDS_Snail16x16=[(7, 7), (8, 7), (8, 8), (7, 8), (6, 8), (6, 7), (6, 6), (7, 6), (8, 6), (9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9),
                (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (9, 10),
                (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4),
                (10, 4), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (10, 11), (9, 11), (8, 11), (7, 11), (6, 11), (5, 11), (4, 11),
                (3, 11), (3, 10), (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
                (11, 3), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (11, 12), (10, 12), (9, 12), (8, 12), (7, 12),
                (6, 12), (5, 12), (4, 12), (3, 12), (2, 12), (2, 11), (2, 10), (2, 9), (2, 8), (2, 7), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2), (3, 2),
                (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2), (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8),
                (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (12, 13), (11, 13), (10, 13), (9, 13), (8, 13), (7, 13), (6, 13), (5, 13), (4, 13), (3, 13), (2, 13),
                (1, 13), (1, 12), (1, 11), (1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1),
                (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7),
                (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (13, 14), (12, 14), (11, 14), (10, 14), (9, 14), (8, 14), (7, 14), (6, 14), (5, 14),
                (4, 14), (3, 14), (2, 14), (1, 14), (0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3),
                (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0),
                (14, 0), (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14),
                (15, 15), (14, 15), (13, 15), (12, 15), (11, 15), (10, 15), (9, 15), (8, 15), (7, 15), (6, 15), (5, 15), (4, 15), (3, 15), (2, 15), (1, 15), (0, 15)]
    
    BMP_COLOR_BG_FLOAT=1.0

    def exportImg6C16x16x4(self, symbol=None) :

        C6SECHMA_16x16x4 = OrderedDict({
            'asof'               : 1,
            EVENT_KLINE_1MIN     : 16,
            EVENT_KLINE_5MIN     : 240,
            EVENT_KLINE_1DAY     : 255,
            EVENT_MONEYFLOW_1MIN : 240,
        })

        seq6C = self.export6C(symbol, d4wished=C6SECHMA_16x16x4)
        if not seq6C: return None

        # TODO: draw the imagex
        img6C = [ [ [MarketState.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(2*16)] # DONOT take [ [[0.0]*6] *16] *16
        # for i in range(240):
        #     x, y = MarketState.COORDS_Snail16x16[i]
        #     img6C[y][x] = seq6C[1 + i]

        # for i in range(240):
        #     x, y = MarketState.COORDS_Snail16x16[i]
        #     img6C[16+y][x] = seq6C[1 +240 + i]
        
        # parition 0: the central 4x4 is KL1min up to 16min, the outter are 240KL5min up to a week
        partition =0
        seq6C_offset =1
        for i in range(16): 
             x, y = MarketState.COORDS_Snail16x16[i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1
        
        for i in range(16, 16 +240): 
             x, y = MarketState.COORDS_Snail16x16[i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1

        # parition 1: the central 1x1 is current datetime, the outter are 255KL1d covers a year
        partition +=1
        x, y = MarketState.COORDS_Snail16x16[0]
        img6C[partition*16 + y][x] = seq6C[0]

        seq6C_offset =C6SECHMA_16x16x4['asof'] + C6SECHMA_16x16x4[EVENT_KLINE_1MIN] + C6SECHMA_16x16x4[EVENT_KLINE_5MIN] # pointer to where EVENT_KLINE_1DAY is
        for i in range(1, 1+C6SECHMA_16x16x4[EVENT_KLINE_1DAY]): 
             x, y = MarketState.COORDS_Snail16x16[i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1

        return self.covertImg6CTo3C(img6C) # return img6C

    bmpSeq=0
    outputSeq=0

    def covertImg6CTo3C(self, img6C) :
        lenX = len(img6C[0])
        lenY = len(img6C)
        img3C = [ [ [MarketState.BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX*2)] for y in range(lenY) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(img6C)
        for y in range(lenY):
            for x in range(lenX) :
                # img3C[y][x], img3C[y][lenX + x] = img6C[y][x][:3], img6C[y][x][3:]
                img3C[y][2*x], img3C[y][2*x +1] = img6C[y][x][:3], img6C[y][x][3:]

        MarketState.bmpSeq +=1
        width = 320
        if 0 == MarketState.bmpSeq % 60:
            imgarray = np.uint8(np.array(img3C)*255)
            bmp = Image.fromarray(imgarray)
            if width > lenX:
                bmp = bmp.resize((width, int(width *1.0/lenX/2 *lenY)), Image.NEAREST)
            # bmp.convert('RGB')
            bmp.save('/mnt/e/bmp/test_%06d.png' % int(MarketState.outputSeq))
            MarketState.outputSeq +=1

        return img3C

    @abstractmethod
    def export4C(self, symbol, d4wished= OrderedDict({ 'asof':1, EVENT_KLINE_1DAY:20 }) ) :
        '''
        @param d4wished OrderedDict to specify number of most recent 4-float of the event category to export
        @return an array_like floats
        '''
        raise NotImplementedError

    @abstractmethod
    def export6C(self, symbol=None) :
        raise NotImplementedError

    # @abstractmethod
    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     raise NotImplementedError
