# encoding: UTF-8

from __future__ import division

from Application import BaseApplication, Iterable
from EventData import *
from MarketData import *

from datetime import datetime
from abc import ABCMeta, abstractmethod
import traceback

import bz2, csv
import copy, pickle
# import tensorflow as tf

EVENT_Perspective  = MARKETDATE_EVENT_PREFIX + 'Persp'   # 错误回报事件

DEFAULT_KLDEPTH_TICK = 0
# DEFAULT_KLDEPTH_TICK = 120
DEFAULT_KLDEPTH_1min = 32
DEFAULT_KLDEPTH_5min = 240 # 96
DEFAULT_KLDEPTH_1day = 260

EXPORT_SIGNATURE= '%dT%dM%dF%dD.%s:200109T17' % (DEFAULT_KLDEPTH_TICK, DEFAULT_KLDEPTH_1min, DEFAULT_KLDEPTH_5min, DEFAULT_KLDEPTH_1day, NORMALIZE_ID)

DEFAULT_MFDEPTH_1min = 60  # 1-hr
DEFAULT_MFDEPTH_5min = 48  # a 4-hr day
DEFAULT_MFDEPTH_1day = 120 # about half a year

########################################################################
class EvictableStack(object):
    def __init__(self, evictSize=0, nildata=None):
        '''Constructor'''
        super(EvictableStack, self).__init__()
        self.__data =[]
        self.__evictSize = evictSize
        self.__dataNIL = copy.copy(nildata) if nildata else None
        self.__stampUpdated = None
        # if self.__dataNIL and self.__evictSize and self.__evictSize >0 :
        #     for i in range(self.__evictSize) :
        #         self.__data.insert(0, nildata)

    def __getitem__(self, index):
        return self.__data[index]

    def __setitem__(self, index, value):
        self.__data[index] = value
    
    @property
    def top(self):
        return self.__data[0] if len(self.__data) >0 else None

    @property
    def evictSize(self):
        if self.__evictSize <0 : self.__evictSize =0
        return self.__evictSize

    @property
    def size(self):
        return len(self.__data) if self.__data else 0

    def resize(self, evictSize):
        self.__evictSize = int(evictSize)

        while self.evictSize >=0 and self.size > self.evictSize:
            del(self.__data[-1])

        return self.evictSize

    @property
    def exportList(self):
        return self._exportList()

    @property
    def stampUpdated(self):
        return self.__stampUpdated if self.__stampUpdated else DT_EPOCH

    def _exportList(self, nilFilled=False):
        data = copy.deepcopy(self.__data)
        if nilFilled :
            fillsize = (self.evictSize - self.size) if self.evictSize >=0 else 0
            data += [self.__dataNIL] *fillsize
        return data

    def overwrite(self, item):
        self.__data[0] =item

    def insert(self, index, item):
        if index <0 or index >= len(self.__data):
            return
        self.__data[index] =item
        while self.evictSize >=0 and self.size > self.evictSize:
            del(self.__data[-1])
        self.__stampUpdated = datetime.now()

    # no pop here: def pop(self):
    #    del(self.__data[-1])

    def push(self, item):
        self.__data.insert(0, item)
        while self.evictSize >=0 and self.size > self.evictSize:
            del(self.__data[-1])
        self.__stampUpdated = datetime.now()

########################################################################
class KLineEx(KLineData):

    EVENT2EXT = {
        EVENT_MONEYFLOW_1MIN: EVENT_KLINE_1MIN,
        EVENT_MONEYFLOW_5MIN: EVENT_KLINE_5MIN,
        EVENT_MONEYFLOW_1DAY: EVENT_KLINE_1DAY,
    }

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'symbol,exchange,date,time,price,netamount,ratioNet,ratioR0,ratioR3cate'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
        super(KLineEx, self).__init__(exchange, symbol)
        
        self.datetime = None
        self.src = []

        self.ratioNet    = EventData.EMPTY_FLOAT   # 净流入率
        self.ratioR0     = EventData.EMPTY_FLOAT   # 主力流入率
        self.ratioR3cate = EventData.EMPTY_FLOAT   # 散户流入率（分钟资金流时）或 行业净流入率（日资金流时）

    @property
    def desc(self) :
        return 'KlEx.%s@%s>%sx%s/%.2f,%.2f%%' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S') if self.datetime else '', self.close, int(self.volume), self.ratioR0, self.ratioNet)

    @abstractmethod
    def hatch(symbol, evType, exchange=None, **kwargs) :
        raise NotImplementedError

    def setEvent(self, ev) :

        if not ev or not isinstance(ev, Event) or not isinstance(ev.data, MarketData) :
            return
        
        if not self.datetime:
            self.datetime = ev.data.datetime
        
        self.src.append(ev.type)
        val = ev.data
        
        if isinstance(val, MoneyflowData) :
            self.close = val.price
            self.ratioNet, self.ratioR0, self.ratioR3cate = val.ratioNet, val.ratioR0, self.ratioR3cate
            if val.ratioNet * val.netamount * val.price > 0:
                self.close  = val.price
                volume = round(val.netamount / val.ratioNet / val.price, 0)
                # DONOT set self.volume as the above volume cacluated was base to today-to-now instead that of this Xmin
                # if self.volume < volume:
                #     self.volume = volume
            return
        
        if isinstance(val, KLineData) :
            self.open  = val.open
            self.high  = val.high
            self.low   = val.low
            self.close = val.close
            self.openInterest = val.openInterest   
            
            if self.volume < val.volume:
                self.volume = val.volume
            return

    def floatXC(self, baseline_Price=1.0, baseline_Volume =1.0, channels=6) :
        '''
        @return float[] for neural network computing
        '''
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0

        # the floats, prioirty first
        ret = [
            floatNormalize_LOG10(self.close, baseline_Price),
            floatNormalize_LOG10(self.volume, baseline_Volume),
            floatNormalize(0.5 + self.ratioNet),                          # priority-H2
            floatNormalize(0.5 + self.ratioR0),                          # priority-H3
            floatNormalize(0.5 + self.ratioR3cate),                          # likely r3=ratioNet-ratioR0
        ]
        #TODO: other optional dims

        channels = int(channels)
        return ret[:channels] if len(ret) >= channels else ret +[0.0]* (channels -len(ret))

########################################################################
class Perspective(MarketData):
    '''
    Data structure of Perspective:
    1. Ticks
    2. 1min KLines
    3. 5min KLines
    4. 1day KLines
    '''
    EVENT_SEQ_KLTICK =  [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]
    EVENT_SEQ_MF = [EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_5MIN, EVENT_MONEYFLOW_1DAY]

    TICKPRICES_TO_EXP = 'price,open,high,low,b1P,b2P,b3P,b4P,b5P,a1P,a2P,a3P,a4P,a5P'
    TICKVOLS_TO_EXP   = 'volume,b1V,b2V,b3V,b4V,b5V,a1V,a2V,a3V,a4V,a5V'
    KLPRICES_TO_EXP = 'open,high,low,close'
    KLVOLS_TO_EXP = 'volume'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol=None, KLDepth_1min=DEFAULT_KLDEPTH_1min, KLDepth_5min=DEFAULT_KLDEPTH_5min, KLDepth_1day=DEFAULT_KLDEPTH_1day, tickDepth=DEFAULT_KLDEPTH_TICK, MFDepth_1min=DEFAULT_MFDEPTH_1min, MFDepth_5min=DEFAULT_MFDEPTH_5min, MFDepth_1day =DEFAULT_MFDEPTH_1day, **kwargs) :
        '''Constructor'''
        super(Perspective, self).__init__(exchange, symbol)

        self._stacks = {
            EVENT_TICK:       EvictableStack(tickDepth, TickData(self.exchange, self.symbol)),
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineEx(self.exchange, self.symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineEx(self.exchange, self.symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_1day, KLineEx(self.exchange, self.symbol)),

            # EVENT_MONEYFLOW_1MIN: EvictableStack(MFDepth_1min, MoneyflowData(self.exchange, self.symbol)),
            # EVENT_MONEYFLOW_5MIN: EvictableStack(MFDepth_5min, MoneyflowData(self.exchange, self.symbol)),
            # EVENT_MONEYFLOW_1DAY: EvictableStack(MFDepth_1day, MoneyflowData(self.exchange, self.symbol)),
        }

        self.__stampLast = None
        self.__focusLast = None
        self.__dayOHLC   = None

        # TODO: evsPerDay temporarily is base on AShare's 4hr/day
        self.__evsPerDay = {
            EVENT_TICK:       3600/2 *4, # assuming every other seconds
            EVENT_KLINE_1MIN: 60*4,
            EVENT_KLINE_5MIN: 12*4,
            EVENT_KLINE_1DAY: 1,
            EVENT_MONEYFLOW_1MIN: 60*4,
            EVENT_MONEYFLOW_5MIN: 12*4,
            EVENT_MONEYFLOW_1DAY: 1,
        }

        self.__overview = {
            'asOf': None, # datetime when this overview is as of
            'netMarketCap': 0.0, # net market cap in volumes, may be taken to calculate the turnover ratio
            'marketCap': 0.0, # market cap in volumes
            'ratioPE': 0.0, # PE
        }

    @property
    def desc(self) :
        str = '%s>' % chopMarketEVStr(self.focus)
        stack = self._stacks[self.focus]
        if stack.size >0:
            str += '%s ' % stack.top.desc
        else: str += 'NIL '

        for i in self.eventTypes :
            str += '%sX%d/%d,' % (chopMarketEVStr(i), self._stacks[i].size, self._stacks[i].evictSize)
        return str

    @property
    def eventTypes(self) : return list(self._stacks.keys())

    @property
    def overview(self) : return copy.copy(self.__overview)

    def updateOverview(self, **kwargs) : 
        self.__overview = {**self.__overview, **kwargs} # kwargs at right will overwrite __overview at left
        return self.overview

    @property
    def asof(self) : return self.getAsOf(None)

    def getAsOf(self, evType=None) :
        if evType and evType in self._stacks.keys() :
            stack = self._stacks[evType]
            if stack.size>0:
                return stack.top.asof
    
        return self.__stampLast if self.__stampLast else DT_EPOCH

    def stampUpdatedOf(self, evType=None) :
        if not evType or not evType in self._stacks.keys():
            return DT_EPOCH
        
        return self._stacks[evType].stampUpdated

    def sizesOf(self, evType=None) :
        if not evType or len(evType) <=0:
            size =0
            esize =0
            for k in self._stacks.keys():
                size += self._stacks[k].size
                esize += self._stacks[k].evcitSize
            return size, esize

        if evType in self._stacks.keys():
            return self._stacks[evType].size, self._stacks[evType].evictSize

        return 0, 0

    def resize(self, evType, evictSize):
        if evType and evType in self._stacks.keys():
            return self._stacks[evType].resize(evictSize)
        return 0

    @property
    def focus(self) :
        return self.__focusLast if self.__focusLast else ''

    @property
    def latestPrice(self) :
        ret =0.0
        # stk = self._stacks[self.__focusLast]
        # if stk and stk.size >0:
        #     ret = stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close
        # else:
        #     for et in self.eventTypes:
        #         stk = self._stacks[et]
        #         if not stk or stk.size <=0:
        #             continue
        #         ret = stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close

        seq = [self.__focusLast]  if self.__focusLast else []
        seq += self.eventTypes
        latestAsOf = None
        for et in seq:
            if EVENT_TICK != et and not EVENT_KLINE_PREFIX in et:
                continue
            stk = self._stacks[et]
            if not stk or stk.size <=0:
                continue

            if latestAsOf and latestAsOf > stk.top.asof:
                continue

            latestAsOf = stk.top.asof
            ret = stk.top.price if isinstance(stk.top, TickData) else stk.top.close

        if latestAsOf is None:
            latestAsOf = DT_EPOCH
        return round(ret, PRICE_DISPLAY_ROUND_DECIMALS), latestAsOf

    @property
    def dailyOHLC_sofar(self) :
        if not self.__dayOHLC:
            dtAsof = self.asof
            if not dtAsof or dtAsof <= DT_EPOCH : 
                return None

            self.__dayOHLC = KLineData(self.exchange, self.symbol)
            self.__dayOHLC.datetime = dtAsof.replace(hour=0,minute=0,second=0,microsecond=0)
            self.__dayOHLC.open = 0.0
            self.__dayOHLC.high = 0.0
            self.__dayOHLC.low  = 0.0
            self.__dayOHLC.volume=0

            for et in [EVENT_KLINE_5MIN, EVENT_KLINE_1MIN]:
                lst = self._stacks[et]._exportList()
                if len(lst) <=0:
                    continue
                lst.reverse()
                for i in lst :
                    if i.asof < self.__dayOHLC.asof :
                        continue

                    self.__dayOHLC.high = max(self.__dayOHLC.high, i.high)
                    self.__dayOHLC.low  = i.low if self.__dayOHLC.low <=0 else min(self.__dayOHLC.low, i.low)
                    if self.__dayOHLC.open <=0 :
                        self.__dayOHLC.open = i.open

                    self.__dayOHLC.close = i.close        
                    self.__dayOHLC.openInterest = i.openInterest
                    self.__dayOHLC.volume =0  # NOT GOOD when built up from 1min+5min: += int(i.volume) 
                    self.__dayOHLC.datetime = i.asof

            #TODO sumup EVENT_TICK

        # self.__dayOHLC.datetime = self.asof.replace(hour=23,minute=59,second=59,mircosecond=0)
        self.__dayOHLC.date = self.__dayOHLC.datetime.strftime('%Y-%m-%d')
        self.__dayOHLC.time = self.__dayOHLC.datetime.strftime('%H:%M:%S')
        return self.__dayOHLC

    def dumps(self) :
        dict = {
            'stacks' : self._stacks,
            'overview' : self.__overview,
            'focus': (self.__stampLast, self.__focusLast),
            'today' : self.__dayOHLC
        }

        return pickle.dumps(dict) # this is bytes

    def loads(self, pickleData) : # load the pikledata exported from dump()
        dict = pickle.loads(pickleData)

        if 'stacks' in dick.keys():
            self._stacks = dict['stacks']

        if 'overview' in dick.keys():
            self.__overview = dict['overview']

        if 'focus' in dick.keys():
            self.__stampLast, self.__focusLast = dict['focus']

        if 'today' in dick.keys():
            self.__dayOHLC = dict['today']

    def push(self, ev) :
        ev, stk = self.__push(ev)

        while stk and stk.evictSize >=0 and stk.size > stk.evictSize:
            del(stk._data[-1])

        if not ev :
            return None

        # special dealing about TICK and KLine
        if ev.type in Perspective.EVENT_SEQ_KLTICK:
            if self.__dayOHLC and self.__dayOHLC.asof < self.asof.replace(hour=0,minute=0,second=0,microsecond=0) :
                self.__dayOHLC = None

            evd = ev.data
            if not self.__dayOHLC :
                if isinstance(evd, KLineData):
                    self.__dayOHLC = evd
                return ev

            if evd.asof > self.__dayOHLC.asof:
                if isinstance(evd, KLineData):
                    self.__dayOHLC.close = float(evd.close)
                    self.__dayOHLC.high = max(self.__dayOHLC.high, self.__dayOHLC.close)
                    self.__dayOHLC.low  = min(self.__dayOHLC.low, self.__dayOHLC.close)
                    # self.__dayOHLC.volume =0 # NOT GOOD when built up from 1min+5min: += int(evd.volume)                
                    self.__dayOHLC.openInterest = float(evd.openInterest)
                    self.__dayOHLC.datetime = evd.asof
                elif isinstance(evd, TickData):
                    self.__dayOHLC.close = float(evd.price)
                    self.__dayOHLC.high = max(self.__dayOHLC.high, self.__dayOHLC.close)
                    self.__dayOHLC.low = min(self.__dayOHLC.low, self.__dayOHLC.close)
                    self.__dayOHLC.datetime = evd.datetime  
                    self.__dayOHLC.openInterest = evd.openInterest

        return ev

    def __push(self, ev) :
        '''
        @return the ev that has been successully pushed into the proper stack, otherwise None
        '''
        if not ev: return None
 
        stk, etOfStack  = None, ev.type
        if etOfStack in self._stacks.keys():
            stk = self._stacks[etOfStack]
        elif etOfStack in KLineEx.EVENT2EXT.keys():
            etOfStack = KLineEx.EVENT2EXT[etOfStack]
            if etOfStack in self._stacks.keys():
                stk = self._stacks[etOfStack]

        if not stk: return None, None
        
        if not self.__stampLast or self.__stampLast < ev.data.datetime :
            self.__stampLast = ev.data.datetime

        latestevd = stk.top
        self.__focusLast = etOfStack

        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            if EVENT_TICK == etOfStack:
                newed = ev.data
            else:
                newed = KLineEx(ev.data.exchange, ev.data.symbol)
                newed.setEvent(ev)

            stk.push(newed)
            return ev, stk

        overwritable = not latestevd.exchange or ('_k2x' in latestevd.exchange or '_t2k' in latestevd.exchange)
        if ev.type == etOfStack:
            if latestevd.exchange and (not ev.data.exchange or len(ev.data.exchange) <=0) :
                overwritable = False

            if EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] and ev.data.datetime == latestevd.datetime and ev.data.volume > latestevd.volume :
                # SINA KL-data was found has such a bug as below: the later polls got bigger volume, so treat the later larger volume as correct data
                # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.915,2.914,2.915,570200.0
                # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.918,2.914,2.918,4575100.0
                # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.918,2.914,2.918,4575100.0
                overwritable = True
        else:
            # this must be an extending, such as MF to fill KLex
            overwritable = True

        if not overwritable:
            return None, stk # not overwritable

        for i in range(stk.size) :
            if ev.data.datetime > stk[i].datetime :
                continue

            if EVENT_TICK == etOfStack:
                if ev.data.datetime == stkdata.datetime :
                    stk[i] = ev.data
                else:
                    stk.insert(i, ev.data)

            stkdata = stk[i]
            if ev.data.datetime == stkdata.datetime :
                stkdata.setEvent(ev)
            else :
                stkdata = KLineEx(ev.data.exchange, ev.data.symbol)
                stkdata.setEvent(ev)
                stk.insert(i, stkdata)

            return ev, stk
        
        if EVENT_TICK == etOfStack:
            stk.insert(-1,  ev.data)
        else:
            stkdata = KLineEx(ev.data.exchange, ev.data.symbol)
            stkdata.setEvent(ev)
            stk.insert(-1, stkdata)

        return ev, stk

    TICK_FLOATS=7
    KLINE_FLOATS=5

    @property
    def fullFloatSize(self):
        # klsize = sum([self._stacks[et].evictSize for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] ])
        # return Perspective.TICK_FLOATS *self._stacks[EVENT_TICK].evictSize + Perspective.KLINE_FLOATS *klsize
        itemsize = sum([self._stacks[et].evictSize for et in self.eventTypes])
        return (itemsize +1) * EXPORT_FLOATS_DIMS

    @property
    def _S1548I4(self):
        '''@return an array_like data as float4C, maybe [] or numpy.array
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return [0.0] * self.fullFloatSize # float4C not available
        
        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        return self.__exportS1548I4(baseline_Price=klbaseline.close, baseline_Volume=klbaseline.volume)
    
    def floatsD4(self, lstsWished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''@return an array_like data as float4C, maybe [] or numpy.array
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None # float4C not available

        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        baseline_Price, baseline_Volume =klbaseline.close, klbaseline.volume

        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        result = []
        for k, v in lstsWished.items():
            if 'asof' ==k and int(v) >0:
                fAsOf = [0.0] * EXPORT_FLOATS_DIMS
                try :
                    stampAsof = self.asof
                    fAsOf[0] = stampAsof.month
                    fAsOf[1] = stampAsof.day
                    fAsOf[2] = stampAsof.weekday()
                    fAsOf[3] = stampAsof.hour *60 +stampAsof.minute
                except: pass
                result += fAsOf # datetime as the first item
                continue

            if not k in self.eventTypes :
                raise ValueError('Perspective.floatsD4() unknown etype[%s]' %k )

            stk = self._stacks[k]
            bV = baseline_Volume
            if k in self.__evsPerDay.keys():
               bV /= self.__evsPerDay[k]

            for i in range(int(v)):
                if i >= stk.size:
                    result += [0.0] * EXPORT_FLOATS_DIMS
                else:
                    fval = stk[i].float4C(baseline_Price=baseline_Price, baseline_Volume= bV)
                    result += fval

        return result

    def export(self, lstsWished= { EVENT_KLINE_1DAY:20 } ) :
        result = {}

        for k, v in lstsWished.items():
            if not k in self.eventTypes :
                continue

            result[k] = self._stacks[k].exportList
            if v>0 and v > len(result[k]):
                del result[k][v:]

        return result

    """
    def float6C(self, lstsWished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''@return a 2D array of floats
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None # float6C not available

        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        baseline_Price, baseline_Volume =klbaseline.close, klbaseline.volume

        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        result = []
        for k, v in lstsWished.items():
            if 'asof' ==k and int(v) >0:
                fAsOf = [0.0] * 6
                try :
                    stampAsof = self.asof
                    fAsOf[0] = (stampAsof.month-1) / 12.0 # normalize to [0.0,1.0]
                    fAsOf[1] = stampAsof.day / 31.0 # normalize to [0.0,1.0]
                    fAsOf[2] = stampAsof.weekday() / 7.0 # normalize to [0.0,1.0]
                    fAsOf[3] = (stampAsof.hour *60 +stampAsof.minute) / (24 *60.0) # normalize to [0.0,1.0]
                except: pass
                result.append(fAsOf) # datetime as the first item
                continue

            if not k in self.eventTypes :
                raise ValueError('Perspective.float6C() unknown etype[%s]' %k )

            stk = self._stacks[k]
            bV = baseline_Volume
            if k in self.__evsPerDay.keys():
               bV /= self.__evsPerDay[k]

            for i in range(int(v)):
                if i >= stk.size:
                    result.append([0.0] * 6)
                else:
                    fval = stk[i].float6C(baseline_Price=baseline_Price, baseline_Volume= bV)
                    result.append(fval)

        return result
    
    def float6Cx(self) :
        '''@return a 2D array of floats
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None # float6C not available

        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        baseline_Price, baseline_Volume =klbaseline.close, klbaseline.volume

        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        result = {
            'asof': [
                (self.asof.month -1) / 12.0, # normalize to [0.0,1.0]
                (self.asof.day -1) / 31.0, # normalize to [0.0,1.0]
                self.asof.weekday() / 7.0, # normalize to [0.0,1.0]
                (self.asof.hour *60 +self.asof.minute) / (24 *60.0), # normalize to [0.0,1.0]
                (int(datetime.strftime(self.asof, '%j')) -1) / 366.0, # the julian day of the year
                0.0
            ],
        }

        # result[EVENT_KLINE_1MIN] = Perspective.__richKL6(self._stacks[EVENT_KLINE_1MIN].exportList, self._stacks[EVENT_MONEYFLOW_1MIN].exportList, baseline_Price, baseline_Volume, self.__evsPerDay[EVENT_KLINE_1MIN])
        # result[EVENT_KLINE_5MIN] = Perspective.__richKL6(self._stacks[EVENT_KLINE_5MIN].exportList, self._stacks[EVENT_MONEYFLOW_5MIN].exportList, baseline_Price, baseline_Volume, self.__evsPerDay[EVENT_KLINE_5MIN])
        # result[EVENT_KLINE_1DAY] = Perspective.__richKL6(self._stacks[EVENT_KLINE_1DAY].exportList, self._stacks[EVENT_MONEYFLOW_1DAY].exportList, baseline_Price, baseline_Volume, 1)
        for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] :
            floats =[]
            for klx in self._stacks[et].exportList:
                floats.append(klx.float6C(baseline_Price, baseline_Volume/self.__evsPerDay[et]))
            result[et] = floats

        return result

    """

    def __richKL6(listKL, listMF, basePrice, baseVol, minsPerDay):
        result =[]
        bV = baseVol /minsPerDay

        for kl in listKL:
            klf = [
                floatNormalize_LOG10(kl.close, basePrice, 1.5),
                floatNormalize_LOG10(kl.volume, bV, 1.5),
                floatNormalize(minsPerDay*(kl.high / kl.close -1)),
                floatNormalize(minsPerDay*(kl.close / kl.low -1)),
                0.0, 0.0
            ]

            while len(listMF)>0 and listMF[0].asof < kl.asof:
                del listMF[0]
                continue

            if len(listMF)>0 and listMF[0].asof == kl.asof:
                mf = listMF[0]
                if minsPerDay >1:
                    klf[4],klf[5] = floatNormalize(0.5 + 10*mf.ratioNet), floatNormalize(0.5 + 10*mf.ratioR0) # in-day KLs
                else:
                    klf[4],klf[5] = floatNormalize(0.5 + 10*mf.ratioNet), minsPerDay*(kl.open / kl.high -1)
            
            result.append(klf)
        return result
        
    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     if symbol and symbol in self._dictPerspective.keys():
    #         return self._dictPerspective[symbol].engorged

    @property
    def TickFloats(self) :
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return [0.0] * self.fullFloatSize # float4C not available
        
        klbaseline = self._stacks[EVENT_KLINE_1DAY].top

        result = []
        stk = self._stacks[EVENT_TICK]
        bV= (klbaseline.volume / self.__evsPerDay[EVENT_TICK])
        for i in range(stk.evictSize):
            if i >= stk.size:
                result += [0.0] * EXPORT_FLOATS_DIMS
            else:
                v = stk[i].float4C(baseline_Price=klbaseline.close, baseline_Volume= bV)
                result += v
        return result
    
    def __exportS1548I4(self, baseline_Price=1.0, baseline_Volume =1.0) :
        '''@return an array_like data as float4C, maybe [] or numpy.array
        '''
        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        fAsOf = [0.0] * EXPORT_FLOATS_DIMS
        try :
            stampAsof = self.asof
            fAsOf[0] = stampAsof.month
            fAsOf[1] = stampAsof.day
            fAsOf[2] = stampAsof.weekday()
            fAsOf[3] = stampAsof.hour *60 +stampAsof.minute
        except: pass

        result = fAsOf # datetime as the first item
        c =1

        for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]:
            stk = self._stacks[et]
            bV= (baseline_Volume / self.__evsPerDay[et])
            for i in range(stk.evictSize):
                if i >= stk.size:
                    result += [0.0] * EXPORT_FLOATS_DIMS
                else:
                    v = stk[i].float4C(baseline_Price=baseline_Price, baseline_Volume= bV)
                    # Perspective.KLINE_FLOATS = len(v)
                    result += v

        return result

    def __data2export(self, mdata, fields) :
        fdata = []
        for f in fields:
            fdata.append(float(mdata.__dict__[f]))
        return fdata

########################################################################
class PerspectiveGenerator(Iterable):

    #----------------------------------------------------------------------
    def __init__(self, perspective):
        super(PerspectiveGenerator, self).__init__()
        self._perspective = perspective
        self.__gen = None

        self._readers = {
            EVENT_TICK:       None,
            EVENT_KLINE_1MIN: None,
            EVENT_KLINE_5MIN: None,
            EVENT_KLINE_1DAY: None,
        }

    def adaptReader(self, reader, grainedEvent = EVENT_KLINE_1MIN):
        self._readers[grainedEvent] = reader

    # -- impl of Iterable --------------------------------------------------------------
    def resetRead(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        succ =False
        for et in self._perspective.eventTypes:
            if not self._readers[et] :
                continue
            if self._readers[et].resetRead() :
                succ = True

        return succ

    def readNext(self):
        '''
        @return next item, mostlikely expect one of Event()
        '''
        for et in self._perspective.eventTypes:
            if not self._readers[et] :
                continue
            ev = next(self._readers[et])
            if not ev or not ev.data:
                return None
            if not ev.type in self._perspective._stacks.keys():
                return ev

            latestevd = self._perspective._stacks[ev.type].top
            if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
                if self._perspective.push(ev) :
                    evPsp = Event(EVENT_Perspective)
                    evPsp.setData(self._perspective)
                    return evPsp
                
        return None

########################################################################
class PerspectiveState(MarketState):

    def __init__(self, exchange):
        """Constructor"""
        super(PerspectiveState, self).__init__(exchange)
        self._dictPerspective ={} # dict of symbol to Perspective
        # self.__dictMoneyflow ={} # dict of symbol to MoneyflowPerspective

    # -- impl of MarketState --------------------------------------------------------------
    def listOberserves(self) :
        return [ s for s in self._dictPerspective.keys()]

    def addMonitor(self, symbol) :
        ''' add a symbol to monitor
        '''
        if symbol and not symbol in self._dictPerspective.keys() :
            self._dictPerspective[symbol] = Perspective(self.exchange, symbol)

    def latestPrice(self, symbol) :
        ''' query for latest price of the given symbol
        @return the price, datetimeAsOf
        '''
        if not symbol in self._dictPerspective.keys() :
            return 0.0, DT_EPOCH
        
        return self._dictPerspective[symbol].latestPrice

    def getAsOf(self, symbol=None, evType =None) :
        ''' 
        @return the datetime as of latest observing
        '''
        if len(self._dictPerspective) <=0: return DT_EPOCH
        if symbol and symbol in self._dictPerspective.keys():
            psp = self._dictPerspective[symbol]
            if psp:
                if evType and evType in psp._stacks.keys():
                    return psp.getAsOf(evType)
                return psp.asof

        return max([pofs.asof for pofs in self._dictPerspective.values()])

    def stampUpdatedOf(self, symbol=None, evType=None) :
        if len(self._dictPerspective) <=0: return DT_EPOCH
        if symbol and symbol in dict:
            return self._dictPerspective[symbol].stampUpdatedOf(evType)

        return max([pofs.asof for pofs in self._dictPerspective.values()])

    def sizesOf(self, symbol, evType =None) :
        ''' 
        @return the size of specified symbol/evType
        '''
        if symbol and symbol in self._dictPerspective.keys():
            return self._dictPerspective[symbol].sizesOf(evType)
        return 0, 0

    def resize(self, symbol, evType, evictSize) :
        if symbol and symbol in self._dictPerspective.keys():
            return self._dictPerspective[symbol].resize(evType, evictSize)

    def descOf(self, symbol) :
        ''' 
        @return the desc of specified symbol
        '''
        strDesc=''
        if symbol and symbol in self._dictPerspective.keys():
            strDesc += self._dictPerspective[symbol].desc
            return strDesc
        
        for s, v in self._dictPerspective.items() :
            strDesc += '%s{%s};' %(s, v.desc)

        return strDesc
        
    def dumps(self, symbol) :
        if symbol and symbol in self._dictPerspective.keys():
            return self._dictPerspective[symbol].dumps()
        return b''

    def loads(self, symbol, pickleData) : # load the pikledata exported from dump()
        if symbol and symbol in self._dictPerspective.keys():
            return self._dictPerspective[symbol].loads(pickleData)

    def dailyOHLC_sofar(self, symbol) :
        ''' 
        @return (date, open, high, low, close) as of today
        '''
        if not symbol in self._dictPerspective.keys():
            return '', 0.0, 0.0, 0.0, 0.0, 0

        kl = self._dictPerspective[symbol].dailyOHLC_sofar
        return kl # .date, kl.open, kl.high, kl.low, kl.close
        
    def updateByEvent(self, ev) :
        ''' 
        @event could be Event(Tick), Event(KLine), Event(Perspective)
        '''
        if EVENT_Perspective == ev.type :
            self._dictPerspective[ev.data.symbol] = ev.data
            return None

        s = ev.data.symbol
        # if ev.type in Perspective.EVENT_SEQ_KLTICK :
        #     if not s in self._dictPerspective.keys() :
        #         self._dictPerspective[s] = Perspective(self.exchange, s)
        #     self._dictPerspective[s].push(ev)

        # if ev.type in MoneyflowPerspective.EVENT_SEQ_KLTICK :
        #     if not s in self.__dictMoneyflow.keys() :
        #         self.__dictMoneyflow[s] = MoneyflowPerspective(self.exchange, s)
        #     self.__dictMoneyflow[s].push(ev)
        if not s in self._dictPerspective.keys() :
            self._dictPerspective[s] = Perspective(self.exchange, s)
        
        return self._dictPerspective[s].push(ev)

    def export(self, symbol, lstsWished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :

        if symbol and symbol in self._dictPerspective.keys():
            return self._dictPerspective[symbol].export(lstsWished)

        raise ValueError('Perspective.export() unknown symbol[%s]' %symbol )


########################################################################
class PerspectiveFormatter(Formatter):
    '''
    '''
    def __init__(self, marketState =None):
        '''Constructor'''
        super(PerspectiveFormatter, self).__init__()

        if marketState:
            self.attach(marketState)

    def validate(self) :
        if not self.mstate or not isinstance(self.mstate, PerspectiveState) :
            return False

        return True

########################################################################
class Formatter_F1548(PerspectiveFormatter):
    F4SECHMA_1548 = OrderedDict({
        'asof':              1,
        EVENT_KLINE_1MIN :  30,
        EVENT_KLINE_5MIN :  96,
        EVENT_KLINE_1DAY : 260,
    })

    def __init__(self):
        '''Constructor'''
        super(Formatter_F1548, self).__init__()

    def doFormat(self, symbol=None) :

        if symbol and symbol in self.mstate._dictPerspective.keys():
            ret = self.mstate._dictPerspective[symbol].floatsD4(self.__class__.F4SECHMA_1548)
        else : 
            raise ValueError('Perspective.floatsD4() unknown symbol[%s]' %symbol )

        if not ret: return [0.0] * 1548
        if isinstance(ret, list) and 1548 ==len(ret):
            return ret

        raise ValueError('%s.format() unexpected ret' % self.__class__.__name__)

########################################################################
class Formatter_base2dImg(PerspectiveFormatter):
    '''
    '''
    BMP_COLOR_BG_FLOAT=1.0

    def __init__(self, imgPathPrefix=None, dem=60):
        '''Constructor'''
        super(Formatter_base2dImg, self).__init__()
        self._imgPathPrefix = imgPathPrefix
        self._dem = dem
        if self._dem <=0: self._dem=60

    def doFormat(self, symbol=None) :

        C6SECHMA_16xx = OrderedDict({
            EVENT_KLINE_1MIN     : -1,
            EVENT_KLINE_5MIN     : -1,
            EVENT_KLINE_1DAY     : -1,
        })

        seqdict = self.mstate.export(symbol, lstsWished=C6SECHMA_16xx)  # = self.export6C(symbol, lstsWished=C6SECHMA_16x16)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=0: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        if len(stk) <=0: return None

        img6C = [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(32)] # DONOT take [ [[0.0]*6] *16] *16

        # parition 0: data[0,:4] as the datetime asof the 1st KL1min, data[1:16,:4] fillin K1min up to an hour
        startRow =0
        dtAsOf = stk[0].asof
        img6C[startRow][0] = Formatter_base2dImg.datetimeTo6C(dtAsOf)
        img6C[startRow+1][0] = img6C[startRow][0]
        img6C[startRow+2][0] = img6C[startRow][0]
        img6C[startRow+3][0] = img6C[startRow][0]

        for x in range(1,16):
            for y in range(0,4):
                i = (x-1) *4 + y
                if i < len(stk) :
                    img6C[startRow + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)

        # parition 1: data[0:16,4:6] fillin 16x6 K5min up to a week
        startRow =5
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        for x in range(0, 16): 
            for y in range(0, 6):
                i = x *6 + y
                if i < len(stk) :
                    img6C[startRow + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        
        # parition 2: data[0:16,10:15] fillin 16x5 K1Day up to 16 week or 1/3yr
        startRow =12
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        for x in range(0, 16): 
            for y in range(0, 5):
                i = x *5 + y
                if i < len(stk) :
                    img6C[startRow + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)

        # parition 3x: TODO KL1w
        '''
        # optional at the moment:data[0:16,10:15] fillin 16x5 K1Day up to 16 week or 1/3yr
        startRow =18
        stk = seqdict[EVENT_KLINE_1DAY]
        bV = baseline_Volume
        for x in range(0, 16): 
            for y in range(0, 10):
                i = x *5 + y + 16*5
                if i < len(stk) :
                    img6C[startRow + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        '''

        return self.covertImg6CTo3C(img6C, dtAsOf) # return img6C

    def datetimeTo6C(dt) :
        return [
            (dt.month-1) / 12.0, # normalize to [0.0,1.0]
            dt.day / 31.0, # normalize to [0.0,1.0]
            dt.weekday() / 7.0, # normalize to [0.0,1.0]
            (dt.hour *60 +dt.minute) / (24 *60.0), # normalize to [0.0,1.0]
            (dt.year %100)/100.0, 
            1.0
            ]

    def covertImg6CTo3C(self, img6C, dtAsOf, expandX =False) :
        img3C = self.__6CTo3C_expandX(img6C) if expandX else self.__6CTo3C_expandY(img6C)

        if self._imgPathPrefix:
            ftime = img6C[0][0]
            yy, mon, day, minute = int(ftime[4] * 100), 1+ int(ftime[0] * 12), int(ftime[1] * 31), int(ftime[3] * 24*60)
            bmpstamp = dtAsOf.strftime('%Y%m%dT%H%M')
            width = 320
            lenX, lenY = len(img3C[0]), len(img3C)
            # if  bmpstamp != self.__bmpstamp  and 0 == minute % 60 :
            if self._dem >0 and 0 == minute % self._dem :
                imgarray = np.uint8(np.array(img3C)*255)
                bmp = Image.fromarray(imgarray)
                if width > lenX:
                    bmp = bmp.resize((width, int(width *1.0/lenX *lenY)), Image.NEAREST)
                # bmp.convert('RGB')
                bmp.save('%s%s_%s.png' % (self._imgPathPrefix, self.id, bmpstamp))
                self.__bmpstamp = bmpstamp

        return img3C

    def __6CTo3C_expandX(self, img6C) :
        lenX, lenY = len(img6C[0]), len(img6C)
        img3C = [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX*2)] for y in range(lenY) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(img6C)
        for y in range(lenY):
            for x in range(lenX) :
                img3C[y][2*x], img3C[y][2*x +1] = img6C[y][x][:3], img6C[y][x][3:]

        return img3C

    def __6CTo3C_expandY(self, img6C) :
        lenX, lenY = len(img6C[0]), len(img6C)
        img3C = [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX)] for y in range(lenY*2) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(img6C)
        for y in range(lenY):
            for x in range(lenX) :
                img3C[2*y][x], img3C[2*y+1][x] = img6C[y][x][:3], img6C[y][x][3:]

        return img3C

########################################################################
class Formatter_2dImg16x32(Formatter_base2dImg):

    def __init__(self, imgDir=None, dem=60):
        '''Constructor'''
        super(Formatter_2dImg16x32, self).__init__(imgDir, dem)

    def doFormat(self, symbol=None) :
        X_LEN, Y_LEN = 32, 16
        C6SECHMA_16x32R = OrderedDict({
            'asof'               : 1,
            EVENT_KLINE_1MIN     : 32,
            EVENT_KLINE_5MIN     : 240,
            EVENT_KLINE_1DAY     : 240,
        })

        seqdict = self.mstate.export(symbol, lstsWished=C6SECHMA_16x32R)  # = self.export6C(symbol, lstsWished=C6SECHMA_16x16)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=0: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        # TODO: draw the imagex
        img6C = [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(X_LEN)] for y in range(Y_LEN)] # DONOT take [ [[0.0]*6] *16] *16

        startRow =0
        # parition 0: pixel[0,0] as the datetime, X_LEN* 2rows -1 KL1min to cover half an hour
        rows =1
        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        if len(stk) <=0: return None

        dtAsOf = stk[0].asof
        img6C[startRow][0] = Formatter_base2dImg.datetimeTo6C(dtAsOf)
        todayYYMMDD = dtAsOf.strftime('%Y%m%d')

        # seq6C_offset =C6SECHMA_16x32R['asof']
        for i in range(1, min(1+ len(stk), X_LEN*rows)): 
             kl = stk[i-1]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD: continue # represent only today's
             x, y = int(i %X_LEN), int(i /X_LEN)
             img6C[startRow + y][x] = kl.floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        startRow +=rows
        
        # parition 1: X_LEN *15rows KL5min to cover a week
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48

        # parition 1.1.: X_LEN *3rows today's KL5min
        rows =2
        for i in range(0, min(len(stk), X_LEN*rows)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD:
                  # split today's out of the days before
                 if i>0: del stk[:i]
                 break
             x, y = int(i %X_LEN), int(i /X_LEN)
             img6C[startRow + y][x] = kl.floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        startRow +=rows

        # parition 1.2.: X_LEN *12rows to cover 4 days before today
        rows =6
        for i in range(0, min(len(stk), X_LEN*rows)): 
             kl = stk[i]
             x, y = int(i %X_LEN), int(i /X_LEN)
             img6C[startRow + y][x] = kl.floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        startRow +=rows

        # parition 3: X_LEN*15 KL1day to cover near a year
        rows =8
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        # seq6C_offset =C6SECHMA_16x32R['asof'] + C6SECHMA_16x32R[EVENT_KLINE_1MIN] + C6SECHMA_16x32R[EVENT_KLINE_5MIN]
        for i in range(0, min(len(stk), X_LEN*rows)): 
             x, y = int(i %X_LEN), int(i /X_LEN)
             img6C[startRow + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        startRow +=rows

        return self.covertImg6CTo3C(img6C, dtAsOf) # return img6C

########################################################################
class Formatter_2dImgSnail16(Formatter_base2dImg):

    def __init__(self, imgDir=None, dem=60):
        '''Constructor'''
        super(Formatter_2dImgSnail16, self).__init__(imgDir, dem)

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
    COORDS16x16=[(7, 7), (8, 7), (8, 8), (7, 8), (6, 8), (6, 7), (6, 6), (7, 6), (8, 6), (9, 6), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 9),
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
    
    def doFormat(self, symbol=None) :
        if not self.mstate or not isinstance(self.mstate, PerspectiveState) :
            raise ValueError('%s could not attach marketState of %s' %(self.__class__.__name__, str(self.mstate)))

        C6SECHMA_16x16 = OrderedDict({
            'asof'               : 1,
            EVENT_KLINE_1MIN     : 16,
            EVENT_KLINE_5MIN     : 240,
            EVENT_KLINE_1DAY     : 255,
            # EVENT_MONEYFLOW_1MIN : 16,
            # EVENT_MONEYFLOW_5MIN : 48*4, # 4days
            # EVENT_MONEYFLOW_1DAY : 48, # 48days
        })

        seqdict = self.mstate.export(symbol, lstsWished=C6SECHMA_16x16)  # = self.export6C(symbol, lstsWished=C6SECHMA_16x16)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=0: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        # TODO: draw the imagex
        # parition 0: the central 4x4 is KL1min up to 16min, the outter are 240KL5min up to a week
        dtAsOf = None
        partition =0
        img6C = [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(16)] # DONOT take [ [[0.0]*6] *16] *16
        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        if not dtAsOf and len(stk) >0 : dtAsOf = stk[0].asof

        datalen = min(len(stk), 16)
        for i in range(datalen): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[i]
             img6C[y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)
        
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        if not dtAsOf and len(stk) >0 : dtAsOf = stk[0].asof

        datalen = min(len(stk), len(Formatter_2dImgSnail16.COORDS16x16) -16)
        for i in range(datalen): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[16 +i]
             img6C[y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)

        # parition 1: the central 1x1 is current datetime, the outter are 255KL1d covers a year
        if not dtAsOf: return None
        partition +=1
        img6C += [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(16)] # append a 16x16 partition
        x, y = Formatter_2dImgSnail16.COORDS16x16[0]
        img6C[partition*16 + y][x] = Formatter_base2dImg.datetimeTo6C(dtAsOf)

        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        for i in range(min(len(stk), len(Formatter_2dImgSnail16.COORDS16x16) -1)): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[1+i]
             img6C[partition*16 + y][x] = stk[i].floatXC(baseline_Price=baseline_Price, baseline_Volume= bV, channels=6)

        '''
        # parition 2 MF: the central 4x4 MF1m, the outter: 48*4MF5m cover 4days, 48 MF1d cover 48days
        partition +=1
        img6C += [ [ [Formatter_base2dImg.BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(16)] # append a 16x16 partition
        x, y = Formatter_2dImgSnail16.COORDS16x16[0]
        img6C[partition*16 + y][x] = seq6C[0]

        seq6C_offset =C6SECHMA_16x16['asof'] + C6SECHMA_16x16[EVENT_KLINE_1MIN] + C6SECHMA_16x16[EVENT_KLINE_5MIN] + C6SECHMA_16x16[EVENT_KLINE_1DAY] # pointer to where EVENT_MONEYFLOW_1MIN is
        snail_loc = 0
        for i in range(C6SECHMA_16x16[EVENT_MONEYFLOW_1MIN]): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[snail_loc + i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1

        snail_loc += C6SECHMA_16x16[EVENT_MONEYFLOW_1MIN]
        for i in range(C6SECHMA_16x16[EVENT_MONEYFLOW_5MIN]): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[snail_loc + i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1

        snail_loc += C6SECHMA_16x16[EVENT_MONEYFLOW_5MIN]
        for i in range(C6SECHMA_16x16[EVENT_MONEYFLOW_1DAY]): 
             x, y = Formatter_2dImgSnail16.COORDS16x16[snail_loc + i]
             img6C[partition*16 + y][x] = seq6C[seq6C_offset]
             seq6C_offset +=1
        '''

        return self.covertImg6CTo3C(img6C, dtAsOf, expandX =True) # return img3C

    # @abstractmethod
    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     raise NotImplementedError


    # __dummy = None
    # def exportF1548(self, symbol=None) :
    #     '''@return an array_like data as float4C, maybe [] or numpy.array
    #     '''
    #     if symbol and symbol in self._dictPerspective.keys():
    #         return self._dictPerspective[symbol]._S1548I4

    #     if not PerspectiveState.__dummy:
    #         PerspectiveState.__dummy = Perspective(self.exchange, 'Dummy')

    #     return [0.0] * PerspectiveState.__dummy.fullFloatSize

    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     if symbol and symbol in self._dictPerspective.keys():
    #         return self._dictPerspective[symbol].engorged
        
    #     return [0.0]
