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

DEFAULT_KLDEPTH_TICK = 0 #  = 120
DEFAULT_KLDEPTH_1min = 60 # 1-hr
DEFAULT_KLDEPTH_5min = 240 # covers a week
DEFAULT_KLDEPTH_1day = 260 # about a year

FORMATTER_KL1d_MIN = 32 # 32-latest-days are minimallly required to generate 'format' for evaluating

EXPORT_SIGNATURE= '%dT%dM%dF%dD.%s:200109T17' % (DEFAULT_KLDEPTH_TICK, DEFAULT_KLDEPTH_1min, DEFAULT_KLDEPTH_5min, DEFAULT_KLDEPTH_1day, NORMALIZE_ID)

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
        EVENT_MONEYFLOW_1WEEK: EVENT_KLINE_1WEEK,
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

        # the floats, prioirty first, recommented to be multiple of 4
        ret = [
            # 1st-4C
            floatNormalize_LOG8(self.close, baseline_Price),
            floatNormalize_LOG8(self.volume, baseline_Volume),
            floatNormalize_LOG8(self.high, baseline_Price),
            floatNormalize_LOG8(self.low, baseline_Price),
            # 2nd-4C
            floatNormalize01(0.5 + self.ratioNet),                         # priority-H2
            floatNormalize01(0.5 + self.ratioR0),                          # priority-H3
            floatNormalize01(0.5 + self.ratioR3cate),                      # likely r3=ratioNet-ratioR0
            floatNormalize_LOG8(self.open, baseline_Price),
        ]
        #TODO: other optional dims

        channels = int(channels)
        return ret[:channels] if len(ret) >= channels else ret +[NORMALIZED_FLOAT_UNAVAIL]* (channels -len(ret))

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
    def __init__(self, exchange, symbol=None, KLDepth_1min=DEFAULT_KLDEPTH_1min, KLDepth_5min=DEFAULT_KLDEPTH_5min, KLDepth_1day=DEFAULT_KLDEPTH_1day, tickDepth=DEFAULT_KLDEPTH_TICK, **kwargs) :
        '''Constructor'''
        super(Perspective, self).__init__(exchange, symbol)

        self._stacks = {
            EVENT_TICK:       EvictableStack(tickDepth, TickData(self.exchange, self.symbol)),
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineEx(self.exchange, self.symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineEx(self.exchange, self.symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_1day, KLineEx(self.exchange, self.symbol)),

            # EVENT_MONEYFLOW_1MIN: EvictableStack(KLDepth_1min, MoneyflowData(self.exchange, self.symbol)),
            # EVENT_MONEYFLOW_5MIN: EvictableStack(KLDepth_5min, MoneyflowData(self.exchange, self.symbol)),
            # EVENT_MONEYFLOW_1DAY: EvictableStack(KLDepth_1day, MoneyflowData(self.exchange, self.symbol)),
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
        if not self.focus or len(self.focus) <0:
            str = 'None>NIL '
        else:
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

            v = stk.top.price if isinstance(stk.top, TickData) else stk.top.close
            if v and v >0.0:
                ret = v
                latestAsOf = stk.top.asof

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

        # align the datetime
        if EVENT_KLINE_PREFIX == etOfStack[: len(EVENT_KLINE_PREFIX)]:
            ev.data.datetime = ev.data.datetime.replace(second=0, microsecond=0)
            if EVENT_KLINE_1DAY == etOfStack:
                ev.data.datetime = ev.data.datetime.replace(hour=23, minute=59, second=59)

        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            if EVENT_TICK == etOfStack:
                newed = ev.data
            else:
                newed = KLineEx(ev.data.exchange, ev.data.symbol)
                newed.setEvent(ev)

            stk.push(newed)
            return ev, stk

        overwritable = not latestevd.exchange or ('_k2x' in latestevd.exchange or '_t2k' in latestevd.exchange)
        if isinstance(latestevd, KLineEx) :
            overwritable = ev.type not in latestevd.src
        elif ev.type == etOfStack:
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
            if ev.data.datetime < stk[i].datetime :
                continue

            if EVENT_TICK == etOfStack:
                if ev.data.datetime == stk[i].datetime :
                    stk[i] = ev.data
                else:
                    stk.insert(i, ev.data)

            if ev.data.datetime == stk[i].datetime :
                stk[i].setEvent(ev)
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

    """
    @property
    def _S1548I4(self):
        '''
        @return an array_like data, maybe [] or numpy.array
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return [0.0] * self.fullFloatSize
        
        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        return self.__exportS1548I4(baseline_Price=klbaseline.close, baseline_Volume=klbaseline.volume)
    
    def floatsD4(self, lstsWished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''@return an array_like data , maybe [] or numpy.array
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None

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

    """

    def export(self, lstsWished= { EVENT_KLINE_1DAY:20 } ) :
        result = {}

        for k, v in lstsWished.items():
            if not k in self.eventTypes :
                continue

            lst = self._stacks[k].exportList
            if len(lst) >0 and isinstance(lst[0], KLineEx): # ensure the KLineEx has been filled by its primary source
                nlst=[]
                for i in lst:
                    if k not in i.src: continue
                    nlst.append(i)
                lst = nlst
            result[k] = lst
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
                floatNormalize_LOG8(kl.close, basePrice, 1.5),
                floatNormalize_LOG8(kl.volume, bV, 1.5),
                floatNormalize01(minsPerDay*(kl.high / kl.close -1)),
                floatNormalize01(minsPerDay*(kl.close / kl.low -1)),
                0.0, 0.0
            ]

            while len(listMF)>0 and listMF[0].asof < kl.asof:
                del listMF[0]
                continue

            if len(listMF)>0 and listMF[0].asof == kl.asof:
                mf = listMF[0]
                if minsPerDay >1:
                    klf[4],klf[5] = floatNormalize01(0.5 + 10*mf.ratioNet), floatNormalize01(0.5 + 10*mf.ratioR0) # in-day KLs
                else:
                    klf[4],klf[5] = floatNormalize01(0.5 + 10*mf.ratioNet), minsPerDay*(kl.open / kl.high -1)
            
            result.append(klf)
        return result
        
    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     if symbol and symbol in self._dictPerspective.keys():
    #         return self._dictPerspective[symbol].engorged

    """
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
    """

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

    def adaptReader(self, reader, gained = EVENT_KLINE_1MIN):
        self._readers[gained] = reader

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
BMP_COLOR_BG_FLOAT  = 0.0

class PerspectiveFormatter(Formatter):
    '''
    '''
    def __init__(self, marketState =None, channels =6, valueUnavail = NORMALIZED_FLOAT_UNAVAIL):
        '''Constructor'''
        super(PerspectiveFormatter, self).__init__(channels =channels, valueUnavail=valueUnavail)

        if marketState:
            self.attach(marketState)

    def validate(self) :
        if not self.mstate or not isinstance(self.mstate, PerspectiveState) :
            return False

        return True

    def __md2floats_KLineEx(self, klineEx, baseline_Price, baseline_Volume) :
        '''
        @return float[] for neural network computing
        '''
        # the floats, prioirty first, recommented to be multiple of 4
        return [
            # 1st-4
            floatNormalize_LOG_PRICE(klineEx.close, baseline_Price),
            floatNormalize_LOG8(klineEx.volume, baseline_Volume),
            floatNormalize_LOG_PRICE(klineEx.high, baseline_Price),
            floatNormalize_LOG_PRICE(klineEx.low, baseline_Price),
            # 2nd-4
            floatNormalize01(0.5 + klineEx.ratioNet),                         # priority-H2
            floatNormalize01(0.5 + klineEx.ratioR0),                          # priority-H3
            floatNormalize01(0.5 + klineEx.ratioR3cate),                      # likely r3=ratioNet-ratioR0
            floatNormalize_LOG_PRICE(klineEx.open, baseline_Price),
        ]
        #TODO: other optional dims

    def marketDataTofloatXC(self, marketData, baseline_Price=1.0, baseline_Volume =1.0) :
        if not isinstance(marketData, KLineEx): 
            return super(self).marketDataTofloatXC(marketData, baseline_Price, baseline_Volume)

        ret = None
        if baseline_Price <=0: baseline_Price=1.0
        if baseline_Volume <=0: baseline_Volume=1.0
        ret = self.__md2floats_KLineEx(marketData, baseline_Price, baseline_Volume)
        return self._complementChannels(ret)

    def saveBMP(self, img3C, imgPathName) :
        width = 320
        lenX, lenY = len(img3C[0]), len(img3C)
        imgarray = np.uint8(np.array(img3C)*255)
        bmp = Image.fromarray(imgarray)
        if width > lenX:
            bmp = bmp.resize((width, int(width *1.0/lenX *lenY)), Image.NEAREST)
        # bmp.convert('RGB')
        bmp.save(imgPathName)
        return imgPathName

########################################################################
class Formatter_F1548(PerspectiveFormatter):
    def __init__(self):
        '''Constructor'''
        super(Formatter_F1548, self).__init__(channels =4)

    def doFormat(self, symbol=None) :
        F4SECHMA_1548 = OrderedDict({
            'asof':              1,
            EVENT_KLINE_1MIN :  30,
            EVENT_KLINE_5MIN :  96,
            EVENT_KLINE_1DAY : 260,
        })

        seqdict = self.mstate.export(symbol, lstsWished=F4SECHMA_1548)  # = self.export6C(symbol, lstsWished=EXP_SECHEMA)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=0: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        result = []
        # parition 0: first item is dtAsOf
        dtAsOf, todayYYMMDD = None, ""
        stk = seqdict[EVENT_KLINE_1MIN]
        if len(stk) >0: 
            dtAsOf = stk[0].asof
        else:
            stk = seqdict[EVENT_KLINE_5MIN]
            if len(stk) >0: 
                dtAsOf = stk[0].asof

        if not dtAsOf: return None
        todayYYMMDD = dtAsOf.strftime('%Y%m%d')

        result = [ # self._channels = 4
            dtAsOf.month,
            dtAsOf.day,
            dtAsOf.weekday(),
            dtAsOf.hour *60 +dtAsOf.minute
        ]

        # parition 1: EVENT_KLINE_1MIN
        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        for i in range(min(len(stk), F4SECHMA_1548[EVENT_KLINE_1MIN])): 
             kl = stk[i]
             result += self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)

        if len(stk) < F4SECHMA_1548[EVENT_KLINE_1MIN]:
            result += [NORMALIZED_FLOAT_UNAVAIL] * self._channels * (F4SECHMA_1548[EVENT_KLINE_1MIN] - len(stk))

        # parition 2: EVENT_KLINE_5MIN
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        for i in range(min(len(stk), F4SECHMA_1548[EVENT_KLINE_5MIN])): 
             kl = stk[i]
             result += self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)

        if len(stk) < F4SECHMA_1548[EVENT_KLINE_5MIN]:
            result += [NORMALIZED_FLOAT_UNAVAIL] * self._channels * (F4SECHMA_1548[EVENT_KLINE_5MIN] - len(stk))

        # parition 3: EVENT_KLINE_1DAY
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        for i in range(min(len(stk), F4SECHMA_1548[EVENT_KLINE_1DAY])): 
             kl = stk[i]
             result += self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)

        if len(stk) < F4SECHMA_1548[EVENT_KLINE_1DAY]:
            result += [NORMALIZED_FLOAT_UNAVAIL] * self._channels * (F4SECHMA_1548[EVENT_KLINE_1DAY] - len(stk))

        return result

        '''
        # -----------------------------------------------
        if symbol and symbol in self.mstate._dictPerspective.keys():
            ret = self.mstate._dictPerspective[symbol].floatsD4(self.__class__.F4SECHMA_1548)
        else : 
            raise ValueError('Perspective.floatsD4() unknown symbol[%s]' %symbol )

        if not ret: return [0.0] * 1548
        if isinstance(ret, list) and 1548 ==len(ret):
            return ret

        raise ValueError('%s.format() unexpected ret' % self.__class__.__name__)
        '''

########################################################################
class Formatter_1d518(PerspectiveFormatter):
    X_LEN = 518
    FLOAT_NaN = 0.0

    EXP_SECHEMA = OrderedDict({
        'asof':              1,
        EVENT_KLINE_1MIN :  60,
        EVENT_KLINE_5MIN :  240,
        EVENT_KLINE_1DAY :  240,
    })

    XOFFSETS = {
        EVENT_KLINE_1MIN :  0,
        'aof0'           :  60,
        EVENT_KLINE_5MIN :  61,
        'aof1'           :  301, # = 61 + 240,
        EVENT_KLINE_1DAY :  302,
    }

    def __init__(self, bmpPathPrefix=None, dem=60, channels=8):
        '''Constructor'''
        super(Formatter_1d518, self).__init__(channels=channels)
        self._bmpPathPrefix = bmpPathPrefix
        self._dem = dem
        if self._dem <=0: self._dem=60

    def doFormat(self, symbol=None) :
        seqdict = self.mstate.export(symbol, lstsWished=Formatter_1d518.EXP_SECHEMA)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <FORMATTER_KL1d_MIN: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        # determine dtAsOf
        dtAsOf =None
        stk = seqdict[EVENT_KLINE_1MIN]
        if len(stk) >0: 
            dtAsOf = stk[0].asof
        else:
            stk = seqdict[EVENT_KLINE_5MIN]
            if len(stk) >0: 
                dtAsOf = stk[0].asof

        if not dtAsOf: return None
        todayYYMMDD = dtAsOf.strftime('%Y%m%d')

        result = [ [Formatter_1d518.FLOAT_NaN for k in range(self._channels)] for x in range(Formatter_1d518.X_LEN) ] # DONOT take [ [[0.0]*6] *16] *16

        # part 0: EVENT_KLINE_1MIN
        stk, bV, roffset = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240, Formatter_1d518.XOFFSETS[EVENT_KLINE_1MIN]
        for i in range(0, min(len(stk), Formatter_1d518.EXP_SECHEMA[EVENT_KLINE_1MIN])): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD: continue # represent only today's
             result[roffset +i] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # part 1: aof0
        roffset = Formatter_1d518.XOFFSETS['aof0']
        result[roffset] = [ dtAsOf.minute/60.0, dtAsOf.hour/24.0, dtAsOf.weekday() / 7.0, dtAsOf.day / 31.0 ] * int ((self._channels +3)/4)

        # part 2: EVENT_KLINE_5MIN
        stk, bV, roffset = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48, Formatter_1d518.XOFFSETS[EVENT_KLINE_5MIN]
        # 2.1 today's 48 KL5min
        for i in range(0, min(len(stk), 48)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD: 
                 if i>0: del stk[:i]
                 break

             result[roffset +i] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # 2.2 KL5min of prev 4days
        for i in range(0, min(len(stk), 48*4)): 
             kl = stk[i]
             result[roffset +48 +i] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # part 3: aof1
        roffset = Formatter_1d518.XOFFSETS['aof1']
        result[roffset] = [ dtAsOf.day / 31.0, (dtAsOf.month-1) / 12.0, dtAsOf.weekday() / 7.0, dtAsOf.hour/24.0 ] * int ((self._channels +3)/4)

        # part 4: EVENT_KLINE_1DAY
        stk, bV, roffset = seqdict[EVENT_KLINE_1DAY], baseline_Volume, Formatter_1d518.XOFFSETS[EVENT_KLINE_1DAY]
        for i in range(0, min(len(stk), Formatter_1d518.EXP_SECHEMA[EVENT_KLINE_1DAY])): 
             kl = stk[i]
             result[roffset +i] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # export BMP for auditing
        if self._bmpPathPrefix and dtAsOf and self._dem >0 and 0 == dtAsOf.minute % self._dem:
            img3C = self.expandRGBLayers(result)
            imgPathName = '%s%s_%s.png' % (self._bmpPathPrefix if self._bmpPathPrefix else '', self.id, dtAsOf.strftime('%Y%m%dT%H%M'))
            self.saveBMP(img3C, imgPathName)

        return result

    def readDateTime(self, imgResult) :
        dt0, dt1 = imgResult[Formatter_1d518.XOFFSETS['aof0']], imgResult[Formatter_1d518.XOFFSETS['aof1']]
        month, day, hour, minute = int(dt1[1]*12 +1.2), int(dt1[0]*31 +0.2), int(dt0[1]*24 +0.2), int(dt0[0]*60 +0.2)
        dt = datetime(year=2030, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        return dt

    def expandRGBLayers(self, img1d) :
        channels, len1d = len(img1d[0]), len(img1d)
        scaleX = int((channels+2)/3)
        X = int(math.sqrt(len1d))
        Y = X
        if X <1: X=1
        while (X * Y) < len1d: Y+=1

        imgRGB = [ [ [ Formatter_1d518.FLOAT_NaN for k in range(3) ] for x in range(X* scaleX + scaleX-1) ] for y in range(Y) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(imgResult)
        pixelEdge = [ 1.0 -Formatter_1d518.FLOAT_NaN ] *3
        for y in range(Y):
            for x in range(1, scaleX) :
                imgRGB[y][x *(X+1) -1] = pixelEdge

            for x in range(X) :
                offset = y * X + x
                if offset >= len1d: continue
                for i in range(scaleX):
                    pixel = img1d[offset][i*3 : (i+1)*3]
                    if len(pixel) < 3: pixel += [ Formatter_1d518.FLOAT_NaN ] * (3-len(pixel))
                    imgRGB[y][i *X +x + i] = pixel
                    
        return imgRGB

########################################################################
class Formatter_base2dImg(PerspectiveFormatter):
    '''
    '''
    FLOAT_NaN = 0.0

    def __init__(self, bmpPathPrefix=None, dem=60, channels=6):
        '''Constructor'''
        super(Formatter_base2dImg, self).__init__(channels=channels)
        self._bmpPathPrefix = bmpPathPrefix
        self._dem = dem
        if self._dem <=0: self._dem=60

    def doFormat(self, symbol=None) :

        EXP_SECHEMA = OrderedDict({
            EVENT_KLINE_1MIN     : -1,
            EVENT_KLINE_5MIN     : -1,
            EVENT_KLINE_1DAY     : -1,
        })

        seqdict = self.mstate.export(symbol, lstsWished=EXP_SECHEMA)  # = self.export6C(symbol, lstsWished=EXP_SECHEMA)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=0: return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        if len(stk) <=0: return None

        imgResult = [ [ [BMP_COLOR_BG_FLOAT for k in range(6)] for x in range(16)] for y in range(32)] # DONOT take [ [[0.0]*6] *16] *16

        # parition 0: data[0,:4] as the datetime asof the 1st KL1min, data[1:16,:4] fillin K1min up to an hour
        startRow =0
        dtAsOf = stk[0].asof
        imgResult[startRow][0] = Formatter_base2dImg.datetimeTo6C(dtAsOf)
        imgResult[startRow+1][0] = imgResult[startRow][0]
        imgResult[startRow+2][0] = imgResult[startRow][0]
        imgResult[startRow+3][0] = imgResult[startRow][0]

        for x in range(1,16):
            for y in range(0,4):
                i = (x-1) *4 + y
                if i < len(stk) :
                    imgResult[startRow + y][x] = self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)

        # parition 1: data[0:16,4:6] fillin 16x6 K5min up to a week
        startRow =5
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        for x in range(0, 16): 
            for y in range(0, 6):
                i = x *6 + y
                if i < len(stk) :
                    imgResult[startRow + y][x] = self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)
        
        # parition 2: data[0:16,10:15] fillin 16x5 K1Day up to 16 week or 1/3yr
        startRow =12
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        for x in range(0, 16): 
            for y in range(0, 5):
                i = x *5 + y
                if i < len(stk) :
                    imgResult[startRow + y][x] = self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)

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
                    imgResult[startRow + y][x] = self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)
        '''

        img3C = self.expand6Cto3C_Y(imgResult)
        if self._bmpPathPrefix and dtAsOf and self._dem >0 and 0 == dtAsOf.minute % self._dem:
            self.saveBMP(img3C, dtAsOf=dtAsOf)

        return img3C

    def datetimeTo6C(dt) :
        return [
            (dt.month-1) / 12.0, # normalize to [0.0,1.0]
            dt.day / 31.0, # normalize to [0.0,1.0]
            dt.weekday() / 7.0, # normalize to [0.0,1.0]
            (dt.hour *60 +dt.minute) / (24 *60.0), # normalize to [0.0,1.0]
            (dt.year %100)/100.0, 
            NORMALIZED_FLOAT_UNAVAIL
            ]

    def expand6Cto3C_X(self, imgResult) :
        lenX, lenY = len(imgResult[0]), len(imgResult)
        img3C = [ [ [BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX*2)] for y in range(lenY) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(imgResult)
        for y in range(lenY):
            for x in range(lenX) :
                img3C[y][2*x], img3C[y][2*x +1] = imgResult[y][x][:3], imgResult[y][x][3:]

        return img3C

    def expand6Cto3C_Y(self, imgResult) :
        lenX, lenY = len(imgResult[0]), len(imgResult)
        img3C = [ [ [BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX)] for y in range(lenY*2) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(imgResult)
        for y in range(lenY):
            for x in range(lenX) :
                img3C[2*y][x], img3C[2*y+1][x] = imgResult[y][x][:3], imgResult[y][x][3:]

        return img3C

    def expandRGBLayers(self, img) :
        channels, lenX, lenY = len(img[0][0]), len(img[0]), len(img)
        scaleY = int((channels+2)/3)
        scaleX = int(math.sqrt(scaleY))
        if scaleX >1:
            s = int(scaleY/scaleX)
            while (s * scaleX) < scaleY: s+=1
            scaleY =s
        else: scaleX =1

        imgRGB = [ [ [BMP_COLOR_BG_FLOAT for k in range(3)] for x in range(lenX* scaleX + scaleX-1)] for y in range(lenY* scaleY  + scaleY-1) ] # DONOT take [ [[0.0]*6] *lenR*2] *len(imgResult)
        pixelEdge = [ 1.0 -BMP_COLOR_BG_FLOAT ] *3
        
        for y in range(1, scaleY) : # the edge lines
            imgRGB[y*(lenY+1) -1] = [ pixelEdge ] * (lenX* scaleX + scaleX-1)

        for y in range(lenY):
            for x in range(1, scaleX) : # the edge lines
                imgRGB[y][x *(lenX+1) -1] = pixelEdge

            for x in range(lenX) :
                v = img[y][x]
                for i in range(scaleY):
                    for j in range(scaleX):
                        coffset = 3* (i*scaleX +j)
                        pixel = v[coffset : coffset+3]
                        if len(pixel) <3: pixel += [ BMP_COLOR_BG_FLOAT ] * (3-len(pixel))
                        posRGB = [i *(lenY+1) +y, j*(lenX+1) +x]
                        imgRGB[posRGB[0]][posRGB[1]] = pixel

        return imgRGB

########################################################################
class Formatter_2dImg32x18(Formatter_base2dImg):
    
    @staticmethod
    def shape(): return (32, 18)

    def __init__(self, imgDir=None, dem=60, channels=8):
        '''Constructor'''
        super(Formatter_2dImg32x18, self).__init__(imgDir, dem, channels=channels)

    def doFormat(self, symbol=None) :
        X_LEN, Y_LEN = Formatter_2dImg32x18.shape()
        EXP_SECHEMA = OrderedDict({
            'asof'               : 1,
            EVENT_KLINE_1MIN     : 60,
            EVENT_KLINE_5MIN     : 240,
            EVENT_KLINE_1DAY     : 240,
        })

        seqdict = self.mstate.export(symbol, lstsWished=EXP_SECHEMA)  # = self.export6C(symbol, lstsWished=EXP_SECHEMA)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <FORMATTER_KL1d_MIN: return None # 32-latest-days are minimallly required for evaluating
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        # determine dtAsOf
        dtAsOf =None
        stk = seqdict[EVENT_KLINE_1MIN]
        if len(stk) >0: 
            dtAsOf = stk[0].asof
        else:
            stk = seqdict[EVENT_KLINE_5MIN]
            if len(stk) >0: 
                dtAsOf = stk[0].asof

        if not dtAsOf: return None
        todayYYMMDD = dtAsOf.strftime('%Y%m%d')

        imgResult = []

        # parition 0: pixel[,0] and [,31] as the datetime, X_LEN-2 KL1min x2rows to cover an hour
        rows =2
        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        rows6C = [ [ [BMP_COLOR_BG_FLOAT for k in range(self._channels)] for x in range(X_LEN)] for y in range(rows)] # DONOT take [ [[0.0]*6] *16] *16
        dtCell = [ dtAsOf.minute/60.0, dtAsOf.hour/24.0, dtAsOf.weekday() / 7.0, dtAsOf.day / 31.0 ] * int ((self._channels +3)/4)
        if len(dtCell) > self._channels: del dtCell[self._channels:]
        for y in range(0, rows):
            for x in [0]:
                rows6C[y][x] = copy.copy(dtCell)
                rows6C[y][X_LEN -1 -x] = rows6C[y][x]

        klPerRow =30
        for i in range(0, min(len(stk), klPerRow *rows)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD: continue # represent only today's
             x, y = int(i %klPerRow), int(i /klPerRow)
             rows6C[y][1+ x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        imgResult += rows6C # imgResult += self.expand6Cto3C_Y(rows6C)
        
        # parition 1: X_LEN *15rows KL5min to cover a week
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        # break line takes current date-time
        # br1 = [ dtAsOf.hour/24.0, dtAsOf.minute/60.0, dtAsOf.weekday() / 7.0] if dtAsOf else [BMP_COLOR_BG_FLOAT] *(self._channels -3)
        # img3C.append([br1] * X_LEN)

        # parition 1.1.: X_LEN *2rows today's 48 KL5min in the center
        rows =2
        klPerRow =24
        rows6C = [ [ [BMP_COLOR_BG_FLOAT for k in range(self._channels)] for x in range(X_LEN)] for y in range(rows)] # DONOT take [ [[0.0]*6] *16] *16
        dtCell = [ dtAsOf.weekday() / 7.0, dtAsOf.hour/24.0, dtAsOf.minute/60.0,  dtAsOf.day / 31.0 ] * int ((self._channels +3)/4)
        if len(dtCell) > self._channels: del dtCell[self._channels:]

        for y in range(0, rows):
            for x in range(int((X_LEN -klPerRow)/2)):
                rows6C[y][x] =copy.copy(dtCell)
                rows6C[y][X_LEN -1 -x] = rows6C[y][x]

        for i in range(0, min(len(stk), klPerRow*rows)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD:
                  # split today's out of the days before
                 if i>0: del stk[:i]
                 break
             x, y = int(i %klPerRow) +int((X_LEN -klPerRow)/2), int(i /klPerRow)
             rows6C[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)
        imgResult += rows6C # img3C += self.expand6Cto3C_Y(rows6C)

        # parition 1.2.: X_LEN *6rows to cover addtional 4 days before today
        rows =6
        klPerRow =32
        rows6C = [ [ [BMP_COLOR_BG_FLOAT for k in range(self._channels)] for x in range(X_LEN)] for y in range(rows)] # DONOT take [ [[0.0]*6] *16] *16
        for i in range(0, min(len(stk), klPerRow*rows)): 
             kl = stk[i]
             x, y = int(i %klPerRow), int(i /klPerRow)
             rows6C[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)
        imgResult += rows6C # img3C += self.expand6Cto3C_Y(rows6C)

        # parition 3: X_LEN*5 KL1day to cover near a year
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume

        # break line takes current date-time
        dtCell = [ dtAsOf.day / 31.0, (dtAsOf.month-1) / 12.0, dtAsOf.weekday() / 7.0, dtAsOf.hour/24.0 ] * int ((self._channels +3)/4)
        if len(dtCell) > self._channels: del dtCell[self._channels:]
        imgResult.append([copy.copy(dtCell)] * X_LEN) # img3C.append([br2] * X_LEN)

        rows =7 # to =5 when KL1week involves, leave 2rows to KL1week
        klPerRow =32
        rows6C = [ [ [BMP_COLOR_BG_FLOAT for k in range(self._channels)] for x in range(X_LEN)] for y in range(rows)] # DONOT take [ [[0.0]*6] *16] *16
        for i in range(0, min(len(stk), klPerRow*rows)): 
             x, y = int(i %klPerRow), int(i /klPerRow)
             rows6C[y][x] = self.marketDataTofloatXC(stk[i], baseline_Price=baseline_Price, baseline_Volume= bV)
        imgResult += rows6C # img3C += self.expand6Cto3C_Y(rows6C)

        # TODO parition 4: X_LEN*2 KL1week to cover near a year

        # if exceed the expected Y_LEN 
        rows = len(imgResult)
        if rows > Y_LEN:
            del imgResult[Y_LEN:]
        else : imgResult += [ [ [BMP_COLOR_BG_FLOAT for k in range(self._channels)] for x in range(X_LEN)] for y in range(Y_LEN - rows)] # DONOT take [ [[0.0]*6] *16] *16

        if self._bmpPathPrefix and dtAsOf and self._dem >0 and 0 == dtAsOf.minute % self._dem:
            img3C = self.expandRGBLayers(imgResult)
            imgPathName = '%s%s_%s.png' % (self._bmpPathPrefix if self._bmpPathPrefix else '', self.id, dtAsOf.strftime('%Y%m%dT%H%M'))
            self.saveBMP(img3C, imgPathName)

        return imgResult

    def readDateTime(self, imgResult) :
        dt0, dtBr = imgResult[0][0], imgResult[10][0]
        month, day, hour, minute = int(dtBr[1]*12 +1.2), int(dtBr[0]*31 +0.2), int(dt0[1]*24 +0.2), int(dt0[0]*60 +0.2)
        dt = datetime(year=2030, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        return dt

########################################################################
class Formatter_Snail32x32(Formatter_base2dImg):

    def __init__(self, imgDir=None, dem=60, channels=8):
        '''Constructor'''
        super(Formatter_Snail32x32, self).__init__(imgDir, dem, channels=channels)
    
    def CORDS_OF_SNAIL(edgelen) :
        import math
        edgelen = 4
        for i in range(edgelen * edgelen):
            if i<=0:
                ret=[(int(edgelen/2-1),int(edgelen/2-1))]
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

            ret.append((int(x+ edgelen/2-1), int(y+ edgelen/2-1)))
            # print('%s=%d ^2 +%d@(%d, %d)' % (i, a, b, x, y))

        print('COORDS_Snail%dx%d=%s' % (edgelen, edgelen, ret))
        
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

    COORDS_Snail8x8=[(3, 3), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (2, 5),
                (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (4, 6),
                (3, 6), (2, 6), (1, 6), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (6, 7), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7), (0, 7)]
    
    COORDS_Snail4x4=[(1, 1), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3)]
    
    def doFormat(self, symbol=None) :
        if not self.mstate or not isinstance(self.mstate, PerspectiveState) :
            raise ValueError('%s could not attach marketState of %s' %(self.__class__.__name__, str(self.mstate)))

        X_LEN, Y_LEN = 32, 32
        EXP_SECHEMA = OrderedDict({
            'asof'               : 1,
            EVENT_KLINE_1MIN     : 60,
            EVENT_KLINE_5MIN     : 240,
            EVENT_KLINE_1DAY     : 240,
        })

        seqdict = self.mstate.export(symbol, lstsWished=EXP_SECHEMA)  # = self.export6C(symbol, lstsWished=EXP_SECHEMA)
        if not seqdict or len(seqdict) <=0 or not EVENT_KLINE_1MIN in seqdict.keys() or not EVENT_KLINE_1DAY in seqdict.keys():
            return None

        stk = seqdict[EVENT_KLINE_1DAY]
        if len(stk) <=FORMATTER_KL1d_MIN : return None
        baseline_Price, baseline_Volume= stk[0].close, stk[0].volume

        # draw the imagex

        # make the dtAsOf as the background
        dtAsOf = None
        for evt in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]:
            stk = seqdict[evt]
            if len(stk) >0 : 
                dtAsOf = stk[0].asof
                if dtAsOf : break

        if not dtAsOf: return None
        todayYYMMDD = dtAsOf.strftime('%Y%m%d')
        cellDt = [ (dtAsOf.minute/80.0 + dtAsOf.hour)/25.0, dtAsOf.weekday() / 8.0, int(dtAsOf.strftime('%j'))/ 400.0, (dtAsOf.year %100) /100.0 ] * int ((self._channels +3)/4)
        imgResult = [ [ copy.copy(cellDt) for x in range(X_LEN)] for y in range(Y_LEN)] # DONOT take [ [[0.0]*6] *16] *16

        # parition 1: the KLex5min at left-top 16x16
        stk, bV = seqdict[EVENT_KLINE_5MIN], baseline_Volume /48
        lefttop = (0, 0)
        #1.1 today's KL5m
        for i in range(0, min(len(stk), 16*16)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD:
                  # split today's out of the days before
                 if i>0: del stk[:i]
                 break
             x, y = Formatter_Snail32x32.COORDS16x16[i]
             x, y = x +lefttop[0], y + lefttop[1]
             imgResult[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        while i < 48:
             x, y = Formatter_Snail32x32.COORDS16x16[i]
             x, y = x +lefttop[0], y + lefttop[1]
             imgResult[y][x] = [0.0] * len(cellDt)
             i += 1

        #1.2 four-day's KL5m to cover a week
        for i in range(0, min(len(stk), 16*16 -48)): 
             kl = stk[i]
             x, y = Formatter_Snail32x32.COORDS16x16[48 + i]
             x, y = x +lefttop[0], y + lefttop[1]
             imgResult[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # parition 2: the KLex1day at right-top 16x16
        stk, bV = seqdict[EVENT_KLINE_1DAY], baseline_Volume
        lefttop = (16, 0)
        for i in range(0, min(len(stk), 16*16)): 
             kl = stk[i]
             x, y = Formatter_Snail32x32.COORDS16x16[i]
             x, y = x +lefttop[0], y + lefttop[1]
             imgResult[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        # parition 3: the KLex1min at left-bottom 8x8
        stk, bV = seqdict[EVENT_KLINE_1MIN], baseline_Volume /240
        lefttop = (0, 16)
        for i in range(0, min(len(stk), 8*8)): 
             kl = stk[i]
             if kl.asof.strftime('%Y%m%d') != todayYYMMDD: continue # represent only today's
             x, y = Formatter_Snail32x32.COORDS8x8[i]
             x, y = x +lefttop[0], y + lefttop[1]
             imgResult[y][x] = self.marketDataTofloatXC(kl, baseline_Price=baseline_Price, baseline_Volume= bV)

        if self._bmpPathPrefix and dtAsOf and self._dem >0 and 0 == dtAsOf.minute % self._dem:
            img3C = self.expandRGBLayers(imgResult)
            imgPathName = '%s%s_%s.png' % (self._bmpPathPrefix if self._bmpPathPrefix else '', self.id, dtAsOf.strftime('%Y%m%dT%H%M'))
            self.saveBMP(img3C, imgPathName)

        return imgResult

    def readDateTime(self, imgResult) :
        # dt0, dt1 = imgResult[Formatter_1d518.XOFFSETS['aof0']], imgResult[Formatter_1d518.XOFFSETS['aof1']]
        # month, day, hour, minute = int(dt1[1]*12 +1.2), int(dt1[0]*31 +0.2), int(dt0[1]*24 +0.2), int(dt0[0]*60 +0.2)
        dt = datetime(year=2030, month=1, day=1, hour=1, minute=1, second=0, microsecond=0) # TODO
        return dt
