# encoding: UTF-8

from __future__ import division

from Application import BaseApplication, Iterable
from EventData import *
from MarketData import *

from datetime import datetime
from abc import ABCMeta, abstractmethod
import traceback

import bz2
import csv
import copy
# import tensorflow as tf

EVENT_Perspective  = MARKETDATE_EVENT_PREFIX + 'Persp'   # 错误回报事件

DEFAULT_KLDEPTH_TICK = 0
# DEFAULT_KLDEPTH_TICK = 120
DEFAULT_KLDEPTH_1min = 30
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
        if nilFilled :
            fillsize = (self.evictSize - self.size) if self.evictSize >=0 else 0
            return self.__data + [self.__dataNIL] *fillsize
        return self.__data

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
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_1day, KLineData(self.exchange, self.symbol)),

            EVENT_MONEYFLOW_1MIN: EvictableStack(MFDepth_1min, MoneyflowData(self.exchange, self.symbol)),
            EVENT_MONEYFLOW_5MIN: EvictableStack(MFDepth_5min, MoneyflowData(self.exchange, self.symbol)),
            EVENT_MONEYFLOW_1DAY: EvictableStack(MFDepth_1day, MoneyflowData(self.exchange, self.symbol)),
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
        str = '%s>%s ' % (self.focus[len(MARKETDATE_EVENT_PREFIX):], self.getAsOf(self.focus).strftime('%Y-%m-%dT%H:%M:%S'))
        for i in self.eventTypes :
            str += '%sX%d/%d,' % (i[len(MARKETDATE_EVENT_PREFIX):], self._stacks[i].size, self._stacks[i].evictSize)
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
            ret = stk.top.price if EVENT_TICK == et else stk.top.close

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

    def push(self, ev) :
        ev = self.__push(ev)
        if not ev :
            return None

        # special dealing about TICK and KLine
        if ev.type in Perspective.EVENT_SEQ_KLTICK:
            if self.__dayOHLC and self.__dayOHLC.asof < self.asof.replace(hour=0,minute=0,second=0,microsecond=0) :
                self.__dayOHLC = None

            evd = ev.data
            if not self.__dayOHLC :
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
        if not ev or not ev.type in self._stacks.keys():
            return None

        if not self.__stampLast or self.__stampLast < ev.data.datetime :
            self.__stampLast = ev.data.datetime

        latestevd = self._stacks[ev.type].top
        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            self._stacks[ev.type].push(ev.data)
            if self._stacks[ev.type].size >0:
                self.__focusLast = ev.type
            return ev

        overwritable = not latestevd.exchange or ('_k2x' in latestevd.exchange or '_t2k' in latestevd.exchange)
        if latestevd.exchange and (not ev.data.exchange or len(ev.data.exchange) <=0) :
            overwritable = False

        if EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] and ev.data.datetime == latestevd.datetime and ev.data.volume > latestevd.volume :
            # SINA KL-data was found has such a bug as below: the later polls got bigger volume, so treat the later larger volume as correct data
            # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.915,2.914,2.915,570200.0
            # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.918,2.914,2.918,4575100.0
            # evmdKL5m,SH510050,AShare,2020-06-19,13:30:00,2.914,2.918,2.914,2.918,4575100.0
            overwritable = True

        if not overwritable:
            return None # not overwritable

        for i in range(self._stacks[ev.type].size) :
            if ev.data.datetime > self._stacks[ev.type][i].datetime :
                continue

            if ev.data.datetime == self._stacks[ev.type][i].datetime :
                self._stacks[ev.type][i] = ev.data
            else :
                self._stacks[ev.type].insert(i, ev.data)
            return ev
        
        self._stacks[ev.type].insert(-1, ev.data)
        while self._stacks[ev.type].evictSize >=0 and self._stacks[ev.type].size > self._stacks[ev.type].evictSize:
            del(self._stacks[ev.type]._data[-1])

        if self._stacks[ev.type].size >0:
            self.__focusLast = ev.type

        return ev

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
    
    def floatsD4(self, d4wished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''@return an array_like data as float4C, maybe [] or numpy.array
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None # float4C not available

        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        baseline_Price, baseline_Volume =klbaseline.close, klbaseline.volume

        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        result = []
        for k, v in d4wished.items():
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

    def export(self, d4wished= { EVENT_KLINE_1DAY:20 } ) :
        result = {}

        for k, v in d4wished.items():
            if not k in self.eventTypes :
                continue

            result[k] = self._stacks[k].exportList
            if v>0 and v > len(result[k]):
                del result[k][v:]

        return result

    def float6C(self, d4wished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''@return a 2D array of floats
        '''
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return None # float6C not available

        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        baseline_Price, baseline_Volume =klbaseline.close, klbaseline.volume

        if baseline_Price <0.01: baseline_Price=1.0
        if baseline_Volume <0.001: baseline_Volume=1.0

        result = []
        for k, v in d4wished.items():
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

        result[EVENT_KLINE_1MIN] = Perspective.__richKL6(self._stacks[EVENT_KLINE_1MIN].exportList, self._stacks[EVENT_MONEYFLOW_1MIN].exportList, baseline_Price, baseline_Volume, self.__evsPerDay[EVENT_KLINE_1MIN])
        result[EVENT_KLINE_5MIN] = Perspective.__richKL6(self._stacks[EVENT_KLINE_5MIN].exportList, self._stacks[EVENT_MONEYFLOW_5MIN].exportList, baseline_Price, baseline_Volume, self.__evsPerDay[EVENT_KLINE_5MIN])
        result[EVENT_KLINE_1DAY] = Perspective.__richKL6(self._stacks[EVENT_KLINE_1DAY].exportList, self._stacks[EVENT_MONEYFLOW_1DAY].exportList, baseline_Price, baseline_Volume, 1)

        return result

    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     if symbol and symbol in self.__dictPerspective.keys():
    #         return self.__dictPerspective[symbol].engorged

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
        self.__dictPerspective ={} # dict of symbol to Perspective
        # self.__dictMoneyflow ={} # dict of symbol to MoneyflowPerspective

    # -- impl of MarketState --------------------------------------------------------------
    def listOberserves(self) :
        return [ s for s in self.__dictPerspective.keys()]

    def addMonitor(self, symbol) :
        ''' add a symbol to monitor
        '''
        if symbol and not symbol in self.__dictPerspective.keys() :
            self.__dictPerspective[symbol] = Perspective(self.exchange, symbol)

    def latestPrice(self, symbol) :
        ''' query for latest price of the given symbol
        @return the price, datetimeAsOf
        '''
        if not symbol in self.__dictPerspective.keys() :
            return 0.0, DT_EPOCH
        
        return self.__dictPerspective[symbol].latestPrice

    def getAsOf(self, symbol=None, evType =None) :
        ''' 
        @return the datetime as of latest observing
        '''
        if len(self.__dictPerspective) <=0: return DT_EPOCH
        if symbol and symbol in self.__dictPerspective.keys():
            psp = self.__dictPerspective[symbol]
            if psp:
                if evType and evType in psp._stacks.keys():
                    return psp.getAsOf(evType)
                return psp.asof

        return max([pofs.asof for pofs in self.__dictPerspective.values()])

    def stampUpdatedOf(self, symbol=None, evType=None) :
        if len(self.__dictPerspective) <=0: return DT_EPOCH
        if symbol and symbol in dict:
            return self.__dictPerspective[symbol].stampUpdatedOf(evType)

        return max([pofs.asof for pofs in self.__dictPerspective.values()])

    def sizesOf(self, symbol, evType =None) :
        ''' 
        @return the size of specified symbol/evType
        '''
        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].sizesOf(evType)
        return 0, 0

    def resize(self, symbol, evType, evictSize) :
        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].resize(evType, evictSize)

    def descOf(self, symbol) :
        ''' 
        @return the desc of specified symbol
        '''
        strDesc=''
        if symbol and symbol in self.__dictPerspective.keys():
            strDesc += self.__dictPerspective[symbol].desc
            return strDesc
        
        for s, v in self.__dictPerspective.items() :
            strDesc += '%s{%s};' %(s, v.desc)

        return strDesc
        
    def dailyOHLC_sofar(self, symbol) :
        ''' 
        @return (date, open, high, low, close) as of today
        '''
        if not symbol in self.__dictPerspective.keys():
            return '', 0.0, 0.0, 0.0, 0.0, 0

        kl = self.__dictPerspective[symbol].dailyOHLC_sofar
        return kl.date, kl.open, kl.high, kl.low, kl.close
        
    def updateByEvent(self, ev) :
        ''' 
        @event could be Event(Tick), Event(KLine), Event(Perspective)
        '''
        if EVENT_Perspective == ev.type :
            self.__dictPerspective[ev.data.symbol] = ev.data
            return None

        s = ev.data.symbol
        # if ev.type in Perspective.EVENT_SEQ_KLTICK :
        #     if not s in self.__dictPerspective.keys() :
        #         self.__dictPerspective[s] = Perspective(self.exchange, s)
        #     self.__dictPerspective[s].push(ev)

        # if ev.type in MoneyflowPerspective.EVENT_SEQ_KLTICK :
        #     if not s in self.__dictMoneyflow.keys() :
        #         self.__dictMoneyflow[s] = MoneyflowPerspective(self.exchange, s)
        #     self.__dictMoneyflow[s].push(ev)
        if not s in self.__dictPerspective.keys() :
            self.__dictPerspective[s] = Perspective(self.exchange, s)
        
        return self.__dictPerspective[s].push(ev)

    # __dummy = None
    # def exportF1548(self, symbol=None) :
    #     '''@return an array_like data as float4C, maybe [] or numpy.array
    #     '''
    #     if symbol and symbol in self.__dictPerspective.keys():
    #         return self.__dictPerspective[symbol]._S1548I4

    #     if not PerspectiveState.__dummy:
    #         PerspectiveState.__dummy = Perspective(self.exchange, 'Dummy')

    #     return [0.0] * PerspectiveState.__dummy.fullFloatSize

    def export4C(self, symbol, d4wished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :
        '''
        @param d4wished to specify number of most recent 4-float of the event category to export
        @return an array_like data as float4C
        '''
        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].floatsD4(d4wished)

        raise ValueError('Perspective.floatsD4() unknown symbol[%s]' %symbol )

    # def engorged(self, symbol=None) :
    #     '''@return dict {fieldName, engorged percentage} to represent the engorged percentage of state data
    #     '''
    #     if symbol and symbol in self.__dictPerspective.keys():
    #         return self.__dictPerspective[symbol].engorged
        
    #     return [0.0]

    def export(self, symbol, d4wished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :

        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].export(d4wished)

        raise ValueError('Perspective.export6Cx() unknown symbol[%s]' %symbol )

    def export6C(self, symbol, d4wished= { 'asof':1, EVENT_KLINE_1DAY:20 } ) :

        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].float6C(d4wished)

        raise ValueError('Perspective.export6Cx() unknown symbol[%s]' %symbol )

    def export6Cx(self, symbol, d4wished) :
        if symbol and symbol in self.__dictPerspective.keys():
            res = self.__dictPerspective[symbol].float6Cx()
            if not res: return None

            ret =[]
            for et in d4wished.keys():
                sz = d4wished[et]
                if sz <=0: continue
                if not et in res.keys():
                    ret += [[0.0 for i in range(6)] for j in range(sz)]
                    continue

                data = res[et]
                if 'asof' == et:
                    ret += [data]
                    continue

                if len(data) > sz:
                    ret += data[:sz]
                else:
                    ret += data
                    ret += [[0.0 for i in range(6)] for j in range(sz-len(data))]

            return ret



        raise ValueError('Perspective.export6Cx() unknown symbol[%s]' %symbol )
