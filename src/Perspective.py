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

DEFAULT_DEPTH_TICK = 0
# DEFAULT_DEPTH_TICK = 120
DEFAULT_DEPTH_1min = 30
DEFAULT_DEPTH_5min = 96
DEFAULT_DEPTH_1day = 260

EXPORT_SIGNATURE= '%dT%dM%dF%dD.%s:200109T17' % (DEFAULT_DEPTH_TICK, DEFAULT_DEPTH_1min, DEFAULT_DEPTH_5min, DEFAULT_DEPTH_1day, NORMALIZE_ID)

########################################################################
class EvictableStack(object):
    def __init__(self, evictSize=0, nildata=None):
        '''Constructor'''
        super(EvictableStack, self).__init__()
        self.__data =[]
        self.__evictSize = evictSize
        self.__dataNIL = copy.copy(nildata) if nildata else None
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

    @property
    def exportList(self):
        return _exportList(self, nilFilled=True)

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

    # no pop here: def pop(self):
    #    del(self.__data[-1])

    def push(self, item):
        self.__data.insert(0, item)
        while self.evictSize >=0 and self.size > self.evictSize:
            del(self.__data[-1])

########################################################################
class Perspective(MarketData):
    '''
    Data structure of Perspective:
    1. Ticks
    2. 1min KLines
    3. 5min KLines
    4. 1day KLines
    '''
    EVENT_SEQ =  [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]
    TICKPRICES_TO_EXP = 'price,open,high,low,b1P,b2P,b3P,b4P,b5P,a1P,a2P,a3P,a4P,a5P'
    TICKVOLS_TO_EXP   = 'volume,b1V,b2V,b3V,b4V,b5V,a1V,a2V,a3V,a4V,a5V'
    KLPRICES_TO_EXP = 'open,high,low,close'
    KLVOLS_TO_EXP = 'volume'

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol=None, KLDepth_1min=DEFAULT_DEPTH_1min, KLDepth_5min=DEFAULT_DEPTH_5min, KLDepth_1day=DEFAULT_DEPTH_1day, tickDepth=DEFAULT_DEPTH_TICK) :
        '''Constructor'''
        super(Perspective, self).__init__(exchange, symbol)

        self._stacks = {
            EVENT_TICK:       EvictableStack(tickDepth, TickData(self.exchange, self.symbol)),
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_1day, KLineData(self.exchange, self.symbol)),
        }

        #TODO: evsPerDay temporarily is base on AShare's 4hr/day
        self._evsPerDay = {
            EVENT_TICK:       3600/2 *4, # assuming every other seconds
            EVENT_KLINE_1MIN: 60*4,
            EVENT_KLINE_5MIN: 12*4,
            EVENT_KLINE_1DAY: 1,
        }

        self.__stampLast = None
        self.__focusLast = None
        self.__dayOHLC = None

    @property
    def desc(self) :
        str = '%s>%s ' % (self.focus[len(MARKETDATE_EVENT_PREFIX):], self.getAsOf(self.focus).strftime('%Y-%m-%dT%H:%M:%S'))
        for i in Perspective.EVENT_SEQ :
            str += '%sX%d/%d,' % (i[len(MARKETDATE_EVENT_PREFIX):], self._stacks[i].size, self._stacks[i].evictSize)
        return str

    @property
    def asof(self) : return self.getAsOf(None)

    def getAsOf(self, evType=None) :
        if not evType or not evType in self._stacks.keys():
            return self.__stampLast if self.__stampLast else DT_EPOCH
        
        stack = self._stacks[evType]
        return DT_EPOCH if stack.size <=0 else stack.top.asof

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

    @property
    def focus(self) :
        return self.__focusLast if self.__focusLast else ''

    @property
    def latestPrice(self) :
        ret =0.0
        stk = self._stacks[self.__focusLast]
        if stk or stk.size >0:
            ret = stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close
        else:
            for et in Perspective.EVENT_SEQ:
                stk = self._readers[self.__focusLast]
                if not stk or stk.size <=0:
                    continue
                ret = stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close
        
        return round(ret, 3)

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

        latestevd = self._stacks[ev.type].top
        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            self._stacks[ev.type].push(ev.data)
            self.__focusLast = ev.type
            if not self.__stampLast or self.__stampLast < ev.data.datetime :
                self.__stampLast = ev.data.datetime
            return ev
        
        if not ev.data.exchange or (latestevd.exchange and not ('_k2x' in latestevd.exchange or '_t2k' in latestevd.exchange)) :
            return None # not overwritable

        self.__focusLast = ev.type
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
        return ev

    TICK_FLOATS=7
    KLINE_FLOATS=5

    @property
    def NNFloatsSize(self):
        # klsize = sum([self._stacks[et].evictSize for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] ])
        # return Perspective.TICK_FLOATS *self._stacks[EVENT_TICK].evictSize + Perspective.KLINE_FLOATS *klsize
        itemsize = sum([self._stacks[et].evictSize for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] ]) + self._stacks[EVENT_TICK].evictSize
        return (itemsize +1) * EXPORT_FLOATS_DIMS

    @property
    def NNFloats(self):
        if self._stacks[EVENT_KLINE_1DAY].size <=0:
            return [0.0] * self.NNFloatsSize # toNNFloats not available
        
        klbaseline = self._stacks[EVENT_KLINE_1DAY].top
        return self.toNNFloats(baseline_Price=klbaseline.close, baseline_Volume=klbaseline.volume)
    
    @abstractmethod
    def toNNFloats(self, baseline_Price=1.0, baseline_Volume =1.0, priceDiffNormalizeMultiple=1.0) :
        '''
        @return float[] for numpy
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

        # self._evsPerDay = {
        #     EVENT_TICK:       3600/2 *4, # assuming every other seconds
        #     EVENT_KLINE_1MIN: 60*4,
        #     EVENT_KLINE_5MIN: 12*4,
        #     EVENT_KLINE_1DAY: 1,
        # }


        # part 1, EVENT_TICK
        stk = self._stacks[EVENT_TICK]
        bV= (baseline_Volume / self._evsPerDay[EVENT_TICK])
        pdNM = 10.0 # = priceDiffNormalizeMultiple *self._evsPerDay[EVENT_TICK]
        for i in range(stk.evictSize):
            if i >= stk.size:
                result += [0.0] * EXPORT_FLOATS_DIMS
            else:
                v = stk[i].toNNFloats(baseline_Price=baseline_Price, baseline_Volume= b, priceDiffNormalizeMultiple =pdNM)
                # Perspective.TICK_FLOATS = len(v)
                result += v

        for et in [EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]:
            stk = self._stacks[et]
            bV= (baseline_Volume / self._evsPerDay[et])
            pdNM = 10.0 # = priceDiffNormalizeMultiple *self._evsPerDay[et]
            for i in range(stk.evictSize):
                if i >= stk.size:
                    result += [0.0] * EXPORT_FLOATS_DIMS
                else:
                    v = stk[i].toNNFloats(baseline_Price=baseline_Price, baseline_Volume= bV, priceDiffNormalizeMultiple =pdNM)
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
        for et in Perspective.EVENT_SEQ:
            if not self._readers[et] :
                continue
            if self._readers[et].resetRead() :
                succ = True

        return succ

    def readNext(self):
        '''
        @return next item, mostlikely expect one of Event()
        '''
        for et in Perspective.EVENT_SEQ:
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

    # -- impl of MarketState --------------------------------------------------------------
    def listOberserves(self) :
        return [ s for s in self.__dictPerspective.keys()]

    def addMonitor(self, symbol) :
        ''' add a symbol to monitor
        '''
        raise NotImplementedError

    def latestPrice(self, symbol) :
        ''' query for latest price of the given symbol
        @return the price
        '''
        if not symbol in self.__dictPerspective.keys() :
            return 0.0
        
        return self.__dictPerspective[symbol].latestPrice

    def getAsOf(self, symbol=None) :
        ''' 
        @return the datetime as of latest observing
        '''
        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].asof

        ret = None
        for s, p in self.__dictPerspective.items() :
            if not ret or ret > p.asof:
                ret = p.asof
        return ret if ret else DT_EPOCH

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
            return

        if not ev.type in Perspective.EVENT_SEQ :
            return
            
        s = ev.data.symbol
        if not s in self.__dictPerspective.keys() :
            self.__dictPerspective[s] = Perspective(self.exchange, s)
        self.__dictPerspective[s].push(ev)

    __dummy = None
    def toNNFloats(self, symbol=None) :
        '''@return an array_like data as toNNFloats, maybe [] or numpy.array
        '''
        if symbol and symbol in self.__dictPerspective.keys():
            return self.__dictPerspective[symbol].NNFloats

        if not PerspectiveState.__dummy:
            PerspectiveState.__dummy = Perspective(self.exchange, 'Dummy')
        return [0.0] * PerspectiveState.__dummy.NNFloatsSize

