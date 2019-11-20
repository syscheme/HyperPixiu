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

########################################################################
class EvictableStack(object):
    def __init__(self, evictSize=0, nildata=None):
        '''Constructor'''
        super(EvictableStack, self).__init__()
        self._data =[]
        self._evictSize = evictSize
        self._dataNIL = copy.copy(nildata) if nildata else None
        # if self._dataNIL and self._evictSize and self._evictSize >0 :
        #     for i in range(self._evictSize) :
        #         self._data.insert(0, nildata)

    @property
    def top(self):
        return self._data[0] if len(self._data) >0 else None

    @property
    def evictSize(self):
        return self._evictSize if self._evictSize else -1

    @property
    def size(self):
        return len(self._data) if self._data else 0

    @property
    def exportList(self):
        return _exportList(self, nilFilled=True)

    def _exportList(self, nilFilled=False):
        if nilFilled :
            fillsize = (self.evictSize - self.size) if self.evictSize >=0 else 0
            return self._data + [self._dataNIL] *fillsize
        return self._data

    def overwrite(self, item):
        self._data[0] =item

    # no pop here: def pop(self):
    #    del(self._data[-1])

    def push(self, item):
        self._data.insert(0, item)
        while self.evictSize >=0 and self.size > self.evictSize:
            del(self._data[-1])

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
    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None, KLDepth_1min=60, KLDepth_5min=240, KLDepth_1day=220, tickDepth=120):
        '''Constructor'''
        super(Perspective, self).__init__(exchange, symbol)

        self._stampAsOf = None
        self._stacks = {
            EVENT_TICK:       EvictableStack(tickDepth, TickData(self.exchange, self.symbol)),
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineData(self.exchange, self.symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_5min, KLineData(self.exchange, self.symbol)),
        }

        self.__stampLast = None
        self.__focusLast = None
        self.__dayOHLC = None

    @property
    def desc(self) :
        str = '%s> ' % self.focus
        for i in [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] :
            str += '%sx%s, ' % (i[4:], self._stacks[i].size)
        return str

    @property
    def asof(self) :
        return self.__stampLast if self.__stampLast else DT_EPOCH

    @property
    def focus(self) :
        return self.__focusLast if self.__focusLast else ''

    @property
    def latestPrice(self) :
        stk = self._stacks[self.__focusLast]
        if stk or stk.size >0:
            return stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close

        for et in Perspective.EVENT_SEQ:
            stk = self._readers[self.__focusLast]
            if not stk or stk.size <=0:
                continue
            return stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close

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
        self.__dayOHLC.date = self.__dayOHLC.datetime.strftime('%Y%m%d')
        self.__dayOHLC.time = self.__dayOHLC.datetime.strftime('%H%M%S')
        return self.__dayOHLC

    def push(self, ev) :
        if not self.__push(ev) :
            return False

        if self.__dayOHLC and self.__dayOHLC.asof < self.asof.replace(hour=0,minute=0,second=0,microsecond=0) :
            self.__dayOHLC = None

        evd = ev.data
        if not self.__dayOHLC :
            self.__dayOHLC = evd
            return True

        if evd.asof > self.__dayOHLC.asof:
            self.__dayOHLC.high = max(self.__dayOHLC.high, evd.high)
            self.__dayOHLC.low  = min(self.__dayOHLC.low, evd.low)
            self.__dayOHLC.close = evd.close        
            self.__dayOHLC.volume =0 # NOT GOOD when built up from 1min+5min: += int(evd.volume)                

            self.__dayOHLC.openInterest = evd.openInterest
            self.__dayOHLC.datetime = evd.asof

        return True

    def __push(self, ev) :
        if not ev.type in self._stacks.keys():
            return False

        latestevd = self._stacks[ev.type].top
        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            self._stacks[ev.type].push(ev.data)
            self.__focusLast = ev.type
            if not self.__stampLast or self.__stampLast < ev.data.datetime :
                self.__stampLast = ev.data.datetime
            return True
        
        if not ev.data.exchange or latestevd.exchange and not '_k2x' in latestevd.exchange and not '_t2k' in latestevd.exchange :
            return False # not overwritable

        self.__focusLast = ev.type
        for i in range(len(self._stacks[ev.type])) :
            if ev.data.datetime > self._stacks[ev.type][i].datetime :
                continue
            if ev.data.datetime == self._stacks[ev.type][i] :
                self._stacks[ev.type][i] = ev.data
                return True
            else :
                self._stacks[ev.type].insert(i, ev.data)
                while self._stacks[ev.type].evictSize >=0 and self._stacks[ev.type].size > self._stacks[ev.type].evictSize:
                    del(self._stacks[ev.type]._data[-1])
                return True
        
        self._stacks[ev.type].insert(-1, ev.data)
        while self._stacks[ev.type].evictSize >=0 and self._stacks[ev.type].size > self._stacks[ev.type].evictSize:
            del(self._stacks[ev.type]._data[-1])
        return True

########################################################################
class PerspectiveGenerator(Iterable):

    #----------------------------------------------------------------------
    def __init__(self, perspective):
        super(PerspectiveGenerator, self).__init__()
        self._perspective = perspective
        self._generator = None

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
class PerspectiveDict(MarketState):

    def __init__(self, exchange):
        """Constructor"""
        super(PerspectiveDict, self).__init__(exchange)
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
