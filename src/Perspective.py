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
__dtEpoch = datetime.utcfromtimestamp(0)

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
        fillsize = (self.evictSize - self.size) if self.evictSize >=0 else 0
        return self._data + [self._dataNIL] *fillsize

    def overwrite(self, item):
        self._data[0] =item

    # no pop here: def pop(self):
    #    del(self._data[-1])

    def push(self, item):
        self._data.insert(0, item)
        while self.evictSize >=0 and self.size > self.evictSize:
            del(self._data[-1])

########################################################################
class Perspective(EventData):
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
        super(Perspective, self).__init__()

        self._stampAsOf = None
        self._symbol    = EventData.EMPTY_STRING
        self._vtSymbol  = EventData.EMPTY_STRING
        self._exchange  = exchange
        if symbol and len(symbol)>0:
            self._symbol = self.vtSymbol = symbol
            if  len(exchange)>0 :
                self._vtSymbol = '.'.join([self._symbol, self._exchange])

        self._stacks = {
            EVENT_TICK:       EvictableStack(tickDepth, TickData(self._exchange, self._symbol)),
            EVENT_KLINE_1MIN: EvictableStack(KLDepth_1min, KLineData(self._exchange, self._symbol)),
            EVENT_KLINE_5MIN: EvictableStack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
            EVENT_KLINE_1DAY: EvictableStack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
        }

        self.__stampLast = None
        self.__focusLast = None

    @property
    def desc(self) :
        str = self.asof.strftime('%Y%m%dT%H%M%S')
        str += ': '
        for i in [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] :
            str += '%sx%s, ' % (i[4:], self._stacks[i].size)
        return str

    @property
    def asof(self) :
        return self.__stampLast if self.__stampLast else __dtEpoch

    @property
    def latestPrice(self) :
        stk = self._readers[self.__focusLast]
        if stk or stk.size >0:
            return stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close

        for et in Perspective.EVENT_SEQ:
            stk = self._readers[self.__focusLast]
            if not stk or stk.size <=0:
                continue
            return stk.top.price if EVENT_TICK == self.__focusLast else stk.top.close

    def push(self, ev) :
        if not ev.type_ in self._stacks.keys():
            return False

        latestevd = self._stacks[ev.type_].top
        if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
            self._stacks[ev.type_].push(ev.data)
            if not self.__stampLast or self.__stampLast < ev.data.datetime :
                self.__stampLast = ev.data.datetime
                self.__focusLast = ev.type_
            return True
        
        if not ev.data.exchange or latestevd.exchange and not '_k2x' in latestevd.exchange and not '_t2k' in latestevd.exchange :
            return False # not overwritable

        for i in range(len(self._stacks[ev.type_])) :
            if ev.data.datetime > self._stacks[ev.type_][i].datetime :
                continue
            if ev.data.datetime == self._stacks[ev.type_][i] :
                self._stacks[ev.type_][i] = ev.data
                return True
            else :
                self._stacks[ev.type_].insert(i, ev.data)
                while self._stacks[ev.type_].evictSize >=0 and self._stacks[ev.type_].size > self._stacks[ev.type_].evictSize:
                    del(self._stacks[ev.type_]._data[-1])
                return True
        
        self._stacks[ev.type_].insert(-1, ev.data)
        while self._stacks[ev.type_].evictSize >=0 and self._stacks[ev.type_].size > self._stacks[ev.type_].evictSize:
            del(self._stacks[ev.type_]._data[-1])
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
            if not ev or not ev.data or not ev.type_ in self._perspective._stacks.keys():
                return None

            latestevd = self._perspective._stacks[ev.type_].top
            if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
                if self._perspective.push(ev) :
                    evPsp = Event(EVENT_Perspective)
                    evPsp.setData(self._perspective)
                    return evPsp
                
        return None

