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
import tensorflow as tf
import copy

EVENT_Perspective  = MARKETDATE_EVENT_PREFIX + 'Persp'   # 错误回报事件

########################################################################
class Stack(object):
    def __init__(self, fixedSize=0, nildata=None):
        '''Constructor'''
        super(Stack, self).__init__()
        self._data =[]
        self._fixedSize =fixedSize
        self._dataNIL = copy.copy(nildata) if nildata else None
        if self._dataNIL and self._fixedSize and self._fixedSize>0 :
            for i in range(self._fixedSize) :
                self._data.insert(0, nildata)

    @property
    def top(self):
        return self._data[0] if len(self._data) >0 else None

    @property
    def tsize(self):
        return self._fixedSize

    @property
    def size(self):
        return len(self._data) if self._data else 0

    @property
    def tolist(self):
        return self._data

    def overwrite(self, item):
        self._data[0] =item

    # no pop here: def pop(self):
    #    del(self._data[-1])

    def push(self, item):
        self._data.insert(0, item)
        while self._fixedSize and self._fixedSize >0 and self.size > self._fixedSize:
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
            EVENT_TICK:       Stack(tickDepth, TickData(self._exchange, self._symbol)),
            EVENT_KLINE_1MIN: Stack(KLDepth_1min, KLineData(self._exchange, self._symbol)),
            EVENT_KLINE_5MIN: Stack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
            EVENT_KLINE_1DAY: Stack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
        }

    @property
    def desc(self) :
        return ''

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
            if not ev or not ev.data :
                return None

            latestevd = self._perspective._stacks[ev.type_].top
            if not latestevd or not latestevd.datetime or ev.data.datetime > latestevd.datetime :
                self._perspective._stacks[ev.type_].push(ev.data)
                evPsp = Event(EVENT_Perspective)
                evPsp.setData(self._perspective)
                return evPsp
                
        return None
