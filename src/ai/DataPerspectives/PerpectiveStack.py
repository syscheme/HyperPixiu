# encoding: UTF-8

from __future__ import division

from marketdata.mdBackEnd import MarketData

from datetime import datetime
from abc import ABCMeta, abstractmethod

########################################################################
class MDPerspective(object):
    '''
    Data structure of Perspective:
    1. Ticks
    2. 1min KLines
    3. 5min KLines
    4. 1day KLines
    '''
    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
        super(MDPerspective, self).__init__()
        self._data = {
            MarketData.EVENT_TICK:   [],
            MarketData.EVENT_KLINE_1MIN: [],
            MarketData.EVENT_KLINE_5MIN: [],
            MarketData.EVENT_KLINE_1DAY: [],
        }
        
        self._stampAsOf = None
        self._symbol    = EventData.EMPTY_STRING
        self._vtSymbol  = EventData.EMPTY_STRING
        self._exchange  = exchange
        if symbol and len(symbol)>0:
            self.symbol = self.vtSymbol = symbol
            if  len(exchange)>0 :
                self.vtSymbol = '.'.join([self.symbol, self.exchange])


########################################################################
class PerspectiveStack(object):
    '''
    Perspective合成器，支持:
    1. 基于x分钟K线输入（X可以是1、5、day	）
    2. 基于Tick输入
    '''

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol, KLDepth_1min=60, KLDepth_5min=240, KLDepth_1day=220, tickDepth=120) :
        '''Constructor'''
        self._depth ={
            MarketData.EVENT_TICK:   tickDepth,
            MarketData.EVENT_KLINE_1MIN: KLDepth_1min,
            MarketData.EVENT_KLINE_5MIN: KLDepth_5min,
            MarketData.EVENT_KLINE_1DAY: KLDepth_1day,
        }

        self._symbol   =symbol
        self._exchange =exchange
        self._mergerTick21Min    = TickToKLineMerger(self.cbMergedKLine1min)
        self._mergerKline1To5m   = KlineToXminMerger(self.cbMergedKLine5min, xmin=5)
        self._mergerKline5mToDay = KlineToXminMerger(self.cbMergedKLineDay,  xmin=240)
        self._currentPersective =  MDPerspective(self._symbol, self._exchange)
        for i in range(self._depth[MarketData.EVENT_TICK]) :
            self._currentPersective._data.ticks = [TickData(self._symbol, self._exchange)] + self._currentPersective._data.ticks
        for i in range(self._depth[MarketData.EVENT_KLINE_1MIN]) :
            self._currentPersective._data.KL_1min = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KL_1min
        for i in range(self._depth[MarketData.EVENT_KLINE_5MIN]) :
            self._currentPersective._data.KL_5min = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KL_5min
        for i in range(self._depth[MarketData.EVENT_KLINE_1DAY]) :
            self._currentPersective._data.KLDepth_1day = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KLDepth_1day

        self._stampStepped = {
            MarketData.EVENT_TICK:   None,
            MarketData.EVENT_KLINE_1MIN: None,
            MarketData.EVENT_KLINE_5MIN: None,
            MarketData.EVENT_KLINE_1DAY: None,
        }
        self._stampNoticed = datetime.now()
        
    #----------------------------------------------------------------------
    def pushKLineXmin(self, kline, kltype =MarketData.EVENT_KLINE_1MIN) :
        self._pushKLine(self, kline, kltype)
        if self._stampNoticed < self._stampStepped :
            self._currentPersective._stampAsOf = self._stampStepped
            self.OnNewPerspective(copy(self._currentPersective))
            self._stampNoticed = datetime.now()

    def OnNewPerspective(self, persective) :
        pass
           
    def OnKLinePrefillWished(self, kltype) :
        # dummy setting the stamp
        self._stampStepped[kltype] = datetime.now()

    #----------------------------------------------------------------------
    # private callbacks for the mergers
    def cbMergedKLine1min(self, kline):
        self._pushKLine(kline, kltype =MarketData.EVENT_KLINE_1MIN)

    def cbMergedKLine5min(self, kline):
        self._pushKLine(kline, kltype =MarketData.EVENT_KLINE_5MIN)

    def cbMergedKLineDay(self, kline):
        self._pushKLine(kline, kltype =MarketData.EVENT_KLINE_1DAY)

    #----------------------------------------------------------------------
    def _pushKLine(self, kline, kltype =MarketData.EVENT_KLINE_1MIN):
        '''X分钟K线更新'''

        merger = None
        nextKLT = MarketData.EVENT_TICK
        if MarketData.EVENT_KLINE_1MIN == kltype:
            nextKLT = MarketData.EVENT_KLINE_5MIN
            merger = self._mergerKline1To5m
        if MarketData.EVENT_KLINE_5MIN == kltype:
            nextKLT = MarketData.EVENT_KLINE_1DAY
            merger = self._mergerKline5mToDay
        if MarketData.EVENT_KLINE_5MIN == kltype:
            nextKLT = MarketData.EVENT_TICK

        klines = self._currentPersective._data.get(kltype)
        if klines and len(klines) >0 :
            if (kline.exchange.find('_k2x') or kline.exchange.find('_t2k')) and \
                klines[0].datetime and kline.datetime < klines[0].datetime:
                return # ignore the late merging

            if klines[0].datetime.minute == kline.datetime.minute and klines[0].datetime.date == kline.datetime.date :
                klines[0] = kline # overwrite the frond Kline
            else:
                klines = [kline] + klines[1:] # shift the existing list and push new kline at the front
                self._stampStepped[kltype] = kline.datetime

        nextKL = None
        if nextKLT != MarketData.EVENT_TICK :
            nextKL = self._currentPersective._data.get(nextKLT)

        if nextKL and not self._stampStepped[nextKLT] :
            self.OnKLinePrefillWished(nextKLT)

        if merger:
            merger.pushKLine(kline)
        
    def pushTick(self, tick):
        '''Tick更新'''
        ticks = self._currentPersective._data[MarketData.EVENT_TICK]
        if ticks[0].datetime and tick.datetime < ticks[0].datetime :
            return

        ticks = [tick] + ticks[1:] # shift the existing list and push new tick at the front
        self._stampStepped[MarketData.EVENT_TICK] = ticks.datetime
        if not self._stampStepped[MarketData.EVENT_KLINE_1MIN] :
            self.OnKLinePrefillWished(MarketData.EVENT_KLINE_1MIN)

        if self._mergerTick21Min:
            self._mergerTick21Min.pushTick(tick)

