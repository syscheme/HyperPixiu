# encoding: UTF-8

from __future__ import division

from marketdata.mdBackEnd import MarketData, TickToKLineMerger, KlineToXminMerger, TickData, KLineData
from event.ecBasic import EventData

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
                self._vtSymbol = '.'.join([self._symbol, self._exchange])


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
        self._mergerTickTo1Min    = TickToKLineMerger(self.cbMergedKLine1min)
        self._mergerKline1To5m   = KlineToXminMerger(self.cbMergedKLine5min, xmin=5)
        self._mergerKline5mToDay = KlineToXminMerger(self.cbMergedKLineDay,  xmin=240)
        self._currentPersective =  MDPerspective(self._symbol, self._exchange)
        for i in range(self._depth[MarketData.EVENT_TICK]) :
            self._currentPersective._data[MarketData.EVENT_TICK] = [TickData(self._exchange, self._symbol)] + self._currentPersective._data[MarketData.EVENT_TICK]
        for i in range(self._depth[MarketData.EVENT_KLINE_1MIN]) :
            self._currentPersective._data[MarketData.EVENT_KLINE_1MIN] = [KLineData(self._exchange, self._symbol)] + self._currentPersective._data[MarketData.EVENT_KLINE_1MIN]
        for i in range(self._depth[MarketData.EVENT_KLINE_5MIN]) :
            self._currentPersective._data[MarketData.EVENT_KLINE_5MIN] = [KLineData(self._exchange, self._symbol)] + self._currentPersective._data[MarketData.EVENT_KLINE_5MIN]
        for i in range(self._depth[MarketData.EVENT_KLINE_1DAY]) :
            self._currentPersective._data[MarketData.EVENT_KLINE_1DAY] = [KLineData(self._exchange, self._symbol)] + self._currentPersective._data[MarketData.EVENT_KLINE_1DAY]

        self._stampStepped = {
            MarketData.EVENT_TICK:   None,
            MarketData.EVENT_KLINE_1MIN: None,
            MarketData.EVENT_KLINE_5MIN: None,
            MarketData.EVENT_KLINE_1DAY: None,
        }
        self._stampNoticed = datetime.now()
        
    #----------------------------------------------------------------------
    def pushKLineXmin(self, kline, kltype =MarketData.EVENT_KLINE_1MIN) :
        self._pushKLine(kline, kltype)
        if self._stampNoticed < self._stampStepped :
            self._currentPersective._stampAsOf = self._stampStepped
            self.OnNewPerspective(copy(self._currentPersective))
            self._stampNoticed = datetime.now()

    @abstractmethod
    def OnNewPerspective(self, persective) :
        pass
           
    @abstractmethod
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
            nextKLT = MarketData.EVENT_TICK # as an invalid option

        klines = self._currentPersective._data.get(kltype)
        if klines and len(klines) >0 :
            if (kline.exchange.find('_k2x') or kline.exchange.find('_t2k')) and klines[0].datetime and kline.datetime < klines[0].datetime:
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

        if self._mergerTickTo1Min:
            self._mergerTickTo1Min.pushTick(tick)

########################################################################
from src.marketdata.mdOffline import TaobaoCvsToEvent
import bz2
import csv

class TestStack(PerspectiveStack):

    #----------------------------------------------------------------------
    def __init__(self, symbol, folder) :
        '''Constructor'''
        if symbol.isdigit() :
            if symbol.startswith('0') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('3') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('6') :
                symbol = "sh%s" % symbol

        super(TestStack, self).__init__("shop37077890", symbol)
        self._dirData = folder
        self._fields = 'date,time,open,high,low,close,volume,ammount'
        self._dataToEvent = TaobaoCvsToEvent(self.pushKLineEvent)

    def pushKLineEvent(self, event) :
        self.pushKLineXmin(event._dict['data'], event.type_)

    def pushKLine1min(self, kline) :
        self.pushKLineXmin(kline, MarketData.EVENT_KLINE_1MIN)

    def _listAllFiles(self, top):
        fnlist = []
        for root, _, files in os.walk(top, topdown=False):
            for name in files:
                fnlist.append('%s/%s' % (root, name))
            # for name in subdirs:
            #     fnlist += self._listAllFiles('%s/%s' % (root, name))
        return fnlist

    def _filterFiles(self):
        """从数据库中读取Bar数据，startDate是datetime对象"""

        fnlist = self._listAllFiles(self._dirData)
        fnlist.sort()
        csvfiles = []
        prev = ""
        trange = None
        searchsymb = self._symbol[2:]
        for name in fnlist:
            if not searchsymb in name:
                continue

            basename = name.split('/')[-1]
            stk = basename.split('.')
            if stk[-1] =='csv' :
                basename = stk[-2] 
            elif stk[-1] =='bz2' and stk[-2] =='csv':
                basename = stk[-3]
            else : continue

            pos = basename.find(searchsymb)
            stampstr = basename[pos+1 + len(searchsymb):] if pos >=0 else basename
            # if 'H' in stampstr:
            #     trange = self._stampRangeinFn['H']
            # elif 'Q' in stampstr :
            #     trange = self._stampRangeinFn['Q']
            # else :
            #     trange = self._stampRangeinFn['T']
            
            # if stampstr < trange[0]:
            #     prev= name
            # elif stampstr <= trange[1]:
            csvfiles.append(name)

        if len(prev) >0 and len(csvfiles) >0 and csvfiles[0] > trange[0]:
            csvfiles = [prev] + csvfiles
        
        return csvfiles

    def debug(self, msg) :
        pass

    def error(self, msg) :
        pass

    def loadFiles(self) :

        self._seqFiles = self._filterFiles()
        fields = self._fields.split(',') if self._fields else None
        for fn in self._seqFiles :
            self.debug('openning input file %s' % (fn))
            extname = fn.split('.')[-1]
            if extname == 'bz2':
                self._importStream = bz2.open(fn, mode='rt') # bz2.BZ2File(fn, 'rb')
            else:
                self._importStream = file(fn, 'rt')

            self._reader = csv.DictReader(self._importStream, fields, lineterminator='\n') if 'csv' in fn else self._importStream
            # self._reader = csv.reader(self._importStream) if 'csv' in fn else self._importStream
            if not self._reader:
                self.warn('failed to open input file %s' % (fn))
                continue

            if len(self._fields) <=0:
                self._fields = self._reader.headers()

            while self._reader:
                try :
                    line = next(self._reader, None)
                    # c+=1 if line
                except Exception as ex:
                    line = None

                if not line:
                    # self.error(traceback.format_exc())
                    self._reader = None
                    self._importStream.close()
                    continue

                try :
                    if line and self._dataToEvent:
                        # self.debug('line: %s' % (line))
                        self._dataToEvent.push(line, MarketData.EVENT_KLINE_1MIN, self._symbol)
                except Exception as ex:
                    pass # self.error(traceback.format_exc())

        if bEOS:
            self.info('reached end, queued dummy Event(End), fake an EOS event')
            newSymbol = '%s.%s' % (self._symbol, self.exchange)
            edata = TickData(self._symbol) if 'Tick' in self._eventType else KLineData(self._symbol)
            edata.date = MarketData.DUMMY_DATE_EOS
            edata.time = MarketData.DUMMY_TIME_EOS
            edata.datetime = MarketData.DUMMY_DT_EOS
            edata.exchange  = MarketData.exchange # output exchange name 
            event = Event(self._eventType)
            event.dict_['data'] = edata
            self.onMarketEvent(event)
            c +=1
            self.onMarketEvent(event)
            self._main.stop()

        return c


if __name__ == '__main__':
    # import vnApp.HistoryData as hd
    import os

    srcDataHome=u'/mnt/h/AShareSample/'
    symbols= ["000540","000623","000630","000709","00072"]

    for s in symbols :
        ps = TestStack(s,srcDataHome)
        files = ps.loadFiles()

