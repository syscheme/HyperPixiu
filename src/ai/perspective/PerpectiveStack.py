# encoding: UTF-8

from __future__ import division

from marketdata.mdBackEnd import MarketData, TickToKLineMerger, KlineToXminMerger, TickData, KLineData
from event.ecBasic import EventData, datetime2float

from datetime import datetime
from abc import ABCMeta, abstractmethod
import traceback

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
            self._symbol = self.vtSymbol = symbol
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
    # def __init__(self, exchange, symbol, KLDepth_1min=10, KLDepth_5min=12, KLDepth_1day=20, tickDepth=20) :
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
        self._currentPersective =  MDPerspective(self._exchange, self._symbol)
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
        self._stampNoticed = None
        
    #----------------------------------------------------------------------
    def pushKLineXmin(self, kline, kltype =MarketData.EVENT_KLINE_1MIN) :
        import copy
        self._pushKLine(kline, kltype)
        asof = self._stampStepped[kltype]
        if not self._stampNoticed or self._stampNoticed < asof :
            self._currentPersective._stampAsOf = asof
            self.OnNewPerspective(self._currentPersective)
            self._stampNoticed = asof

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

        if kltype in self._currentPersective._data.keys() and len(self._currentPersective._data[kltype]) >0 :
            peek = self._currentPersective._data[kltype][-1]

            if (kline.exchange.find('_k2x') or kline.exchange.find('_t2k')) and peek.datetime and kline.datetime < peek.datetime:
                return # ignore the late merging

            if peek.datetime and (peek.datetime.minute == kline.datetime.minute and peek.datetime.date == kline.datetime.date) :
                self._currentPersective._data[kltype][-1] = kline # overwrite the frond Kline
            else:
                del(self._currentPersective._data[kltype][0])
                self._currentPersective._data[kltype].append(kline)
                # klines.insert(0,kline) # shift the existing list and push new kline at the front
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
from src.marketdata.mdOffline import CapturedToKLine
import bz2
import csv
import tensorflow as tf

class TestStack(PerspectiveStack):

    #----------------------------------------------------------------------
    def __init__(self, symbol, srcFolder, destFolder) :
        '''Constructor'''
        if symbol.isdigit() :
            if symbol.startswith('0') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('3') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('6') :
                symbol = "sh%s" % symbol

        super(TestStack, self).__init__("shop37077890", symbol)
        self._srcFolder, self._destFolder = srcFolder, destFolder
        self._writer = tf.python_io.TFRecordWriter(self._destFolder + "/" + self._currentPersective.vtSymbol +".dpst")

        self._fields = 'date,time,open,high,low,close,volume,ammount'
        self._cvsToKLine = CapturedToKLine(self.OnKLine)

    def __enter__(self) :
        return self

    def __exit__(self) :
        if self._writer :
            self._writer.close()

    def OnKLine(self, kl, eventType) :
        self.pushKLineXmin(kl, eventType)

    def pushKLine1min(self, kline) :
        self.pushKLineXmin(kline, MarketData.EVENT_KLINE_1MIN)

    def KLinePartiToMetrix(self, klines) :
        lst = [] # metrx = []
        for kl in klines :
            dti = int(datetime2float(kl.datetime)) if kl.datetime else 0
            # nlst = [int(dti/86400), int(dti%86400), int(kl.open*1000), int(kl.high*1000), int(kl.low*1000), int(kl.close*1000), int(kl.volume)]
            nlst = [float(dti), float(kl.open*1000), float(kl.high), float(kl.low), float(kl.close*1000), float(kl.volume)]
            lst.extend(nlst) # metrx.append(nlst)
        return lst

    def ticksPartiToMetrix(self, ticks) :
        metrx = []

        for tk in ticks :
            # TODO metrx.append(value =[datetime2float(kl.datetime), float(kl.open), float(kl.high), float(kl.low), float(kl.close), float(kl.volume)])
            pass
        return metrx

    @abstractmethod
    def OnNewPerspective(self, persective) :
        dti = int(datetime2float(persective._stampAsOf)) if persective._stampAsOf else 0
        partiTick = self.ticksPartiToMetrix(persective._data[MarketData.EVENT_TICK])
        partiK1m = self.KLinePartiToMetrix(persective._data[MarketData.EVENT_KLINE_1MIN])
        partiK5m = self.KLinePartiToMetrix(persective._data[MarketData.EVENT_KLINE_5MIN])
        partiK1d = self.KLinePartiToMetrix(persective._data[MarketData.EVENT_KLINE_1DAY])

        example = tf.train.Example(features=tf.train.Features(feature={
                    "asof": tf.train.Feature(int64_list=tf.train.Int64List(value=[dti])),
                    "Tick": tf.train.Feature(float_list=tf.train.FloatList(value=partiTick)),
                    "K1m": tf.train.Feature(float_list=tf.train.FloatList(value=partiK1m)),
                    "K5m": tf.train.Feature(float_list=tf.train.FloatList(value=partiK5m)),
                    "K1d": tf.train.Feature(float_list=tf.train.FloatList(value=partiK1d)),
                    }
                    ))
        self._writer.write(example.SerializeToString())

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

        fnlist = self._listAllFiles(self._srcFolder)
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
                    if line and self._cvsToKLine:
                        # self.debug('line: %s' % (line))
                        self._cvsToKLine.pushCvsRow(line, MarketData.EVENT_KLINE_1MIN, self._exchange, self._symbol)
                except Exception as ex:
                    print(traceback.format_exc())

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

    srcDataHome=u'/mnt/e/AShareSample/'
    destDataHome=u'/mnt/e/AShareSample/'
    symbols= ["000540","000623"]

    for s in symbols :
        with TestStack(s,srcDataHome,destDataHome) as ps :
            ps.loadFiles()




 