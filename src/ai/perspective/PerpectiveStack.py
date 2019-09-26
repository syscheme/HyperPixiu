# encoding: UTF-8

from __future__ import division

from marketdata.mdBackEnd import MarketData, TickToKLineMerger, KlineToXminMerger, TickData, KLineData
from event.ecBasic import EventData, datetime2float

from datetime import datetime
from abc import ABCMeta, abstractmethod
import traceback

from src.marketdata.mdOffline import CapturedToKLine
import bz2
import csv
import tensorflow as tf

########################################################################
class psStack(object):
    def __init__(self, fixedSize=0, nildata=None):
        '''Constructor'''
        super(psStack, self).__init__()
        self._data =[]
        self._fixedSize =fixedSize
        if nildata and self._fixedSize and self._fixedSize>0 :
            for i in range(self._fixedSize) :
                self._data.insert(0,nildata)

    @property
    def top(self):
        return self._data[0] if len(self._data) >0 else None

    @property
    def size(self):
        return len(self._data) if self._data else 0

    @property
    def tolist(self):
        return self._data

    def overwrite(self, item):
        self._data[0] =item

    def pop(self):
        del(self._data[-1])

    def push(self, item):
        self._data.insert(0, item)
        while self._fixedSize and self._fixedSize >0 and self.size > self._fixedSize:
            self.pop()

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
    def __init__(self, exchange, symbol =None, KLDepth_1min=60, KLDepth_5min=240, KLDepth_1day=220, tickDepth=120):
        '''Constructor'''
        super(MDPerspective, self).__init__()

        self._stampAsOf = None
        self._symbol    = EventData.EMPTY_STRING
        self._vtSymbol  = EventData.EMPTY_STRING
        self._exchange  = exchange
        if symbol and len(symbol)>0:
            self._symbol = self.vtSymbol = symbol
            if  len(exchange)>0 :
                self._vtSymbol = '.'.join([self._symbol, self._exchange])

        self._data = {
            MarketData.EVENT_TICK:   psStack(tickDepth, TickData(self._exchange, self._symbol)),
            MarketData.EVENT_KLINE_1MIN: psStack(KLDepth_1min, KLineData(self._exchange, self._symbol)),
            MarketData.EVENT_KLINE_5MIN: psStack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
            MarketData.EVENT_KLINE_1DAY: psStack(KLDepth_5min, KLineData(self._exchange, self._symbol)),
        }

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
        self._currentPersective =  MDPerspective(self._exchange, self._symbol, KLDepth_1min=60, KLDepth_5min=240, KLDepth_1day=220, tickDepth=120)
        self._incremental = {
            MarketData.EVENT_TICK:   None,
            MarketData.EVENT_KLINE_1MIN: None,
            MarketData.EVENT_KLINE_5MIN: None,
            MarketData.EVENT_KLINE_1DAY: None,
        }

        self._stampStepped = {
            MarketData.EVENT_TICK:   None,
            MarketData.EVENT_KLINE_1MIN: None,
            MarketData.EVENT_KLINE_5MIN: None,
            MarketData.EVENT_KLINE_1DAY: None,
        }
        self._stampNoticed = None
        
    #----------------------------------------------------------------------
    def pushKLineXmin(self, kline, kltype =MarketData.EVENT_KLINE_1MIN) :
        self._pushKLine(kline, kltype)
        asof = self._stampStepped[kltype]
        if not self._stampNoticed or self._stampNoticed < asof :
            self._currentPersective._stampAsOf = asof
            self.OnIncrementalPerspective(self._currentPersective, self._incremental)
            self.commitIncremental()
            self._stampNoticed = asof

    @abstractmethod
    def OnIncrementalPerspective(self, persective, incremental) :
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

    def commitIncremental(self):
        for dtype in [MarketData.EVENT_TICK, MarketData.EVENT_KLINE_1MIN, MarketData.EVENT_KLINE_5MIN, MarketData.EVENT_KLINE_1DAY] :
            newdata = self._incremental[dtype]
            if not newdata :
                continue

            updated = False
            top = self._currentPersective._data[dtype].top
            if top and top.datetime and top.datetime == newdata.datetime :
                self._currentPersective._data[kltype].overwrite(newdata) # overwrite the frond Kline
                updated = True
            
            if not updated :
                self._currentPersective._data[dtype].push(newdata)

            self._incremental[dtype] = None # reset the _incremental if has commited
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

        if kltype in self._currentPersective._data.keys() :
            top = self._currentPersective._data[kltype].top
            added = False
            if top and top.datetime:
                if (kline.exchange.find('_k2x') or kline.exchange.find('_t2k')) and kline.datetime < top.datetime:
                    return # ignore the late merging

            #     if top.datetime.minute == kline.datetime.minute and top.datetime.date == top.datetime.date :
            #         self._currentPersective._data[kltype].overwrite(kline) # overwrite the frond Kline
            #         added = True
            
            # if not added :
            #     self._currentPersective._data[kltype].push(kline)
        self._incremental[kltype]  = kline
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
        self._cFrames =0
        self._fields = 'date,time,open,high,low,close,volume,ammount'
        self._cvsToKLine = CapturedToKLine(self.OnKLine)

    def __enter__(self) :
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) :
        if self._writer :
            self._writer.close()

    def OnKLine(self, kl, eventType) :
        self.pushKLineXmin(kl, eventType)

    def pushKLine1min(self, kline) :
        self.pushKLineXmin(kline, MarketData.EVENT_KLINE_1MIN)

    def _klineToListInt(self, kl) :
        dti = int(datetime2float(kl.datetime)) if kl.datetime else 0
        return [int(dti/86400), int(dti%86400), int(kl.open*1000), int(kl.high*1000), int(kl.low*1000), int(kl.close*1000), int(kl.volume)]

    def KLinePartiToTfFeature(self, klines) :
        lst = []
        for kl in klines :
            # nlst = [float(dti), float(kl.open*1000), float(kl.high), float(kl.low), float(kl.close*1000), float(kl.volume)]
            # nlst = [int(dti/86400), int(dti%86400), int(kl.open*1000), int(kl.high*1000), int(kl.low*1000), int(kl.close*1000), int(kl.volume)]
            lst.extend(self._klineToListInt(kl))
        # return tf.train.Feature(float_list=tf.train.FloatList(value=lst))
        return tf.train.Feature(int64_list=tf.train.Int64List(value=lst)) # the sizeof(Int64List） is about 1/3 of sizeof(FloatList）

    def _tickToListInt(self, tk) :
        dti = int(datetime2float(tk.datetime)) if tk.datetime else 0
        nlst = [ int(dti/86400), int(dti%86400), 
                    int(tk.price*1000), int(tk.volume), int(tk.openInterest), 
                    int(tk.open*1000), int(tk.high*1000), int(tk.low*1000), int(tk.prevClose*1000), # no tk.close always
                    int(tk.b1P*1000), int(tk.b1V), int(tk.a1P*1000), int(tk.a1V),
                    int(tk.b2P*1000), int(tk.b2V), int(tk.a2P*1000), int(tk.a2V),
                    int(tk.b3P*1000), int(tk.b3V), int(tk.a3P*1000), int(tk.a3V),
                    int(tk.b4P*1000), int(tk.b4V), int(tk.a4P*1000), int(tk.a4V),
                    int(tk.b5P*1000), int(tk.b5V), int(tk.a5P*1000), int(tk.a5V)
                    ]
        return nlst
        
    def ticksPartiToTfFeature(self, ticks) :
        lst = []
        for tk in ticks :
            lst.extend(self._tickToListInt(tk))

        return tf.train.Feature(int64_list=tf.train.Int64List(value=lst)) # the sizeof(Int64List） is about 1/3 of sizeof(FloatList）

    @abstractmethod
    def OnIncrementalPerspective(self, persective, incremental) :

        dti = int(datetime2float(persective._stampAsOf)) if persective._stampAsOf else 0

        frame = None
        if self._cFrames <=0 :
            # the first frame
            partiTick = self.ticksPartiToTfFeature(persective._data[MarketData.EVENT_TICK].tolist)
            partiK1m = self.KLinePartiToTfFeature(persective._data[MarketData.EVENT_KLINE_1MIN].tolist)
            partiK5m = self.KLinePartiToTfFeature(persective._data[MarketData.EVENT_KLINE_5MIN].tolist)
            partiK1d = self.KLinePartiToTfFeature(persective._data[MarketData.EVENT_KLINE_1DAY].tolist)
            frame = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[dti])),
                        "Tick": partiTick,
                        "K1m": partiK1m,
                        "K5m": partiK5m,
                        "K1d": partiK1d,
                        }
                        ))

        if frame :
            self._writer.write(frame.SerializeToString())
            self._cFrames+=1

        frame = None
        # the incremental frame
        NilFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[])) 
        inc = [NilFeature, NilFeature, NilFeature, NilFeature]
        if incremental[MarketData.EVENT_TICK] :
            inc[0] = tf.train.Feature(int64_list=tf.train.Int64List(value=self._tickToListInt(incremental[MarketData.EVENT_TICK])))
        c =0
        for dtype in [MarketData.EVENT_KLINE_1MIN, MarketData.EVENT_KLINE_5MIN, MarketData.EVENT_KLINE_1DAY] :
            c+=1
            newdata = incremental[dtype]
            if not newdata :
                continue
            inc[c] = tf.train.Feature(int64_list=tf.train.Int64List(value=self._klineToListInt(incremental[dtype])))

        frame = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[dti])),
                        "Tick": inc[0],
                        "K1m": inc[1],
                        "K5m": inc[2],
                        "K1d": inc[3],
                        }
                        ))

        if frame :
            self._writer.write(frame.SerializeToString())
            self._cFrames+=1

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
        c=0
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
                
                if c>1000:
                    return c

                try :
                    line = next(self._reader, None)
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
                        c+=1
                except Exception as ex:
                    print(traceback.format_exc())
        return c


if __name__ == '__main__':
    # import vnApp.HistoryData as hd
    import os

    srcDataHome=u'/mnt/h/AShareSample/'
    destDataHome=u'/mnt/h/AShareSample/'
    symbols= ["000540","000623"]

    for s in symbols :
        with TestStack(s,srcDataHome,destDataHome) as ps :
            ps.loadFiles()

 