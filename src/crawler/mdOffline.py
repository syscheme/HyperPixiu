# encoding: UTF-8

from __future__ import division

# from vnApp.MainRoutine import *
# from vnApp.Account import *
from vnApp.MarketData import *
from vnApp.EventChannel import *
# from vnApp.DataRecorder import *

from copy import copy
from datetime import datetime
from threading import Thread
import sys
from multiprocessing.dummy import Pool
from time import sleep
from datetime import datetime

import json

########################################################################
class CapturedToKLine(object):

    def __init__(self, sink):
        """Constructor"""
        super(CapturedToKLine, self).__init__()
        self._sink = sink

    @property
    def fields(self) :
        return 'date,time,open,high,low,close,volume,ammount'

    @abstractmethod
    def OnKLine(self, eventType, kline, dataOf =None):
        if self._sink:
            self._sink(kline, eventType)

    @abstractmethod
    def pushCvsRow(self, csvrow, eventType, exchange, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError

        kl = KLineData(exchange, symbol)
        kl.open = float(csvrow['open'])
        kl.high = float(csvrow['high'])
        kl.low = float(csvrow['low'])
        kl.close = float(csvrow['close'])
        kl.volume = float(csvrow['volume'])
        kl.date = datetime.strptime(csvrow['date'], '%Y/%m/%d').strftime('%Y%m%d')
        kl.time = csvrow['time']+":00"
        
        kl.datetime = datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:%S')
        dataOf = kl.datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:00')
        self.OnKLine(eventType, kl, dataOf)

########################################################################
class McCvsToKLine(CapturedToKLine):

    def __init__(self, sink):
        """Constructor"""
        super(McCvsToKLine, self, sink).__init__(sink)

    @abstractmethod
    def pushCvsRow(self, csvrow, eventType =None, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError

        kl = KLineData('', symbol)
        kl.open = float(csvrow['Open'])
        kl.high = float(csvrow['High'])
        kl.low = float(csvrow['Low'])
        kl.close = float(csvrow['Close'])
        kl.volume = float(csvrow['TotalVolume'])
        kl.date = datetime.strptime(csvrow['Date'], '%Y-%m-%d').strftime('%Y%m%d')
        kl.time = csvrow['Time']+":00"
        
        kl.datetime = datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:%S')
        dataOf = kl.datetime.strptime(kl.date + ' ' + kl.time, '%Y%m%d %H:%M:00')
        self.OnKLine(eventType, kl, dataOf)

########################################################################
class McCvsToEvent(DataToEvent):

    def __init__(self, sink):
        """Constructor"""
        super(McCvsToEvent, self).__init__(sink)
        self._dataToKline = McCvsToKLine(_cbKLine)

    def _cbKLine(self, eventType, kline, dataOf =None):
        self._updateEvent(self, eventType, kline, dataOf)

    @abstractmethod
    def push(self, csvrow, eventType =None, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError
        self._dataToKline.pushCvsRow(csvrow, eventType, symbol)

########################################################################
class TaobaoCvsToEvent(DataToEvent):

    def __init__(self, sink):
        """Constructor"""
        super(TaobaoCvsToEvent, self).__init__(sink)

    @property
    def fields(self) :
        return 'date,time,open,high,low,close,volume,ammount'

    @abstractmethod
    def push(self, csvrow, eventType =None, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError
        else :
            eData = KLineData('', symbol)
            eData.open = float(csvrow['open'])
            eData.high = float(csvrow['high'])
            eData.low = float(csvrow['low'])
            eData.close = float(csvrow['close'])
            eData.volume = float(csvrow['volume'])
            eData.date = datetime.strptime(csvrow['date'], '%Y/%m/%d').strftime('%Y%m%d')
            eData.time = csvrow['time']+":00"
        
        eData.datetime = datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:%S')
        dataOf = eData.datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:00')
        self._cbMarketEvent(eventType, eData, dataOf)

########################################################################
# from vnApp.marketdata.mdHuobi import HuobiToEvent

class mdOffline(MarketData):
    '''
    Loader to import offline data, usually in csv format, to convert and post MarketData into event channel
    the consumer could be either BackTest or DataRecorder
    '''

    EOS = -1 # end of Stream/Sequence

    DataToEventClasses = {
        'MultiCharts'  : McCvsToEvent,
        'shop37077890' : TaobaoCvsToEvent,
#        'huobi'        : HuobiToEvent,
    }

    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        """Constructor
            setting schema:
                {
                    "id": "backtest",
                    "source": "MultiCharts", // Taobao shopXXXX, tushare etc
                    "homeDir": "/tmp/data"
                    "exchange": "huobi", // the original exchange
                    "events": {
                        "tick": "True", // to trigger tick events
                    }
                },
        """

        super(mdOffline, self).__init__(program, settings)

        # self._btApp = btApp
        self._main      = program
        self._dirData  = settings.homeDir('.')
        self._timerStep = settings.timerStep(0)
        try :
            tmp = settings.startDate('2000-01-01')
            self._dtStart = datetime.strptime(tmp, '%Y-%m-%d')
        except :
            self._dtStart = datetime(year=1900, month=1, day=1, hour=0)
        try :
            tmp   = settings.endDate('2999-12-31')
            self._dtEnd = datetime.strptime(tmp, '%Y-%m-%d') +timedelta(hours=23,minutes=59,seconds=59)
        except :
            self._dtEnd = MarketData.DUMMY_DT_EOS

        # filter the csv files
        self._stampRangeinFn = {
            'T': [self._dtStart.strftime('%Y%m%dT000000'), self._dtEnd.strftime('%Y%m%dT000000')],
            'H': ['%dH%d' % (self._dtStart.year, int((self._dtStart.month+5) /6)), '%dH%d' % (self._dtEnd.year, int((self._dtEnd.month+5) /6))],
            'Q': ['%dQ%d' % (self._dtStart.year, int((self._dtStart.month+2) /3)), '%dQ%d' % (self._dtEnd.year, int((self._dtEnd.month+2) /3))],
        }

        self._symbol    = settings.symbol('')
        self._eventType = MARKETDATE_EVENT_PREFIX + settings.event('KL1m')
        self._batchSize = 10
        self._fields = 'date,time,open,high,low,close,volume,ammount'
        self._dataToEvent = None

        classname = settings.className('')
        if classname in mdOffline.DataToEventClasses :
            convertor = mdOffline.DataToEventClasses[classname]
            self._dataToEvent = convertor(self.onMarketEvent)

    @property
    def exchange(self) :
        # for Backtest, we take self._exchange as the source exchange to read
        # and (self._exchange + MarketData.TAG_BACKTEST) as output exhcnage to post into eventChannel
        return self._exchange

    @property
    def fields(self) :
        if self._dataToEvent:
            return self._dataToEvent.fields
        return 'date,time,open,high,low,close,volume,ammount'

    #----------------------------------------------------------------------
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
        for name in fnlist:
            if not self._symbol in name:
                continue

            basename = name.split('/')[-1]
            stk = basename.split('.')
            if stk[-1] =='csv' :
                basename = stk[-2] 
            elif stk[-1] =='bz2' and stk[-2] =='csv':
                basename = stk[-3]
            else : continue

            pos = basename.find(self._symbol)
            stampstr = basename[pos+1 + len(self._symbol):] if pos >=0 else basename
            if 'H' in stampstr:
                trange = self._stampRangeinFn['H']
            elif 'Q' in stampstr :
                trange = self._stampRangeinFn['Q']
            else :
                trange = self._stampRangeinFn['T']
            
            if stampstr < trange[0]:
                prev= name
            elif stampstr <= trange[1]:
                csvfiles.append(name)

        if len(prev) >0 and len(csvfiles) >0 and csvfiles[0] > trange[0]:
            csvfiles = [prev] + csvfiles
        
        return csvfiles, trange

    # @abstractmethod
    # def addFolders_YYYYHh(self, dir): # dateStart, dateEnd, eventType=None, symbol=None) : # return a list of sorted filenames
    #     '''
    #     filter out a sequence of sorted full filenames, 
    #     its first file may includes some data piror to dateStart and its last may include some data after dateEnd
    #     '''
    #     ret = []
    #     for root, subdirs, _ in os.walk(dir):
    #         subdirs.sort()
    #         for d in subdirs:
    #             tks = self._dateStart.split('-')
    #             fldStart = '%sH%s' % (tks[0], 1 if int(tks[1]) <7 else 2)
    #             tks = self._dateEnd.split('-')
    #             fldEnd = '%sH%s' % (tks[0], 1 if int(tks[1]) <7 else 2)
    #             if d < fldStart or d > fldEnd:
    #                 continue

    #             ret += self._symolsInFolder('%s/%s' % (root, d))

    #     ret.sort()
    #     return ret

    # @abstractmethod
    # def _symolsInFolder(self, dir): # dateStart, dateEnd, eventType=None, symbol=None) : # return a list of sorted filenames
    #     '''
    #     filter out a sequence of sorted full filenames, 
    #     its first file may includes some data piror to dateStart and its last may include some data after dateEnd
    #     '''
    #     ret = []
    #     for root, subdirs, files in os.walk(dir):
    #         for name in files:
    #             if self._symbol in name and ('csv' in name or 'bz2' in name) :
    #                 pathname = '%s/%s' %(root, name)
    #                 ret.append(unicode(pathname,'utf-8'))    
    #             # if '1min' in name and '.bz2' in name:
    #             #     ret.append('%s/%s' %(self._homeDir, name))
    #         for d in subdirs:
    #             ret += self._symolsInFolder('%s/%s' % (root, d))

    #     ret.sort()
    #     return ret

    #----------------------------------------------------------------------
    def onMarketEvent(self, event) :
        data = event.data
        if data.datetime != MarketData.DUMMY_DT_EOS:
            if data.datetime < self._dtStart or data.datetime > self._dtEnd:
                return # out of range

        data.exchange = self.exchange
        data.vtSymbol = '%s.%s' % (data.symbol, self.exchange)
        self._main._eventLoop.put(event)
        self.debug('posted %s' % data.desc)

    def connect(self):

        self._seqFiles, trange = self. _filterFiles() # addFolders_YYYYHh(self._homeDir)
        self._idxNextFiles = 0
        self._importStream = None
        self._reader = None
        self.debug('ftrange %s associated files to import: %s' % (trange, self._seqFiles))

    def doAppStep(self):

        bEOS = False
        c =0
        while c < self._batchSize:
            if not self._reader :
                # open the file
                if self._idxNextFiles >= len(self._seqFiles):
                    bEOS = True
                    break

                fn = self._seqFiles[self._idxNextFiles]
                self._idxNextFiles +=1
                self.debug('openning input file %s' % (fn))
                extname = fn.split('.')[-1]
                if extname == 'bz2':
                    self._importStream = bz2.BZ2File(fn, 'r')
                else:
                    self._importStream = file(fn, 'r')
                fields = self.fields.split(',') if self.fields else None
                self._reader = csv.DictReader(self._importStream, fields) if 'csv' in fn else self._importStream
                if not self._reader:
                    self.warn('failed to open input file %s' % (fn))
                    continue

                if len(self._fields) <=0:
                    self._fields = self._reader.headers()

            try :
                line = next(self._reader, None)
                if line: c+=1 
            except:
                line = None

            if not line:
                self.error(traceback.format_exc())
                self._reader = None
                self._importStream.close()
                continue

            try :
                if line and self._dataToEvent:
                    # self.debug('line: %s' % (line))
                    self._dataToEvent.push(line, self._eventType, self._symbol)
            except :
                self.error(traceback.format_exc())

        if bEOS:
            self.info('reached end, queued dummy Event(End), fake an EOS event')
            newSymbol = '%s.%s' % (self._symbol, self.exchange)
            edata = TickData(self._symbol) if 'Tick' in self._eventType else KLineData(self._symbol)
            edata.date = MarketData.DUMMY_DATE_EOS
            edata.time = MarketData.DUMMY_TIME_EOS
            edata.datetime = MarketData.DUMMY_DT_EOS
            edata.exchange  = MarketData.exchange # output exchange name 
            event = Event(self._eventType)
            event.setData(edata)
            self.onMarketEvent(event)
            c +=1
            self.onMarketEvent(event)
            self._main.stop()

        return c


########################################################################
if __name__ == '__main__':
    # loadMcCsv('examples/AShBacktesting/IF0000_1min.csv', MINUTE_DB_NAME, 'IF0000')
    # hd.loadMcCsvBz2('examples/AShBacktesting/IF0000_1min.csv.bz2', MINUTE_DB_NAME, 'IF0000')
    # loadMcCsv('rb0000_1min.csv', MINUTE_DB_NAME, 'rb0000')
    
    print('-'*20)

    # dirname(dirname(abspath(file)))
    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/LD_shop37077890.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        exit(-1)

    me = MainRoutine(settings)

    me.addMarketData(mdOffline, settings['offlinedata'][0])
    recorder = me.createApp(DataRecorder, settings['datarecorder'])  
    # recorder = me.createApp(CsvRecorder, settings['datarecorder'])
    # me.createApp(Zipper, settings['datarecorder'])
    me.info(u'主引擎创建成功')

    me.start()
    me.loop()
