# encoding: UTF-8

from __future__ import division

from vnApp.MainRoutine import *
from vnApp.Account import *
from vnApp.MarketData import *
from vnApp.EventChannel import *
from vnApp.DataRecorder import *

from copy import copy
from datetime import datetime
from threading import Thread
from Queue import Queue, Empty
from multiprocessing.dummy import Pool
from time import sleep
from datetime import datetime

import json

########################################################################
class McCvsToEvent(DataToEvent):

    def __init__(self, sink):
        """Constructor"""
        super(McCvsToEvent, self).__init__(sink)

    @abstractmethod
    def push(self, csvrow, eventType =None, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError
        else :
            eData = KLineData(symbol)
            eData.open = float(csvrow['Open'])
            eData.high = float(csvrow['High'])
            eData.low = float(csvrow['Low'])
            eData.close = float(csvrow['Close'])
            eData.volume = csvrow['TotalVolume']
            eData.date = datetime.strptime(csvrow['Date'], '%Y-%m-%d').strftime('%Y%m%d')
            eData.time = csvrow['Time']
            eData.datetime = datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:%S')
            dataOf = eData.datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:00')
        self._updateEvent(self, eventType, eData, dataOf)

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
from vnApp.marketdata.mdHuobi import HuobiToEvent

class mdOffline(MarketData):
    '''
    Loader to import offline data, usually in csv format, to convert and post MarketData into event channel
    the consumer could be either BackTest or DataRecorder
    '''

    EOS = -1 # end of Stream/Sequence

    DataToEventClasses = {
        'MultiCharts'  : McCvsToEvent,
        'shop37077890' : TaobaoCvsToEvent,
        'huobi'        : HuobiToEvent,
    }

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
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

        super(mdOffline, self).__init__(mainRoutine, settings)

        # self._btApp = btApp
        self._main = mainRoutine
        self._homeDir   = settings.homeDir('.')
        self._timerStep = settings.timerStep(0)
        self._dateStart = settings.startDate('2000-01-01')
        self._dateEnd   = settings.endDate('2999-12-31')
        self._symbol   = settings.symbol('')
        self._eventType   = MARKETDATE_EVENT_PREFIX + settings.event('KL1m')
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
    @abstractmethod
    def addFolders_YYYYHh(self, dir): # dateStart, dateEnd, eventType=None, symbol=None) : # return a list of sorted filenames
        '''
        filter out a sequence of sorted full filenames, 
        its first file may includes some data piror to dateStart and its last may include some data after dateEnd
        '''
        ret = []
        for root, subdirs, _ in os.walk(dir):
            subdirs.sort()
            for d in subdirs:
                tks = self._dateStart.split('-')
                fldStart = '%sH%s' % (tks[0], 1 if int(tks[1]) <7 else 2)
                tks = self._dateEnd.split('-')
                fldEnd = '%sH%s' % (tks[0], 1 if int(tks[1]) <7 else 2)
                if d < fldStart or d > fldEnd:
                    continue

                ret += self._symolsInFolder('%s/%s' % (root, d))

        ret.sort()
        return ret

    @abstractmethod
    def _symolsInFolder(self, dir): # dateStart, dateEnd, eventType=None, symbol=None) : # return a list of sorted filenames
        '''
        filter out a sequence of sorted full filenames, 
        its first file may includes some data piror to dateStart and its last may include some data after dateEnd
        '''
        ret = []
        for root, subdirs, files in os.walk(dir):
            for name in files:
                if self._symbol in name and ('csv' in name or 'bz2' in name) :
                    pathname = '%s/%s' %(root, name)
                    ret.append(unicode(pathname,'utf-8'))    
                # if '1min' in name and '.bz2' in name:
                #     ret.append('%s/%s' %(self._homeDir, name))
            for d in subdirs:
                ret += self._symolsInFolder('%s/%s' % (root, d))

        ret.sort()
        return ret

    #----------------------------------------------------------------------
    def onMarketEvent(self, event) :
        data = event.dict_['data']
        if data.datetime != MarketData.DUMMY_DT_EOS:
            datestr = data.datetime.strftime('%Y-%m-%d')
            if datestr < self._dateStart or datestr > self._dateEnd:
                return # out of range

        data.exchange = self.exchange
        self._main._eventChannel.put(event)
        self.debug('posted %s' % data.desc)

    def connect(self):
        self._seqFiles = self.addFolders_YYYYHh(self._homeDir)
        self._idxNextFiles = 0
        self._importStream = None
        self._reader = None
        self.debug('files to import: %s' % self._seqFiles)

    def step(self):

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
                line = None
                line = self._reader.next()
                c+=1
            except :
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
            event.dict_['data'] = edata
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
    
    srcDataHome=u'/bigdata/sourcedata/shop37077890.taobao/股票1分钟csv'
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
 #   recorder = me.addApp(DataRecorder, settings['datarecorder'])
    recorder = me.addApp(CsvRecorder, settings['datarecorder'])
 #   me.addApp(Zipper, settings['datarecorder'])
    me.info(u'主引擎创建成功')

    me.start()
    me.loop()

    input()


    # hd.loadTaobaoCsvBz2(srcDataHome +'/2012.7-2012.12/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2012.1-2012.6/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2011.7-2011.12/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2011.1-2011.6/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    folders = ['2012H2', '2012H1', '2011H2', '2011H1']
    # symbols= ["601607","601611","601618","601628","601633","601668","601669","601688",
    #     "601718","601727","601766","601788","601800","601808","601818","601828","601838","601857",
    #     "601866","601877","601878","601881","601888","601898","601899","601901","601919","601933",
    #     "601939","601958","601985","601988","601989","601991","601992","601997","601998","603160",
    #     "603260","603288","603799","603833","603858","603993" ]

    symbols= ["601000"]

    for s in symbols :
        for f in folders :
            try :
                csvfn = '%s/SH%s.csv.bz2' % (f, s)
                sym = 'A%s' % s
                hd.loadTaobaoCsvBz2(srcDataHome +'/'+csvfn, MINUTE_DB_NAME, sym)
            except :
                pass



 