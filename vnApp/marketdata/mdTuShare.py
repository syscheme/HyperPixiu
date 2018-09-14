# encoding: UTF-8

from __future__ import division

from vnApp.MainRoutine import *
from vnApp.Account import *
from vnApp.MarketData import *
from vnApp.EventChannel import *
from vnApp.DataRecorder import *

from copy import copy
from datetime import datetime
from Queue import Queue, Empty
from time import time, sleep
from datetime import datetime
import errno

import json
import shutil
import tushare as ts # pip install tushare
import tarfile

########################################################################


config = open('config.json')
setting = json.load(config)

SYMBOLS = setting['SYMBOLS']

#----------------------------------------------------------------------
def downloadTodayData():
    """download all the txn happened today of SYMBOLS"""

    cwd = os.getcwd()
    stampStart = time()

    # step 1. folder of today
    shutil.rmtree('tmp', ignore_errors=True) 
    mkdir_p('tmp')
    os.chdir('tmp')
    day_asof = datetime.today().strftime('%Y%m%d')
    path = "ts" + day_asof
    mkdir_p('./' + path)

    # step 1.1 index data
    for i in range(1,16):
        try:
            ts.get_index().to_csv(path+"/index_ov.csv", header=True, index=False, encoding='utf-8')
            break
        except IOError as exc:  # Python >2.5
            sleep(30*i)
            pass

    # step 1.2 all ticks' day overview
    for i in range(1,16):
        try:
            ts.get_today_all().to_csv(path+"/all.csv", header=True, index=False, encoding='utf-8')
            break
        except IOError as exc:  # Python >2.5
            sleep(30*i)
            pass

    # step 2. create the folder of the date
    mkdir_p(path + "/txn")
    for symbol in SYMBOLS:
        downloadTxnsBySymbol(symbol, folder=path +"/txn")

    # step 3. create the folder of the date
    mkdir_p(path + "/sina")

    # step 4. compress the tar ball
    tar = tarfile.open(cwd +'/ts' + day_asof + '.tar.bz2', 'w:bz2')
    tar.add(path)
    tar.close()

    elapsed = (time() - stampStart) * 1000

    print("took %dmsec to download today's data" % elapsed)
    os.chdir(cwd)


#----------------------------------------------------------------------
def downloadTxnsBySymbol(symbol, folder):
    """download all the txn happened today"""
    for i in range(1,16):
        try:
            df = ts.get_today_ticks(symbol)
            df.to_csv(folder + "/" + symbol+'.csv', header=True, index=False, encoding='utf-8')
            break
        except IOError as exc:  # Python >2.5
            sleep(30*i)
            pass

def mkdir_p(path):
    try:
        os.makedirs(path, 0777)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

downloadTodayData()

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
        super(TaobaoCvsToEvent, self).__init__(sink = None)

    @property
    def fields(self) :
        return 'date,time,open,high,low,close,volume,ammount'

    @abstractmethod
    def push(self, csvrow, eventType =None, symbol =None) :
        if eventType == MarketData.EVENT_TICK :
            raise NotImplementedError
        else :
            eData = KLineData(symbol)
            eData.open = float(csvrow['open'])
            eData.high = float(csvrow['high'])
            eData.low = float(csvrow['low'])
            eData.close = float(csvrow['close'])
            eData.volume = csvrow['volume']
            eData.date = datetime.strptime(csvrow['date'], '%Y/%m/%d').strftime('%Y%m%d')
            eData.time = csvrow['time']+":00"
        
        eData.datetime = datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:%S')
        dataOf = eData.datetime.strptime(eData.date + ' ' + eData.time, '%Y%m%d %H:%M:00')
        self._updateEvent(self, eventType, eData, dataOf)

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
    def filteredFiles(self, dateStart, dateEnd, eventType=None, symbol=None) : # return a list of sorted filenames
        '''
        filter out a sequence of sorted full filenames, 
        its first file may includes some data piror to dateStart and its last may include some data after dateEnd
        '''
        ret = []
        for _, _, files in os.walk(self._homeDir):
            for name in files:
                if '1min' in name and '.bz2' in name:
                    ret.append('%s/%s' %(self._homeDir, name))

        ret.sort()
        return ret

    #----------------------------------------------------------------------
    def onMarketEvent(self, event) :
        event.dict_['data'].exchange = self.exchange
        self._main._eventChannel.put(event)
        self.debug('posted %s' % event.dict_['data'].desc)

    def connect(self):
        self._seqFiles = self.filteredFiles(self._dateStart, self._dateEnd)
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
                self._reader = csv.DictReader(self._importStream, self.fields) if 'csv' in fn else self._importStream
                if not self._reader:
                    self.warn('failed to open input file %s' % (fn))
                    continue

                if len(self._fields) <=0:
                    self._fields = self._reader.headers()

            try :
                line = None
                line = self._reader.next()
            except :
                self.error(traceback.format_exc())
                self._reader.close()
                self._reader = None
                continue

            try :
                if line and self._dataToEvent:
                    self.debug('line: %s' % (line))
                    self._dataToEvent.push(line)
            except :
                self.error(traceback.format_exc())

        if bEOS:
            self.info('reached end, queued dummy Event(End), fake an EOS event')
            # newSymbol = '%s.%s' % (symbol, self.exchange)
            # edata = TickData(self._symbol) if 'Tick' in self._eventType else KLineData(self._symbol)
            # edata.date = MarketData.DUMMY_DATE_EOS
            # edata.time = MarketData.DUMMY_TIME_EOS
            # edata.datetime = MarketData.DUMMY_DT_EOS
            # edata.exchange  = MarketData.exchange # output exchange name 
            # event = Event(self._eventType)
            # event.dict_['data'] = edata
            # self.postMarketEvent(event); c +=1

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
    # recorder = me.createApp(DataRecorder, settings['datarecorder'])
    recorder = me.createApp(CsvRecorder, settings['datarecorder'])
    me.createApp(Zipper, settings['datarecorder'])
    me.info(u'主引擎创建成功')

    me.start()
    startDate=datetime.now() - timedelta(60)
    data = recorder.loadRecentMarketData('ethusdt', startDate)
    data = recorder.loadRecentMarketData('ethusdt', startDate, MarketData.EVENT_TICK)
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



 