# encoding: UTF-8

'''
'''
from __future__ import division

from Application import BaseApplication, Iterable
from EventData import *
from MarketData import *
'''
from .language import text
'''

import json
import csv
import os
import copy
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import sys
if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty
from pymongo.errors import DuplicateKeyError

EVENT_TOARCHIVE  = EVENT_NAME_PREFIX + 'toArch'

########################################################################
class Recorder(BaseApplication):
    """数据记录, the base DR is implmented as a csv Recorder
        configuration:
            "datarecorder": {
                "dbNamePrefix": "dr", // the preffix of DB name to save: <dbNamePrefix>Tick, <dbNamePrefix>K1min
            }
    """
     
    DEFAULT_DBPrefix = 'dr'

    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        """Constructor"""
        super(Recorder, self).__init__(program, settings)
        self._dbNamePrefix = settings.dbNamePrefix(Recorder.DEFAULT_DBPrefix) if settings else Recorder.DEFAULT_DBPrefix

        # 配置字典
        self._dictDR = OrderedDict() # categroy -> { fieldnames, ....}

        # 负责执行数据库插入的单独线程相关
        self._queRowsToRecord = Queue()  # 队列 of (category, Data)

    def setDataDir(self, dataDir):
        if len(dataDir) > 3:
            self._dataPath = dataDir

    # impl of BaseApplication
    #----------------------------------------------------------------------
    def doAppInit(self): # return True if succ
        if not super(Recorder, self).doAppInit() :
            return False
        return True

    def doAppStep(self):
        cStep =0
        while self.isActive:
            try:
                category, row = self._queRowsToRecord.get(block=False, timeout=0.05)
                self._saveRow(category, row)
                cStep +=1
            except: # Empty:
                break

        return cStep >0

    #----------------------------------------------------------------------
    def pushRow(self, category, row):
        self._queRowsToRecord.put((category, row))

    @abstractmethod
    def configIndex(self, category, definition, unique=False):
        """定义某category collection的index"""
        raise NotImplementedError

    @abstractmethod
    def _saveRow(self, category, dataDict) :
        # coll = self.findCollection(self, category)
        raise NotImplementedError

    def registerCollection(self, category, params= {}):
        '''
           for example recorder.registerCollection(self._recCatgDPosition, params= {'index': [('date', ASCENDING), ('time', ASCENDING)], 'columns' : ['date','symbol']})
        '''
        if not category in self._dictDR.keys() :
            self._dictDR[category] = OrderedDict()
        coll = self._dictDR[category]
        coll['params'] = params
        return coll

    def findCollection(self, category) :
        return self._dictDR[category] if category in self._dictDR.keys() else None

########################################################################
import csv
class CsvRecorder(Recorder):
    """数据记录引擎,
    configuration:
        "datarecorder": {
            ...
            "dbNamePrefix": "dr", // the preffix of DB name to save: <dbNamePrefix>Tick, <dbNamePrefix>K1min

            // for csv recorder
            "min2flush" : 0.3,
            "days2roll" : 1.0,
            "days2archive"  : 0.0028,
        }
    """

    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        """Constructor"""
        super(CsvRecorder, self).__init__(program, settings)

        self._min2flush  = settings.min2flush(1.0)
        self._days2roll  = settings.days2roll(1.0)
        self._days2zip   = settings.days2archive(7.0)
        if self._days2zip < self._days2roll *2:
            self._days2zip = self._days2roll *2

    # impl/overwrite of Recorder
    #----------------------------------------------------------------------
    def registerCollection(self, category, params= {}):
        coll = super(CsvRecorder, self).registerCollection(category, params)

        # perform the csv registration
        pos = category.rfind('/')
        if pos >0:
            dir = self._dbNamePrefix + category[:pos]
            fn  = category[pos+1:]
        else:
            dir =""
            fn = self._dbNamePrefix + category

        coll['dir'] = '%s/%s' % (self.dataRoot, dir)
        coll['fn'] = fn
        coll['f'] = None
        coll['c'] = 0
        coll['o'] = None
        if params:
            coll['params'] = params

        self._checkAndRoll(coll)

        return coll

    @abstractmethod
    def configIndex(self, category, definition, unique=False):
        """定义某category collection的index"""
        pass # nothing to do as csv doesn't support index

    def _saveRow(self, category, row) :
        collection =  self.findCollection(category)
        if not collection:
            self.debug('collection[%s] not registered, ignore' % category)
            return

        if not 'w' in collection.keys():
            colnames = []
            orderedcols = []
            if 'params' in collection and 'columns' in collection['params']:
                orderedcols = collection['params']['columns']
                
            tmp = row.keys()
            for i in orderedcols:
                if i in tmp:
                    colnames.append(i)
                    tmp.remove(i)

            tmp.sort()
            colnames += tmp
            collection['w'] =csv.DictWriter(collection['f'], colnames)

        w = collection['w']
        
        if collection['c'] <=0:
            w.writeheader()
            collection['c'] +=1
            self.debug('catg[%s/%s] header saved' % (collection['dir'], collection['fn']))

        w.writerow(row)
        collection['c'] +=1
        self.debug('catg[%s/%s] row saved: %s' % (collection['dir'], collection['fn'], row))

        self._checkAndRoll(collection)

    # --private methods----------------------------------------------------------------
    def _checkAndRoll(self, collection) :

        dtNow = datetime.now()
        if collection['f'] :
            if not 'flush' in collection.keys() or (collection['flush']+timedelta(minutes=self._min2flush)) < dtNow:
                collection['f'].flush()
                collection['flush'] =dtNow
                self.debug('flushed: %s/%s' % (collection['dir'], collection['fn']))

            if collection['o'] :
                dtToRoll = collection['o']+timedelta(hours=self._days2roll*24)
                if self._days2roll >=1 and (self._days2roll-int(self._days2roll)) <1/24 : # make the roll occur at midnight
                    dtToRoll = datetime(dtToRoll.year, dtToRoll.month, dtToRoll.day, 23, 59, 59, 999999)

                if dtToRoll > dtNow :
                    return collection

                self.debug('rolling %s/%s' % (collection['dir'], collection['fn']))
        
        stampToZip = (dtNow - timedelta(hours=self._days2zip*24)).strftime('%Y%m%dT%H%M%S')
        stampThis   = dtNow.strftime('%Y%m%dT%H%M%S')
        try :
            os.makedirs(collection['dir'])
        except:
            pass

        # check if there are any old data to archive
        for _, _, files in os.walk(collection['dir']):
            for name in files:
                stk = name.split('.')
                if stk[-1] !='csv' or collection['fn'] != name[:len(collection['fn'])]:
                    continue
                if stk[-2] <stampToZip:
                    fn = '%s/%s' % (collection['dir'], name)
                    self.postEvent(EVENT_TOARCHIVE, fn)
                    self.debug('schedule to archive: %s' % fn)

        fname = '%s/%s.%s.csv' % (collection['dir'], collection['fn'], stampThis)
        size =0
        try :
            size = os.fstat(fname).st_size
        except:
            pass
        
        try :
            del collection['w']
        except:
            pass
        collection['f'] = open(fname, 'wb' if size <=0 else 'ab') # Just use 'w' mode in 3.x
        collection['c'] = 0 if size <=0 else 1000 # just a dummy non-zeron
        collection['o'] = dtNow
        self.debug('file[%s] opened: size=%s' % (fname, size))
        return collection

########################################################################
class Playback(Iterable):
    """The reader part of HistoryData
    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    DUMMY_DATE_START = '19900101T000000'
    DUMMY_DATE_END   = '39991231T000000'

    def __init__(self, symbol, startDate =DUMMY_DATE_START, endDate=DUMMY_DATE_END, category =None, exchange=None):
        """Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        """
        super(Playback, self).__init__()
        self._symbol = symbol
        self._category = category if category else EVENT_KLINE_1MIN
        self._exchange = exchange if exchange else 'na'
        self._dbNamePrefix = Recorder.DEFAULT_DBPrefix
        
        self._startDate = startDate if startDate else Playback.DUMMY_DATE_START
        self._endDate   = endDate if endDate else Playback.DUMMY_DATE_END

    # -- impl of Iterable --------------------------------------------------------------
    @abstractmethod
    def resetRead(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        self.__lastMarketClk = None

    @abstractmethod
    def readNext(self):
        '''
        @return next item, mostlikely expect one of Event()
        '''
        if not self.isActive or not self.pendingSize <=0: 
            raise StopIteration

        event = None
        try :
            event = self.popPending(block = False, timeout = 0.1)
        except Exception:
            pass
        return event

    def _testAndGenerateMarketHourEvent(self, ev):
        if self.__lastMarketClk and (self.__lastMarketClk + timedelta(hours=1)) > ev.data.asof:
            return None
        self.__lastMarketClk = ev.data.asof
        self.__lastMarketClk = self.__lastMarketClk.replace(minute=0, second=0, microsecond=0)
        evdMH = EventData()
        evdMH.datetime = self.__lastMarketClk
        evMH = Event(EVENT_MARKET_HOUR)
        evMH.setData(evdMH)
        self.enquePending(evMH)
        return evdMH

########################################################################
class CsvPlayback(Playback):
    DIGITS=set('0123456789')

    #----------------------------------------------------------------------
    def __init__(self, symbol, folder, fields, category =None, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END) :

        super(CsvPlayback, self).__init__(symbol, startDate, endDate, category)
        self._folder = folder
        self._fields = fields

        self._csvfiles =[]
        self._cvsToEvent = DictToKLine(self._category, symbol)
        # self._merger1minTo5min =None
        self._merger1minTo5min = KlineToXminMerger(self._cbMergedKLine5min, xmin=5)
        self._merger5minTo1Day = KlineToXminMerger(self._cbMergedKLine1Day, xmin=60*24-10)
        self._dtEndOfDay = None

#        if not self._fields and 'mdKL' in self._category:
#            self._fields ='' #TODO
        self._fieldnames = self._fields.split(',') if self._fields else None

    def _cbMergedKLine5min(self, klinedata):
        if not klinedata: return
        ev = Event(EVENT_KLINE_5MIN)
        ev.setData(klinedata)
        self.enquePending(ev)
        asofDay = klinedata.asof.replace(hour=23, minute=59, second=59)
        if self._merger5minTo1Day :
            self._merger5minTo1Day.pushKLineData(klinedata, asofDay)

    def _cbMergedKLine1Day(self, klinedata):
        if not klinedata: return
        ev = Event(EVENT_KLINE_1DAY)
        # klinedata.datetime = klinedata.datetime.replace(hour=23, minute=59, second=59)
        ev.setData(klinedata)
        self.enquePending(ev)

    # -- Impl of Playback --------------------------------------------------------------
    def resetRead(self):
        super(CsvPlayback, self).resetRead()

        self._csvfiles =[]
        self._reader =None

        # # filter the csv files
        prev = ""
        for _, _, files in os.walk(self._folder):
            files.sort()
            for name in files:
                stk = name.split('.')
                if not 'csv' in stk or self._symbol.lower() != name[:len(self._symbol)].lower():
                    continue

                fn = '%s/%s' % (self._folder, name)
                stampstr = stk[0][len(self._symbol):]
                pos = next((i for i, ch  in enumerate(stampstr) if ch in CsvPlayback.DIGITS), None)
                if not pos :
                    self._csvfiles.append(fn)
                    continue

                if 'Y' == stampstr[pos] : pos +=1
                stampstr = stampstr[pos:]
                if stampstr < self._startDate :
                    prev = fn
                elif stampstr <= self._endDate:
                    self._csvfiles.append(fn)

        if len(prev) >0:
            self._csvfiles = [prev] + self._csvfiles
        
        return len(self._csvfiles) >0

    def readNext(self):
        '''
        @return True if busy at this step
        '''
        try :
            ev = self.popPending(block = False, timeout = 0.1)
            if ev: return ev
        except Exception:
            pass

        row = None
        while not row:
            while not self._reader:
                if not self._csvfiles or len(self._csvfiles) <=0:
                    self._iterableEnd = True
                    return None

                fn = self._csvfiles[0]
                del(self._csvfiles[0])

                self.debug('openning input file %s' % (fn))
                extname = fn.split('.')[-1]
                if extname == 'bz2':
                    self._importStream = bz2.open(fn, mode='rt') # bz2.BZ2File(fn, 'rb')
                else:
                    self._importStream = file(fn, 'rt')

                self._reader = csv.DictReader(self._importStream, self._fieldnames, lineterminator='\n') if 'csv' in fn else self._importStream
                if not self._reader:
                    self.warn('failed to open input file %s' % (fn))

            if not self._reader:
                self._iterableEnd = True
                return None

            # if not self._fieldnames or len(self._fieldnames) <=0:
            #     self._fieldnames = self._reader.headers()

            try :
                row = next(self._reader, None)
            except Exception as ex:
                row = None

            if not row:
                # self.error(traceback.format_exc())
                self._reader = None
                self._importStream.close()

        ev = None
        try :
            if row and self._cvsToEvent:
                # print('line: %s' % (line))
                ev = self._cvsToEvent.convert(row, self._exchange, self._symbol)
                if ev:
                    evdMH = self._testAndGenerateMarketHourEvent(ev)
                    if  self._merger1minTo5min :
                        self._merger1minTo5min.pushKLineEvent(ev)

                    if evdMH :
                        if self._dtEndOfDay and self._dtEndOfDay < evdMH.asof :
                            if  self._merger1minTo5min :
                                self._merger1minTo5min.flush()
                            if  self._merger5minTo1Day :
                                self._merger5minTo1Day.flush()
                            self._dtEndOfDay = None

                        if not self._dtEndOfDay :
                            self._dtEndOfDay = evdMH.asof.replace(hour=23,minute=59,second=59)

                    # because the generated is always as of the previous event, so always deliver those pendings in queue first
                    if self.pendingSize >0 :
                        evout = self.popPending()
                        self.enquePending(ev)
                        return evout

        except Exception as ex:
            self.logexception(ex)

        return ev

########################################################################
class MongoRecorder(Recorder):
    """数据记录引擎
    """
    #----------------------------------------------------------------------
    def __init__(self, program, settings):
        """Constructor"""
        super(MongoRecorder, self).__init__(program, settings)

    # impl/overwrite of Recorder
    #----------------------------------------------------------------------
    def registerCollection(self, category, params= {}):
        coll = super(MongoRecorder, self).registerCollection(category, params)

        # perform the registration
        pos = category.rfind('/')
        if pos >0:
            dbName = self._dbNamePrefix + category[:pos]
            cn  = category[pos+1:]
        else:
            dbName = self._dbNamePrefix + self._id
            cn =  category

        coll['dbName'] = dbName
        coll['collectionName'] = cn
        if params:
            coll['params'] = params
            if 'index' in params.keys():
                # self.configIndex(collectionName, [('date', ASCENDING), ('time', ASCENDING)], True, dbName) #self._dbNamePrefix +e
                self.configIndex(cn, params['index'], True, dbName) #self._dbNamePrefix +e

        return coll

    def _saveRow(self, category, row) :
        collection =  self.findCollection(category)
        if not collection:
            self.debug('collection[%s] not registered, ignore' % category)
            return

        try:
            self.dbInsert(collection['collectionName'], row, collection['dbName'])
            self.debug('DB %s[%s] inserted: %s' % (collection['dbName'], collection['collectionName'], row))
        except DuplicateKeyError:
            self.error('键值重复插入失败：%s' %traceback.format_exc())

    @abstractmethod
    def configIndex(self, category, definition, unique=False):
        """定义某category collection的index"""
        # TODO

########################################################################
class MongoPlayback(Playback):

    def __init__(self, program, settings, symbol=None):
        super(MongoPlayback, self).__init__(program, settings)

    #--- impl of BaseApplication -----------------------
    def doAppStep(self):
        '''
        @return True if busy at this step
        '''
        if not self._lst :
            return False
        
        try :
            data = next(lst)
        except Exception:
            self._lst = None
            return True
        
        ev = KLineData(node['ds'], symbol)
        ev.__dict__ = data
            # if self._cvsToEvent:
            #     # self.debug('line: %s' % (line))
            #     ev = self._cvsToEvent.pushCvsRow(line, self._category, self._exchange, self._symbol)
        self.__queue.put(ev)
        return True

    # -- Impl of Playback --------------------------------------------------------------
    def resetRead(self):
        category = self._category[len(MARKETDATE_EVENT_PREFIX):]
        stampSince = self._startDate
        stampTill  = self._endDate

        if not category in self._dictDR.keys() or not symbol in self._dictDR[category].keys():
            return False

        node = self._dictDR[category][symbol]
        if not 'coll' in node.keys() :
            return False

        flt = {'datetime':{'$gte':startDate}}
        self._lst = self.dbQuery(node['coll']['collectionName'], flt, 'datetime', ASCENDING, node['coll']['dbName'])

########################################################################
class MarketRecorder(BaseApplication):
    """数据记录引擎, the base DR is implmented as a csv Recorder"""
     
    #close,date,datetime,high,low,open,openInterest,time,volume
    #,date,datetime,high,price,volume,low,lowerLimit,openInterest,open,prevClose,time,upperLimit,volume
    CSV_LEADING_COLUMNS=['datetime','price','close', 'volume', 'high','low','open']

    #----------------------------------------------------------------------
    def __init__(self, program, settings, recorder=None):
        """Constructor"""

        super(MarketRecorder, self).__init__(program, settings)

        if recorder :
            self._recorder = recorder
        else : 
            rectype = settings.type('csv')
            if rectype == 'mongo' :
                self._recorder = MongoRecorder(program, settings)
            else:
                self._recorder = CsvRecorder(program, settings)

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def doAppInit(self): # return True if succ
        if not self._recorder:
            return False
        return self._recorder.doAppInit()

    @abstractmethod
    def start(self):
        if not self._recorder:
            return
        
        self._recorder.start()
        
        # Tick记录配置
        self._subscribeMarketData(self._settings.ticks, MarketData.EVENT_TICK)
        # 分钟线记录配置
        self._subscribeMarketData(self._settings.kline1min, MarketData.EVENT_KLINE_1MIN)

    @abstractmethod
    def doAppStep(self):
        if not self._recorder:
            return 0
        return self._recorder.step()

    #----------------------------------------------------------------------
    def _subscribeMarketData(self, settingnode, eventType): # , dict, cbFunc) :
        if not self._recorder or len(settingnode({})) <=0:
            return

        eventCatg = eventType[len(MARKETDATE_EVENT_PREFIX):]
        for i in settingnode :
            try:
                symbol = i.symbol('')
                category='%s/%s' % ( eventCatg, symbol)
                if len(symbol) <=3 or self._recorder.findCollection(category):
                    continue

                ds = i.ds('')
                if len(ds) >0 :
                    dsrc = self._engine.getMarketData(ds)
                    if dsrc:
                        dsrc.subscribe(symbol, eventType)

                self.subscribeEvent(eventType, self.onMarketEvent)
                self._recorder.registerCollection(category, params= {'ds': ds, 'index': [('date', ASCENDING), ('time', ASCENDING)], 'columns' : self.CSV_LEADING_COLUMNS})

            except Exception as e:
                self.logexception(e)

    #----------------------------------------------------------------------
    def onMarketEvent(self, event):
        """处理行情事件"""
        eventType = event.type

        if  MARKETDATE_EVENT_PREFIX != eventType[:len(MARKETDATE_EVENT_PREFIX)] :
            return

        category = eventType[len(MARKETDATE_EVENT_PREFIX):]
        eData = event.data # this is a TickData or KLineData
        # if tick.sourceType != MarketData.DATA_SRCTYPE_REALTIME:
        if MarketData.TAG_BACKTEST in eData.exchange :
            return

        category = '%s/%s' % (category, eData.symbol)
        collection= self._recorder.findCollection(category)
        if not collection:
            return

        if not eData.datetime : # 生成datetime对象
            try :
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S')
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S.%f')
            except:
                pass

        self.debug('On%s: %s' % (category, eData.desc))

        row = eData.__dict__
        try :
            del row['exchange']
            del row['symbol']
            del row['vtSymbol']
        except:
            pass

        self._recorder.pushRow(category, row)
    
########################################################################
import bz2

class Zipper(BaseApplication):
    """数据记录引擎"""
    def __init__(self, program, settings):
        super(Zipper, self).__init__(program, settings)
        self._threadWished = True  # this App must be Threaded

        self._queue = Queue()                    # 队列
        self.subscribeEvent(EVENT_TOARCHIVE, self.onToArchive)

    # impl of BaseApplication
    #----------------------------------------------------------------------
    def doAppStep(self):
        cStep =0
        while self.isActive:
            try:
                fn = self._queue.get(block=True, timeout=0.2)
                f = file(fn, 'rb')
                ofn = fn + '.bz2'
                try :
                    os.fstat(ofn)
                    continue # output file exists, skip
                except:
                    pass

                self.debug('zipping: %s to %s' % (fn, ofn))
                everGood = False
                with f:
                    with bz2.BZ2File(ofn, 'w') as z:
                        while True:
                            data = None
                            data = f.read(1024000)
                            if data != None :
                                everGood = True
                            if len(data) <=0 : break
                            z.write(data)

                if everGood: os.remove(fn)
                cStep +=1
            except: # Empty:
                break

        return False # this doesn't matter as this is threaded

    def onToArchive(self, event) :
        self._push(event.data)

    def _push(self, filename) :
        self._queue.put(filename)


