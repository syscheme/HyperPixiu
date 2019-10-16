# encoding: UTF-8

'''
'''
from __future__ import division

from Application import BaseApplication
from EventData import *
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
import sys
if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty
from pymongo.errors import DuplicateKeyError

EVENT_TOARCHIVE  = EVENT_NAME_PREFIX + 'toArch'

########################################################################
class DataRecorder(BaseApplication):
    """数据记录引擎, the base DR is implmented as a csv Recorder
        configuration:
            "datarecorder": {
                "dbNamePrefix": "dr", // the preffix of DB name to save: <dbNamePrefix>Tick, <dbNamePrefix>K1min
            }
    """
     
    DEFAULT_DBPrefix = 'dr'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(DataRecorder, self).__init__(mainRoutine, settings)
        self._dbNamePrefix = settings.dbNamePrefix(DataRecorder.DEFAULT_DBPrefix)

        # 配置字典
        self._dictDR = OrderedDict()

        # 负责执行数据库插入的单独线程相关
        self._queRowsToRecord = Queue()  # 队列 of (category, Data)

    def setDataDir(self, dataDir):
        if len(dataDir) > 3:
            self._dataPath = dataDir

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ
        return super(DataRecorder, self).init()

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self):
        cStep =0
        while True:
            try:
                category, row = self._queRowsToRecord.get(block=False, timeout=0.05)
                self.saveRow(category, row)
                cStep +=1
            except: # Empty:
                break
        return cStep

    #----------------------------------------------------------------------
    def pushRow(self, category, row):
        self._queRowsToRecord.put((category, row))

    @abstractmethod
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

    @abstractmethod
    def saveRow(self, category, dataDict) :
        coll = self.findCollection(self, category)
        pass

########################################################################
import csv
class CsvRecorder(DataRecorder):
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
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(CsvRecorder, self).__init__(mainRoutine, settings)

        self._min2flush  = settings.min2flush(1.0)
        self._days2roll  = settings.days2roll(1.0)
        self._days2zip   = settings.days2archive(7.0)
        if self._days2zip < self._days2roll *2:
            self._days2zip = self._days2roll *2

    #----------------------------------------------------------------------
    # impl of DataRecorder
    #----------------------------------------------------------------------
    @abstractmethod
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
    def saveRow(self, category, row) :
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

    #----------------------------------------------------------------------
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

    @abstractmethod
    def filterCollections(self, symbol, startDate, endDate=None, eventType =MarketData.EVENT_KLINE_1MIN):
        """从数据库中读取Bar数据，startDate是datetime对象"""

        mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
        csvfiles =[]
        if not mdEvent in self._dictDR.keys() or not symbol in self._dictDR[mdEvent].keys():
            return csvfiles

        node = self._dictDR[mdEvent][symbol]
        if not 'coll' in node.keys() :
            return csvfiles

        # filter the csv files
        stampSince = startDate.strftime('%Y%m%dT000000')
        stampTill  = (endDate +timedelta(hours=24)).strftime('%Y%m%dT000000') if endDate else '39991231T000000'

        collection = node['coll']
        prev = ""
        for _, _, files in os.walk(collection['dir']):
            files.sort()
            for name in files:
                stk = name.split('.')
                if stk[-1] !='csv' or collection['fn'] != name[:len(collection['fn'])]:
                    continue
                fn = '%s/%s' % (collection['dir'], name)
                if stk[-2] <stampSince :
                    prev = fn
                elif stk[-2] <stampTill:
                    csvfiles.append(fn)

        if len(prev) >0:
            csvfiles = [prev] + csvfiles

        return csvfiles, stampSince, stampTill

########################################################################
class MongoRecorder(DataRecorder):
    """数据记录引擎
    """
    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(MongoRecorder, self).__init__(mainRoutine, settings)

    #----------------------------------------------------------------------
    # impl of DataRecorder
    #----------------------------------------------------------------------
    @abstractmethod
    def registerCollection(self, category, params= {}):
        coll = super(MongoRecorder, self).registerCollection(category, params)

        # perform the csv registration
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
                # self.dbEnsureIndex(collectionName, [('date', ASCENDING), ('time', ASCENDING)], True, dbName) #self._dbNamePrefix +e
                self.dbEnsureIndex(cn, params['index'], True, dbName) #self._dbNamePrefix +e

        return coll

    @abstractmethod
    def saveRow(self, category, row) :
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
    def loadRecentMarketData(self, symbol, startDate, eventType =MarketData.EVENT_KLINE_1MIN):
        """从数据库中读取Bar数据，startDate是datetime对象"""

        mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
        if not mdEvent in self._dictDR.keys() or not symbol in self._dictDR[mdEvent].keys():
            return []

        node = self._dictDR[mdEvent][symbol]
        if not 'coll' in node.keys() :
            return []
            
        flt = {'datetime':{'$gte':startDate}}
        lst = self.dbQuery(node['coll']['collectionName'], flt, 'datetime', ASCENDING, node['coll']['dbName'])
        
        ret = []
        for data in lst:
            v = KLineData(node['ds'], symbol)
            v.__dict__ = data
            ret.append(v)
        return ret


########################################################################
class MarketRecorder(BaseApplication):
    """数据记录引擎, the base DR is implmented as a csv Recorder"""
     
    #close,date,datetime,high,low,open,openInterest,time,volume
    #,date,datetime,high,price,volume,low,lowerLimit,openInterest,open,prevClose,time,upperLimit,volume
    CSV_LEADING_COLUMNS=['datetime','price','close', 'volume', 'high','low','open']

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings, recorder=None):
        """Constructor"""

        super(MarketRecorder, self).__init__(mainRoutine, settings)

        if recorder :
            self._recorder = recorder
        else : 
            rectype = settings.type('csv')
            if rectype == 'mongo' :
                self._recorder = MongoRecorder(mainRoutine, settings)
            else:
                self._recorder = CsvRecorder(mainRoutine, settings)

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ
        if not self._recorder:
            return False
        return self._recorder.init()

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
    def step(self):
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
        eventType = event.type_

        if  MARKETDATE_EVENT_PREFIX != eventType[:len(MARKETDATE_EVENT_PREFIX)] :
            return

        mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
        eData = event.dict_['data'] # this is a TickData or KLineData
        # if tick.sourceType != MarketData.DATA_SRCTYPE_REALTIME:
        if MarketData.TAG_BACKTEST in eData.exchange :
            return

        category = '%s/%s' % (mdEvent, eData.symbol)
        collection= self._recorder.findCollection(category)
        if not collection:
            return

        if not eData.datetime : # 生成datetime对象
            try :
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S')
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S.%f')
            except:
                pass

        self.debug('On%s: %s' % (mdEvent, eData.desc))

        row = eData.__dict__
        try :
            del row['exchange']
            del row['symbol']
            del row['vtSymbol']
        except:
            pass

        self._recorder.pushRow(category, row)
    
    # @abstractmethod
    # def filterCollections(self, symbol, startDate, endDate=None, eventType =MarketData.EVENT_KLINE_1MIN):
    #     """从数据库中读取Bar数据，startDate是datetime对象"""

    #     mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
    #     csvfiles =[]
    #     if not mdEvent in self._dictDR.keys() or not symbol in self._dictDR[mdEvent].keys():
    #         return csvfiles

    #     node = self._dictDR[mdEvent][symbol]
    #     if not 'coll' in node.keys() :
    #         return csvfiles

    #     # filter the csv files
    #     stampSince = startDate.strftime('%Y%m%dT000000')
    #     stampTill  = (endDate +timedelta(hours=24)).strftime('%Y%m%dT000000') if endDate else '39991231T000000'

    #     collection = node['coll']
    #     prev = ""
    #     for _, _, files in os.walk(collection['dir']):
    #         files.sort()
    #         for name in files:
    #             stk = name.split('.')
    #             if stk[-1] !='csv' or collection['fn'] != name[:len(collection['fn'])]:
    #                 continue
    #             fn = '%s/%s' % (collection['dir'], name)
    #             if stk[-2] <stampSince :
    #                 prev = fn
    #             elif stk[-2] <stampTill:
    #                 csvfiles.append(fn)

    #     if len(prev) >0:
    #         csvfiles = [prev] + csvfiles

    #     return csvfiles, stampSince, stampTill

    # @abstractmethod
    # def loadMarketData(self, symbol, startDate, endDate=None, eventType =MarketData.EVENT_KLINE_1MIN):
    #     """从数据库中读取Bar数据，startDate是datetime对象"""

    #     mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
    #     csvfiles, stampSince, stampTill = self.filterCollections(symbol, startDate, endDate, eventType)
    #     ret = []
    #     if len(csvfiles) <=0:
    #         return ret

    #     DataClass = TickData if mdEvent == 'Tick' else KLineData

    #     for fn in csvfiles :
    #         with open(fn, 'rb') as f:
    #             reader = csv.DictReader(f)
    #             for row in reader:
    #                 if fn == csvfiles[0] and row['datetime'] <stampSince:
    #                     continue
    #                 if fn == csvfiles[-1] and ['datetime'] >=stampTill:
    #                     break
    #                 data = DataClass('')
    #                 data.__dict__  = row
    #                 ret.append(data)
    #     return ret

########################################################################
import bz2

class Zipper(BaseApplication):
    """数据记录引擎"""
    def __init__(self, mainRoutine, settings):
        super(Zipper, self).__init__(mainRoutine, settings)
        self._queue = Queue()                    # 队列
        self.subscribeEvent(EVENT_TOARCHIVE, self.onToArchive)

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def step(self):
        cStep =0
        while True:
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

        return cStep

    def onToArchive(self, event) :
        self._push(event.dict_['data'])

    def _push(self, filename) :
        self._queue.put(filename)



########################################################################
class IterableData(object):
    """The reader part of HistoryData
    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    def __init__(self):
        """Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        """
        super(IterableData, self).__init__()
        self._generator = self.generate()

    def __iter__(self):
        if not self._generator :
            raise NotImplementedError()
        return self

    @abstractmethod
    def generate(self):
        # dummyrow = [2.0]*5
        i=0
        while True:
            if i >10:
                raise StopIteration
            yield np.array([np.random.normal(scale=10)]*5, dtype=np.float)
            i+=1
 
    def __next__(self):
        if not self._generator :
            raise NotImplementedError()
        return next(self._generator)

    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        print("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

    @abstractmethod
    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        pass
