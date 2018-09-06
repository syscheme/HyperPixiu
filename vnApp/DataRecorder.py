# encoding: UTF-8

'''
本文件中实现了行情数据记录引擎，用于汇总TICK数据，并生成K线插入数据库。

使用DR_setting.json来配置需要收集的合约，以及主力合约代码。
'''
from __future__ import division

from .MainRoutine import *
from .MarketData import *
from .EventChannel import datetime2float
from .language import text

import json
import csv
import os
import copy
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from Queue import Queue, Empty
from pymongo.errors import DuplicateKeyError
import shelve

EVENT_TOARCHIVE  = EVENT_NAME_PREFIX + 'toArch'

########################################################################
class DataRecorder(BaseApplication):
    """数据记录引擎"""
     
    DEFAULT_DBPrefix = 'dr'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(DataRecorder, self).__init__(mainRoutine, settings)

        self._dbNamePrefix = settings.dbNamePrefix(self.DEFAULT_DBPrefix)
        self._dataPath     = settings.dataPath('./data')
        
        # 配置字典
        self._dictDR = OrderedDict()

        # 负责执行数据库插入的单独线程相关
        self._queueMarketData = Queue()                    # 队列
        # self.thread = Thread(target=self.run)   # 线程

    #----------------------------------------------------------------------
    def _subscribeMarketData(self, settingnode, eventType): # , dict, cbFunc) :
        if len(settingnode({})) <=0:
            return

        mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
        if not mdEvent in self._dictDR.keys() :
            self._dictDR[mdEvent] = OrderedDict()
        dict = self._dictDR[mdEvent]

        for i in settingnode :
            try:
                symbol = i.symbol('')
                ds = i.ds('')
                if len(symbol) <=3 or len(ds)<=0 or symbol in dict:
                    continue

                self._engine.getMarketData(ds).subscribe(symbol, eventType)
                if len(dict) <=0:
                    self.subscribeEvent(eventType, self.onMarketEvent)

                if symbol in dict:
                    continue

                # 保存到配置字典中
                d = {
                        'symbol': symbol,
                        'ds': ds,
                    }

                dict[symbol] = d
            except Exception as e:
                self.logexception(e)

    def subscribe(self):
        """加载配置"""

        # Tick记录配置
        self._subscribeMarketData(self._settings.ticks, MarketData.EVENT_TICK)

        # 分钟线记录配置
        self._subscribeMarketData(self._settings.kline1min, MarketData.EVENT_KLINE_1MIN)

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

        if not mdEvent in self._dictDR or not eData.symbol in self._dictDR[mdEvent]:
            return

        if not 'coll' in self._dictDR[mdEvent][eData.symbol].keys() :
            return

        collection = self._dictDR[mdEvent][eData.symbol]['coll']

        if not eData.datetime : # 生成datetime对象
            try :
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S')
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S.%f')
            except:
                pass

        self.debug('On%s: %s' % (mdEvent, eData.desc))
        self._queueMarketData.put((collection, eData.__dict__))
    
    @abstractmethod
    def saveMarketData(self, collection, row) :
            # 这里采用MongoDB的update模式更新数据，在记录tick数据时会由于查询
            # 过于频繁，导致CPU占用和硬盘读写过高后系统卡死，因此不建议使用
            #flt = {'datetime': d['datetime']}
            #self._engine.dbUpdate(dbName, collectionName, d, flt, True)
        # 使用insert模式更新数据，可能存在时间戳重复的情况，需要用户自行清洗
        try :
            del row['exchange']
            del row['symbol']
            del row['vtSymbol']
        except:
            pass

        try:
            self.dbInsert(collection['collectionName'], row, collection['dbName'])
            self.debug('DB %s[%s] inserted: %s' % (collection['dbName'], collection['collectionName'], row))
        except DuplicateKeyError:
            self.error('键值重复插入失败：%s' %traceback.format_exc())

    @abstractmethod
    def openCollection(self, dbName, collectionName) :
        collection = {
            'dbName': dbName,
            'collectionName' : collectionName
        }
        self.dbEnsureIndex(collectionName, [('date', ASCENDING), ('time', ASCENDING)], True, dbName) #self._dbNamePrefix +e

        return collection

    @abstractmethod
    def saveObject(self, category, id, obj):
        """保存对象到硬盘"""
        try :
            os.makedirs(self._dataPath + '/objects')
        except:
            pass

        fn = '%s/objects/%s' % (self._dataPath, category)
        f = shelve.open(fn)
        f[id] = obj
        f.close()

    @abstractmethod
    def loadObject(self, category, id):
        """读取对象"""
        try :
            fn = '%s/objects/%s' % (self._dataPath, category)
            f = shelve.open(fn)
            if id in f :
                return f[id].value
        except Exception as ex:
            print("loadObject() error: %s %s" % (ex, traceback.format_exc()))

        return None

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ
        return super(DataRecorder, self).init()

    @abstractmethod
    def start(self):
        # 载入设置，订阅行情
        self.debug('start() subcribing')
        self.subscribe()
        for e in self._dictDR.keys() :
            dict = self._dictDR[e]
            for symbol in dict.keys() :
                collName = '%s.%s' %(symbol, dict[symbol]['ds'])
                dict[symbol]['coll'] = self.openCollection(self._dbNamePrefix +e, collName)

    @abstractmethod
    def step(self):
        while True:
            try:
                collection, d = self._queueMarketData.get(block=False, timeout=0.05)
                self.saveMarketData(collection, d)
            except: # Empty:
                break

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
import csv

class CsvRecorder(DataRecorder):
    """数据记录引擎"""
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(CsvRecorder, self).__init__(mainRoutine, settings)
        self._dbNamePrefix = settings.dbNamePrefix('dr')
        self._min2flush  = settings.min2flush(1.0)
        self._days2roll  = settings.days2roll(1.0)
        self._days2zip   = settings.days2archive(7.0)
        if self._days2zip < self._days2roll *2:
            self._days2zip = self._days2roll *2

    #close,date,datetime,high,low,open,openInterest,time,volume
    #,date,datetime,high,price,volume,low,lowerLimit,openInterest,open,prevClose,time,upperLimit,volume
    SORT_KEYS=['datetime','price','close', 'volume', 'high','low','open']

    @abstractmethod
    def saveMarketData(self, collection, row) :
        try :
            del row['exchange']
            del row['symbol']
            del row['vtSymbol']
        except:
            pass

        if not 'w' in collection.keys():
            colnames = []
            tmp = row.keys()
            for i in self.SORT_KEYS:
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
            self.debug('header[%s/%s] saved' % (collection['dir'], collection['name']))
        w.writerow(row)
        collection['c'] +=1
        self.debug('row[%s/%s] saved: %s' % (collection['dir'], collection['name'], row))

        self._checkAndRoll(collection)

    def _checkAndRoll(self, collection) :

        dtNow = datetime.now()
        if collection['f'] :
            if not 'flush' in collection.keys() or (collection['flush']+timedelta(minutes=self._min2flush)) < dtNow:
                collection['f'].flush()
                collection['flush'] =dtNow
                self.debug('flushed: %s/%s' % (collection['dir'], collection['name']))

            if collection['o'] :
                dtToRoll = collection['o']+timedelta(hours=self._days2roll*24)
                if self._days2roll >=1 and (self._days2roll-int(self._days2roll)) <1/24 : # make
                    dtToRoll = datetime(dtToRoll.year, dtToRoll.month, dtToRoll.day, 23, 59, 59, 999999)

                if dtToRoll > dtNow :
                    return collection
        
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
                if stk[-1] !='csv' or collection['name'] != name[:len(collection['name'])]:
                    continue
                if stk[-2] <stampToZip:
                    fn = '%s/%s' % (collection['dir'], name)
                    self.postEvent(EVENT_TOARCHIVE, fn)
                    self.debug('schedule to archive: %s' % fn)

        fname = '%s/%s.%s.csv' % (collection['dir'], collection['name'], stampThis)
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
    def openCollection(self, dbName, collectionName) :
        col = {
            'dir' : '%s/%s' % (self._dataPath, dbName),
            'name': collectionName,
            'f': None,
            'c': 0,
            'o': None
        }

        return self._checkAndRoll(col)

    @abstractmethod
    def loadRecentMarketData(self, symbol, startDate, eventType =MarketData.EVENT_KLINE_1MIN):
        """从数据库中读取Bar数据，startDate是datetime对象"""

        mdEvent = eventType[len(MARKETDATE_EVENT_PREFIX):]
        if not mdEvent in self._dictDR.keys() or not symbol in self._dictDR[mdEvent].keys():
            return []

        node = self._dictDR[mdEvent][symbol]
        if not 'coll' in node.keys() :
            return []

        # filter the csv files
        stampSince   = startDate.strftime('%Y%m%dT000000')
        collection = node['coll']
        csvfiles = []
        prev = ""
        for _, _, files in os.walk(collection['dir']):
            files.sort()
            for name in files:
                stk = name.split('.')
                if stk[-1] !='csv' or collection['name'] != name[:len(collection['name'])]:
                    continue
                fn = '%s/%s' % (collection['dir'], name)
                if stk[-2] <stampSince:
                    prev = fn
                else :
                    csvfiles.append(fn)
        if len(prev) >0:
            csvfiles = [prev] + csvfiles
        
        ret = []
        DataClass = TickData if mdEvent == 'Tick' else KLineData

        for fn in csvfiles :
            with open(fn, 'rb') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if fn == prev and row['datetime'] <stampSince:
                        continue
                    data = DataClass('')
                    data.__dict__  = row
                    ret.append(data)
        return ret

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
            except: # Empty:
                break

    def onToArchive(self, event) :
        self._push(event.dict_['data'])

    def _push(self, filename) :
        self._queue.put(filename)

