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

import sys, re
if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty
import bz2

EVENT_TOARCHIVE  = EVENT_NAME_PREFIX + 'toArch'
H5DSET_DEFAULT_ARGS={ 'compression': 'lzf' } # note lzf is good at speed, but HDFExplorer doesn't support it. Change it to gzip if want to view

def listAllFiles(folder, depthAllowed=5):
    ret =[]
    if depthAllowed <=0:
        return ret

    for root, subdirs, files in os.walk(folder, topdown=False):
        for name in files:
            ret.append(os.path.join(root, name))
        for name in subdirs:
            ret += listAllFiles(os.path.join(root, name), depthAllowed -1)

    return ret

########################################################################
class Recorder(BaseApplication):
    """数据记录, the base DR is implmented as a csv Recorder
        configuration:
            "datarecorder": {
                "dbNamePrefix": "dr", // the prefix of DB name to save: <dbNamePrefix>Tick, <dbNamePrefix>K1min
            }
    """
     
    DEFAULT_DBPrefix = 'dr'

    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs):
        """Constructor"""
        super(Recorder, self).__init__(program, **kwargs)

        self._dbNamePrefix = self.getConfig('dbNamePrefix', Recorder.DEFAULT_DBPrefix)

        # 配置字典
        self._dictDR = OrderedDict() # categroy -> { fieldnames, ....}

        # 负责执行数据库插入的单独线程相关
        self.__queRowsToRecord = Queue()  # 队列 of (category, Data)

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
                category, row = self.__queRowsToRecord.get(block=False, timeout=0.05)
                self._saveRow(category, row)
                cStep +=1
            except Empty:
                break
            except Exception as ex: # Empty:
                self.logexception(ex)
                break

        return cStep

    def OnEvent(self, event):
        pass # do nothing for Recorder unless it is an auto-recorder

    #----------------------------------------------------------------------
    def pushRow(self, category, row):
        if row and not isinstance(row, dict) :
            row = row.__dict__
        self.__queRowsToRecord.put((category, row))

    @abstractmethod
    def configIndex(self, category, definition, unique=False):
        """定义某category collection的index"""
        raise NotImplementedError

    @abstractmethod
    def _saveRow(self, category, dataDict) :
        # coll = self.findCollection(self, category)
        raise NotImplementedError

    def registerCategory(self, category, params={}):
        '''
           for example Account will call recorder.registerCategory('DailyPosition', params= {'index': [('date', INDEX_ASCENDING), ('time', INDEX_ASCENDING)], 'columns' : ['date','symbol']})
        '''
        if 'columns' in params.keys() and isinstance(params['columns'], str) :
            params['columns'] = params['columns'].split(',')
        if not category in self._dictDR.keys() :
            self._dictDR[category] = OrderedDict()
        coll = self._dictDR[category]
        coll['params'] = params
        return coll

    def findCollection(self, category) :
        return self._dictDR[category] if category in self._dictDR.keys() else None

########################################################################
import csv
import logging
import sys, os
class TaggedCsvRecorder(Recorder):
    '''
    This recorder write lines like csv but put a TAG as record-type at the beginning of line
    configuration:
        "datarecorder": {
            ...
            "dbNamePrefix": "dr", // the prefix of DB name to save: <dbNamePrefix>Tick, <dbNamePrefix>K1min

            // for csv recorder
            "minFlushInterval" : 0.3,
            "daysToRoll" : 1.0,
            "days2archive"  : 0.0028,
        }
    '''

    LOGFMT_GENERAL = '%(message)s' # the writer simply take logger to output file   

    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs):
        """Constructor"""
        super(TaggedCsvRecorder, self).__init__(program, **kwargs)

        self._minFlushInterval = self.getConfig('minFlushInterval', 1)
        self._daysToRoll  = self.getConfig('daysToRoll', 1)
        self._daysToZip   = self.getConfig('daysToZip', 7)
        filename          = self.getConfig('filename', '%s_%s.tcsv' % (self._program.progId, datetime.now().strftime('%m%d')))
        self._filepath    = self.getConfig('filepath', '%s%s' % (self.dataRoot, filename))
        self._fileMB      = self.getConfig('fileMB', 50) # 50MB before rolling
        self._fileCount   = self.getConfig('fileCount', 50) # maximal 50 rolling files before evicting
        tmp = min(self._daysToRoll *2, self._daysToRoll +2)
        if self._daysToZip < tmp:
            self._daysToZip = tmp
        if self._fileMB <=1:
            self._fileMB =10

        # employing the logger
        self.__fakedcsv   = logging.Logger(name=self.ident) #getLogger()
        self.__1stRow   = True     
        
        try :
            statinfo = os.stat(self._filepath)
            if statinfo and statinfo.st_size >10:
                self.__1stRow = False
        except :
            try :
                os.makedirs(os.path.dirname(self._filepath))
            except:
                pass

        self.__hdlrFile = logging.handlers.RotatingFileHandler(self._filepath, maxBytes=self._fileMB*1024*1024, backupCount=self._fileCount) # 20MB about to 3MB after bzip2
        self.__hdlrFile.rotator  = self.__rotator
        self.__hdlrFile.namer    = self.__rotating_namer
        self.__hdlrFile.setLevel(logging.DEBUG)
        self.__hdlrFile.setFormatter(logging.Formatter('%(message)s')) # only the message itself with NO stamp and so on
        self.__fakedcsv.addHandler(self.__hdlrFile)

    def __rotating_namer(self, name):
        return name + ".bz2"

    def __rotator(self, source, dest):
        self.__1stRow   = True
        stampStart = datetime.now()     
        with open(source, "rb") as sf:
            data = sf.read()
            compressed = bz2.compress(data, 9)
            with open(dest, "wb") as df:
                df.write(compressed)
        os.remove(source)
        self.info('rotated %s to %s, took %s' % (source, dest, (datetime.now() - stampStart)))


    # impl of BaseApplication
    #----------------------------------------------------------------------
    def doAppInit(self): # return True if succ
        if not super(TaggedCsvRecorder, self).doAppInit() :
            return False

        if not self.__fakedcsv :
            return False

        return True

    # impl/overwrite of Recorder
    #----------------------------------------------------------------------
    def configIndex(self, category, definition, unique=False):
        """定义某category collection的index"""
        pass # nothing to do as csv doesn't support index

    def _saveRow(self, category, row) :
        columns = None
        if self.__1stRow :
            self.__1stRow = False
            for k, v in self._dictDR.items():
                if not 'params' in v.keys() or not 'columns' in v['params'].keys():
                    continue
                if k == category:
                    columns = v['params']['columns']
                headerLine = ('!%s,' % k) + ','.join(v['params']['columns'])
                self.__fakedcsv.info(headerLine)

        if not columns and category in self._dictDR.keys():
            columns = self._dictDR[category]['params']['columns']

        cols = []
        if columns :
            for col in columns:
                try :
                    v = row[col]
                except:
                    v=''
                cols.append(v)
        else:
            self.warn('category[%s] registration not found, simply taking the current row' % category)
            colnames = row.keys()
            self._dictDR[category] = { 'params':{'columns': colnames} }
            headerLine = ('!%s,' % category) + ','.join(colnames)
            self.__fakedcsv.info(headerLine)
            cols=row.values()
            # for k, v in row.items():
            #     cols.append(v)
            
        line = '%s,%s' % (category, ','.join([str(c) for c in cols]))
        self.__fakedcsv.info(line)
        return line

    # --private methods----------------------------------------------------------------
    def __checkAndRoll(self, collection) :

        dtNow = datetime.now()
        if collection['f'] :
            if not 'flush' in collection.keys() or (collection['flush']+timedelta(minutes=self._minFlushInterval)) < dtNow:
                collection['f'].flush()
                collection['flush'] =dtNow
                self.debug('flushed: %s/%s' % (collection['dir'], collection['fn']))

            if collection['o'] :
                dtToRoll = collection['o']+timedelta(hours=self._daysToRoll*24)
                if self._daysToRoll >=1 and (self._daysToRoll-int(self._daysToRoll)) <1/24 : # make the roll occur at midnight
                    dtToRoll = datetime(dtToRoll.year, dtToRoll.month, dtToRoll.day, 23, 59, 59, 999999)

                if dtToRoll > dtNow :
                    return collection

                self.debug('rolling %s/%s' % (collection['dir'], collection['fn']))
        
        stampToZip = (dtNow - timedelta(hours=self._daysToZip*24)).strftime('%Y%m%dT%H%M%S')
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
            size = os.stat(fname).st_size
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
class TcsvFilter(MetaObj):
    def __init__(self, tcsvFilePath, categroy, symbol=None):
        super(MetaObj, self).__init__()
        self.__gen=None
        self._tcsvFilePath = tcsvFilePath
        self.__category = categroy.strip()
        self.__symbol = symbol.strip() if symbol else None

    def __iter__(self):
        fnlist = TcsvFilter.buildUpFileList(self._tcsvFilePath)
        self.__gen = TcsvFilter.funcGen(fnlist, self.__category, self.__symbol)
        return self

    def __next__(self):
        if not self.__gen:
            self.__iter__()
            # raise StopIteration
        return next(self.__gen)

    def read(self, n=0):
        try:
            return next(self)
        except StopIteration:
            pass
        
        return ''

    def buildUpFileList(tcsvFilePath):
        fnlist = []
        bz2dict = {}

        dirname = os.path.realpath(os.path.dirname(tcsvFilePath))
        basename = os.path.basename(tcsvFilePath)
        if basename[-5:] != '.tcsv':
            basename += '.tcsv'
        lenBN = len(basename)
        for root, _, files in os.walk(dirname, topdown=False):
            for name in files:
                if basename != name[:lenBN]:
                    continue
                suffix = name[lenBN:]
                if len(suffix) <=0:
                    fnlist.append(os.path.join(root,name))
                    continue
                m = re.match(r'\.([0-9]*)\.bz2', suffix)
                if m :
                    bz2dict[int(m.group(1))] = name

        items = list(bz2dict.items())
        items.sort() 

        # insert into fnlist reversly
        for k,v in items:
            fnlist.insert(0, v)

        fnlist = [os.path.join(dirname, x) for x in fnlist]
        return fnlist

    def funcGen(fnlist, tag, symbol=None):
        headers = None
        if ',' != tag[-1]: tag +=','
        taglen = len(tag)
        idxSymbol =-1
        if symbol and len(symbol) <=0:
            self.__symbol =None

        for fn in fnlist:
            if fn[-4:] == '.bz2':
                stream = bz2.open(fn, mode='rt')
            elif fn[-5:] == '.tcsv':
                stream = open(fn, mode='rt')

            row= True # dummy non-None
            while row:
                try :
                    row = next(stream, None).strip()
                except Exception as ex:
                    row = None
                
                if not row:
                    stream.close()
                    continue

                if len(row) <=0:
                    continue

                if '!' ==row[0] :
                    if headers or tag != row[1:taglen+1]:
                        continue

                    headers = row.split(',')
                    del(headers[0])
                    yield row[taglen+1:]+'\n'

                    if idxSymbol<0 and symbol and 'symbol' in headers:
                        idxSymbol = headers.index('symbol')

                    continue
                
                if not headers or tag != row[:taglen]:
                    continue

                if idxSymbol>=0 and row.split(',')[1+idxSymbol] !=symbol:
                    continue

                row = row[taglen:] +'\n'
                yield row

        raise StopIteration

########################################################################
class Playback(Iterable):
    """The reader part of HistoryData
    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    DUMMY_DATE_START = '19900101T000000'
    DUMMY_DATE_END   = '39991231T000000'

    def __init__(self, symbol, startDate =DUMMY_DATE_START, endDate=DUMMY_DATE_END, category =None, exchange=None, **kwargs):
        """Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        """
        super(Playback, self).__init__(**kwargs)
        self._symbol = symbol
        self._category = category if category else EVENT_KLINE_1MIN
        self._exchange = exchange if exchange else ''
        self._dbNamePrefix = Recorder.DEFAULT_DBPrefix
        
        self._startDate = startDate if startDate else Playback.DUMMY_DATE_START
        self._endDate   = endDate if endDate else Playback.DUMMY_DATE_END
        self.__dtStart, self.__dtEnd = None, None
        try :
            self.__dtStart = datetime.strptime(self._startDate, '%Y%m%dT%H%M%S')
            self.__dtStart = datetime.strptime(self._startDate, '%Y-%m-%dT%H:%M:%S')
        except:
            pass

        try :
            self.__dtEnd = datetime.strptime(self._endDate, '%Y%m%dT%H%M%S')
            self.__dtEnd = datetime.strptime(self._endDate, '%Y-%m-%dT%H:%M:%S')
        except:
            pass

        self._dictCategory = {} # eventType/category to { columeNames: [], coverter: func() }
        self.setId('%s.%s.%s-%s' %(self._symbol, self._category, self._startDate, self._endDate) )

    @property
    def datetimeRange(self) : return self.__dtStart, self.__dtEnd

    @property
    def categories(self) : return self._dictCategory

    # -- impl of Iterable --------------------------------------------------------------
    def resetRead(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        self.__lastMarketClk = None
        return True

    def readNext(self):
        '''
        @return next item, mostlikely expect one of Event()
        '''
        if self.pendingSize <=0: 
            raise StopIteration

        event = None
        try :
            event = self.popPending(block = False, timeout = 0.1)
        except Exception:
            event = None

        return event

    def _testAndGenerateMarketHourEvent(self, ev):
        if self.__lastMarketClk and (self.__lastMarketClk + timedelta(hours=1)) > ev.data.asof:
            return None
        self.__lastMarketClk = ev.data.asof
        self.__lastMarketClk = self.__lastMarketClk.replace(minute=0, second=0, microsecond=0)
        evdMH = MarketData(self._exchange, self._symbol)
        evdMH.datetime = self.__lastMarketClk
        evdMH.date = evdMH.datetime.strftime('%Y-%m-%d')
        evdMH.time = evdMH.datetime.strftime('%H:%M:%S')
        evMH = Event(EVENT_MARKET_HOUR)
        evMH.setData(evdMH)
        self.enquePending(evMH)
        return evdMH

    def registerConverter(self, categroy, converter, columns=None):
        if not converter or not categroy or len(categroy) <=0:
            return

        if not columns:
            columns=[]
            # try :
            #     columns=converter.fields.split(',')
            # except: pass

        if not categroy in self._dictCategory:
            self._dictCategory[categroy] = {
                'columns': columns,
                'converter': converter
            }
        else:
            self._dictCategory[categroy]['converter'] = converter
            if len(columns) >0:
                self._dictCategory[categroy]['columns'] = columns

########################################################################
class PlaybackApp(BaseApplication):
    
    #----------------------------------------------------------------------
    def __init__(self, program, playback, **kwargs):
        """Constructor"""
        super(PlaybackApp, self).__init__(program, **kwargs)
        self.__playback = playback

    # impl of BaseApplication
    #----------------------------------------------------------------------
    def doAppInit(self): # return True if succ
        if not super(PlaybackApp, self).doAppInit() :
            return False

        if self.__playback is None :
            return False
        
        self.__playback.setProgram(self.program)
        return True

    def doAppStep(self):
        if not self.isActive or not self.__playback:
            return False

        ev = None
        try :
            ev = next(self.__playback)
        except StopIteration:
            self.__playback = NoneTrue
            self.info('hist-read: end of playback')
        except Exception as ex:
            self.logexception(ex)

        if ev:
            self.postEvent(ev)
        return True

    def OnEvent(self, event):
        pass # do nothing for Recorder unless it is an auto-recorder

    #----------------------------------------------------------------------


########################################################################
class CsvPlayback(Playback):
    DIGITS=set('0123456789')

    #----------------------------------------------------------------------
    def __init__(self, symbol, folder, fields, category =None, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END, **kwargs) :

        super(CsvPlayback, self).__init__(symbol, startDate, endDate, category, **kwargs)
        self._folder = folder
        self._fields = fields

        self.__csvfiles =[]
        self.__csvToKL1m = KLineData.hatch # DictToKLine(self._category, symbol)

        self._merger1minTo5min = None
        self._merger5minTo1Day = None
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

        self.__csvfiles =[]
        self.__reader =None
        self._merger1minTo5min = KlineToXminMerger(self._cbMergedKLine5min, xmin=5)
        self._merger5minTo1Day = KlineToXminMerger(self._cbMergedKLine1Day, xmin=60*24-10)
        self._dtEndOfDay = None

        # filter the csv files
        self.debug('search dir %s for csv files' % self._folder)
        prev = ""
        files = listAllFiles(self._folder)
        files = [os.path.realpath(fn) for fn in files]
        files = list(set(files))
        files.sort()

        for fn in files:
            try :
                os.stat(fn)
            except :
                continue

            name = os.path.basename(fn)
            stk = name.split('.')
            if len(stk) <=1 or not 'csv' in stk or self._symbol.lower() != name[:len(self._symbol)].lower():
                continue
            if not stk[-1].lower() in ['csv', 'bz2'] :
                continue

            self.debug('checking if file %s is valid' % fn)
            stampstr = stk[0][len(self._symbol):]
            pos = next((i for i, ch  in enumerate(stampstr) if ch in CsvPlayback.DIGITS), None)
            if not pos :
                self.__csvfiles.append(fn)
                continue

            if 'Y' == stampstr[pos] : pos +=1
            stampstr = stampstr[pos:]
            if stampstr < self._startDate :
                prev = fn
            elif stampstr <= self._endDate:
                self.__csvfiles.append(fn)

        if len(prev) >0:
            self.__csvfiles = [prev] + self.__csvfiles
        
        self.info('associated file list: %s' % self.__csvfiles)
        return len(self.__csvfiles) >0

    def openReader(self):
        while not self.__reader:
            if not self.__csvfiles or len(self.__csvfiles) <=0:
                self._iterableEnd = True
                return None

            fn = self.__csvfiles[0]
            del(self.__csvfiles[0])

            self.info('openning input file %s' % (fn))
            extname = fn.split('.')[-1]
            if extname == 'bz2':
                self._streamIn = bz2.open(fn, mode='rt') # bz2.BZ2File(fn, 'rb')
            else:
                self._streamIn = open(fn, 'rt')

            self.__reader = csv.DictReader(self._streamIn, self._fieldnames, lineterminator='\n') if 'csv' in fn else self._streamIn
            if not self.__reader:
                self.warn('failed to open input file %s' % (fn))
        
        return self.__reader

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
            self.openReader()

            if not self.__reader:
                self._iterableEnd = True
                return None

            # if not self._fieldnames or len(self._fieldnames) <=0:
            #     self._fieldnames = self.__reader.headers()

            try :
                row = next(self.__reader, None)
            except Exception as ex:
                row = None

            if not row:
                # self.error(traceback.format_exc())
                self.__reader = None
                self._streamIn.close()

        ev = None
        try :
            if row and self.__csvToKL1m:
                # print('line: %s' % (line))
                row = {'evType': EVENT_KLINE_1MIN,
                        'exchange': self._exchange, 'symbol':self._symbol,
                       **row }
                ev = self.__csvToKL1m(**row) # self.__csvToKL1m.convert(row, self._exchange, self._symbol)
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
# playback wrapper on an opened stream
class TaggedCsvStream(Playback):
    
    def __init__(self, stream, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END, symbol=None, **kwargs):

        super(TaggedCsvStream, self).__init__(symbol, startDate, endDate, None, **kwargs)

        self.__strmFileIn = stream

    def close(self): # to be stream-like
        self.__strmFileIn= None

    # -- Impl of Playback --------------------------------------------------------------
    def resetRead(self):
        super(TaggedCsvStream, self).resetRead()
        return not self.__strmFileIn is None

    def readNext(self):
        '''
        @return True if busy at this step
        '''
        try :
            ev = self.popPending(block = False, timeout = 0.1)
            if ev: return ev
        except Exception:
            pass
        None
        row, ev = None, None
        while not row:
            if not self.__strmFileIn:
                self._iterableEnd = True
                return None

            try :
                row = self.__strmFileIn.readline()
            except Exception as ex:
                row = None

            if not row:
                self.__strmFileIn.close()
                self.__strmFileIn = None
                continue

            if not isinstance(row, str): row = row.decode()
            row = row.strip()
            if len(row) <=0:
                continue

            if '!' == row[0] : # this is a header line
                tokens = row.split(',')
                if len(tokens) <=1: continue

                categroy = tokens[0][1:]
                del(tokens[0])
                if not categroy in self._dictCategory.keys():
                    self._dictCategory[categroy] = {
                        'columns': [],
                        'converter': None
                    }

                self._dictCategory[categroy]['columns'] = tokens

                # if idxSymbol<0 and symbol and 'symbol' in tokens:
                #     idxSymbol = tokens.index('symbol')
                continue

            # now deal with the value lines
            tokens = row.split(',')
            if len(tokens) <=1: continue
            categroy = tokens[0]

            if not categroy in self._dictCategory.keys(): continue
            node = self._dictCategory[categroy]
            converter = node['converter']
            if not converter:
                continue
            
            # TODO: ????
            ev = None
            try :
                dict={}
                cols = node['columns']
                if isinstance(cols,str):
                    cols = cols.split(',')
                for i in range(len(cols)) :
                    dict[cols[i]] = tokens[1+i]
                
                if len(dict) <=0: 
                    self.warn('columns unknown, ignored: %s' % (row))
                    continue

                dict['evType'] = categroy
                ev = converter(**dict) # converter.convert(dict)

                # because the merged-event is always as of the previous event, so always deliver
                # those pendings in queue first
                if self.pendingSize >0 :
                    evout = self.popPending()
                    self.enquePending(ev)
                    return evout

            except Exception as ex:
                self.error('convert row[%s] err: %s' % (row, ex))

        return ev

########################################################################
class TaggedCsvPlayback(Playback):
    
    def __init__(self, tcsvFilePath, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END, symbol=None, **kwargs):

        super(TaggedCsvPlayback, self).__init__(symbol, startDate, endDate, None, **kwargs)

        self._tcsvFilePath = tcsvFilePath

        self._dictCategory = {} # eventName/category to { columeNames: [], coverter: func() }
        self.__fnlist =[]
        self._strmTcsv = None

    def openTcsvStream(self):

        while not self._strmTcsv:
            if not self.__fnlist or len(self.__fnlist) <=0:
                self._iterableEnd = True
                return None

            fn = self.__fnlist[0]
            del(self.__fnlist[0])

            self.info('openning input file %s' % (fn))
            extname = fn.split('.')[-1]
            strm = None
            if extname == 'bz2':
                strm = bz2.open(fn, mode='rt') # bz2.BZ2File(fn, 'rb')
            else:
                strm = open(fn, 'rt')

            if not strm:
                self.warn('failed to open input file %s' % (fn))
                break

            dtStart, dtEnd = self.datetimeRange
            self._strmTcsv = TaggedCsvStream(strm, startDate =dtStart, endDate=dtEnd, program=self.program)
            for cat, v in self._dictCategory.items():
                self.registerConverter(cat, **v)

        return self._strmTcsv

    # -- Impl of Playback --------------------------------------------------------------
    def resetRead(self):
        super(TaggedCsvPlayback, self).resetRead()

        fnpatt = os.path.basename(self._tcsvFilePath)
        allFiles = listAllFiles(os.path.dirname(self._tcsvFilePath))
        allFiles.sort()
        self.__fnlist =[]
        for fn in allFiles:
            if fnmatch.fnmatch(os.path.basename(fn), fnpatt):
                self.__fnlist.append(fn)

        self.info('associated file list by %s: %s' % (self._tcsvFilePath, ','.join(self.__fnlist)))
        return len(self.__fnlist) >0

    def readNext(self):
        '''
        @return True if busy at this step
        '''
        try :
            ev = self.popPending(block = False, timeout = 0.1)
            if ev: return ev
        except Exception:
            pass
        
        ev = None
        while not ev:
            self.openTcsvStream()

            if not self._strmTcsv:
                self._iterableEnd = True
                return None

            try :
                ev = next(self._strmTcsv)
            except Exception as ex:
                ev = None

            if not ev:
                self._dictCategory = self._strmTcsv.categories # inherit the current categories for next stream
                self._strmTcsv.close()
                self._strmTcsv = None
                continue
        
        return ev

import tarfile, fnmatch

########################################################################
class TaggedCsvInTarball(TaggedCsvPlayback):
    def __init__(self, fnTarball, memberPattern=None, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END, symbol=None, **kwargs):
        super(TaggedCsvInTarball, self).__init__(symbol, startDate, endDate, None, **kwargs)
        self.__fnTarball  = fnTarball
        self.__memPattern = memberPattern if memberPattern else '*.tcsv*'
        self.__memlist = [] # list of tar.member

        self.__tar = None
        self.__currentMem = None

    @property
    def currentMember(self): 
        return self.__currentMem.name if self.__currentMem else ''

    # -- overwrite of TaggedCsvInTarball --------------------------------------------------------------
    def resetRead(self):
        super(TaggedCsvPlayback, self).resetRead()

        self.__memlist = []
        self.__currentMem = None
        bz2dict1 = {}
        bz2dict2 = {}

        self.__tar = tarfile.open(self.__fnTarball)
        if self.__tar:
            for member in self.__tar.getmembers():
                basename = os.path.basename(member.name)
                if not fnmatch.fnmatch(basename, self.__memPattern):
                    continue
                
                if '.bz2' == basename[-4:]:
                    # old fn format: xxxx.tcsv.1.bz2
                    m = re.match(r'.*\.tcsv\.([0-9]*)\.bz2', basename)
                    if m :
                        bz2dict1[int(m.group(1))] = member
                        continue

                    # new fn format: xxxx.YYYYmmddHHMMSS.tcsv.bz2
                    m = re.match(r'.*\.([0-9]*)\.tcsv\.bz2', basename)
                    if m :
                        bz2dict2[int(m.group(1))] = member
                        continue

                # the main/last tcsv
                if '.tcsv' == basename[-5:]:
                    self.__memlist.append(member) 
                    continue

                # old unzipped fn format: xxxx.tcsv.1
                m = re.match(r'.*\.tcsv\.([0-9]*)', basename)
                if m :
                    bz2dict1[int(m.group(1))] = member
                    continue


        items = list(bz2dict1.items())
        items.sort() 

        # insert into fnlist reversly
        for k,v in items:
            self.__memlist.insert(0, v)

        items = list(bz2dict2.items())
        items.sort()

        # append into fnlist
        for k,v in items:
            self.__memlist.insert(-1, v)

        if len(self.__memlist) >0:
            self.info('resetRead() associated [%s] in tarball[%s]: %s' % (self.__memPattern, self.__fnTarball, ','.join([m.name for m in self.__memlist])))
            return True
        
        self.warn('resetRead() failed to associated [%s] in tarball[%s]' % (self.__memPattern, self.__fnTarball))
        return False

    def openTcsvStream(self):
        while not self._strmTcsv:
            if not self.__memlist or len(self.__memlist) <=0:
                self._iterableEnd = True
                return None

            self.__currentMem = self.__memlist[0]
            del self.__memlist[0]

            self.debug('extracting memberfile[%s]' % self.__currentMem.name)
            extname = self.__currentMem.name.split('.')[-1]

            self._strmTcsv = self.__tar.extractfile(self.__currentMem)
            if extname == 'bz2':
                self._strmTcsv = bz2.open(self._strmTcsv, mode='rt')

            if not self._strmTcsv:
                self.warn('failed to open memberfile[%s]' % self.__currentMem.name)
                break

            dtStart, dtEnd = self.datetimeRange
            self._strmTcsv = TaggedCsvStream(self._strmTcsv, startDate =dtStart, endDate=dtEnd, program=self.program)
            if not self._strmTcsv: continue

            for cat, v in self._dictCategory.items():
                self._strmTcsv.registerConverter(cat, **v)

            self.info('opened memberfile[%s]' % self.__currentMem.name)
                
        return self._strmTcsv


########################################################################
class PlaybackMux(Playback):
    
    def __init__(self, program=None, startDate =Playback.DUMMY_DATE_START, endDate=Playback.DUMMY_DATE_END):

        super(PlaybackMux, self).__init__(symbol=None, program=program, startDate =startDate, endDate=endDate)
        self.__dictStrmPB = {} # dict of Playback to recentEvent
        self.__seqNextPB =[]

    def addStream(self, playback):
        if playback and not playback in self.__dictStrmPB.keys():
            self.__dictStrmPB[playback] = None
    
    @property
    def size(self): return len(self.__dictStrmPB)

    # -- Impl of Playback --------------------------------------------------------------
    def resetRead(self):
        super(PlaybackMux, self).resetRead()
        self.__seqNextPB =[]
        return len(self.__dictStrmPB) >0

    DUMMY_SORT_DATE = datetime.strptime('19000101T000000', '%Y%m%dT%H%M%S')

    def __strmSortKey(self, strm) :
        if strm in self.__dictStrmPB.keys():
            ev = self.__dictStrmPB[strm]
            return ev.data.asof if ev else DUMMY_SORT_DATE

        return DUMMY_SORT_DATE

    def readNext(self):
        '''
        @return True if busy at this step
        '''
        while not self._iterableEnd:
            strmsToEvict = []

            if len(self.__seqNextPB) < len(self.__dictStrmPB):
                for strm, ev in self.__dictStrmPB.items():
                    if ev: continue
                    
                    try :
                        ev = next(strm)
                        if not ev:
                            strmsToEvict.append(strm)
                            continue

                        self.__dictStrmPB[strm] = ev
                        
                        # insert into self.__seqNextPB, earist ev.asof first
                        n =0
                        for n in range(len(self.__seqNextPB)):
                            npb = self.__seqNextPB[n]
                            nev =  self.__dictStrmPB[npb]
                            if ev.data.asof < nev.data.asof:
                                break
                        self.__seqNextPB.insert(n, strm)

                    except StopIteration:
                        strmsToEvict.append(strm)
                        continue

            for sd in strmsToEvict:
                del self.__dictStrmPB[sd]
                if sd in self.__seqNextPB: self.__seqNextPB.remove(sd)
                self.info('evicted stream[%s] that reached end, %d-stream remain' % (sd.id, len(self.__dictStrmPB)))

            if len(self.__dictStrmPB) <=0:
                self._iterableEnd = True
                self.info('all streams reached end')
                break

            if len(self.__seqNextPB) < len(self.__dictStrmPB):
                continue

            strmEariest = self.__seqNextPB[0]
            del self.__seqNextPB[0]
            ev = self.__dictStrmPB[strmEariest]
            self.__dictStrmPB[strmEariest] = None

            dtStart, dtEnd = self.datetimeRange
            if ev.data.asof < dtStart: continue
            if ev.data.asof > dtEnd: continue

            # lst = ['%#%s' % (i.id, self.__dictStrmPB[i].data.asof.strftime('%Y%m%dT%H%M%S')) for i in self.__seqNextPB]
            # self.debug('outgoing ev[%s] nextPB: %s' % (ev.desc, ';'.join(lst[:5])))
            return ev

        raise StopIteration

    def readNext2(self):
        '''
        @return True if busy at this step
        '''
        while not self._iterableEnd:
            strmsToEvict = []

            for strm, ev in self.__dictStrmPB.items():
                if ev: continue
                
                try :
                    ev = next(strm)
                    if not ev:
                        strmsToEvict.append(strm)
                        continue

                    self.__dictStrmPB[strm] = ev
                    
                    # # insert into self.__seqNextPB, earist ev.asof first
                    # n =0
                    # for n in range(len(self.__seqNextPB)):
                    #     npb = self.__seqNextPB[n]
                    #     nev =  self.__dictStrmPB[npb]
                    #     if ev.data.asof < nev.data.asof:
                    #         break
                    # self.__seqNextPB.insert(n, strm)

                except StopIteration:
                    strmsToEvict.append(strm)
                    continue

            for sd in strmsToEvict:
                del self.__dictStrmPB[sd]
                self.__seqNextPB.remove(sd)
                self.info('evicted stream[%s] that reached end, %d-stream remain' % (sd.id, len(self.__dictStrmPB)))

            if len(self.__dictStrmPB) <=0:
                self._iterableEnd = True
                self.info('all streams reached end')
                break

            self.__seqNextPB = list(self.__dictStrmPB.keys())
            self.__seqNextPB.sort(key = self.__strmSortKey)

            strmEariest = self.__seqNextPB[0]
            ev = self.__dictStrmPB[strmEariest]
            self.__dictStrmPB[strmEariest] = None
            if not ev: continue

            dtStart, dtEnd = self.datetimeRange
            if ev.data.asof < dtStart: continue
            if ev.data.asof > dtEnd: continue

            return ev

        raise StopIteration

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
    def registerCategory(self, category, params= {}):
        coll = super(MongoRecorder, self).registerCategory(category, params)

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
                # self.configIndex(collectionName, [('date', INDEX_ASCENDING), ('time', INDEX_ASCENDING)], True, dbName) #self._dbNamePrefix +e
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
class MarketRecorder(BaseApplication):
    """数据记录引擎, the base DR is implmented as a csv Recorder"""
     
    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime,price,close,volume,high,low,open'

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
                self._recorder = TaggedCsvRecorder(program, settings)

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
                self._recorder.registerCategory(category, params= {'ds': ds, 'index': [('date', INDEX_ASCENDING), ('time', INDEX_ASCENDING)], 'columns' : MarketRecorder.COLUMNS})

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
                eData.datetime = datetime.strptime('T'.join([eData.date, eData.time]), '%Y-%m-%dT%H:%M:%S')
                eData.datetime = datetime.strptime('T'.join([eData.date, eData.time]), '%Y-%m-%dT%H:%M:%S.%f')
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
                    os.stat(ofn)
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


