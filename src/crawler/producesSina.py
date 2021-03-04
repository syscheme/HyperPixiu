# encoding: UTF-8

from __future__ import division

from Simulator import IdealTrader_Tplus1, SAMPLES_PER_H5FRAME # , ShortSwingScanner
from EventData import Event, EventData
from MarketData import *
from Perspective import PerspectiveState
from Application import listAllFiles
import HistoryData as hist
from crawler.crawlSina import *
import h5tar, h5py, pickle, bz2

from datetime import datetime, timedelta
from time import sleep
import os, copy
from io import StringIO
import fnmatch

def defaultNextYield(retryNo) :
    return min(10.0* (2 ** retryNo), 300.0) if retryNo >0 else 0.1

########################################################################
class SinaMux(hist.PlaybackMux) :

    MDEVENTS_FROM_ADV = [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY] # [EVENT_TICK, EVENT_KLINE_1MIN]
    SRCPATH_PATTERN_ONLINE="$ONLINE$"

    def __init__(self, program, startDate =None, endDate=None, **kwargs):
        '''Constructor
        '''
        self.__symbols = []

        super(SinaMux, self).__init__(program, startDate =startDate, endDate=endDate)
        self.__cachedFiles=[]

    @property
    def symbols(self) : return self.__symbols
    
    @property
    def cachedFiles(self) : return self.__cachedFiles

    def setSymbols(self, symbols) :
        if isinstance(symbols, str):
            symbols = symbols.split(',')
            
        self.__symbols = [SinaCrawler.fixupSymbolPrefix(s) for s in symbols]
        return self.symbols

    def populateSymbolList(tarballPath):
        tar = tarfile.open(tarballPath)

        symbolList = []
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if not basename.split('.')[-1] in ['json']: 
                continue

            symbol = basename[:basename.index('_')]
            symbolList.append(symbol)

        #no self: self.debug('%d symbols found in tarball[%s]: %s' % (len(symbolList), tarballPath, ','.join(symbolList)))
        return symbolList

    def __jsonStrToPlayback(self, content, symbol, evtype, dtStart =None, dtEnd=None):

        edseq =[]

        # dispatch the convert func up to evtype from tar filename
        if EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)]:
            edseq = SinaCrawler.convertToKLineDatas(symbol, content)
        elif EVENT_MONEYFLOW_1MIN == evtype:
            edseq = SinaCrawler.convertToMoneyFlow(symbol, content, True)
        elif EVENT_MONEYFLOW_1DAY == evtype:
            edseq = SinaCrawler.convertToMoneyFlow(symbol, content, False)

        if len(edseq) <=0: return None, edseq

        pb, c = hist.Playback(symbol, program=self.program), 0
        pb.setId('%s_%s.json' % (symbol, evtype))
        for ed in edseq:
            # TODO if dtStart and ed.asof < dtStart:
            if dtEnd and ed.asof > dtEnd:
                del edseq[c:]
                break
            ev = Event(evtype)
            ev.setData(ed)
            pb.enquePending(ev)
            c+=1
        
        return pb, edseq

    def importJsonSequence(self, jContent, symbol, evtype, dtStart =None, dtEnd=None):
        pb, dataseq = self.__jsonStrToPlayback(jContent, symbol, evtype, dtStart =dtStart, dtEnd =dtEnd)
        if not pb : return None

        pb.setId('json:%s.%s' % (symbol, evtype))
        self.addStream(pb)
        self.info('added substrm[%s] into mux' % (pb.id))
        return pb.id

    def __jsonToPlayback(self, jsonFstream, symbol, evtype, dtStart =None, dtEnd=None):

        content = jsonFstream.read().decode()
        return self.__jsonStrToPlayback(content, symbol, evtype, dtStart, dtEnd)

    def __extractJsonTarball(self, tarballName, evtype):
        tar = tarfile.open(tarballName)

        foundlist = []
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if not basename.split('.')[-1] in ['json']: 
                continue

            symbol = basename[:basename.index('_')]

            if (len(self.symbols)>0 and not symbol in self.symbols) or symbol in EXECLUDE_LIST:
                continue

            self.debug('memberFile[%s] in %s matched et[%s]' % (member.name, tarballName, evtype))
            foundlist.append(member.name)
            with tar.extractfile(member) as f:
                pb = self.__jsonToPlayback(f, symbol, evtype)
                if not pb: continue
                pb.setId('%s@%s' % (basename, os.path.basename(tarballName)))
                
                self.addStream(pb)
                self.info('added substrm[%s] into mux' % (pb.id))
    
            if len(foundlist) >= len(self.symbols) : # if self.symbols and self.symbols == symbol:
                break # do not scan the tar anymore

    def __extractAdvisorTarball(self, tarballName):
        pb = hist.TaggedCsvInTarball(tarballName, memberPattern='advisor_*.tcsv*', program=self.program)
        pb.setId('%s' % (os.path.basename(tarballName)))

        pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        pb.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

        self.addStream(pb)

    def __advmdToPlayback(self, advmdtcsvStrm, symbol, evtype) :
        pb = hist.TaggedCsvStream(advmdtcsvStrm, program=self.program)
        pb.setId('advmd%s_%s.tcsv' % (symbol, evtype))

        pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
        return pb

    def __extractAdvMdTarball(self, tarballName):
        subStrmsAdded = []
        tar = tarfile.open(tarballName)
        memlist = []

        for member in tar.getmembers():
            '''
            # tar tvfj ../advmd_20200720.A300-tc.tar.bz2 
            -rw-rw-rw- root/root     20590 2020-09-12 16:26 SH600037_KL1d20200720.tcsv
            -rw-rw-rw- root/root     21461 2020-09-12 16:26 SH600037_KL1m20200720.tcsv
            -rw-rw-rw- root/root      6264 2020-09-12 16:26 SH600037_KL5m20200720.tcsv
            '''
            basename = os.path.basename(member.name)
            if not fnmatch.fnmatch(basename, 'S[HZ][0-9]*_*.tcsv'):
                continue
            
            tokens = basename.split('.')[0].split('_')
            if len(tokens) <2: continue
            symbol, evtype = tokens[0], tokens[1]
            if (len(self.symbols)>0 and not symbol in self.symbols) or symbol in EXECLUDE_LIST:
                continue

            if '2020' in evtype: evtype=evtype[:evtype.index('2020')] # a fixup for old data
            if not MARKETDATE_EVENT_PREFIX in evtype: evtype = MARKETDATE_EVENT_PREFIX + evtype

            if not evtype in SinaMux.MDEVENTS_FROM_ADV : 
                continue # exclude some events as we collected in some other ways

            self.debug('memberFile[%s] in %s matched' % (member.name, tarballName))
            memlist.append(member) 

        for member in memlist:
            f = tar.extractfile(member)
            pb = self.__advmdToPlayback(f, symbol, evtype)
            if not pb : continue
            pb.setId('%s@%s' % (os.path.basename(member.name), os.path.basename(tarballName)))

            self.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))
            subStrmsAdded.append(pb.id)
        
        return subStrmsAdded

    def __extractedFolder_advmd(self, folderName):
        subStrmsAdded =[]
        allfiles = listAllFiles(folderName, depthAllowed=1)

        memlist = []
        for fn in allfiles:
            '''
            -rw-rw-rw- root/root     20590 2020-09-12 16:26 SH600037_KL1d20200720.tcsv
            -rw-rw-rw- root/root     21461 2020-09-12 16:26 SH600037_KL1m20200720.tcsv
            -rw-rw-rw- root/root      6264 2020-09-12 16:26 SH600037_KL5m20200720.tcsv
            '''
            basename = os.path.basename(fn)
            if not fnmatch.fnmatch(basename, 'S[HZ][0-9]*_*.tcsv'):
                continue
            
            tokens = basename.split('.')[0].split('_')
            if len(tokens) <2: continue
            symbol, evtype = tokens[0], tokens[1]
            if (len(self.symbols)>0 and not symbol in self.symbols) or symbol in EXECLUDE_LIST:
                continue

            if '202' in evtype: evtype=evtype[:evtype.index('202')] # a fixup for old data starting since year 2020
            if not MARKETDATE_EVENT_PREFIX in evtype: evtype = MARKETDATE_EVENT_PREFIX + evtype

            if not evtype in SinaMux.MDEVENTS_FROM_ADV : 
                continue # exclude some events as we collected in some other ways

            self.debug('file[%s] matched' % (fn))
            memlist.append(fn) 

        for fn in memlist:
            f = open(fn, "r")
            pb = self.__advmdToPlayback(f, symbol, evtype)
            if not pb : continue
            pb.setId('%s' % (fn))

            self.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))
            subStrmsAdded.append(pb.id)

        return subStrmsAdded

    def __extractedFolder_sinaJson(self, folderName, evtype):
        
        subStrmsAdded=[]
        allfiles = listAllFiles(folderName, depthAllowed=2)

        for fn in allfiles:
            '''
            -rw-rw-rw- root/root     20590 2020-09-12 16:26 SZ002588_MF1m20200817.json
            '''
            basename = os.path.basename(fn)
            if not fnmatch.fnmatch(basename, 'S[HZ][0-9]*_%s*.json' % chopMarketEVStr(evtype)):
                continue
            
            tokens = basename.split('.')[0].split('_')
            if len(tokens) <2: continue
            symbol, _ = tokens[0], tokens[1]
            if (len(self.symbols)>0 and not symbol in self.symbols) or symbol in EXECLUDE_LIST:
                continue

            # if '202' in evtype: evtype=evtype[:evtype.index('202')] # a fixup for old data starting since year 2020
            # if not MARKETDATE_EVENT_PREFIX in evtype: evtype = MARKETDATE_EVENT_PREFIX + evtype
            self.debug('file[%s] matched' % (fn))

            pb = None
            with open(fn, "rb") as f:
                pb = self.__jsonToPlayback(f, symbol, evtype)

            if not pb :
                self.warn('NULL substrm of file[%s] skipped' % (fn))
                continue

            pb.setId('%s' % (fn))
            self.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))
            subStrmsAdded.append(pb.id)

        return subStrmsAdded

    def loadOnline(self, evtype, symbol, nSampleLast =1, saveAs=None, saveDir =None, nextYield=defaultNextYield): # return True if succ

        if not symbol or len(symbol) <=0 or not evtype in [EVENT_KLINE_1DAY, EVENT_KLINE_5MIN, EVENT_MONEYFLOW_1DAY, EVENT_MONEYFLOW_1MIN] :
            self.debug('%s of %s does not support online query as source' % (evtype, symbol))
            return None, None, []

        crawl = SinaCrawler(self.program, None)
        dtStart, dtEnd = self.datetimeRange
        days = (datetime.now() - dtStart).days +2
        if days > 1000: days =0 # seems not specified
        if days > 500: days =500
        if days < 240:
            days =240 # online query of XX1d will always cover minimal one year
            if dtEnd < datetime.now():
                days += (datetime.now() - dtEnd).days

        self.debug('taking online query as source of %s/%s of %ddays' % (evtype, symbol, days))
        httperr, dataseq = 500, []
        saveFilename = saveAs
        if (not saveFilename or len(saveFilename) <=0) and saveDir and len(saveDir) >0:
            saveYYMMDD = (datetime.now() - timedelta(hours=15, minutes=30)).strftime('%Y%m%d') # 16hr fit to the latest date
            saveFilename = os.path.join(saveDir, '%s_%s%s.json' %(symbol, chopMarketEVStr(evtype), saveYYMMDD)) if saveDir else None

        for retryNo in range(10) :
            if evtype == EVENT_KLINE_1DAY :
                httperr, dataseq = crawl.GET_RecentKLines(symbol, 240, days, saveAs=saveFilename)
            elif evtype == EVENT_KLINE_5MIN :
                httperr, dataseq = crawl.GET_RecentKLines(symbol, 5, 48*min(10, days), saveAs=saveFilename)
            elif evtype == EVENT_MONEYFLOW_1DAY :
                httperr, dataseq = crawl.GET_MoneyFlow(symbol, days, False, saveAs=saveFilename)
            elif evtype == EVENT_MONEYFLOW_1MIN :
                httperr, dataseq = crawl.GET_MoneyFlow(symbol, 240*min(5, days), True, saveAs=saveFilename)

            if 200 == httperr:
                self.__cachedFiles.append(saveFilename)
                break

            if httperr in [408, 456]:
                secYield = nextYield(retryNo)
                self.error("load() query(%s:%s) failed, err(%s) len(%d), yield %ssec" %(symbol, evtype, httperr, len(dataseq), secYield))
                if secYield >0:
                    sleep(secYield)
                    continue

            self.error("load() query(%s:%s) failed, err(%s) len(%d)" %(symbol, evtype, httperr, len(dataseq)))
            return httperr, None, []

        # succ at query
        if not isinstance(dataseq, list) or len(dataseq) <=0:
            self.warn("load() query(%s:%s) %s got empty list" %(symbol, evtype, httperr))
            return httperr, None, []

        pb, c = hist.Playback(symbol, program=self.program), 0
        pb.setId('Online.%s/%s' % (evtype, symbol))
        for i in dataseq:
            if dtEnd and i.asof > dtEnd:
                del dataseq[c:]
                break

            ev = Event(evtype)
            ev.setData(i)
            pb.enquePending(ev)
            c+=1

        self.addStream(pb)
        self.info('load() added online query(%s:%s) result len[%d]' % (evtype, symbol, c))

        return httperr, pb.id, dataseq[- min(len(dataseq), nSampleLast) :]

    def loadOfflineJson(self, evtype, symbol, filename, nSampleLast =1): # return True if succ
        
        dtStart, dtEnd = self.datetimeRange
        with open(filename, "rb") as f:
            pb, dataseq = self.__jsonToPlayback(f, symbol, evtype, dtEnd)
            if not pb :
                self.warn('NULL substrm of file[%s] skipped' % (fn))
                return None, edseq

            pb.setId(filename)
            self.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))

        return pb.id, dataseq[- min(len(dataseq), nSampleLast) :]

    def loadJsonH5t(self, evtype, symbol, filename, mfn = None, nSampleLast =1): # return True if succ
        '''
        '''
        
        pb, dataseq = None, []

        dtStart, dtEnd = self.datetimeRange

        if not mfn or len(mfn) <=0:
            evtag = chopMarketEVStr(evtype)

            memberfiles=[]
            try:
                memberfiles = h5tar.list_utf8(filename)
            except Exception:
                self.error('failed to list member files in %s by %s and %s' % (filename, symbol, evtype))

            for mf in memberfiles:
                if mf['size'] < 10: continue
                bfn = os.path.basename(mf['name'])
                if '.json' != bfn[-5:] or not evtag in bfn or not symbol in bfn:
                    continue

                mfn = mf['name']
                break

        if not mfn or len(mfn) <=0:
            self.error('failed to find suitable member file in %s by %s and %s' % (filename, symbol, evtype))
            return None, []

        jbody = h5tar.read_utf8(filename, mfn)
        if not jbody or len(jbody) <=0: return None, []

        pb, dataseq = self.__jsonStrToPlayback(jbody, symbol, evtype, dtEnd)
        if not pb : return None, []

        pb.setId('%s@%s' % (mfn, filename))
        self.addStream(pb)
        self.info('added substrm[%s] into mux' % (pb.id))

        return pb.id if pb else None, dataseq[- min(len(dataseq), nSampleLast) :]

    def loadOffline(self, evtype, fnSearch, minFn=None): # return True if succ
        
        subStrmsAdded =[]
        dirOnly, parentDir = False, os.path.dirname(fnSearch)
        if '/' == fnSearch[-1]:
            fnSearch = fnSearch[:-1]
            dirOnly, parentDir = True, os.path.dirname(fnSearch)

        fnAll = listAllFiles(parentDir, fileOnly = not dirOnly)
        fnSearch = os.path.basename(fnSearch)
        srcPaths = []
        for fn in fnAll:
            if dirOnly :
                if '/' != fn[-1] : continue
                fn = fn[:-1]
            if fnmatch.fnmatch(os.path.basename(fn), fnSearch):
                if minFn and len(minFn) >8 and minFn > fn: continue # skip some old files
                srcPaths.append(fn)

        self.info('associated %d paths of event[%s]: %s' % (len(srcPaths), evtype, ','.join(srcPaths)))

        for tn in srcPaths:
            bname = os.path.basename(tn)
            if '.json' == bname[-5:].lower() :
                symbol = bname[:bname.index('_')]
                if not symbol in self.__symbols:
                    continue

                pbId, edseq = self.loadOfflineJson(evtype, symbol, tn)
                continue

            if 'sina' == bname[:4].lower() :
                self.__extractedFolder_sinaJson(tn, evtype) if dirOnly else self.__extractJsonTarball(tn, evtype)
                continue

            if 'realtime' == evtype :
                if 'advisor' == bname[:len('advisor')] :
                    subStrmsAdded += self.__extractAdvisorTarball(tn) # self.__extractAdvisorStreams(tn)
                elif 'advmd' == bname[:len('advmd')] : # the pre-filtered tcsv from original advisor.tcsv
                    subStrmsAdded += self.__extractedFolder_advmd(tn) if dirOnly else self.__extractAdvMdTarball(tn)
                continue
        
        return subStrmsAdded

########################################################################
class SinaMerger(hist.PlaybackApp) :

    def __init__(self, program, playback, **kwargs):
        '''Constructor
        '''
        super(SinaMerger, self).__init__(program, playback, **kwargs)

        self.__marketState = PerspectiveState(exchange="AShare")
        self.__mfMerger   = {} # dict of symbol to SinaMF1mToXm
        self.__delayedQuit =100

    def doAppInit(self): # return True if succ
        if not super(SinaMerger, self).doAppInit() :
            return False
        
        # subscribing, see OnEvent()
        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
        self.subscribeEvents([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_5MIN, EVENT_MONEYFLOW_1DAY])

        self.pb.load()
        for s in self.pb.symbols:
            self.__mfMerger[s] = SinaMF1mToXm(self.__onMF5mMerged, 5)

        return True

    def doAppStep(self):
        ev = None

        if self.__delayedQuit <=0:
            self.program.stop()
            return 0

        try :
            ev = next(self.pb)
            if not ev or not ev.data.symbol in self.pb.symbols:
                return 1

            # self.debug('filtered ev: %s' % ev.desc)
            ev = self.__marketState.updateByEvent(ev)
            if ev: 
                # yes, for merging only, a more straignt-foward way is to directly call self._recorder.pushRow(ev.type, ev.data) here
                # postEvent() then do recording in OnEvent() seems wordy, but allow other applications, which may join the prog, to be
                # able to process the merged events at the same time while merging
                self.postEvent(ev)

            if ev and EVENT_MONEYFLOW_1MIN == ev.type:
                self.__mfMerger[ev.data.symbol].pushMF1m(ev.data)

        except StopIteration:
            self.__delayedQuit -=1
        
        return 1

    def __onMF5mMerged(self, mf5m):
        ev = Event(EVENT_MONEYFLOW_5MIN)
        ev.setData(mf5m)
        ev = self.__marketState.updateByEvent(ev)
        if ev: 
            self.postEvent(ev)
            self.debug("onMF5mMerged() merged: %s ->psp: %s" % (mf5m.desc, self.__marketState.descOf(mf5m.symbol)))

def _makeupMux(simulator, dirOffline):

    dtStart, _ = simulator._wkHistData.datetimeRange
    symbol = simulator._tradeSymbol
    # associatedEvents=[]

    # part.1 the weekly tcsv collection by advisors that covers KL5m, MF1m, and Ticks
    # in the filename format suchm as SZ000001_sinaWk20200629.tcsv
    fnAll = listAllFiles(dirOffline)
    fnFilter_SinaWeek = '%s_sinaWk[0-9]*.tcsv' % symbol
    fnFilter_SinaJson = {
        EVENT_KLINE_5MIN: '%s_KL5m[0-9]*.json' % symbol,
        EVENT_KLINE_1DAY: '%s_KL1d[0-9]*.json' % symbol,
        EVENT_MONEYFLOW_1MIN: '%s_MF1m[0-9]*.json' % symbol,
        EVENT_MONEYFLOW_1DAY: '%s_MF1d[0-9]*.json' % symbol,
    }

    for fn in fnAll:
        bfn = os.path.basename(fn)
        if fnmatch.fnmatch(bfn, fnFilter_SinaWeek):
            simulator.debug("_makeupMux() loading offline sinaWeek: %s" %(fn))
            f = open(fn, "rb")
            pb = hist.TaggedCsvStream(f, program=simulator.program)
            pb.setId(bfn)
            # pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
            # pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
            pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
            pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)

            pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
            pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
            pb.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)
            simulator._wkHistData.addStream(pb)
            # associatedEvents += [EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_5MIN, EVENT_MONEYFLOW_1DAY]
            continue

        for et in [EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]:
            if fnmatch.fnmatch(bfn, fnFilter_SinaJson[et]):
                simulator.debug("_makeupMux() loading offline %s json: %s" %(et, fn))
                f = open(fn, "rb")
                pb = hist.TaggedCsvStream(f, program=simulator.program)
                pb.setId(bfn)


    # part.2 the online daily-data that supply a long term data that might be not covered by the above advisors
    crawl = SinaCrawler(simulator.program, None)
    days = (datetime.now() - dtStart).days +2
    if days > 300: days =300

    # part.2.1 EVENT_KLINE_1DAY
    evtype = EVENT_KLINE_1DAY
    simulator.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
    httperr, dataseq = crawl.GET_RecentKLines(symbol, 240, days)
    if 200 != httperr or len(dataseq) <=0:
        simulator.error("_makeupMux() GET_RecentKLines(%s:%s) failed, err(%s) len(%d)" %(symbol, evtype, httperr, len(dataseq)))
    else:
        # succ at query
        pb, c = hist.Playback(symbol, program=simulator.program), 0
        for i in dataseq:
            ev = Event(evtype)
            ev.setData(i)
            pb.enquePending(ev)
            c+=1

        simulator._wkHistData.addStream(pb)
        simulator.info('_makeupMux() added online query as source of event[%s] len[%d]' % (evtype, c))

    evtype = EVENT_MONEYFLOW_1DAY
    simulator.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
    httperr, dataseq = crawl.GET_MoneyFlow(symbol, days, False)
    if 200 != httperr or len(dataseq) <=0:
        simulator.error("_makeupMux() GET_MoneyFlow(%s:%s) failed, err(%s) len(%d)" %(symbol, evtype, httperr, len(dataseq)))
    else:
        # succ at query
        pb, c = hist.Playback(symbol, program=simulator.program), 0
        for i in dataseq:
            ev = Event(evtype)
            ev.setData(i)
            pb.enquePending(ev)
            c+=1

        simulator._wkHistData.addStream(pb)
        simulator.info('_makeupMux() added online query as source of event[%s] len[%d]' % (evtype, c))

    return simulator._wkHistData.size >0

########################################################################
class Sina_Tplus1(IdealTrader_Tplus1):
    '''
    Sina_Tplus1 extends IdealTrader_Tplus1 based on online and offline data collected from Sina
    '''
    def __init__(self, program, trader, symbol, dirOfflineData, **kwargs):
        '''Constructor
        '''
        mux = hist.PlaybackMux(program=program) # not start/end data specified, startDate =startDate, endDate=endDate)

        super(Sina_Tplus1, self).__init__(program, trader, histdata=mux, **kwargs) # mux will be kept as self._wkHistData

        self._dirOfflineData = dirOfflineData
        self._tradeSymbol = symbol

    def doAppInit(self): # return True if succ
        # load the offline tcsv streams and the online daily streams int self._wkHistData as a PlaybackMux
        if not _makeupMux(self, self._dirOfflineData):
            return False

        if not super(Sina_Tplus1, self).doAppInit() :
            return False
        
        return True

    # Directly takes that of IdealTrader_Tplus1
    #  - def OnEvent(self, ev)
    #  - def resetEpisode(self)
    #  - def OnEpisodeDone(self, reachedEnd=True)
    #  - def doAppStep(self)

"""
########################################################################
class SinaSwingScanner(ShortSwingScanner):
    '''
    ShortSwingScanner extends OfflineSimulator by scanning the MarketEvents occurs up to several days, determining
    the short trend
    '''
    def __init__(self, program, trader, symbol, dirOfflineData, f4schema=None, **kwargs):
        '''Constructor
        '''
        mux = hist.PlaybackMux(program=program) # not start/end data specified, startDate =startDate, endDate=endDate)

        super(SinaSwingScanner, self).__init__(program, trader, mux, **kwargs)

        self._dirOfflineData = dirOfflineData
        self._tradeSymbol = symbol

    def doAppInit(self): # return True if succ
        # load the offline tcsv streams and the online daily streams int self._wkHistData as a PlaybackMux
        if not _makeupMux(self, self._dirOfflineData):
            return False

        if not super(SinaSwingScanner, self).doAppInit() :
            return False

        return True
"""

########################################################################
# utility funcs
def listAllSymbols(prog, maxRetryAt456=20):

    lstSH, lstSZ = [], []
    md = SinaCrawler(prog, None)

    httperr, retryNo =408, 0
    for i in range(maxRetryAt456):
        httperr, lstSH = md.GET_AllSymbols('SH')
        if 2 == int(httperr/100): break

        prog.warn('SH-resp(%d) len=%d' %(httperr, len(lstSH)))
        if httperr in [408, 456]:
            retryNo += 1
            sleep(defaultNextYield(retryNo))
            continue

    prog.info('SH-resp(%d) len=%d' %(httperr, len(lstSH)))
    if len(lstSH) <=0:
        return lstSH, lstSZ

    httperr, retryNo =408, 1
    for i in range(maxRetryAt456):
        httperr, lstSZ = md.GET_AllSymbols('SZ')
        if 2 == int(httperr/100): break

        prog.warn('SZ-resp(%d) len=%d' %(httperr, len(lstSZ)))
        if httperr in [408, 456]:
            retryNo += 1
            sleep(defaultNextYield(retryNo))
            continue

    prog.info('SZ-resp(%d) len=%d' %(httperr, len(lstSZ)))
    return lstSH, lstSZ

def __txtfile2list(filename):
    # populate all strategies under the current vn package
    filepath = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(filepath, filename)

    lst =[]
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n\r\t ').upper()
            if line[:2] not in ['SH', 'SZ']: continue
            lst.append(line) 
    
    return lst

def listAllETFs():
    return __txtfile2list('AllETFs.txt')

def listAllIndexs():
    return __txtfile2list('AllIndex.txt')

########################################################################
def readArchivedDays(prog, dirArchived, symbol, YYYYMMDDs):
    YYYYMMDDs = copy.copy(YYYYMMDDs)
    if isinstance(YYYYMMDDs, str):
        YYYYMMDDs = [YYYYMMDDs]

    YYYYMMDDs.sort()

    all_lines=''
    readtxn = ''
    for yymmdd in YYYYMMDDs:
        fnArch = os.path.join(dirArchived, 'SinaMDay_%s.h5t' % yymmdd)
        memName = '%s_day%s.tcsv' %(symbol, yymmdd)
        try :
            lines = ''
            lines = h5tar.read_utf8(fnArch, memName)
            if lines and len(lines) >0 :
                all_lines += '\n' + lines
            readtxn += '%s(%dB)@%s, ' % (memName, len(lines), fnArch)
        except:
            prog.error('readArchivedDays() failed to read %s from %s' % (memName, fnArch))

    prog.info('readArchivedDays() read %s' % readtxn) 
    return all_lines # take celery's compression instead of return bz2.compress(all_lines.encode('utf8'))

########################################################################
def populateMuxFromWeekDir(prog, dirArchived, symbol, dtStart = None):

    mux = SinaMux(program=prog) # the result to return
    mux.setId(dirArchived)

    wkStart, yymmddStart = '', ''
    if dtStart:
        year, weekNo, YYYYMMDDs = sinaWeekOf(dtStart)
        wkStart = 'Sina%04dW%02d_' % (year, weekNo)
        yymmddStart = dtStart.strftime('%Y%m%d')

    allfiles = listAllFiles(dirArchived)
    fnSinaDays, fnSinaWeeks = [], []
    for fn in allfiles:
        bn = os.path.basename(fn)
        if fnmatch.fnmatch(bn, 'SinaDay_*.h5t') and bn >= 'SinaDay_%s' %yymmddStart :
            fnSinaDays.append(fn)
            continue

        if fnmatch.fnmatch(bn, 'Sina*W*_*-*.h5t'):
            fnSinaWeeks.append(fn)
            continue

    fnSinaWeeks.sort()
    fnSinaDays.sort()
    allfiles =[]
    ev1dIncluded = []

    for fn in fnSinaWeeks:
        bn = os.path.basename(fn)
        mw = re.match(r'Sina([0-9]*)W([0-9]*)_([0-9]*)-([0-9]*).h5t', bn)
        if not mw : continue

        yymmdd = '%s%s' % (mw.group(1), mw.group(3))
        if yymmdd > yymmddStart : 
            idxFnD = 0
            for fnD in fnSinaDays :
                bnD = os.path.basename(fnD)
                m = re.match(r'SinaDay_([0-9]*).h5t', bnD)
                if m and m.group(1) >= yymmdd: break

                idxFnD += 1 
                if not m or m.group(1) < yymmddStart: continue

                memName = '%s_day%s.tcsv' %(symbol, m.group(1))
                lines = h5tar.read_utf8(fnD, memName)
                if len(lines) <=0: continue
                pb = hist.TaggedCsvStream(StringIO(lines), program=prog)
                pb.setId('%s@%s' % (symbol, bnD))
                pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
                pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
                pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
                pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)

                pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
                pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
                pb.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)
                mux.addStream(pb)
            
            if idxFnD >0:
                del fnSinaDays[: idxFnD]

        wmem = '%s_%sW%s.tcsv' %(symbol, mw.group(1), mw.group(2))
        lines = h5tar.read_utf8(fn, wmem)
        if len(lines) <=0: continue
        pb = hist.TaggedCsvStream(StringIO(lines), program=prog)
        pb.setId('%s@%s' % (symbol, bn))
        pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)

        pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        pb.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

        if not mux.subStreamIds or len(mux.subStreamIds) <=0:
            mux._setDateRange(yymmdd + 'T000000')
        mux.addStream(pb)

        if len(ev1dIncluded) <2:
            mlst = h5tar.list_utf8(fn) # this mem-name listing is a bit stupid but MF1d/KL1d may have days-not-open
            for wmem in mlst:
                wmem = wmem['name']
                if not symbol in wmem: continue
                evt = MARKETDATE_EVENT_PREFIX + wmem.split('/')[0]
                if not evt in [EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1DAY] or evt in ev1dIncluded: continue
                lines = h5tar.read_utf8(fn, wmem)
                if len(lines) <=0: continue
                if '.json' == wmem[-5:] :
                    mux.importJsonSequence(lines, symbol, evt)
                    ev1dIncluded.append(evt)
                elif '.csv' == wmem[-4:] :
                    pb = hist.CsvStream(symbol, StringIO(lines), MoneyflowData.COLUMNS if EVENT_MONEYFLOW_PREFIX in evt else KLineData.COLUMNS, evtype =evt, program=prog)
                    pb.setId('csv:%s.%s' % (symbol, evt))
                    mux.addStream(pb)
                    
                    ev1dIncluded.append(evt)

                if len(ev1dIncluded) >=2: break

        yymmddStart = '%s%s' % (mw.group(1), mw.group(4))

    subStrmIds = mux.subStreamIds
    if prog: prog.info('populateArchivedDir() populated read %d substrms: %s' % (len(subStrmIds), ','.join(subStrmIds)))
    return mux

########################################################################
def sinaWeekOf(dtInWeek=None):
    '''
    dtInWeek = 2020-12-21(Mon) ~ 2020-12-27(Sun) all lead to the week Monday2020-12-21 ~ Sunday2020-12-27
    '''
    if not dtInWeek:
        dtInWeek = datetime.now()

    year, weekNo, wday = dtInWeek.isocalendar()
    monday = (dtInWeek - timedelta(days = (wday +6) %7)).replace(hour=0,minute=0,second=0,microsecond=0)
    friday = monday + timedelta(days = 6) - timedelta(microseconds=1)
    
    YYYYMMDDs = [ (monday + timedelta(days=i, hours=1)).strftime('%Y%m%d') for i in range(7) ]
    return year, weekNo, YYYYMMDDs

# ----------------------------------------------------------------------
def archiveWeek(dirArchived, symbols, dtInWeek=None, prog=None):
    year, weekNo, YYYYMMDDs = sinaWeekOf(dtInWeek)
    fnOut = os.path.join(dirArchived, 'Sina%04dW%02d_%s-%s.h5t' % (year, weekNo, YYYYMMDDs[0][4:], YYYYMMDDs[4][4:]))

    if symbols and not isinstance(symbols, list):
        symbols = symbols.split(',')
    
    if not symbols or len(symbols) <=0:
        # populate all symbols from SinaMDay_%s.h5t
        symbols = []
        for yymmdd in YYYYMMDDs:
            fnMDay = os.path.join(dirArchived, 'SinaMDay_%s.h5t' % yymmdd)
            try :
                mlst = h5tar.list_utf8(fnMDay)
                for m in mlst:
                    s = m['name'].split('_')[0]
                    if not s in symbols: symbols.append(s)
            except: pass

    if prog: prog.debug('archiveWeek() determined week of %s is %s, archiving %d symbols into %s' % (dtInWeek, ','.join(YYYYMMDDs), len(symbols), fnOut))

    slist=[]
    for symbol in symbols: 
        linesMday=''
        readtxn = []

        evt1ds = ['MF1d', 'KL1d']
        json1ds, jsonName1ds = [None, None], [None, None]
        csv1ds, csvName1ds = [None, None], [None, None]

        for yymmdd in YYYYMMDDs:
            fnMDay = os.path.join(dirArchived, 'SinaMDay_%s.h5t' % yymmdd)
            memName = '%s_day%s.tcsv' %(symbol, yymmdd)
            try :
                lines = ''
                lines = h5tar.read_utf8(fnMDay, memName)
                if lines and len(lines) >0 :
                    linesMday += '\n' + lines
                readtxn.append('%s(%dB)@%s' % (memName, len(lines), fnMDay))
            except:
                if prog: prog.error('archiveWeek() failed to read %s from %s' % (memName, fnMDay))

            for i in range(len(evt1ds)):
                if json1ds[i]: continue # we take the eariest of 1d
                fn = os.path.join(dirArchived, 'Sina%s_%s.h5t' % (evt1ds[i], yymmdd))
                memName = '%s_%s%s.json' %(symbol, evt1ds[i], yymmdd)
                try :
                    lines = ''
                    os.stat(fn) # check if file exists
                    lines = h5tar.read_utf8(fn, memName)
                    if not lines or len(lines) <=0 : continue

                    json1ds[i] = lines
                    jsonName1ds[i] = memName
                    readtxn.append('%s(%dB)@%s' % (memName, len(lines), fn))

                except Exception as ex:
                    if prog: prog.logexception(ex, 'archiveWeek() failed to read %s from %s' % (memName, fn))

        # write the week into fnOut
        if linesMday and len(linesMday) >0:
            h5tar.write_utf8(fnOut, '%s_%04dW%02d.tcsv' % (symbol, year, weekNo), linesMday, createmode='a')

        for i in range(len(evt1ds)):
            if not json1ds[i] or len(json1ds[i]) <=0: continue
            # h5tar.write_utf8(fnOut, '%s/%s' % (evt1ds[i], jsonName1ds[i]), json1ds[i], createmode='a')

            memName = '%s/%s.csv' % (evt1ds[i], jsonName1ds[i][:-5]) # replace the file extname
            colnames = []
            edseq = []
            if 'KL1d' == evt1ds[i]:
                colnames = KLineData.COLUMNS
                edseq    = SinaCrawler.convertToKLineDatas(symbol, json1ds[i])
            elif 'MF1d' == evt1ds[i]:
                colnames = MoneyflowData.COLUMNS
                edseq    = SinaCrawler.convertToMoneyFlow(symbol, json1ds[i], False)
            
            if isinstance(colnames, str):
                colnames = colnames.split(',')

            fcsv = StringIO()
            fcsv.write(','.join(colnames) +'\r\n') # the head line
            for ed in edseq:
                row = ed.__dict__
                cols = [str(row[col]) for col in colnames]
                fcsv.write(','.join(cols) +'\r\n')

            strdata = fcsv.getvalue()
            h5tar.write_utf8(fnOut, memName, strdata, createmode='a')
            readtxn.append('convert[%s]to %s(%d)' % (jsonName1ds[i], memName, len(strdata)))

        if prog: prog.debug('archiveWeek() %s archived %s' % (fnOut, ','.join(readtxn)))
        slist.append(symbol)

    return fnOut, slist

########################################################################
def determineLastDays(prog, nLastDays =7, todayYYMMDD= None):
    lastYYMMDDs = []
    if not todayYYMMDD:
        todayYYMMDD = datetime.now().strftime('%Y%m%d')

    symbol = 'SH000001'  # 上证指数
    playback = SinaMux(prog)

    lastDays = []
    httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, 'SH000001', nLastDays+3)
    lastDays.reverse()
    for i in lastDays:
        yymmdd = i.asof.strftime('%Y%m%d')
        if yymmdd > todayYYMMDD:
            continue
        lastYYMMDDs.append(yymmdd)
        if len(lastYYMMDDs) >= nLastDays:
            break
    
    prog.debug('determineLastDays() last %d trade-days are %s according to %s' % (nLastDays, ','.join(lastYYMMDDs), symbol))
    return lastYYMMDDs


# ----------------------------------------------------------------------
def convertJsonTarToCsvH5t(jsonTarfn, csvh5, evtype):

    colnames = MoneyflowData.COLUMNS
    if isinstance(colnames, str) :
        colnames = colnames.split(',')

    tar = tarfile.open(jsonTarfn)
    for member in tar.getmembers():
        basename = os.path.basename(member.name)
        if not basename.split('.')[-1] in ['json']: 
            continue

        basename = basename.split('.')[0]
        symbol = basename.split('_')[0]
        memName = '%s.csv' % basename # replace the file extname

        try :
            with tar.extractfile(member) as f:
                content = f.read().decode('utf-8')
                edseq   = SinaCrawler.convertToMoneyFlow(symbol, content, False, maxEvents=-1)

                fcsv = StringIO()
                fcsv.write(','.join(colnames) +'\r\n') # the head line
                for ed in edseq:
                    row = ed.__dict__
                    cols = [str(row[col]) for col in colnames]
                    fcsv.write(','.join(cols) +'\r\n')

                h5tar.write_utf8(csvh5, memName, fcsv.getvalue(), createmode='a')
        except Exception as ex:
            print('convertJsonTarToCsvH5t() converting %s caught: %s' %(member.name, ex))

####################################
from time import sleep
if __name__ == '__main__':

    # convertJsonTarToCsvH5t('/mnt/i/ETF-u20hp01.haswell/SinaMF1d_20200620.tar.bz2', '/mnt/e/AShareSample/SinaMF1d_20200620.h5t', EVENT_MONEYFLOW_1DAY)
    convertJsonTarToCsvH5t(sys.argv[1], sys.argv[2], EVENT_MONEYFLOW_1DAY)

    # ret = listAllETFs()
    # ret = listAllIndexs()

    from Application import Program

    prog = Program(name='test', argvs=[])
    prog._heartbeatInterval =-1
    prog.setLogLevel('debug')

    dirArched = '/mnt/e/AShareSample/hpx_archived/sina'
    symbol = 'SZ002008'

    mux = populateMuxFromWeekDir(prog, dirArched, symbol, dtStart = None)
    try :
        while True:
            ev = next(mux)
            if ev: print(ev.desc)
    except: pass
    exit(0)

    dtInWeek = datetime(year=2020, month=12, day=21) # a Monday
    # dtInWeek = datetime(year=2020, month=12, day=26) # a Satday
    # dtInWeek = datetime(year=2020, month=12, day=27) # a Sunday
    archiveWeek(dirArched, [symbol, 'SH510050'], dtInWeek, prog)
    # archiveWeek(dirArched, None, dtInWeek, prog)

    alllines = readArchivedDays(prog, dirArched, symbol, ['20201221', '20201222'])
    # print(alllines)

    dirTickets = '/mnt/e/AShareSample/hpx_archived/tickets'
    with bz2.open(os.path.join(dirTickets, 'Tickets_%s.tcsv.bz2' % symbol), 'wt', encoding='utf-8') as f:
        f.write(alllines)

