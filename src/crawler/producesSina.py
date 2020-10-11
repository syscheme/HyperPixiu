# encoding: UTF-8

from __future__ import division

from Application import BaseApplication
from Simulator import IdealTrader_Tplus1, ShortSwingScanner
from EventData import Event, EventData
from MarketData import *
from Perspective import PerspectiveState
import HistoryData as hist
from crawler.crawlSina import *

from datetime import datetime, timedelta
import os
import fnmatch

########################################################################
class TcsvMerger(BaseApplication) :

    MDEVENTS_FROM_ADV = [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY] # [EVENT_TICK, EVENT_KLINE_1MIN]

    def __init__(self, program, tarNamePat_KL5m, tarNamePat_MF1m, startDate =None, endDate=None, tarNamePat_RT=None, tarNamePat_KL1d=None, tarNamePat_MF1d=None, **kwargs):
        '''Constructor
        '''
        super(TcsvMerger, self).__init__(program, **kwargs)
        self.__tnPattern = {
            EVENT_KLINE_5MIN:     tarNamePat_KL5m,
            EVENT_KLINE_1DAY:     tarNamePat_KL1d,
            EVENT_MONEYFLOW_1MIN: tarNamePat_MF1m,
            EVENT_MONEYFLOW_1DAY: tarNamePat_MF1d,
            'realtime':           tarNamePat_RT,
        }

        self.__tarballs = {}
        self.__symbols = []

        self.__mux = hist.PlaybackMux(program=self.program, startDate =startDate, endDate=endDate)
        self.__delayedQuit =100
        self.__marketState = PerspectiveState(exchange="AShare")

        self.__mfMerger   = {} # dict of symbol to SinaMF1mToXm

    @property
    def symbols(self) : return self.__symbols

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

    def __jsonToPlayback(self, jsonFstream, symbol, evtype):

        content = jsonFstream.read().decode()
        edseq =[]

        # dispatch the convert func up to evtype from tar filename
        if EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)]:
            edseq = SinaCrawler.convertToKLineDatas(symbol, content)
        elif EVENT_MONEYFLOW_1MIN == evtype:
            edseq = SinaCrawler.convertToMoneyFlow(symbol, content, True)
        elif EVENT_MONEYFLOW_1DAY == evtype:
            edseq = SinaCrawler.convertToMoneyFlow(symbol, content, False)

        if len(edseq) <=0: return None

        pb = hist.Playback(symbol, program=self.program)
        pb.setId('%s_%s.json' % (symbol, evtype))
        for ed in edseq:
            ev = Event(evtype)
            ev.setData(ed)
            pb.enquePending(ev)
        
        return pb

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
                
                self.__mux.addStream(pb)
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

        self.__mux.addStream(pb)

    def __advmdToPlayback(self, advmdtcsvStrm, symbol, evtype) :
        pb = hist.TaggedCsvStream(advmdtcsvStrm, program=self.program)
        pb.setId('advmd%s_%s.tcsv' % (symbol, evtype))

        pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
        return pb

    def __extractAdvMdTarball(self, tarballName):
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

            if not evtype in TcsvMerger.MDEVENTS_FROM_ADV : 
                continue # exclude some events as we collected in some other ways

            self.debug('memberFile[%s] in %s matched' % (member.name, tarballName))
            memlist.append(member) 

        for member in memlist:
            f = tar.extractfile(member)
            pb = self.__advmdToPlayback(f, symbol, evtype)
            if not pb : continue
            pb.setId('%s@%s' % (os.path.basename(member.name), os.path.basename(tarballName)))

            self.__mux.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))

    def __extractedFolder_advmd(self, folderName):
        allfiles = hist.listAllFiles(folderName, depthAllowed=1)

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

            if not evtype in TcsvMerger.MDEVENTS_FROM_ADV : 
                continue # exclude some events as we collected in some other ways

            self.debug('file[%s] matched' % (fn))
            memlist.append(fn) 

        for fn in memlist:
            f = open(fn, "r")
            pb = self.__advmdToPlayback(f, symbol, evtype)
            if not pb : continue
            pb.setId('%s' % (fn))

            self.__mux.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))

    def __extractedFolder_sinaJson(self, folderName, evtype):
        allfiles = hist.listAllFiles(folderName, depthAllowed=2)

        for fn in allfiles:
            '''
            -rw-rw-rw- root/root     20590 2020-09-12 16:26 SZ002588_MF1m20200817.json
            '''
            basename = os.path.basename(fn)
            if not fnmatch.fnmatch(basename, 'S[HZ][0-9]*_%s*.json' % evtype[len(MARKETDATE_EVENT_PREFIX):]):
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
            self.__mux.addStream(pb)
            self.info('added substrm[%s] into mux' % (pb.id))

    def doAppInit(self): # return True if succ
        if not super(TcsvMerger, self).doAppInit() :
            return False

        # subscribing, see OnEvent()
        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
        self.subscribeEvents([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_5MIN, EVENT_MONEYFLOW_1DAY])

        for s in self.__symbols:
            self.__mfMerger[s] = SinaMF1mToXm(self.__onMF5mMerged, 5)

        for evtype in self.__tnPattern.keys():
            if not self.__tnPattern[evtype]:
                self.__tarballs[evtype] = None
                continue

            self.__tarballs[evtype] = []
            fnSearch=self.__tnPattern[evtype]

            dirOnly, parentDir = False, os.path.dirname(fnSearch)
            if '/' == fnSearch[-1]:
                fnSearch = fnSearch[:-1]
                dirOnly, parentDir = True, os.path.dirname(fnSearch)

            fnAll = hist.listAllFiles(parentDir, fileOnly = not dirOnly)
            fnSearch = os.path.basename(fnSearch)
            for fn in fnAll:
                if dirOnly :
                    if '/' != fn[-1] : continue
                    fn = fn[:-1]
                if fnmatch.fnmatch(os.path.basename(fn), fnSearch):
                    self.__tarballs[evtype].append(fn)

            self.info('associated %d paths of event[%s]: %s' % (len(self.__tarballs[evtype]), evtype, ','.join(self.__tarballs[evtype])))

            for tn in self.__tarballs[evtype]:
                bname = os.path.basename(tn)

                if 'sina' == bname[:4].lower() :
                    self.__extractedFolder_sinaJson(tn, evtype) if dirOnly else self.__extractJsonTarball(tn, evtype)
                if 'realtime' == evtype :
                    if 'advisor' == bname[:len('advisor')] :
                        self.__extractAdvisorTarball(tn) # self.__extractAdvisorStreams(tn)
                    elif 'advmd' == bname[:len('advmd')] : # the pre-filtered tcsv from original advisor.tcsv
                        self.__extractedFolder_advmd(tn) if dirOnly else self.__extractAdvMdTarball(tn)

        if len(self.symbols) >0 and None in [self.__tarballs[EVENT_KLINE_1DAY], self.__tarballs[EVENT_MONEYFLOW_1DAY]]:
            crawl = SinaCrawler(self.program, None)
            dtStart, _ = self.__mux.datetimeRange
            days = (datetime.now() - dtStart).days +2
            if days > 500: days =500

            evtype = EVENT_KLINE_1DAY
            if not self.__tarballs[evtype] :
                self.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
                for s in self.symbols:
                    httperr, dataseq = crawl.GET_RecentKLines(s, 240, days)
                    if 200 != httperr or len(dataseq) <=0:
                        self.error("doAppInit() GET_RecentKLines(%s:%s) failed, err(%s) len(%d)" %(s, evtype, httperr, len(dataseq)))
                        continue

                    # succ at query
                    pb, c = hist.Playback(s, program=self.program), 0
                    for i in dataseq:
                        ev = Event(evtype)
                        ev.setData(i)
                        pb.enquePending(ev)
                        c+=1

                    self.__mux.addStream(pb)
                    self.info('added online query as source of event[%s] len[%d]' % (evtype, c))

            evtype = EVENT_MONEYFLOW_1DAY
            if not self.__tarballs[evtype] :
                self.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
                for s in self.symbols:
                    httperr, dataseq = crawl.GET_MoneyFlow(s, days, False)
                    if 200 != httperr or len(dataseq) <=0:
                        self.error("doAppInit() GET_MoneyFlow(%s:%s) failed, err(%s) len(%d)" %(s, evtype, httperr, len(dataseq)))
                        continue

                    # succ at query
                    pb, c = hist.Playback(s, program=self.program), 0
                    for i in dataseq:
                        ev = Event(evtype)
                        ev.setData(i)
                        pb.enquePending(ev)
                        c+=1

                    self.__mux.addStream(pb)
                    self.info('added online query as source of event[%s] len[%d]' % (evtype, c))

        self.info('inited mux with %d substreams' % (self.__mux.size))
        return self.__mux.size >0

    def OnEvent(self, event):
        # see notes on postEvent() in doAppStep()
        # if self._recorder: 
        #     self._recorder.pushRow(event.type, event.data)
        pass
    
    def doAppStep(self):
        ev = None

        if self.__delayedQuit <=0:
            self.program.stop()
            return 0

        try :
            ev = next(self.__mux)
            if not ev or not ev.data.symbol in self.__symbols:
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
    fnAll = hist.listAllFiles(dirOffline)
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
