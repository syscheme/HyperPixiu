import HistoryData as hist
from MarketData  import *
from Perspective import PerspectiveState
from EventData   import datetime2float
from Application import *
from TradeAdvisor import EVENT_ADVICE, DictToAdvice
from crawler import crawlSina as sina

import os, sys, fnmatch, tarfile, re

EXECLUDE_LIST = ["SH600005"]
SYMBOL='SZ002008'

class SinaMerger(BaseApplication) :
    def __init__(self, program, recorder, symbol, tarNamePat_KL5m, tarNamePat_MF1m, startDate =None, endDate=None, tarNamePat_Tick=None, tarNamePat_KL1d=None, tarNamePat_MF1d=None, **kwargs):
        '''Constructor
        '''
        super(SinaMerger, self).__init__(program, **kwargs)
        self.__tnPattern = {
            EVENT_KLINE_5MIN:     tarNamePat_KL5m,
            EVENT_KLINE_1DAY:     tarNamePat_KL1d,
            EVENT_MONEYFLOW_1MIN: tarNamePat_MF1m,
            EVENT_MONEYFLOW_1DAY: tarNamePat_MF1d,
            EVENT_TICK:           tarNamePat_Tick,
        }

        self.__tarballs = {}
        self._recorder =  recorder
        self._symbolLookFor = symbol

        if self._recorder:
            rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

        self.__mux = hist.PlaybackMux(program=program, startDate =startDate, endDate=endDate)
        self.__delayedQuit =100
        self.__marketState = PerspectiveState(exchange="AShare")

    def __extractJsonStreams(self, tarballName, evtype):
        tar = tarfile.open(tarballName)

        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if not basename.split('.')[-1] in ['json']: 
                continue

            symbol = basename[:basename.index('_')]

            if symbol in EXECLUDE_LIST or (self._symbolLookFor and self._symbolLookFor != symbol):
                continue

            self.debug('memberFile[%s] in %s matched et[%s]' % (member.name, tarballName, evtype))
            edseq =[]
            with tar.extractfile(member) as f:
                content =f.read().decode()

                # dispatch the convert func up to evtype from tar filename
                if EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)]:
                    edseq = sina.SinaCrawler.convertToKLineDatas(symbol, content)
                elif EVENT_MONEYFLOW_1MIN == evtype:
                    edseq = sina.SinaCrawler.convertToMoneyFlow(symbol, content, True)
                elif EVENT_MONEYFLOW_1DAY == evtype:
                    edseq = sina.SinaCrawler.convertToMoneyFlow(symbol, content, False)

            pb = hist.Playback(symbol, program=self.program)
            for ed in edseq:
                ev = Event(evtype)
                ev.setData(ed)
                pb.enquePending(ev)

            self.__mux.addStream(pb)
            if self._symbolLookFor and self._symbolLookFor == symbol:
                break # do not scan the tar anymore

    def __extractAdvisorStreams(self, tarballName):
        tar = tarfile.open(tarballName)
        bz2dict = {}
        memlist = []

        for member in tar.getmembers():
            # ./out/advisor.BAK20200615T084501/advisor_15737.tcsv
            # ./out/advisor.BAK20200615T084501/advisor_15737.tcsv.1.bz2
            # ./out/advisor.BAK20200615T084501/Advisor.json
            basename = os.path.basename(member.name)
            if not fnmatch.fnmatch(basename, 'advisor_*.tcsv*'):
                continue
            
            if '.tcsv' == basename[-5:]:
               memlist.append(member) 
               continue

            m = re.match(r'advisor_.*\.([0-9]*)\.bz2', basename)
            if m :
                bz2dict[int(m.group(1))] = member

        items = list(bz2dict.items())
        items.sort() 
        # insert into fnlist reversly
        for k,v in items:
            memlist.insert(0, v)

        for member in memlist:
            f = tar.extractfile(member)
            if '.bz2' == member.name[-4:]:
                f = bz2.open(f, mode='rt')

            pb = hist.TaggedCsvStream(f, program=self.program)
            pb.registerConverter(EVENT_KLINE_1MIN, DictToKLine(EVENT_KLINE_1MIN, SYMBOL))
            pb.registerConverter(EVENT_KLINE_5MIN, DictToKLine(EVENT_KLINE_5MIN, SYMBOL))
            pb.registerConverter(EVENT_KLINE_1DAY, DictToKLine(EVENT_KLINE_1DAY, SYMBOL))
            pb.registerConverter(EVENT_TICK,       DictToTick(SYMBOL))

            pb.registerConverter(EVENT_MONEYFLOW_1MIN, DictToMoneyflow(EVENT_MONEYFLOW_1MIN, SYMBOL))
            pb.registerConverter(EVENT_MONEYFLOW_1DAY, DictToMoneyflow(EVENT_MONEYFLOW_1DAY, SYMBOL))

            self.__mux.addStream(pb)

    def doAppInit(self): # return True if succ
        if not super(SinaMerger, self).doAppInit() :
            return False

        # subscribing, see OnEvent()
        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
        self.subscribeEvents([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY])

        for evtype in self.__tnPattern.keys():
            if not self.__tnPattern[evtype]:
                self.__tarballs[evtype] = None
                continue

            self.__tarballs[evtype] = []
            fnSearch=self.__tnPattern[evtype]

            fnAll = hist.listAllFiles(os.path.dirname(fnSearch))
            fnSearch = os.path.basename(fnSearch)
            for fn in fnAll:
                if fnmatch.fnmatch(os.path.basename(fn), fnSearch):
                    self.__tarballs[evtype].append(fn)

            self.info('associated %d tarballs of event[%s]: %s' % (len(self.__tarballs[evtype]), evtype, ','.join(self.__tarballs[evtype])))

            for tn in self.__tarballs[evtype]:
                bname = os.path.basename(tn)
                if 'sina' == bname[:4].lower() :
                    self.__extractJsonStreams(tn, evtype)
                elif EVENT_TICK == evtype and 'advisor' == bname[:len('advisor')] :
                    self.__extractAdvisorStreams(tn)

        if self._symbolLookFor and len(self._symbolLookFor) >5 and None in [self.__tarballs[EVENT_KLINE_1DAY], self.__tarballs[EVENT_MONEYFLOW_1DAY]]:
            crawl = sina.SinaCrawler(self.program, None)
            dtStart, _ = self.__mux.datetimeRange
            days = (datetime.now() - dtStart).days +2
            if days > 500: days =500

            evtype = EVENT_KLINE_1DAY
            if not self.__tarballs[evtype] :
                self.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
                httperr, dataseq = crawl.GET_RecentKLines(self._symbolLookFor, 240, days)
                if 200 != httperr:
                    self.error("doAppInit() GET_RecentKLines(%s:%s) failed, err(%s)" %(self._symbolLookFor, evtype, httperr))
                elif len(dataseq) >0: 
                    # succ at query
                    pb, c = hist.Playback(self._symbolLookFor, program=self.program), 0
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
                httperr, dataseq = crawl.GET_MoneyFlow(self._symbolLookFor, days, False)
                if 200 != httperr:
                    self.error("doAppInit() GET_MoneyFlow(%s:%s) failed, err(%s)" %(self._symbolLookFor, evtype, httperr))
                elif len(dataseq) >0: 
                    # succ at query
                    pb, c = hist.Playback(self._symbolLookFor, program=self.program), 0
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
        if self._recorder: 
            self._recorder.pushRow(event.type, event.data)
    
    def doAppStep(self):
        ev = None

        if self.__delayedQuit <=0:
            self.program.stop()
            return 0

        try :
            ev = next(self.__mux)
            # self.debug('filtered ev: %s' % ev.desc)
            ev = self.__marketState.updateByEvent(ev)
            if ev: 
                # yes, for merging only, a more straignt-foward way is to directly call self._recorder.pushRow(ev.type, ev.data) here
                # postEvent() then do recording in OnEvent() seems wordy, but allow other applications, which may join the prog, to be
                # able to process the merged events at the same time while merging
                self.postEvent(ev)
        except StopIteration:
            self.__delayedQuit -=1
        
        return 1

if __name__ == '__main__':

    sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Advisor.json']
    thePROG = Program()
    thePROG._heartbeatInterval =-1
    srcFolder = '/mnt/e/AShareSample/SinaWeek'

    SYMBOL='SZ002008'

    tarNamePats={
        'tarNamePat_KL5m' : None, #'%s/SinaKL5m_*.tar.bz2' %srcFolder,
        'tarNamePat_MF1m' : None, #'%s/SinaMF1m_*.tar.bz2' %srcFolder,
        'tarNamePat_Tick' : '%s/advisor.BAK*.tar.bz2' %srcFolder,
        # 'tarNamePat_KL1d' : '%s/SinaKL1d*.tar.bz2' %srcFolder,
        # 'tarNamePat_MF1d' : '%s/SinaMF1d*.tar.bz2' %srcFolder,
    }

    rec    = thePROG.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, '%s.tcsv' % SYMBOL))
    merger = thePROG.createApp(SinaMerger, recorder =rec, symbol=SYMBOL, startDate='20200601T000000', endDate='20200609T235959', **tarNamePats)

    '''
    acc = thePROG.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    tdrCore = thePROG.createApp(BaseTrader, configNode ='trader', objectives=objectives, account=acc)
    objectives = tdrCore.objectives
    SYMBOL = objectives[0]

    TEST_f4schema = {
            'asof':1, 
            EVENT_KLINE_5MIN     : 2,
            EVENT_MONEYFLOW_1MIN : 10,
    }

    tdrWraper = thePROG.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=histReader, f4schema=TEST_f4schema)
    '''

    thePROG.start()
    thePROG.setLogLevel('debug')
    thePROG.loop()
    thePROG.stop()


