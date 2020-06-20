import HistoryData as hist
from MarketData import *
from Perspective import PerspectiveState
from EventData   import datetime2float
from Application import *
from TradeAdvisor import EVENT_ADVICE, DictToAdvice
from crawler import crawlSina as sina

import os, fnmatch, tarfile

EXECLUDE_LIST = ["SH600005"]

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

        self.__mux = hist.PlaybackMux(program=program, startDate =startDate, endDate=endDate)
        self.__delayedQuit =100
        self.__marketState = PerspectiveState(exchange="AShare")

    def doAppInit(self): # return True if succ
        if not super(SinaMerger, self).doAppInit() :
            return False

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

            memberExts = ['json']
            for tn in self.__tarballs[evtype]:
                tar = tarfile.open(tn)

                for member in tar.getmembers():
                    basename = os.path.basename(member.name)
                    if not basename.split('.')[-1] in memberExts: 
                        continue

                    symbol = basename[:basename.index('_')]

                    if symbol in EXECLUDE_LIST or (self._symbolLookFor and self._symbolLookFor != symbol):
                        continue

                    self.debug('memberFile[%s] in %s matched et[%s]' % (member.name, tn, evtype))
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

                    pb = hist.Playback(symbol, program=thePROG)
                    for ed in edseq:
                        ev = Event(evtype)
                        ev.setData(ed)
                        pb.enquePending(ev)

                    self.__mux.addStream(pb)
                    if self._symbolLookFor and self._symbolLookFor == symbol:
                        break # do not scan the tar anymore

                    # if self.__mux.size > 5: break # TODO: DELETE THIS LINE

        self.info('inited mux with %d substreams' % (self.__mux.size))
        return self.__mux.size >0

    def OnEvent(self, event):
        pass
    
    def doAppStep(self):
        ev = None

        if self.__delayedQuit <=0:
            self.program.stop()
            return 0

        try :
            ev = next(self.__mux)
            self.debug('filtered ev: %s' % ev.desc)
            ev = self.__marketState.updateByEvent(ev)
            if ev: 
                self._recorder.pushRow(ev.type, ev.data)
        except StopIteration:
            self.__delayedQuit -=1
        
        return 1

if __name__ == '__main__':

    thePROG = Program()
    thePROG._heartbeatInterval =-1
    srcFolder = '/mnt/e/AShareSample/SinaWeek'

    SYMBOL='SZ002008'
    
    tarNamePats={
        'tarNamePat_KL5m' : '%s/SinaKL5m_*.tar.bz2' %srcFolder,
        'tarNamePat_MF1m' : '%s/SinaMF1m_*.tar.bz2' %srcFolder,
        # 'tarNamePat_Tick' : '%s/advisor.BAK*.tar.bz2' %srcFolder,
        # 'tarNamePat_KL1d' : '%s/SinaKL1d*.tar.bz2' %srcFolder,
        # 'tarNamePat_MF1d' : '%s/SinaMF1d*.tar.bz2' %srcFolder,
    }

    rec    = thePROG.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, '%s.tcsv' % SYMBOL))
    rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

    merger = thePROG.createApp(SinaMerger, recorder =rec, symbol=SYMBOL, startDate='20200601T000000', **tarNamePats)

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


