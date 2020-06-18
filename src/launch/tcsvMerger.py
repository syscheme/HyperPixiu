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
    def __init__(self, program, recorder, fnSearch, symbolLookFor=None, **kwargs):
        '''Constructor
        '''
        super(SinaMerger, self).__init__(program, **kwargs)
        self.__fnSearch = fnSearch
        self._recorder =  recorder
        self._symbolLookFor = symbolLookFor

        self.__mux = hist.PlaybackMux(program=program)
        self.__delayedQuit =100
        self.__marketState = PerspectiveState(exchange="AShare")

    def doAppInit(self): # return True if succ
        if not super(SinaMerger, self).doAppInit() :
            return False

        fnAll = hist.listAllFiles(os.path.dirname(self.__fnSearch))
        fnSearch = os.path.basename(self.__fnSearch)
        tarballs = []
        for fn in fnAll:
            if fnmatch.fnmatch(os.path.basename(fn), fnSearch):
                tarballs.append(fn)

        for tn in tarballs:
            tar = tarfile.open(tn)

            # determine the eventype by tarball name
            bname = os.path.basename(tn)
            evtype = EVENT_KLINE_5MIN
            memberExts = ['json']
            if 'MF1m' in bname:
                evtype = EVENT_MONEYFLOW_1MIN
            if 'MF1d' in bname:
                evtype = EVENT_MONEYFLOW_1DAY
            if 'KL1d' in bname:
                evtype = EVENT_KLINE_1DAY

            for member in tar.getmembers():
                basename = os.path.basename(member.name)
                if not basename.split('.')[-1] in memberExts: 
                    continue

                symbol = basename[:basename.index('_')]

                if symbol in EXECLUDE_LIST or (self._symbolLookFor and self._symbolLookFor != symbol):
                    continue

                self.debug('member[%s] matched in %s' % (member.name, tn))
                edseq =[]
                with tar.extractfile(member) as f:
                    content =f.read().decode()

                    # dispatch the convert func up to evtype from tar filename
                    if EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)]:
                        edseq = sina.SinaCrawler.convertToKLineDatas(symbol, content)
                    elif EVENT_MONEYFLOW_1MIN == evtype:
                        edseq = sina.SinaCrawler.convertToMoneyFlow(symbol, content, True)

                pb = hist.Playback(symbol, program=thePROG)
                for ed in edseq:
                    ev = Event(evtype)
                    ev.setData(ed)
                    pb.enquePending(ev)

                self.__mux.addStream(pb)
                # if self.__mux.size > 5: break # TODO: DELETE THIS LINE

        self.info('inited mux with %d streams' % (self.__mux.size))
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

    fnSearch = '/mnt/e/AShareSample/Sina*.tar.bz2'

    rec    = thePROG.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, 'P%s.tcsv' % thePROG.pid))
    rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

    merger = thePROG.createApp(SinaMerger, recorder =rec, fnSearch = fnSearch, symbolLookFor='SZ002008')

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


