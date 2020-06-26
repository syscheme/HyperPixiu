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

class SinaMerger(sina.TcsvMerger) :
    def __init__(self, program, recorder, tarNamePat_KL5m, tarNamePat_MF1m, startDate =None, endDate=None, tarNamePat_Tick=None, tarNamePat_KL1d=None, tarNamePat_MF1d=None, **kwargs):
        super(SinaMerger, self).__init__(program, tarNamePat_KL5m, tarNamePat_MF1m, startDate, endDate, tarNamePat_Tick, tarNamePat_KL1d, tarNamePat_MF1d, **kwargs)
        self.__dictRec =  {}

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

    def OnEvent(self, event):
        # see notes on postEvent() in doAppStep()
        if not MARKETDATE_EVENT_PREFIX in event.type :
            return

        symbol = event.data.symbol
        if len(self.symbols) >0 and not symbol in self.symbols:
            return

        if not symbol in self.__dictRec.keys():
            rec = self.program.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, '%s_sinaMerged.tcsv' % symbol))
            rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})
            rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})

            self.program.initApp(rec)
            self.__dictRec[symbol]=rec

        rec = self.__dictRec[symbol]
        if rec: 
            rec.pushRow(event.type, event.data)

    # def doAppStep(self):
    #     c = super(self.__class__, self).doAppStep()
    #     for rec in self.__dictRec.values():
    #         c += rec.doAppStep()
    #     return c
    
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

    # rec    = thePROG.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, '%s.tcsv' % SYMBOL))
    merger = thePROG.createApp(SinaMerger, recorder =None, symbol=SYMBOL, startDate='20200601T000000', endDate='20200630T235959', **tarNamePats)
    merger.setSymbols('SZ002008,SZ002080,SZ002007,SZ002106')

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


