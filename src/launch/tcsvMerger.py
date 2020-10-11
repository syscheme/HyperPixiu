import HistoryData as hist
from MarketData  import *
from Perspective import PerspectiveState
from EventData   import datetime2float
from Application import *
from TradeAdvisor import EVENT_ADVICE
from crawler.producesSina import SinaMerger, SinaMux


from datetime import datetime, timedelta
import os

########################################################################
class SinaWeek(SinaMerger) :
    '''
    to merge the market events collected in the recent week
    '''
    def __init__(self, program, tarNamePat_KL5m, tarNamePat_MF1m, dayInWeek =None, tarNamePat_RT=None, tarNamePat_KL1d=None, tarNamePat_MF1d=None, **kwargs):

        self._dtStart = None
        if dayInWeek and len(dayInWeek) >=8:
            for i in range(1):
                try: 
                    self._dtStart = datetime.strptime(dayInWeek, '%Y%m%d') 
                    break
                except: pass
                try: 
                    self._dtStart = datetime.strptime(dayInWeek, '%Y-%m-%d') 
                    break
                except: pass

        if not self._dtStart:
            self._dtStart = datetime.now()
        
        self._dtStart = self._dtStart.replace(hour=0, minute=0, second=0, microsecond=0)
        self._dtStart -= timedelta(days=self._dtStart.weekday()) # adjust to Monday
        dtEnd   = self._dtStart + timedelta(days=7) - timedelta(microseconds=1)

        playback = SinaMux(program, tarNamePat_KL5m=tarNamePat_KL5m, tarNamePat_MF1m=tarNamePat_MF1m, startDate =self._dtStart.strftime('%Y%m%dT000000'), endDate=dtEnd.strftime('%Y%m%dT235959'), tarNamePat_RT=tarNamePat_RT,tarNamePat_KL1d=tarNamePat_KL1d,tarNamePat_MF1d=tarNamePat_MF1d)
        super(SinaWeek, self).__init__(program, playback=playback, **kwargs)
        self.__dictRec =  {}
    
    def setSymbols(self, objectives) :
        self.pb.setSymbols(objectives)

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
            pb.setId('%s@%s' % (os.path.basename(member.name), os.path.basename(tarballName)))

            pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
            pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)
            
            # pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
            # pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
            # pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
            # pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
            # pb.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

            self.__mux.addStream(pb)

    def OnEvent(self, event):
        # see notes on postEvent() in doAppStep()
        if not MARKETDATE_EVENT_PREFIX in event.type :
            return

        symbol = event.data.symbol
        if len(self.pb.symbols) >0 and not symbol in self.pb.symbols:
            return

        if not symbol in self.__dictRec.keys():
            rec = self.program.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(thePROG.outdir, '%s_sinaWk%s.tcsv' % (symbol, self._dtStart.strftime('%Y%m%d'))))
            rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
            rec.registerCategory(EVENT_MONEYFLOW_5MIN, params={'columns': MoneyflowData.COLUMNS})
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
    
########################################################################

if __name__ == '__main__':

    allSymbols='SZ000001,SH601066,SZ000860,SZ399006,SZ399102,SZ399306' #'SH601377,SZ000636,SH510050,SH510500,SH510300' # sample
    # sys.argv += ['-x', 'SH601377,SZ000636']
    dayInWeek = datetime.now().strftime('%Y%m%d')
    dayInWeek = '20200817'
    srcFolder = '/tmp/SinaWeek.20200817' # '/mnt/e/AShareSample/SinaWeek.20200817'

    if '-x' in sys.argv :
        pos = sys.argv.index('-x')
        allSymbols = sys.argv[pos+1]
        del sys.argv[pos:pos+2]

    if '-d' in sys.argv :
        pos = sys.argv.index('-d')
        dayInWeek = sys.argv[pos+1]
        del sys.argv[pos:pos+2]

    if '-s' in sys.argv :
        pos = sys.argv.index('-s')
        srcFolder = sys.argv[pos+1]
        del sys.argv[pos:pos+2]

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Advisor.json']

    thePROG = Program()
    thePROG._heartbeatInterval =-1

    muxPathPatterns={
        'tarNamePat_KL5m' : '%s/SinaKL5m_*/' % srcFolder, # '%s/SinaKL5m_*.tar.bz2' %srcFolder,
        'tarNamePat_MF1m' : '%s/SinaMF1m_*/' % srcFolder, # '%s/SinaMF1m_*.tar.bz2' %srcFolder,
        # 'tarNamePat_RT'   : '%s/advmd_*/' % srcFolder,    # '%s/advmd_*.tar.bz2' % srcFolder, # '%s/advisor_*.tar.bz2' %srcFolder,
        'tarNamePat_KL1d' : '%s/SinaKL1d_*/' % srcFolder, # '%s/SinaKL1d_*.tar.bz2' %srcFolder,
        'tarNamePat_MF1d' : '%s/SinaMF1d_*/' % srcFolder, # '%s/SinaMF1d_*.tar.bz2' %srcFolder,
    }

    allSymbols = allSymbols.split(',')
    if len(allSymbols) <=0:
        symbolListBy = muxPathPatterns['tarNamePat_KL5m']
        fnAll = hist.listAllFiles(os.path.dirname(symbolListBy))
        symbolListBy = os.path.basename(symbolListBy)
        fnMatched = []
        for fn in fnAll:
            if fnmatch.fnmatch(os.path.basename(fn), symbolListBy):
                fnMatched.append(fn)

        if len(fnMatched) >0:
            fnMatched.sort()
            allSymbols = SinaWeek.populateSymbolList(fnMatched[-1])

    merger = thePROG.createApp(SinaWeek, dayInWeek=dayInWeek, **muxPathPatterns)
    merger.setSymbols(allSymbols) # ('SH601377,SZ000636,SH510050,SH510500,SH510300') #symoblist[:20]   

    thePROG.start()
    thePROG.setLogLevel('debug')
    thePROG.loop()
    thePROG.stop()


