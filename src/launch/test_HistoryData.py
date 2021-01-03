import unittest
import HistoryData as hist
import Perspective as psp
import MarketData as md
from EventData import datetime2float
from Application import *
from TradeAdvisor import EVENT_ADVICE, AdviceData
from crawler import crawlSina as sina

import h5py

class Foo(BaseApplication) :
    def __init__(self, program, settings):
        super(Foo, self).__init__(program, settings)
        self.__step =0

    def doAppInit(self): # return True if succ
        if not super(Foo, self).doAppInit() :
            return False
        return True

    def OnEvent(self, event):
        self.info("Foo.OnEvent %s" % event)
    
    def doAppStep(self):
        self.__step +=1
        self.info("Foo.step %d" % self.__step)

import tarfile

PROGNAME = os.path.basename(__file__)[0:-3]
SYMBOL ='000001'
thePROG = Program(PROGNAME)

hs300s= [
        "600000","600008","600009","600010","600011","600015","600016","600018","600019","600023",
        "600025","600028","600029","600030","600031","600036","600038","600048","600050","600061",
        "600066","600068","600085","600089","600100","600104","600109","600111","600115","600118",
        "600153","600157","600170","600176","600177","600188","600196","600208","600219","600221",
        "600233","600271","600276","600297","600309","600332","600339","600340","600346","600352",
        "600362","600369","600372","600373","600376","600383","600390","600398","600406","600415",
        "600436","600438","600482","600487","600489","600498","600516","600518","600519","600522",
        "600535","600547","600549","600570","600583","600585","600588","600606","600637","600660",
        "600663","600674","600682","600688","600690","600703","600704","600705","600739","600741",
        "600795","600804","600809","600816","600820","600837","600867","600886","600887","600893",
        "600900","600909","600919","600926","600958","600959","600977","600999","601006","601009",
        "601012","601018","601021","601088","601099","601108","601111","601117","601155","601166",
        "601169","601186","601198","601211","601212","601216","601225","601228","601229","601238",
        "601288","601318","601328","601333","601336","601360","601377","601390","601398","601555",
        "601600","601601","601607","601611","601618","601628","601633","601668","601669","601688",
        "601718","601727","601766","601788","601800","601808","601818","601828","601838","601857",
        "601866","601877","601878","601881","601888","601898","601899","601901","601919","601933",
        "601939","601958","601985","601988","601989","601991","601992","601997","601998","603160",
        "603260","603288","603799","603833","603858","603993","000001","000002","000060","000063",
        "000069","000100","000157","000166","000333","000338","000402","000413","000415","000423",
        "000425","000503","000538","000540","000559","000568","000623","000625","000627","000630",
        "000651","000671","000709","000723","000725","000728","000768","000776","000783","000786",
        "000792","000826","000839","000858","000876","000895","000898","000938","000959","000961",
        "000963","000983","001965","001979","002007","002008","002024","002027","002044","002050",
        "002065","002074","002081","002085","002142","002146","002153","002202","002230","002236",
        "002241","002252","002294","002304","002310","002352","002385","002411","002415","002450",
        "002456","002460","002466","002468","002470","002475","002493","002500","002508","002555",
        "002558","002572","002594","002601","002602","002608","002624","002625","002673","002714",
        "002736","002739","002797","002925","300003","300015","300017","300024","300027","300033",
        "300059","300070","300072","300122","300124","300136","300144","300251","300408","300433"
        ]

class TestHistoryData(unittest.TestCase):

    def _test_baseApp(self):
        thePROG.createApp(Foo, None)
        thePROG.createApp(Foo, None)

        applst = thePROG.listApps()
        print('listed all BaseApplication: %s\n' % applst)

        thePROG.start()
        thePROG.loop()
        thePROG.stop()

    def _test_NonBlockingLoop(self):
        thePROG._heartbeatInterval =-1

        thePROG.createApp(Foo, None)
        thePROG.createApp(Foo, None)

        applst = thePROG.listApps()
        print('listed all BaseApplication: %s\n' % applst)

        thePROG.start()
        thePROG.loop()
        thePROG.stop()

    def _test_CsvPlayback(self):
        reader = hist.CsvPlayback(symbol=SYMBOL, folder='/mnt/e/AShareSample/%s' % SYMBOL, fields='date,time,open,high,low,close,volume,ammount')
        for i in reader :
            print('Row: %s\n' % i.desc)

    def _test_TaggedCsvPlayback(self):
        reader = hist.TaggedCsvPlayback(tcsvFilePath='/mnt/e/AShareSample/advisor/advisor_15737.tcsv', program=thePROG)
        reader.registerConverter(EVENT_ADVICE, AdviceData.hatch, AdviceData.COLUMNS)
        for i in reader :
            print('Row: %s' % i.desc)

    def _test_PlaybackMux(self):
        r1 = hist.TaggedCsvPlayback(tcsvFilePath='/mnt/e/AShareSample/advisor/advisor_15737.tcsv', program=thePROG)
        r1.registerConverter(EVENT_ADVICE, AdviceData.hatch, AdviceData.COLUMNS)

        r2 = hist.CsvPlayback(symbol='SH510050', folder='/mnt/e/AShareSample/ETF', fields='date,time,open,high,low,close,volume,ammount', program=thePROG)
        r3 = hist.CsvPlayback(symbol='SH510300', folder='/mnt/e/AShareSample/ETF', fields='date,time,open,high,low,close,volume,ammount', program=thePROG)
        r4 = hist.CsvPlayback(symbol='SH510500', folder='/mnt/e/AShareSample/ETF', fields='date,time,open,high,low,close,volume,ammount', program=thePROG)

        r5 = hist.TaggedCsvInTarball(fnTarball='/mnt/e/AShareSample/advisor/advisor.BAK20200615T084501.tar.bz2', program=thePROG)
        r5.registerConverter(EVENT_ADVICE, AdviceData.hatch, AdviceData.COLUMNS)
        r5.registerConverter(md.EVENT_KLINE_1MIN, hist.KLineData.hatch, hist.KLineData.COLUMNS) # r5.registerConverter(md.EVENT_KLINE_1MIN, hist.DictToKLine(md.EVENT_KLINE_1MIN, SYMBOL))

        mux = hist.PlaybackMux(program=thePROG)
        # mux.addStream(r1)
        # mux.addStream(r2)
        # mux.addStream(r3)
        # mux.addStream(r4)
        mux.addStream(r5)

        for i in mux :
            print('Row: %s' % i.desc)

    def test_SavedSinaData(self):
        mux = hist.PlaybackMux(program=thePROG)

        fn = '/mnt/e/AShareSample/SinaKL5m_20200609.tar.bz2'
        fn = '/mnt/e/AShareSample/SinaMF1m_20200609.tar.bz2'
        evtype = md.EVENT_KLINE_5MIN
        if fn.index('MF1m') >0:
            evtype = md.EVENT_MONEYFLOW_1MIN

        tar = tarfile.open(fn)
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if not basename.split('.')[-1] in ['txt', 'json', 'csv', 'tcsv']: 
                continue

            symbol = basename[:basename.index('_')]
            if symbol in ["SH600005"]: continue
            edseq =[]
            with tar.extractfile(member) as f:
                content =f.read().decode()

                # dispatch the convert func up to evtype from tar filename
                if md.EVENT_KLINE_5MIN == evtype:
                    edseq = sina.SinaCrawler.convertToKLineDatas(symbol, content)
                elif md.EVENT_MONEYFLOW_1MIN == evtype:
                    edseq = sina.SinaCrawler.convertToMoneyFlow(symbol, content, True)

            pb = hist.Playback(symbol, program=thePROG)
            for ed in edseq:
                ev = Event(evtype)
                ev.setData(ed)
                pb.enquePending(ev)
            
            mux.addStream(pb)
            if mux.size > 5:
                break

        for i in mux :
            print('Row: %s' % i.desc)

#                print("file[%s] has %d bytes, %d lines" % (member.name, len(content), 1+content.count(b'\n')))
        # for ev in evSeq:
        #     if isinstance(ev, Event)
        #         self.enquePending(self, ev)

    def _test_Tarball(self):
        fn = '/mnt/e/AShareSample/advisor/advisor.BAK20200615T084501.tar.bz2'
        tar = tarfile.open(fn)
        for member in tar.getmembers():
            if not member.name.split('.')[-1] in ['txt', 'json', 'csv', 'log', 'tcsv']: 
                continue
            with tar.extractfile(member) as f:
                content=f.read()
                print("file[%s] has %d bytes, %d lines" % (member.name, len(content), content.count(b'\n')))

    def _test_Perspective(self):
        thePROG._heartbeatInterval =-1
        ps = psp.Perspective('AShare', SYMBOL)
        
        histdata = psp.PerspectiveGenerator(ps)
        histdata.setProgram(thePROG)

        reader = hist.CsvPlayback(symbol=SYMBOL, folder='/mnt/e/AShareSample/%s' % SYMBOL, fields='date,time,open,high,low,close,volume,ammount')
        reader.setProgram(thePROG)

        histdata.adaptReader(reader, md.EVENT_KLINE_1MIN)
        marketstate = psp.PerspectiveState('AShare')

        for i in histdata :
            if psp.EVENT_Perspective != i.type :
                thePROG.info('evnt: %s' % i.desc) 
                continue

            thePROG.info('Psp: %s' % i.desc)
            marketstate.updateByEvent(i)
            s = i.data.symbol
            price, asofP = marketstate.latestPrice(s)
            thePROG.debug('-> state: asof[%s] symbol[%s] lastPrice[%s] OHLC%s\n' % (asofP.strftime('%Y%m%dT%H:%M:%S'), s, price, marketstate.dailyOHLC_sofar(s)))

    def _test_PerspToHDF5(self):
        thePROG._heartbeatInterval =-1
        ps = psp.Perspective('AShare', SYMBOL)

        # https://www.jianshu.com/thePROG/998c861d32e3
        # https://www.jianshu.com/thePROG/ae12525450e8
        h5file = h5py.File('/tmp/psp%s.h5pd'%SYMBOL, 'w')

        marketState_Len= 1552
        chunk_size= 256

        X = h5file.create_dataset(shape=(0, 1, marketState_Len),            #数据集的维度
                              maxshape = (None, 1, marketState_Len),          #数据集的允许最大维度　
                              dtype=float, name='marketState_train', compression='szip',   #数据类型、是否压缩(szip better than gzip, https://support.hdfgroup.org/doc_resource/SZIP/ but not supported bz2/bcolz)，以及数据集的名字
                              chunks=(chunk_size, 1, marketState_Len))
        
        histdata = psp.PerspectiveGenerator(ps)
        histdata.setProgram(thePROG)

        reader = hist.CsvPlayback(symbol=SYMBOL, folder='/mnt/e/AShareSample/%s.REDU' % SYMBOL, fields='date,time,open,high,low,close,volume,ammount')
        reader.setProgram(thePROG)

        histdata.adaptReader(reader, md.EVENT_KLINE_1MIN)
        marketstate = psp.PerspectiveState('AShare')

        c =0
        for i in histdata :
            if psp.EVENT_Perspective != i.type :
                thePROG.info('evnt: %s' % i.desc) 
                continue

            thePROG.info('Psp: %s' % i.desc)
            marketstate.updateByEvent(i)
            s = i.data.symbol
            nnfloats = [0.0] * md.EXPORT_FLOATS_DIMS
            nnfloats[0] = datetime2float(marketstate.getAsOf(s))
            price, asofP = marketstate.latestPrice(s)
            nnfloats[1] = price
            # nnfloats += marketstate.exportF1548(s)
            fmtr = psp.Formatter_F1548()
            market_state = self._marketState.format(fmtr, s)

            if 0 == c %chunk_size:
                X.resize(X.shape[0]+chunk_size, axis=0)
            X[c,:,:] = nnfloats
            c+=1

            thePROG.debug('-> state: asof[%s] symbol[%s] lastPrice[%s] OHLC%s\n' % (asofP.strftime('%Y%m%dT%H:%M:%S'), s, price, marketstate.dailyOHLC_sofar(s)))

if __name__ == '__main__':
    unittest.main()

