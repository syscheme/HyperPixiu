import unittest

import HistoryData as hist
import Perspective as psp
import MarketData as md
from Application import *
from Account import *

PROGNAME = os.path.basename(__file__)[0:-3]

class TestAccount(unittest.TestCase):

    def test_AccApp(self):
        p = Program(PROGNAME)
        p._heartbeatInterval =-1

        p.createApp(Account_AShare, None)
        pdict = psp.PerspectiveDict('AShare')
        p.addObj(pdict)
        print('listed all Objects: %s\n' % p.listByType(MetaObj))

        p.start()
        p.loop()
        p.stop()

    def _test_Perspective(self):
        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)
        pdict = psp.PerspectiveDict('AShare')

        for i in pg :
            print('Psp: %s' % i.desc)
            pdict.updateByEvent(i)
            s = i.data._symbol
            print('-> state: asof[%s] lastPrice[%s] OHLC%s\n' % (pdict.getAsOf(s).strftime('%Y%m%d %H:%M:%S'), pdict.latestPrice(s), pdict.dailyOHLC_sofar(s)))

    def _test_Account(self):
        p = Program(PROGNAME)
        p._heartbeatInterval =-1

        p.createApp(Foo, None)
        p.createApp(Foo, None)


        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)
        pdict = psp.PerspectiveDict('AShare')

        for i in pg :
            print('Psp: %s' % i.desc)
            pdict.updateByEvent(i)
            s = i.data._symbol
            print('-> state: asof[%s] lastPrice[%s] OHLC%s\n' % (pdict.getAsOf(s).strftime('%Y%m%d %H:%M:%S'), pdict.latestPrice(s), pdict.dailyOHLC_sofar(s)))


if __name__ == '__main__':
    unittest.main()

