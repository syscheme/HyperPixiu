import unittest
import HistoryData as hist
import Perspective as psp
import MarketData as md
from Application import *

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

PROGNAME = os.path.basename(__file__)[0:-3]

class TestHistoryData(unittest.TestCase):

    def _test_baseApp(self):
        p = Program(PROGNAME)

        p.createApp(Foo, None)
        p.createApp(Foo, None)

        applst = p.listApps()
        print('listed all BaseApplication: %s\n' % applst)

        p.start()
        p.loop()
        p.stop()

    def _test_NonBlockingLoop(self):
        p = Program(PROGNAME)
        p._heartbeatInterval =-1

        p.createApp(Foo, None)
        p.createApp(Foo, None)

        applst = p.listApps()
        print('listed all BaseApplication: %s\n' % applst)

        p.start()
        p.loop()
        p.stop()

    def _test_playback(self):
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        for i in hpb :
            print('Row: %s\n' % i.desc)

    def test_Perspective(self):
        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/h/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)
        pdict = psp.PerspectiveDict('AShare')

        for i in pg :
            if psp.EVENT_Perspective != i.type :
                print('evnt: %s' % i.desc) 
                continue

            print('Psp: %s' % i.desc)
            pdict.updateByEvent(i)
            s = i.data._symbol
            print('-> state: asof[%s] lastPrice[%s] OHLC%s\n' % (pdict.getAsOf(s).strftime('%Y%m%d %H:%M:%S'), pdict.latestPrice(s), pdict.dailyOHLC_sofar(s)))

    def func1(self, a=None, **kwargs):
        print('func1(a=%s, kwargs=%s)\n' %(a, kwargs))

    def func2(self, **kwargs):
        self.func1(kwargs)
        self.func1(**kwargs)

    def _test_kwargs(self):
        self.func2(a='123',b=45,c=78) 
        self.func2(b=45,c=78)

if __name__ == '__main__':
    unittest.main()

