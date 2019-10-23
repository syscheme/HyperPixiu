import unittest
import HistoryData as hist
import Perspective as psp
import MarketData as md
from Application import *

class Foo(BaseApplication) :
    def __init__(self, program, settings):
        super(Foo, self).__init__(program, settings)
        self.__step =0

    def init(self): # return True if succ
        return True

    def OnEvent(self, event):
        self.info("Foo.OnEvent %s" % event)
    
    def step(self):
        self.__step +=1
        self.info("Foo.step %d" % self.__step)

PROGNAME = os.path.basename(__file__)[0:-3]

class TestHistoryData(unittest.TestCase):

    def _test_baseApp(self):
        p = Program(PROGNAME)

        p.createApp(Foo, None)
        p.createApp(Foo, None)

        applst = p.listAppsOfType(BaseApplication)
        print('listed all BaseApplication: %s\n' % applst)

        p.start()
        p.loop()
        p.stop()

    def _test_NonBlockingLoop(self):
        p = Program(PROGNAME)
        p._heartbeatInterval =-1

        p.createApp(Foo, None)
        p.createApp(Foo, None)

        applst = p.listAppsOfType(BaseApplication)
        print('listed all BaseApplication: %s\n' % applst)

        p.start()
        p.loop()
        p.stop()

    def _test_playback(self):
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        for i in hpb :
            print('Row: %s\n' % i.desc)

    def test_PerspectiveGenerator(self):
        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/h/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)

        for i in pg :
            print('Psp: %s\n' % i.desc)

if __name__ == '__main__':
    unittest.main()

