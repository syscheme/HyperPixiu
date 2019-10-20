import unittest
import HistoryData as hist
import Perspective as psp
import MarketData as md
from Application import *

'''
class TestBaseApp(unittest.TestCase):
    #Test Application.py

    class Foo(BaseApplication) :
        def __init__(self, program, settings):
            super(Foo, self).__init__(program, settings)
            self.__step =0

        def init(self): # return True if succ
            return True

        def OnEvent(self, event):
            print("Foo.OnEvent %s\n" % event)
        
        def step(self):
            self.__step +=1
            print("Foo.step %d\n" % self.__step)

    def test_basic(self):
        p = Program()
        p.createApp(Foo, None)
        p.start()
        p.loop()
        p.stop()

    def test_NonBlocking(self):
        p = Program()
        p.__heartbeatInterval =-1
        p.createApp(Foo, None)
        p.start()
        p.loop()
        p.stop()

    def test_multi(self):
        """Test method multi(a, b)"""
        self.assertEqual(6, multi(2, 3))

    def test_divide(self):
        """Test method divide(a, b)"""
        self.assertEqual(2, divide(6, 3)) 
        self.assertEqual(2.5, divide(5, 2))

'''

class TestHistoryData(unittest.TestCase):

    def test_playback(self):
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        for i in hpb :
            print('Row: %s\n' % i.desc)

    def test_PerspectiveGenerator(self):
        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)

        for i in pg :
            print('Psp: %s\n' % i.desc)

if __name__ == '__main__':
    unittest.main()

