import unittest
from HistoryData import *
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
        p = Program()
        p.createApp(CsvPlayback, None)
        p.start()
        p.loop()
        p.stop()


if __name__ == '__main__':
    unittest.main()

