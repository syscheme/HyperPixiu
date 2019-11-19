import unittest

import HistoryData as hist
import Perspective as psp
import MarketData as md
from Application import *
from Account import *
from Trader import *

PROGNAME = os.path.basename(__file__)[0:-3]

class TestAccount(unittest.TestCase):

    def test_AccApp(self):
        p = Program()
        p._heartbeatInterval =-1

        p.createApp(Account_AShare)
        pdict = psp.PerspectiveDict('AShare')
        p.addObj(pdict)
        print('listed all Objects: %s\n' % p.listByType(MetaObj))

        p.start()
        p.loop()
        p.stop()

    def _test_TraderApp(self):
        p = Program(PROGNAME)
        p._heartbeatInterval =-1

        acc = p.createApp(Account_AShare)
        pdict = psp.PerspectiveDict('AShare')
        p.addObj(pdict)
        tdr = p.createApp(BaseTrader)
        print('listed all Objects: %s\n' % p.listByType(MetaObj))

        p.start()
        p.loop()
        p.stop()

if __name__ == '__main__':
    unittest.main()

