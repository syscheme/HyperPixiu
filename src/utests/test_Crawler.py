import unittest

from crawler.crawlSina import *

import os
PROGNAME = os.path.basename(__file__)[0:-3]

class TestCrawler(unittest.TestCase):

    def test_Sina(self):
        p = Program()
        p._heartbeatInterval =-1

        mc = p.createApp(SinaCrawler, configNode ='sina') # md = SinaCrawler(p, None);
        # _, result = md.searchKLines("000002", EVENT_KLINE_5MIN)
        # _, result = md.getRecentTicks('sh601006,sh601005,sh000001,sz000001')
        # _, result = md.getSplitRate('sh601006')
        # print(result)
        mc.subscribe(['601006','sh601005','sh000001','000001'])

        p.start()
        p.loop()
        p.stop()

if __name__ == '__main__':
    unittest.main()

