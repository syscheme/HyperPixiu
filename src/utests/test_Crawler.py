from Application import Program
import unittest

from crawler.crawlSina import *
import sys, os

class TestCrawler(unittest.TestCase):
    import sys, os

    def test_Sina(self):
        conffile = os.path.dirname(os.path.abspath(__file__)) + '/../..'
        conffile = os.path.realpath(conffile) + '/conf/utests.json'
        sys.argv += ['-f', conffile]
        p = Program()
        p._heartbeatInterval =-1

        mc = p.createApp(SinaCrawler, configNode ='SinaCrawler') # md = SinaCrawler(p, None);
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

