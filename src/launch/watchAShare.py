from Application import Program
from MarketData import TickData, KLineData, EVENT_TICK, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY
import HistoryData as hist

from   RemoteEvent import RedisEE
from crawler.crawlSina import *
import sys, os

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/watchAShare.json']

    p = Program()
    p._heartbeatInterval =0.2 # yield at idle for 200msec

    objectives  = p.getConfig('objectives', ['510050'])
    objectives = [s('') for s in objectives] # convert to string list
    if len(objectives) <=0:
        p.error('no objectives specified')
        quit()

    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder')
    rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

    sina = p.createApp(SinaCrawler, configNode ='sina', recorder=rec, objectives=objectives) # md = SinaCrawler(p, None);
    sina._eventsToPost  = [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY]
    sina._eventsToPost += [EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY]
    sina._timeoutToPostEvent =9999999.9 # dummy

    # always create remote EventEnd in a crawler
    # revents = p.createApp(ZmqEE, configNode ='remoteEvents/zmq')
    revents = p.createApp(RedisEE, configNode ='remoteEvents/redis')

    revents.registerOutgoing([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
    revents.registerOutgoing([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY])

    p.start()
    # if sina.isActive:
    p.loop()
    p.stop()

