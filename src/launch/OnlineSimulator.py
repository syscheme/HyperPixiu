from Application import Program
from MarketData import TickData, KLineData, EVENT_TICK, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY
from Account import Account_AShare
import HistoryData as hist

from hpGym.GymTrader import *

from crawler.crawlSina import *
import sys, os, platform

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Trader.json']

    p = Program()
    p._heartbeatInterval = -1 # TODO 0.2 # yield at idle for 200msec

    sourceCsvDir = None
    SYMBOL = ''
    try:
        jsetting = p.jsettings('trader/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('trader/objectives')
        if not jsetting is None:
            SYMBOL = jsetting([SYMBOL])[0]
    except Exception as ex:
        SYMBOL =''

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    if 'SYMBOL' in os.environ.keys():
        SYMBOL = os.environ['SYMBOL']

    if 'Windows' in platform.platform() and '/mnt/' == sourceCsvDir[:5] and '/' == sourceCsvDir[6]:
        drive = '%symbol:' % sourceCsvDir[5]
        sourceCsvDir = sourceCsvDir.replace(sourceCsvDir[:6], drive)

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)

    p.info('taking input dir %s for symbol[%s]' % (sourceCsvDir, SYMBOL))
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=sourceCsvDir, fields='date,time,open,high,low,close,volume,ammount')

    gymtdr = p.createApp(GymTrader, configNode ='trader', tradeSymbol=SYMBOL, account=acc)
    p.info('all objects registered piror to OfflineSimulator: %s' % p.listByType())

    trainer = p.createApp(OfflineSimulator, configNode ='trader', trader=gymtdr, histdata=csvreader) # the simulator with brain loaded to verify training result

    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(trainer.outdir, 'TraderSt1_%s.tcsv' % SYMBOL))
    trainer.setRecorder(rec)
    rec.registerCategory(EVENT_TICK, params={'columns': TickData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1MIN, params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_5MIN, params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1DAY, params={'columns': KLineData.COLUMNS})

    mc = p.createApp(SinaCrawler, configNode ='crawler', recorder=rec) # md = SinaCrawler(p, None);
    mc.subscribe([SYMBOL])

    p.start()
    p.loop()
    
    p.stop()

