# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from advisors.dnn import DnnAdvisor_S1548I4A3
from crawler.producesSina import Sina_Tplus1, SinaSwingScanner, populateMuxFromArchivedDir, balanceSamples

import sys, os, platform
import random

RFGROUP_PREFIX  = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

if __name__ == '__main__':

    # sys.argv += ['-z', '-b', '/mnt/e/h5_to_h5b/RFrmD4M1X5_SZ159949.h5']

    if '-b' in sys.argv :
        idx = sys.argv.index('-b') +1
        if idx >0 and idx < len(sys.argv):
            h5fn = sys.argv[idx]
            compress = '-z' in sys.argv
            balanceSamples(h5fn, compress)
            quit()

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json']

    p = Program()
    p._heartbeatInterval =-1

    evMdSource  = p.getConfig('marketEvents/source', None) # market data event source
    advisorType = p.getConfig('advisor/type', "remote")
    ideal       = p.getConfig('trader/backTest/ideal', None) # None

    if "remote" != advisorType:
        # this is a local advisor, so the trader's source of market data event must be the same of the local advisor
        pass
        # jsetting = p.jsettings('advisor/eventSource')
        # if not jsetting is None:
        #     evMdSource = jsetting(None)

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives=objectives, account=acc)
    objectives = tdrCore.objectives
    SYMBOL = objectives[0]

    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, '%s_P%s.tcsv' % (SYMBOL, p.pid)))
    revents = None

    # determine the Playback instance
    # evMdSource = '/mnt/e/AShareSample/hpx_archived/sina' # TEST-CODE
    # evMdSource = '/mnt/e/AShareSample/ETF.2013-2019' # TEST-CODE
    evMdSource = Program.fixupPath(evMdSource)
    basename = os.path.basename(evMdSource)
    if os.path.isdir(evMdSource) :
        try :
            os.stat(os.path.join(evMdSource, 'h5tar.py'))
            histReader = populateMuxFromArchivedDir(p, evMdSource, symbol=SYMBOL)
        except:
            # csvPlayback can only cover one symbol
            p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
            histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
    elif '.tcsv' in basename :
        p.info('taking TaggedCsvPlayback on %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.TaggedCsvPlayback(program=p, symbol=SYMBOL, tcsvFilePath=evMdSource)
        histReader.setId('%s@%s' % (SYMBOL, basename))
        histReader.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        histReader.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

    elif '.tar.bz2' in basename :
        p.info('taking TaggedCsvInTarball on %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.TaggedCsvInTarball(program=p, symbol=SYMBOL, fnTarball=evMdSource, memberPattern='%s_evmd_*.tcsv' %SYMBOL )
        histReader.setId('%s@%s' % (SYMBOL, basename))
        histReader.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        histReader.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

    tdrWraper = None
    if 'remote' == advisorType :
        p.error('sim_offline only takes local advisor')
        quit()
        # revents = p.createApp(ZmqEE, configNode ='remoteEvents')
        # revs = [EVENT_ADVICE]
        # if not evMdSource:
        #     revs += [EVENT_TICK, EVENT_KLINE_1MIN]
        # revents.subscribe(revs)

    # ideal ='T+1' #TEST-CODE

    if 'T+1' == ideal :
        tdrWraper = p.createApp(IdealTrader_Tplus1, configNode ='trader', trader=tdrCore, histdata=histReader) # ideal trader to generator ReplayFrames
    elif 'SinaT+1' == ideal :
        tdrWraper = p.createApp(Sina_Tplus1, configNode ='trader', trader=tdrCore, symbol='SZ000001', dirOfflineData='/mnt/e/AShareSample/SinaWeek.20200629')
    elif 'FuturePrice' == ideal :
        tdrWraper = p.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=histReader) # ShortSwingScanner to classify future price
    elif 'SinaFP' == ideal :
        tdrWraper = p.createApp(SinaSwingScanner, configNode ='trader', trader=tdrCore, symbol='SZ000001', dirOfflineData='/mnt/e/AShareSample/SinaWeek.20200629')
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor_S1548I4A3, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._enableMStateSS = False # MUST!!!
        advisor._exchange = tdrCore.account.exchange

        p.info('all objects registered piror to simulator: %s' % p.listByType())
        tdrWraper = p.createApp(OfflineSimulator, configNode ='trader', trader=tdrCore, histdata=histReader) # the simulator with brain loaded to verify training result

    tdrWraper.setRecorder(rec)

    p.start()
    if tdrWraper.isActive :
        p.loop()
    p.stop()
