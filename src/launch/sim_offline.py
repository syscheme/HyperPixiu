# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
import ReplaySample as rs
from crawler.producesSina import Sina_Tplus1, populateMuxFromWeekDir
from advisors.dnn import DnnAdvisor
import h5tar

import sys, os, platform, re
from io import StringIO
import random

########################################################################
if __name__ == '__main__':

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
    # evMdSource = '/mnt/e/AShareSample/Sina2021W' # TEST-CODE
    # evMdSource = '/mnt/e/AShareSample/ETF.2013-2019' # TEST-CODE
    evMdSource = Program.fixupPath(evMdSource)
    basename = os.path.basename(evMdSource)
    MF1d_toAdd = []
    # MF1d_toAdd = ['/mnt/e/AShareSample/SinaMF1d_20200620.h5t', '/mnt/e/AShareSample/Sina2021W/Sina2021W01_0104-0108.h5t']
    # MF1d_toAdd = ['/mnt/e/AShareSample/Sina2021W/Sina2021W01_0104-0108.h5t']
    
    if os.path.isdir(evMdSource) :
        try :
            os.stat(os.path.join(evMdSource, 'h5tar.py'))
            histReader = populateMuxFromWeekDir(p, evMdSource, symbol=SYMBOL)
        except Exception as ex:
            # csvPlayback can only cover one symbol
            p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
            histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
            if not MF1d_toAdd or len(MF1d_toAdd) <=0:
                try :
                    fnMF1d = os.path.realpath(os.path.join(evMdSource, 'SinaMF1d.h5t'))
                    os.stat(fnMF1d)
                    MF1d_toAdd = [ fnMF1d ]
                except: pass

            for mf in MF1d_toAdd:
                try :
                    os.stat(mf)
                except: continue

                bn = os.path.basename(mf)
                pb = None
                if 'SinaMF1d_' in bn and 'h5t' == bn.split('.')[-1]:
                    memfn = bn.split('.')[0].split('_')[-1]
                    memfn = '%s_MF1d%s.csv' % (SYMBOL, memfn)
                    lines = h5tar.read_utf8(mf, memfn)
                    if len(lines) <=0: continue

                    pb = hist.CsvStream(SYMBOL, StringIO(lines), MoneyflowData.COLUMNS, evtype=EVENT_MONEYFLOW_1DAY, program=p)
                    pb.setId('%s@%s' % (memfn, mf))
                else:
                    m = re.match(r'Sina([0-9]*)W[0-9]*_([0-9]*)-([0-9]*).h5t', bn)
                    if not m: continue
                    mlst = h5tar.list_utf8(mf)
                    memfn = None
                    for mem in mlst:
                        if 'MF1d' in mem['name'] and SYMBOL in mem['name']:
                            memfn = mem['name']
                            break
                    if not memfn: continue
                    lines = h5tar.read_utf8(mf, memfn)
                    if len(lines) <=0: continue

                    pb = hist.CsvStream(SYMBOL, StringIO(lines), MoneyflowData.COLUMNS, evtype=EVENT_MONEYFLOW_1DAY, program=p)
                    pb.setId('%s@%s' % (memfn, mf))
                
                if not pb: continue
                if not isinstance(histReader, hist.PlaybackMux):
                    mux = hist.PlaybackMux(program=p)
                    mux.addStream(histReader)
                    histReader = mux

                histReader.addStream(pb)
                p.info('mux-ed MF1d[%s]' % (pb.id))

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
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._enableMStateSS = False # MUST!!!
        advisor._exchange = tdrCore.account.exchange

        p.info('all objects registered piror to simulator: %s' % p.listByType())
        tdrWraper = p.createApp(OfflineSimulator, configNode ='trader', trader=tdrCore, histdata=histReader) # the simulator with brain loaded to verify training result

    tdrWraper.setRecorder(rec)

    p.start()
    if tdrWraper.isActive :
        p.loop()
    p.stop()
