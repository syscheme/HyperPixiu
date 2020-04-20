from Application import Program

from Account import Account_AShare
from Trader import BaseTrader
from BackTest import OnlineSimulator
import HistoryData as hist

from TradeAdvisor import EVENT_ADVICE
from advisors.dnn import DnnAdvisor_S1548I4A3

from crawler.crawlSina import *
from RemoteEvent import ZmqProxy

import sys, os, platform

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json']

    p = Program()
    p._heartbeatInterval = 0.1 # 0.1 yield at idle for 100msec

    evMdSource  = p.getConfig('marketEvents/source', None) # market data event source
    advisorType = p.getConfig('advisor/type', "dnn.S1548I4A3")
    objectives = p.getConfig('trader/objectives', ['SH510050'])

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    if 'SYMBOL' in os.environ.keys():
        SYMBOL = os.environ['SYMBOL']
        if len(SYMBOL) >0:
            objectives.remove(SYMBOL)
            objectives = [SYMBOL] + objectives
    
    if len(objectives) <=0:
        p.error('no objectives specified')
        quit()

    SYMBOL = objectives[0]

    # evMdSource = Program.fixupPath(evMdSource)
    acc     = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    # tdrCore = p.createApp(GymTrader, configNode ='trader', tradeSymbol=SYMBOL, account=acc)
    tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives=objectives, account=acc)

    p.info('all objects registered piror to OnlineSimulator: %s' % p.listByType())
    simulator = p.createApp(OnlineSimulator, configNode ='trader', trader=tdrCore) # the simulator with brain loaded to verify training result
    rec     = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(simulator.outdir, 'online_%s.tcsv' % SYMBOL))
    simulator.setRecorder(rec)

    if 'remote' == advisorType :
        revents = p.createApp(ZmqProxy, configNode ='remoteEvents/zmq')
        revents.subscribeIncoming([EVENT_ADVICE])
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor_S1548I4A3, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._exchange = tdrCore.account.exchange

    if False: # 'sina' != evMdSource
        pass
    else:
        mc = p.createApp(SinaCrawler, configNode ='crawler', marketState = tdrCore._marketState, recorder=rec) # md = SinaCrawler(p, None);
        mc._postCaptured = True
        mc.subscribe([SYMBOL])

    p.start()
    p.loop()
    
    p.stop()

