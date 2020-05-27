# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from hpGym.GymTrader import *

from   TradeAdvisor import EVENT_ADVICE
from   advisors.dnn  import DnnAdvisor_S1548I4A3
from   Application   import *
import HistoryData as hist
from   RemoteEvent import ZmqEE, RedisEE
from   crawler.crawlSina import *

import sys, os, platform
RFGROUP_PREFIX = 'ReplayFrame:'
OUTFRM_SIZE = 8*1024
import random

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Advisor.json']

    p = Program()
    p._heartbeatInterval =0.2

    advType     = p.getConfig('advisor/type', "dnn.S1548I4A3")
    evMdSource  = p.getConfig('marketEvents/source', 'sina') # market data event source
    exchange    = p.getConfig('marketEvents/exchange', 'AShare')
    objectives  = p.getConfig('advisor/objectives', [])

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

    rec    = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, '%s.tcsv' % p.progId))

    # now read config: revents.registerOutgoing([EVENT_ADVICE])
    p.info('all objects registered piror to Advisor: %s' % p.listByType())
    advisor = p.createApp(DnnAdvisor_S1548I4A3, configNode ='advisor', objectives=objectives, recorder=rec)
    advisor._exchange = exchange
    objectives = advisor.objectives

    for dummy in range(1): # dummy one-round loop
        if '/' in evMdSource and len(objectives)>0: # evMdSource looks like a local file or directory
            SYMBOL = objectives[0] # csvPlayback can only cover one symbol
            evMdSource = Program.fixupPath(evMdSource)
            p.info('taking input dir %s for symbol[%s]' % (evMdSource, SYMBOL))

            # instance a HistData but no remote EventEnd needed
            csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
            pbApp = p.createApp(hist.PlaybackApp, playback= csvreader)

            p._heartbeatInterval =-1
            break

        # remote EventEnd should be wished when market source is a crawler
        # revents = p.createApp(ZmqEE, configNode ='remoteEvents/zmq')
        revents = p.createApp(RedisEE, configNode ='remoteEvents/redis')

        if 'sina' == evMdSource:
            mc = p.createApp(SinaCrawler, configNode ='sina', marketState = advisor.marketState, recorder=rec)
            mc._postCaptured = True
            mc.subscribe(objectives)
            break

    p.start()

    if advisor.isActive:
        p.loop()
    
    p.stop()
