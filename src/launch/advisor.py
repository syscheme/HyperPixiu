# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from hpGym.GymTrader import *

from dnn.TradeAdvisor import DnnAdvisor
from Application import *
import HistoryData as hist
from RemoteEvent import ZeroMqProxy
from crawler.crawlSina import *

import sys, os, platform
RFGROUP_PREFIX = 'ReplayFrame:'
OUTFRM_SIZE = 8*1024
import random

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Advisor.json']

    p = Program()
    p._heartbeatInterval =-1

    eventSource = None
    try:
        jsetting = p.jsettings('advisor/eventSource')
        if not jsetting is None:
            eventSource = jsetting(None)
    except:
        pass

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    rec    = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.logdir, '%s.tcsv' % p.progId))
    revents = p.createApp(ZeroMqProxy, configNode ='remoteEvents')
    revents.registerOutgoing([EVENT_ADVICE, EVENT_KLINE_1MIN]) # should be revents.registerOutgoing(EVENT_ADVICE)

    p.info('all objects registered piror to Advisor: %s' % p.listByType())
    advisor = p.createApp(DnnAdvisor, configNode ='advisor', objectives=objectives, recorder=rec)
    objectives = advisor.objectives

    if 'sina' == eventSource:
        mc = p.createApp(SinaCrawler, configNode ='sina', marketState = advisor.marketState, recorder=rec)
        mc._postCaptured = True
        mc.subscribe(objectives)
    elif '/' in eventSource and len(objectives)>0: # eventSource looks like a file or directory
        SYMBOL = objectives[0] # csvPlayback can only cover one symbol
        eventSource = Program.fixupPath(eventSource)
        p.info('taking input dir %s for symbol[%s]' % (eventSource, SYMBOL))
        csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=eventSource, fields='date,time,open,high,low,close,volume,ammount')
        pbApp = p.createApp(hist.PlaybackApp, playback= csvreader)

    p.start()
    p.loop()
    
    p.stop()
