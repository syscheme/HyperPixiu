# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from hpGym.GymTrader import *

from TradeAdvisor import *
from Application import *
import HistoryData as hist
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
    SYMBOL = ''
    try:
        jsetting = p.jsettings('advisor/eventSource')
        if not jsetting is None:
            eventSource = jsetting(None)

        jsetting = p.jsettings('advisor/objectives')
        if not jsetting is None:
            SYMBOL = jsetting([SYMBOL])[0]
    except Exception as ex:
        SYMBOL =''

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    if 'SYMBOL' in os.environ.keys():
        SYMBOL = os.environ['SYMBOL']

    rec    = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.logdir, 'advisor_%s.tcsv' % SYMBOL))
    p.info('all objects registered piror to Advisor: %s' % p.listByType())
    advisor = p.createApp(NeuralNetAdvisor, configNode ='advisor', objectives=[SYMBOL], recorder=rec)

    if 'sina' == eventSource:
        mc = p.createApp(SinaCrawler, configNode ='sina', marketState = advisor.marketState, recorder=rec)
        mc._postCaptured = True
        mc.subscribe([SYMBOL])
    elif '/' in eventSource: # eventSource looks like a file or directory
        sourceCsvDir = Program.fixupPath(eventSource)
        p.info('taking input dir %s for symbol[%s]' % (eventSource, SYMBOL))
        csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=sourceCsvDir, fields='date,time,open,high,low,close,volume,ammount')
        pbApp = p.createApp(hist.PlaybackApp, playback= csvreader)

    p.start()
    p.loop()
    
    p.stop()
