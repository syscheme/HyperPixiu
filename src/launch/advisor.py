# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from hpGym.GymTrader import *

from TradeAdvisor import *
from Application import *
import HistoryData as hist

import sys, os, platform
RFGROUP_PREFIX = 'ReplayFrame:'
OUTFRM_SIZE = 8*1024
import random

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Advisor.json']

    p = Program()
    p._heartbeatInterval =-1

    sourceCsvDir = None
    SYMBOL = ''
    try:
        jsetting = p.jsettings('advisor/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('advisor/objectives')
        if not jsetting is None:
            SYMBOL = jsetting([SYMBOL])[0]
    except Exception as ex:
        SYMBOL =''

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    if 'SYMBOL' in os.environ.keys():
        SYMBOL = os.environ['SYMBOL']

    sourceCsvDir = Program.fixupPath(sourceCsvDir)

    p.info('taking input dir %s for symbol[%s]' % (sourceCsvDir, SYMBOL))
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=sourceCsvDir, fields='date,time,open,high,low,close,volume,ammount')
    pbApp = p.createApp(hist.PlaybackApp, playback= csvreader)

    rec    = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.logdir, 'advisor_%s.tcsv' % SYMBOL))
    p.info('all objects registered piror to Advisor: %s' % p.listByType())

    advisor = p.createApp(NeuralNetAdvisor, configNode ='advisor', objectives=[SYMBOL], recorder=rec)

    p.start()
    p.loop()
    
    p.stop()
