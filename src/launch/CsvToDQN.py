# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from hpGym.GymTrader import *

from Account import Account_AShare
from Application import *
import HistoryData as hist

import sys, os, platform

if __name__ == '__main__':
    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/CsvToDQN.json']

    p = Program()
    p._heartbeatInterval =-1

    sourceCsvDir = None
    SYMBOL = ''
    try:
        jsetting = p.jsettings('trainer/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('trainer/objectives')
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

    gymtdr = p.createApp(GymTrader, configNode ='trainer', tradeSymbol=SYMBOL, account=acc)
    p.info('all objects registered piror to Simulator: %s' % p.listByType())

    trainer = p.createApp(IdealDayTrader, configNode ='trainer', trader=gymtdr, histdata=csvreader) # ideal trader to generator ReplayFrames
    # trainer = p.createApp(Simulator, configNode ='trainer', trader=gymtdr, histdata=csvreader) # the simulator with brain loaded to verify training result
    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(trainer.outdir, 'CsvToRF_%s.tcsv' % SYMBOL))
    trainer.setRecorder(rec)

    p.start()
    p.loop()
    
    p.stop()
