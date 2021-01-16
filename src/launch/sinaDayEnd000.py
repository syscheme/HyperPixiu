# encoding: UTF-8
'''
   SinaDayEnd 
   step 1. reads
       - pre-downloaded KL5m from dir path/to/screen-dataset/S[HZ]NNNNNN_KL5mYYYYMMDD.json
       - pre-downloaded MF1m from dir path/to/screen-dataset/S[HZ]NNNNNN_MF1mYYYYMMDD.json
       - online KL1d
       - online MF1d
    and takes SinaSwingScanner to generate ReplayFrames of 1 week ago, each includes 
       - KLex5m combines KL5m and MF5m to cover a week
       - KLex1d combines KL1d and MF1d to cover minimal half a year
       - 'predictions' on the price of future 1day, 2day and 5day in possible%
    
    step 2. take the state of 10:00, 11:00, 13:30, 14:30, 15:00 of today to prediction of 1day, 2day and 5day into SinaDayEnd_YYYYMMDD.tcsv output
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from advisors.dnn import DnnAdvisor_S1548I4A3
from crawler.producesSina import Sina_Tplus1, SinaSwingScanner

import sys, os, platform
RFGROUP_PREFIX  = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

import random

########################################################################
class SinaDayEnd(SinaSwingScanner):

    def __init__(self, program, **kwargs):
        '''Constructor
        '''
        super(SinaDayEnd, self).__init__(program, **kwargs)
        self.__dictPredict={} # dict of time to (state, predictions)

    def schedulePreidictions(self, timesToPredict):
        for t in timesToPredict:
            self.__dictPredict[t] = {'state': None, 'pred': None }

    def loadModel(self, modelName):
        # TODO
        pass


if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json' ]

    SINA_TODAY = datetime.now().strftime('%Y-%m-%d')
    if 'SINA_TODAY' in os.environ.keys():
        SINA_TODAY = [os.environ['SINA_TODAY']]
    SINA_TODAY = datetime.strptime(SINA_TODAY, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0)

    p = Program()
    p._heartbeatInterval =-1

    evMdSource  = p.getConfig('marketEvents/source', None) # market data event source
    ideal       = p.getConfig('trader/backTest/ideal', None) # None
    modelName   = p.getConfig('scanner/model', 'ModelName') # None

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives=objectives, account=acc)
    objectives = tdrCore.objectives
    SYMBOL = objectives[0]

    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, 'SinaDayEnd_%s.tcsv' % SINA_TODAY.strftime('%Y%m%d')))
    revents = None

    # determine the Playback instance
    evMdSource = Program.fixupPath(evMdSource)
    basename = os.path.basename(evMdSource)
    if '.tcsv' in basename :
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
    else:
        # csvPlayback can only cover one symbol
        p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')

    tdrWraper = p.createApp(SinaDayEnd, configNode ='trader', trader=tdrCore, symbol='SZ000001', dirOfflineData='/mnt/e/AShareSample/SinaWeek.20200629')

    tdrWraper.setRecorder(rec)
    # rec.registerCategory('PricePred', params= {'columns' : 
    #  1d01p,1d12p,1d25p,1d5pp,1d01n,1d12n,1d2pn',
    #  2d01p,2d12p,2d25p,2d5pp,2d01n,2d12n,2d2pn',
    #  5d01p,5d12p,5d25p,5d5pp,5d01n,5d12n,5d2pn',]})
     
    # subscribe the prediction of 10:00, 11:00, 13:30, 14:30, 15:00 of today
    tdrWraper.schedulePreidictions([
        SINA_TODAY.replace(hour=10),
        SINA_TODAY.replace(hour=11),
        SINA_TODAY.replace(hour=13, minute=30),
        SINA_TODAY.replace(hour=14, minute=30),
        SINA_TODAY.replace(hour=15)])

    tdrWraper.loadModel(modelName)

    p.start()
    if tdrWraper.isActive :
        p.loop()
    p.stop()
