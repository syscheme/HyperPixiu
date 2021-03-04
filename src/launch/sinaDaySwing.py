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
from crawler.producesSina import SinaMux, Sina_Tplus1, SinaSwingScanner

import sys, os, platform, re
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

####################################
def swingOnExtracedJsonFolder():
    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json' ]

    CLOCK_TODAY= datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    SINA_TODAY = CLOCK_TODAY.strftime('%Y-%m-%d')
    SINA_TODAY = '2020-10-09'
    if 'SINA_TODAY' in os.environ.keys():
        SINA_TODAY = [os.environ['SINA_TODAY']]

    SINA_TODAY = datetime.strptime(SINA_TODAY, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=0)
    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    #################
    objectives=['SZ000002', 'SH600029']
    evMdSource = '/mnt/e/AShareSample/screen-dataset'
    dirArchived = os.path.join(os.environ['HOME'], 'wkspaces/hpx_archived/sina') 
    #################

    p = Program()
    p._heartbeatInterval =-1

    modelName   = p.getConfig('scanner/model', 'ModelName') # None

    todayYYMMDD = SINA_TODAY.strftime('%Y%m%d')

    evMdSource  = Program.fixupPath(evMdSource)
    allFiles    = listAllFiles(evMdSource)

    for SYMBOL in objectives:
        p.stop()
        p._heartbeatInterval =-1

        acc     = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
        tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives = [SYMBOL], account=acc)

        rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, 'SinaDayEnd_%s.tcsv' % SINA_TODAY.strftime('%Y%m%d')))
        revents = None

        # determine the Playback instance
        playback   = SinaMux(p, endDate=SINA_TODAY.strftime('%Y%m%dT%H%M%S')) # = p.createApp(SinaMux, **srcPathPatternDict)
        playback.setSymbols(objectives)
        nLastDays = 5

        bnRegex, filelst = '%s_KL1d([0-9]*).json' % SYMBOL, []
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < todayYYMMDD: continue
            filelst.append(str(fn))
        
        daysTolst = (CLOCK_TODAY - SINA_TODAY).days + 1 + nLastDays
        if len(filelst) <=0:
            httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, SYMBOL, daysTolst, evMdSource)
        else:
            filelst.sort()
            _, lastDays = playback.loadOfflineJson(EVENT_KLINE_1DAY, SYMBOL, filelst[0], daysTolst)

        dtStart = lastDays[0].asof
        lastDays.reverse()
        for i in range(len(lastDays)):
            if todayYYMMDD > lastDays[i].asof.strftime('%Y%m%d') :
                if i < len(lastDays) - nLastDays :
                    dtStart = lastDays[i + nLastDays].asof
                break

        startYYMMDD = dtStart.strftime('%Y%m%d')
        p.info('determined %d-Tdays before %s was %s' % (nLastDays, SINA_TODAY.strftime('%Y-%m-%d'), startYYMMDD))
        
        bnRegex, filelst = '%s_MF1d([0-9]*).json' % SYMBOL, []
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < todayYYMMDD: continue
            filelst.append(fn)
        
        if len(filelst) <=0:
            playback.loadOnline(EVENT_MONEYFLOW_1DAY, SYMBOL, 0, evMdSource)
        else:
            filelst.sort()
            playback.loadOfflineJson(EVENT_MONEYFLOW_1DAY, SYMBOL, filelst[0], 1 + nLastDays)

        playback.loadOffline(EVENT_KLINE_5MIN, '%s/%s_KL5m*.json' % (evMdSource, SYMBOL), '%s/%s_KL5m%s.json' % (evMdSource, SYMBOL, startYYMMDD))
        playback.loadOffline(EVENT_MONEYFLOW_1MIN, '%s/%s_MF1m*.json' % (evMdSource, SYMBOL), '%s/%s_MF1m%s.json' % (evMdSource, SYMBOL, startYYMMDD))
        p.info('inited mux with %d substreams' % (playback.size))
        
        tdrWraper  = p.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=playback, symbol=SYMBOL) # = p.createApp(SinaDayEnd, configNode ='trader', trader=tdrCore, symbol=SYMBOL, dirOfflineData=evMdSource)
        tdrWraper.setTimeRange(dtStart = dtStart)
        tdrWraper.setSampling(os.path.join(p.outdir, 'SwingTrainingDS_%s.h5' % SINA_TODAY.strftime('%Y%m%d')))

        tdrWraper.setRecorder(rec)

        p.start()
        if tdrWraper.isActive :
            p.loop()

        p.stop()

        statesOfMoments = tdrWraper.stateOfMoments
        p.warn('TODO: predicting based on statesOf: %s and output to tcsv for summarizing' % ','.join(list(statesOfMoments.keys())))
        # rec.registerCategory('PricePred', params= {'columns' : 
        #  1d01p,1d12p,1d25p,1d5pp,1d01n,1d12n,1d2pn',
        #  2d01p,2d12p,2d25p,2d5pp,2d01n,2d12n,2d2pn',
        #  5d01p,5d12p,5d25p,5d5pp,5d01n,5d12n,5d2pn',]})

####################################
def memberfnInH5tar(fnH5tar, symbol):
    # we knew the member file would like SinaMF1d_20201010/SH600029_MF1d20201010.json@/root/wkspaces/hpx_archived/sina/SinaMF1d_20201010.h5t
    # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
    mfn = os.path.basename(fnH5tar)[:-4]
    idx = mfn.index('_')
    asofYYMMDD = mfn[1+idx:]
    return '%s/%s_%s%s.json' % (mfn, symbol, mfn[4:idx], asofYYMMDD), asofYYMMDD

def swingOnH5tars():
    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json' ]

    CLOCK_TODAY= datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    SINA_TODAY = CLOCK_TODAY.strftime('%Y-%m-%d')
    SINA_TODAY = '2020-10-09'
    if 'SINA_TODAY' in os.environ.keys():
        SINA_TODAY = [os.environ['SINA_TODAY']]

    SINA_TODAY = datetime.strptime(SINA_TODAY, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=0)
    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    #################
    objectives=['SZ000002', 'SH600029']
    dirCache = '/tmp/aaa'
    dirArchived = os.path.join(os.environ['HOME'], 'wkspaces/hpx_archived/sina') 
    #################

    p = Program()
    p._heartbeatInterval =-1

    modelName   = p.getConfig('scanner/model', 'ModelName') # None

    todayYYMMDD = SINA_TODAY.strftime('%Y%m%d')

    dirArchived = Program.fixupPath(dirArchived)
    allFiles    = listAllFiles(dirArchived)

    for SYMBOL in objectives:
        p.stop()
        p._heartbeatInterval =-1

        acc     = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
        tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives = [SYMBOL], account=acc)

        rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, 'SinaDayEnd_%s.tcsv' % SINA_TODAY.strftime('%Y%m%d')))
        revents = None

        # step 1. build up the Playback Mux
        playback   = SinaMux(p, endDate=SINA_TODAY.strftime('%Y%m%dT%H%M%S')) # = p.createApp(SinaMux, **srcPathPatternDict)
        playback.setSymbols(objectives)
        nLastDays = 5

        # 1.a  KL1d and determine the date of n-open-days ago
        bnRegex, filelst = 'SinaKL1d_([0-9]*).h5t', []
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < todayYYMMDD: continue
            filelst.append(str(fn))
        
        caldays = (CLOCK_TODAY - SINA_TODAY).days

        daysTolst = int(caldays/7) *5 + (caldays %7) + 1 + nLastDays
        filelst.sort()
        lastDays =[]
        for fn in filelst:
            try :
                _, lastDays = playback.loadJsonH5t(EVENT_KLINE_1DAY, SYMBOL, fn, None, daysTolst)
                break
            except Exception as ex:
                p.logexception(ex, fn)

        if len(lastDays) <=0:
            httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, SYMBOL, daysTolst, dirCache)
                
        dtStart = lastDays[0].asof
        lastDays.reverse()
        for i in range(len(lastDays)):
            if todayYYMMDD > lastDays[i].asof.strftime('%Y%m%d') :
                if i < len(lastDays) - nLastDays :
                    dtStart = lastDays[i + nLastDays].asof
                break

        startYYMMDD = dtStart.strftime('%Y%m%d')
        p.info('loaded KL1d and determined swing-start with %d-Tdays to %s was %s' % (nLastDays, SINA_TODAY.strftime('%Y-%m-%d'), startYYMMDD))
        
        # 1.b  MF1d
        bnRegex, filelst = 'SinaMF1d_([0-9]*).h5t', []
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < todayYYMMDD: continue
            filelst.append(str(fn))

        loaded=False
        
        filelst.sort()
        for fn in filelst:
            try :
                # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
                mfn, _ = memberfnInH5tar(fn, SYMBOL)
                playback.loadJsonH5t(EVENT_MONEYFLOW_1DAY, SYMBOL, fn, mfn, 1 + nLastDays)
                loaded = True
                break
            except Exception as ex:
                p.logexception(ex, fn)

        if not loaded:
            playback.loadOnline(EVENT_MONEYFLOW_1DAY, SYMBOL, 0, dirCache)

        # 1.c  KL5m
        bnRegex, filelst = 'SinaKL5m_([0-9]*).h5t', []
        # because one download of KL5m covered 5days, so take dtStart directly
        startYYMMDD = dtStart.strftime('%Y%m%d') 
        latestDay = todayYYMMDD
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < startYYMMDD or m.group(1) > todayYYMMDD: continue
            try :
                # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
                mfn, latestDay = memberfnInH5tar(fn, SYMBOL)
                playback.loadJsonH5t(EVENT_KLINE_5MIN, SYMBOL, fn, mfn)
            except Exception as ex:
                p.logexception(ex, fn)
        
        if CLOCK_TODAY == SINA_TODAY and latestDay < todayYYMMDD:
            playback.loadOnline(EVENT_KLINE_5MIN, SYMBOL, 0, dirCache)

        # 1.c  MF1m
        bnRegex, filelst = 'SinaMF1m_([0-9]*).h5t', []
        # because one download of KL5m covered 1day, so take dtStart directly
        # no need to (dtStart - timedelta(days=5)).strftime('%Y%m%d')
        startYYMMDD = dtStart.strftime('%Y%m%d')
        latestDay = todayYYMMDD
        for fn in allFiles:
            m = re.match(bnRegex, os.path.basename(fn))
            if not m or m.group(1) < startYYMMDD or m.group(1) > todayYYMMDD : continue
            try :
                # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
                mfn, latestDay = memberfnInH5tar(fn, SYMBOL)
                playback.loadJsonH5t(EVENT_MONEYFLOW_1MIN, SYMBOL, fn, mfn)
            except Exception as ex:
                p.logexception(ex, fn)

        if CLOCK_TODAY == SINA_TODAY and latestDay < todayYYMMDD:
            playback.loadOnline(EVENT_MONEYFLOW_1MIN, SYMBOL, 0, dirCache)

        p.info('inited mux with %d substreams' % (playback.size))
        
        tdrWraper  = p.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=playback, symbol=SYMBOL) # = p.createApp(SinaDayEnd, configNode ='trader', trader=tdrCore, symbol=SYMBOL, dirOfflineData=evMdSource)
        tdrWraper.setTimeRange(dtStart = dtStart)
        tdrWraper.setSampling(os.path.join(p.outdir, 'SwingTrainingDS_%s.h5' % SINA_TODAY.strftime('%Y%m%d')))

        tdrWraper.setRecorder(rec)

        p.start()
        if tdrWraper.isActive :
            p.loop()

        p.stop()

        statesOfMoments = tdrWraper.stateOfMoments
        p.warn('TODO: predicting based on statesOf: %s and output to tcsv for summarizing' % ','.join(list(statesOfMoments.keys())))
        # rec.registerCategory('PricePred', params= {'columns' : 
        #  1d01p,1d12p,1d25p,1d5pp,1d01n,1d12n,1d2pn',
        #  2d01p,2d12p,2d25p,2d5pp,2d01n,2d12n,2d2pn',
        #  5d01p,5d12p,5d25p,5d5pp,5d01n,5d12n,5d2pn',]})

####################################
if __name__ == '__main__':
    # swingOnExtracedJsonFolder()
    swingOnH5tars()