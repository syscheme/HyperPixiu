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
# from __future__ import absolute_import, unicode_literals
from celery import shared_task
import sys, os, re
if __name__ == '__main__':
    sys.path.append(".")
    from worker import thePROG
else:
    from .worker import thePROG

from Application import *
from Perspective import *
import HistoryData as hist
from crawler.producesSina import SinaMux, Sina_Tplus1, SinaSwingScanner

import h5py

@shared_task
def add(x, y):
    return x + y

@shared_task
def mul(x, y):
    return x * y

@shared_task
def xsum(numbers):
    return sum(numbers)

# ===================================================
def memberfnInH5tar(fnH5tar, symbol):
    # we knew the member file would like SinaMF1d_20201010/SH600029_MF1d20201010.json@/root/wkspaces/hpx_archived/sina/SinaMF1d_20201010.h5t
    # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
    mfn = os.path.basename(fnH5tar)[:-4]
    idx = mfn.index('_')
    asofYYMMDD = mfn[1+idx:]
    return '%s/%s_%s%s.json' % (mfn, symbol, mfn[4:idx], asofYYMMDD), asofYYMMDD

def saveSnapshot(filename, h5group, dsName, snapshot, ohlc):
    compressed = bz2.compress(snapshot)

    with h5py.File(filename, 'a') as h5file:

        if h5group in h5file.keys() :
            g = h5file[h5group]
        else:
            g = h5file.create_group(h5group) 
            g.attrs['desc']         = 'pickled market state via bzip2 compression'

        if dsName in g.keys(): del g[dsName]

        npbytes = np.frombuffer(compressed, dtype=np.uint8)
        sns = g.create_dataset(dsName, data=np.frombuffer(compressed, dtype=np.uint8))
        sns.attrs['size'] = len(snapshot)
        sns.attrs['csize'] = len(compressed)
        sns.attrs['open'] = ohlc[0]
        sns.attrs['high'] = ohlc[1]
        sns.attrs['low']  = ohlc[2]
        sns.attrs['price'] = ohlc[3]
        sns.attrs['generated'] = datetime.now().strftime('%Y%m%dT%H%M%S')
        
        thePROG.info('saved snapshot[%s] %dB->%dz into %s' % (dsName, sns.attrs['size'], sns.attrs['csize'], filename))

# ===================================================
@shared_task
def downloadToday(SYMBOL, todayYYMMDD =None):

    CLOCK_TODAY= datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    SINA_TODAY = CLOCK_TODAY.strftime('%Y-%m-%d') if not todayYYMMDD else todayYYMMDD
    if 'SINA_TODAY' in os.environ.keys():
        SINA_TODAY = [os.environ['SINA_TODAY']]

    SINA_TODAY = datetime.strptime(SINA_TODAY, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=0)

    dirCache = '/tmp/aaa'
    dirArchived = os.path.join(os.environ['HOME'], 'wkspaces/hpx_archived/sina') 

    todayYYMMDD = SINA_TODAY.strftime('%Y%m%d')
    dirArchived = Program.fixupPath(dirArchived)
    
    # step 1. build up the Playback Mux
    playback   = SinaMux(thePROG, endDate=SINA_TODAY.strftime('%Y%m%dT%H%M%S')) # = thePROG.createApp(SinaMux, **srcPathPatternDict)
    playback.setId('Dayend.%s' % SYMBOL)
    playback.setSymbols([SYMBOL])
    nLastDays = 5

    # 1.a  KL1d and determine the date of n-open-days ago
    caldays = (CLOCK_TODAY - SINA_TODAY).days

    daysTolst = int(caldays/7) *5 + (caldays %7) + 1 + nLastDays
    lastDays =[]
    httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, SYMBOL, daysTolst, dirCache)
    if len(lastDays) <=0 :
        return "456 busy"
            
    dtStart = lastDays[0].asof
    lastDays.reverse()
    for i in range(len(lastDays)):
        if todayYYMMDD > lastDays[i].asof.strftime('%Y%m%d') :
            if i < len(lastDays) - nLastDays :
                dtStart = lastDays[i + nLastDays].asof
            break

    startYYMMDD = dtStart.strftime('%Y%m%d')
    todayYYMMDD = lastDays[0].asof.strftime('%Y%m%d')

    thePROG.info('loaded KL1d and determined %d-Tdays pirior to %s was %s, adjusted today as %s' % (nLastDays, SINA_TODAY.strftime('%Y-%m-%d'), startYYMMDD, todayYYMMDD))
    
    # 1.b  MF1d
    httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1DAY, SYMBOL, 0, dirCache)

    # 1.c  KL5m
    httperr, _, _ = playback.loadOnline(EVENT_KLINE_5MIN, SYMBOL, 0, dirCache)

    # 1.c  MF1m
    # bnRegex, filelst = 'SinaMF1m_([0-9]*).h5t', []
    # # because one download of KL5m covered 1day, so take dtStart directly
    # # no need to (dtStart - timedelta(days=5)).strftime('%Y%m%d')
    # startYYMMDD = dtStart.strftime('%Y%m%d')
    # latestDay = todayYYMMDD
    # for fn in allFiles:
    #     m = re.match(bnRegex, os.path.basename(fn))
    #     if not m or m.group(1) < startYYMMDD or m.group(1) > todayYYMMDD : continue
    #     try :
    #         # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
    #         mfn, latestDay = memberfnInH5tar(fn, SYMBOL)
    #         playback.loadJsonH5t(EVENT_MONEYFLOW_1MIN, SYMBOL, fn, mfn)
    #     except Exception as ex:
    #         thePROG.logexception(ex, fn)

    # if CLOCK_TODAY == SINA_TODAY and latestDay < todayYYMMDD:
    httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1MIN, SYMBOL, 0, dirCache)

    thePROG.info('inited mux with %d substreams' % (playback.size))
    psptMarketState = PerspectiveState(SYMBOL)
    stampOfState, momentsToSample = None, ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:30:00', '15:00:00']
    snapshot = {}
    snapshoth5fn = os.path.join(dirCache, '%s_sns.h5' % (SYMBOL))

    while True:
        try :
            ev = next(playback)
            if not ev or MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] : continue

            symbol = ev.data.symbol
            if ev.data.datetime <= SINA_TODAY:
                if not psptMarketState.updateByEvent(ev) or symbol != SYMBOL :
                    continue

                stamp    = psptMarketState.getAsOf(symbol)
                price, _ = psptMarketState.latestPrice(symbol)
                ohlc     = psptMarketState.dailyOHLC_sofar(symbol)

                if not ohlc or todayYYMMDD != stamp.strftime('%Y%m%d'): # or not today
                    continue

                if len(momentsToSample) >0 and stamp.strftime('%H:%M:00') in momentsToSample:
                    snapshot = {
                        'ident'   : '%s@%s' % (symbol, stamp.strftime('%Y%m%dT%H%M%S')),
                        'ohlc'   : [ohlc.open, ohlc.high, ohlc.low, price],
                        'snapshot': psptMarketState.dumps(symbol)
                    }

                if stampOfState and stampOfState == stamp:
                    continue
                
                stampOfState = stamp

                if snapshot and len(snapshot) >0:
                    h5group = snapshot['ident']
                    h5group = h5group[:h5group.index('T')]
                    saveSnapshot(snapshoth5fn, h5group= h5group, dsName=snapshot['ident'], snapshot=snapshot['snapshot'], ohlc=snapshot['ohlc'])
                    snapshot ={}
        
        except StopIteration:
            thePROG.info('hist-read: end of playback')
            break
        except Exception as ex:
            thePROG.logexception(ex)
            break

    if snapshot and len(snapshot) >0:
        h5group = snapshot['ident']
        h5group = h5group[:h5group.index('T')]
        saveSnapshot(snapshoth5fn, h5group= h5group, dsName=snapshot['ident'], snapshot=snapshot['snapshot'], ohlc=snapshot['ohlc'])

    return snapshoth5fn, playback.cachedFiles
    '''
        # -----------------------------------
    
    tdrWraper  = thePROG.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=playback, symbol=SYMBOL) # = thePROG.createApp(SinaDayEnd, configNode ='trader', trader=tdrCore, symbol=SYMBOL, dirOfflineData=evMdSource)
    tdrWraper.setTimeRange(dtStart = dtStart)
    tdrWraper.setSampling(os.path.join(thePROG.outdir, 'SwingTrainingDS_%s.h5' % SINA_TODAY.strftime('%Y%m%d')))

    tdrWraper.setRecorder(rec)

    thePROG.start()
    if tdrWraper.isActive :
        thePROG.loop()

    thePROG.stop()

    statesOfMoments = tdrWraper.stateOfMoments
    thePROG.warn('TODO: predicting based on statesOf: %s and output to tcsv for summarizing' % ','.join(list(statesOfMoments.keys())))
    # rec.registerCategory('PricePred', params= {'columns' : 
    #  1d01p,1d12p,1d25p,1d5pp,1d01n,1d12n,1d2pn',
    #  2d01p,2d12p,2d25p,2d5pp,2d01n,2d12n,2d2pn',
    #  5d01p,5d12p,5d25p,5d5pp,5d01n,5d12n,5d2pn',]})
    '''

####################################
if __name__ == '__main__':
    h5, cached = downloadToday('SZ000002')
    print('%s, %s' % (h5, cached))