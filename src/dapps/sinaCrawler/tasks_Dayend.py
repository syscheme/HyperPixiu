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
    from worker import thePROG, getLogin
else:
    from .worker import thePROG, getLogin

from Application import *
from Perspective import *
import HistoryData as hist
from crawler.producesSina import SinaMux, Sina_Tplus1, SinaSwingScanner
from dapps.CeleryDefs import RetryableError, Retryable

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
    fnH5tar = os.path.realpath(fnH5tar)
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
__accLogin, __accHome = None, None
@shared_task(bind=True, base=Retryable)
def downloadToday(self, SYMBOL, todayYYMMDD =None):
    global __accLogin, __accHome
    if not __accLogin:
        __accLogin, __accHome = getLogin()

    h5, cached = __downloadSymbol(SYMBOL)
    print('%s, %s' % (h5, cached))

def __downloadSymbol(SYMBOL, todayYYMMDD =None):

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

    daysTolst = int(caldays/7) *5 + (caldays %7) + 5 + nLastDays
    lastDays =[]
    httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, SYMBOL, daysTolst, dirCache)
    if 456 == httperr:
        raise RetryableError(httperr, "blocked by sina at %s@%s" %(EVENT_KLINE_1DAY, SYMBOL))
            
    lastDays.reverse()
    tmp = lastDays
    lastDays = []
    for i in tmp:
        if i.asof.strftime('%Y%m%d') > todayYYMMDD:
            continue
        lastDays.append(i)
        if len(lastDays) >= nLastDays:
            break

    if len(lastDays) <=0:
        raise ValueError("failed to get recent %d days" %nLastDays)
    
    dtStart = lastDays[-1].asof
    startYYMMDD = dtStart.strftime('%Y%m%d')
    todayYYMMDD = lastDays[0].asof.strftime('%Y%m%d')
    
    thePROG.info('loaded KL1d and determined %d-Tdays pirior to %s, %ddays: %s ~ %s' % (nLastDays, SINA_TODAY.strftime('%Y-%m-%d'), len(lastDays), startYYMMDD, todayYYMMDD))
    
    # 1.b  MF1d
    httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1DAY, SYMBOL, 0, dirCache)
    if 456 == httperr:
        raise RetryableError(httperr, "blocked by sina at %s@%s" %(EVENT_MONEYFLOW_1DAY, SYMBOL))

    # 1.c  KL5m
    httperr, _, _ = playback.loadOnline(EVENT_KLINE_5MIN, SYMBOL, 0, dirCache)
    if 456 == httperr:
        raise RetryableError(httperr, "blocked by sina at %s@%s" %(EVENT_KLINE_5MIN, SYMBOL))

    # 1.c  MF1m
    httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1MIN, SYMBOL, 0, dirCache)
    if 456 == httperr:
        raise RetryableError(httperr, "blocked by sina at %s@%s" %(EVENT_MONEYFLOW_1MIN, SYMBOL))

    for i in lastDays[1:]:
        offline_mf1m = os.path.join(dirArchived, 'SinaMF1m_%s.h5t' % i.asof.strftime('%Y%m%d'))
        try :
            size = os.stat(offline_mf1m).st_size
            if size <=0: continue

            # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
            mfn, latestDay = memberfnInH5tar(offline_mf1m, SYMBOL)
            playback.loadJsonH5t(EVENT_MONEYFLOW_1MIN, SYMBOL, offline_mf1m, mfn)
        except Exception as ex:
            thePROG.logexception(ex, offline_mf1m)

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
            raise ex

    if snapshot and len(snapshot) >0:
        h5group = snapshot['ident']
        h5group = h5group[:h5group.index('T')]
        saveSnapshot(snapshoth5fn, h5group= h5group, dsName=snapshot['ident'], snapshot=snapshot['snapshot'], ohlc=snapshot['ohlc'])

    return snapshoth5fn, playback.cachedFiles

####################################
if __name__ == '__main__':
    downloadToday('SZ000002')
