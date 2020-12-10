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
from __future__ import absolute_import, unicode_literals
from celery import shared_task
import sys, os, re
import shutil

from dapps.sinaCrawler.worker import thePROG
from dapps.celeryCommon import RetryableError, Retryable, getMappedAs

from Application import *
from Perspective import *
from MarketData import *
from EventData import Event
import HistoryData as hist
from crawler.producesSina import SinaMux, Sina_Tplus1, SinaSwingScanner
import crawler.crawlSina as sina

import h5py

MAPPED_USER, MAPPED_HOME = getMappedAs()

@shared_task
def add(x, y):
    return x + y

@shared_task
def mul(x, y):
    return x * y

@shared_task
def xsum(numbers):
    return sum(numbers)

def __rmfile(fn) :
    os.remove(fn)

# ===================================================
def memberfnInH5tar(fnH5tar, symbol):
    # we knew the member file would like SinaMF1d_20201010/SH600029_MF1d20201010.json@/root/wkspaces/hpx_archived/sina/SinaMF1d_20201010.h5t
    # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
    fnH5tar = os.path.realpath(fnH5tar)
    mfn = os.path.basename(fnH5tar)[:-4]
    idx = mfn.index('_')
    asofYYMMDD = mfn[1+idx:]
    return '%s_%s%s.json' % (symbol, mfn[4:idx], asofYYMMDD), asofYYMMDD
    # return '%s/%s_%s%s.json' % (mfn, symbol, mfn[4:idx], asofYYMMDD), asofYYMMDD

def saveSnapshot(filename, h5group, snapshot, ohlc):
    compressed = bz2.compress(snapshot)

    with h5py.File(filename, 'a') as h5file:

        if h5group in h5file.keys(): del h5file[h5group]

        g = h5file.create_group(h5group) 
        g.attrs['desc'] = '%s: pickled market state via bzip2 compression' % h5group

        npbytes = np.frombuffer(compressed, dtype=np.uint8)
        sns = g.create_dataset(TAG_SNAPSHORT, data=np.frombuffer(compressed, dtype=np.uint8))
        g.attrs['size'] = len(snapshot)
        g.attrs['csize'] = len(compressed)
        g.attrs['open'] = ohlc[0]
        g.attrs['high'] = ohlc[1]
        g.attrs['low']  = ohlc[2]
        g.attrs['price'] = ohlc[3]
        g.attrs['generated'] = datetime.now().strftime('%Y%m%dT%H%M%S')
        
        thePROG.debug('saved snapshot[%s] %dB->%dz into %s' % (h5group, g.attrs['size'], g.attrs['csize'], filename))
        return True
    
    return False

def __publishFiles(srcfiles) :
    destPubDir = os.path.join(MAPPED_HOME, "hpx_publish")
    pubed = []
    for fn in srcfiles:
        try:
            bn = os.path.basename(fn)
            destFn = os.path.join(destPubDir, bn)
            shutil.copyfile(fn, destFn + "~")
            shutil.move(destFn + "~", destFn)
            pubed.append(bn)
            __rmfile(fn)
        except Exception as ex:
            thePROG.logexception(ex, 'publishFile[%s]' % fn)

    return pubed, destPubDir


# ===================================================
@shared_task(bind=True, base=Retryable, default_retry_delay=10.0)
def downloadToday(self, SYMBOL, todayYYMMDD =None, excludeMoneyFlow=False):
    global MAPPED_USER, MAPPED_HOME
    result = __downloadSymbol(SYMBOL, todayYYMMDD, excludeMoneyFlow)

    thePROG.info('downloadToday() result: %s' % result) 
    # sample: {'symbol': 'SZ000002', 'date': '20201127', 'snapshot': 'SZ000002_sns.h5', 'cachedJsons': ['SZ000002_KL1d20201128.json', 'SZ000002_MF1d20201128.json', 'SZ000002_KL5m20201128.json', 'SZ000002_MF1m20201128.json'], 'tcsv': 'SZ000002_day20201127.tcsv'}
    return result

def __downloadSymbol(SYMBOL, todayYYMMDD =None, excludeMoneyFlow=False):

    CLOCK_TODAY= datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    SINA_TODAY = CLOCK_TODAY.strftime('%Y-%m-%d') if not todayYYMMDD else todayYYMMDD
    if 'SINA_TODAY' in os.environ.keys():
        SINA_TODAY = [os.environ['SINA_TODAY']]

    SINA_TODAY = datetime.strptime(SINA_TODAY, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=0)

    dirCache = '/tmp/sina_cache'
    try:
        os.mkdir(dirCache)
    except: pass
    dirArchived = os.path.join(os.environ['HOME'], 'wkspaces/hpx_archived/sina') 

    todayYYMMDD = SINA_TODAY.strftime('%Y%m%d')
    dirArchived = Program.fixupPath(dirArchived)
    
    # step 1. build up the Playback Mux
    playback   = SinaMux(thePROG, endDate=SINA_TODAY.strftime('%Y%m%dT%H%M%S')) # = thePROG.createApp(SinaMux, **srcPathPatternDict)
    playback.setId('Dayend.%s' % SYMBOL)
    playback.setSymbols([SYMBOL])

    nLastDays = 5 +1

    # 1.a  KL1d and determine the date of n-open-days ago
    caldays = (CLOCK_TODAY - SINA_TODAY).days

    daysTolst = int(caldays/7) *5 + (caldays %7) + 5 + nLastDays
    lastDays =[]

    httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, SYMBOL, daysTolst, saveAs=os.path.join(dirCache, '%s_%s%s.json' %(SYMBOL, chopMarketEVStr(EVENT_KLINE_1DAY), todayYYMMDD)))
    if httperr in [408, 456]:
        raise RetryableError(httperr, "blocked by sina at %s@%s, resp(%s)" %(EVENT_KLINE_1DAY, SYMBOL, httperr))
            
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
    if not excludeMoneyFlow:
        httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1DAY, SYMBOL, 0, saveAs=os.path.join(dirCache, '%s_%s%s.json' %(SYMBOL, chopMarketEVStr(EVENT_MONEYFLOW_1DAY), todayYYMMDD)))
        if httperr in [408, 456] :
            raise RetryableError(httperr, "blocked by sina at %s@%s, resp(%s)" %(EVENT_MONEYFLOW_1DAY, SYMBOL, httperr))

    # 1.c  KL5m
    httperr, _, _ = playback.loadOnline(EVENT_KLINE_5MIN, SYMBOL, 0, saveAs=os.path.join(dirCache, '%s_%s%s.json' %(SYMBOL, chopMarketEVStr(EVENT_KLINE_5MIN), todayYYMMDD)))
    if httperr in [408, 456] :
        raise RetryableError(httperr, "blocked by sina at %s@%s, resp(%s)" %(EVENT_KLINE_5MIN, SYMBOL, httperr))

    # 1.c  MF1m
    if not excludeMoneyFlow:
        httperr, _, _ = playback.loadOnline(EVENT_MONEYFLOW_1MIN, SYMBOL, 0, saveAs=os.path.join(dirCache, '%s_%s%s.json' %(SYMBOL, chopMarketEVStr(EVENT_MONEYFLOW_1MIN), todayYYMMDD)))
        if httperr in [408, 456] :
            raise RetryableError(httperr, "blocked by sina at %s@%s, resp(%s)" %(EVENT_MONEYFLOW_1MIN, SYMBOL, httperr))

        for i in lastDays[1:]:
            offline_mf1m = os.path.join(dirArchived, 'SinaMF1m_%s.h5t' % i.asof.strftime('%Y%m%d'))
            try :
                size = os.stat(offline_mf1m).st_size
                if size <=0: continue

                # instead to scan the member file list of h5t, just directly determine the member file and read it, which save a lot of time
                mfn, latestDay = memberfnInH5tar(offline_mf1m, SYMBOL)
                playback.loadJsonH5t(EVENT_MONEYFLOW_1MIN, SYMBOL, offline_mf1m, mfn)
            except FileNotFoundError as ex:
                thePROG.warn('no offline file avail: %s' %offline_mf1m)
            except Exception as ex:
                thePROG.logexception(ex, offline_mf1m)

    thePROG.info('inited mux with %d substreams' % (playback.size))

    psptMarketState = PerspectiveState(SYMBOL)

    stampOfState, momentsToSample = None, ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:30:00', '15:00:00']
    snapshot = {}
    snapshoth5fn = os.path.join(dirCache, '%s_sns%s.h5' % (SYMBOL, todayYYMMDD))

    fnTcsv = os.path.join(dirCache, '%s_day%s.tcsv' % (SYMBOL, todayYYMMDD))
    try:
        os.remove(fnTcsv)
    except: pass

    rec = thePROG.createApp(hist.TaggedCsvRecorder, filepath = fnTcsv)
    rec.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_5MIN, params={'columns': MoneyflowData.COLUMNS})
    rec.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

    def __onMF5mMerged(mf5m) :
        ev = Event(EVENT_MONEYFLOW_5MIN)
        ev.setData(mf5m)
        psptMarketState.updateByEvent(ev)
        if todayYYMMDD == mf5m.asof.strftime('%Y%m%d') :
            rec.pushRow(ev.type, ev.data)

    mf1mTo5m = sina.SinaMF1mToXm(__onMF5mMerged, 5)

    savedSns= []
    while True:
        try :
            rec.doAppStep() # to flush the recorder
            ev = next(playback)
            if not ev or MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] : continue

            symbol = ev.data.symbol
            if ev.data.datetime <= SINA_TODAY:
                ev = psptMarketState.updateByEvent(ev)
                if not ev or symbol != SYMBOL :
                    continue

                if EVENT_MONEYFLOW_1MIN == ev.type:
                    mf1mTo5m.pushMF1m(ev.data)

                stamp    = psptMarketState.getAsOf(symbol)
                price, _ = psptMarketState.latestPrice(symbol)
                ohlc     = psptMarketState.dailyOHLC_sofar(symbol)

                if not ohlc or todayYYMMDD != stamp.strftime('%Y%m%d'): # or not today
                    continue

                rec.pushRow(ev.type, ev.data)

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
                    h5ident = '%s.%s' %(TAG_SNAPSHORT, snapshot['ident'])
                    if saveSnapshot(snapshoth5fn, h5group=h5ident, snapshot=snapshot['snapshot'], ohlc=snapshot['ohlc']):
                        savedSns.append(h5ident)

                    snapshot ={}
        
        except StopIteration:
            thePROG.info('hist-read: end of playback')
            break
        except Exception as ex:
            thePROG.logexception(ex)
            break # NOT sure why StopIteration not caught above but fell here # raise ex

    for i in range(2): rec.doAppStep() # to flush the recorder

    if snapshot and len(snapshot) >0:
        h5ident = '%s.%s' %(TAG_SNAPSHORT, snapshot['ident'])
        if saveSnapshot(snapshoth5fn, h5group=h5ident, snapshot=snapshot['snapshot'], ohlc=snapshot['ohlc']):
            savedSns.append(h5ident)

    cachedJsons = playback.cachedFiles
    thePROG.info('cached %s, generated %s and snapshots:%s, publishing' % (','.join(cachedJsons), fnTcsv, ','.join(savedSns)))
    dirNameLen = len(dirCache) +1
    pubDir, bns = __publishFiles([snapshoth5fn, fnTcsv] + playback.cachedFiles)

    # map to the arguments of sinaMaster.commitToday()
    return {
        'symbol': SYMBOL,
        'login': MAPPED_USER,
        'asofYYMMDD': todayYYMMDD,
        'fnSnapshot': snapshoth5fn[dirNameLen:], 
        'fnJsons': [x[dirNameLen:] for x in cachedJsons],
        'fnTcsv': fnTcsv[dirNameLen:],
        'lastDays': [[x.asof.strftime('%Y%m%d'), x.open, x.high, x.low, x.close, x.volume] for x in lastDays]
    }

####################################
if __name__ == '__main__':
    # downloadToday('SH510300', excludeMoneyFlow=True)
    downloadToday('SZ002008')
