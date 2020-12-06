# encoding: UTF-8

from __future__ import absolute_import, unicode_literals
from celery import shared_task

from MarketData import MARKETDATE_EVENT_PREFIX
import h5tar, h5py, pickle, bz2

from dapps.sinaMaster.worker import thePROG

from dapps.celeryCommon import RetryableError, Retryable, getMappedAs
import crawler.crawlSina as sina
import crawler.producesSina as prod

import dapps.sinaCrawler.tasks_Dayend as CTDayend

import sys, os, re, glob
from datetime import datetime, timedelta
from time import sleep

SYMBOL_LIST_HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,open,high,low,close,volume"
EOL = "\r\n"
SINA_USERS_ROOT = '/mnt/data/hpwkspace/users'
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

import math
__totalAmt1W =1
def activityOf(item):

    ret = item['amount'] / __totalAmt1W
    if ret >0.0:
        ret = 10* math.sqrt(math.sqrt(ret))
    if item['turnoverratio'] >0.2 :
        ret += math.sqrt(math.sqrt(item['turnoverratio']))
    else: ret /=2

    '''
        excel with SQRT(SQRT($E2))+SQRT(SQRT($I2))*10, top2000 is good to cover my interests
        mktcap,nmc单位：万元
        volume单位：股
        turnoverratio: %
        turnover =close*volume/nmc, for example:
        symbol,name,mktcap,nmc,turnoverratio,close,volume
        SH600519,贵州茅台,211140470.0262,211140470.0262,0.06669,1680.790,837778
        SZ002797,第一创业,4668866.4,3891166.4,4.8251,11.110,168994316
        SZ300008,天海防务,834254.064765,645690.804517,8.58922,8.690,63820266

        turnover(SH600519) =1680.790*837778/211140470.0262 =6.66915/万   vs turnoverratio=0.06669%
        turnover(SZ002797) =11.110*168994316/3891166.4     =482.51/万    vs turnoverratio=4.8251%
        turnover(SZ300008) =8.690*63820266/645690.804517   =858.92/万    vs turnoverratio=8.58922%
    '''

    return ret

def __writeCsv(f, sybmolLst) :
    line = SYMBOL_LIST_HEADERSEQ + EOL
    f.write(line)
    for i in sybmolLst:
        line = ','.join([str(i[k]) for k in SYMBOL_LIST_HEADERSEQ.split(',')]) + EOL
        f.write(line)

@shared_task(bind=True, base=Retryable)
def listAllSymbols(self):

    lstSHZ = []
    fnCachedLst = os.path.join(MAPPED_HOME, 'hpx_publish', 'lstSHZ_%s' % datetime.now().strftime('%Y%m%d'))
    try :
        fn = fnCachedLst + '.pkl.bz2'
        st = os.stat(fn)
        ctime = datetime.fromtimestamp(st.st_ctime)
        if st.st_size >1000 and (ctime.isoweekday() >5 or ctime.hour >=16): 
            with bz2.open(fn, 'rb') as f:
                lstSHZ = pickle.load(f)
    except Exception as ex:
        pass

    if len(lstSHZ) <=2000:
        lstSH, lstSZ = __listAllSymbols()
        lstSHZ = {} # temporarily via dict
        for i in lstSH + lstSZ:
            lstSHZ[i['symbol']] =i
        lstSHZ = list(lstSHZ.values())

        totalAmt1W=0
        for i in lstSHZ:
            totalAmt1W += i['amount'] /10000.0

        if totalAmt1W >1.0: # for activityOf() 
            global __totalAmt1W
            __totalAmt1W = totalAmt1W

        noneST = list(filter(lambda x: not 'ST' in x['name'], lstSHZ))
        noneST.sort(key=activityOf)
        noneST.reverse()

        STs = list(filter(lambda x: 'ST' in x['name'], lstSHZ))
        STs.sort(key=activityOf)
        STs.reverse()

        lstSHZ = noneST + STs
        for fn in glob.glob(os.path.join(MAPPED_HOME, 'hpx_publish') + "/lstSHZ_*.pkl.bz2") :
            try :
                os.remove(fn)
            except Exception as ex:
                pass

        with bz2.open(fnCachedLst + '.pkl.bz2', 'wb') as f:
            f.write(pickle.dumps(lstSHZ))

        with bz2.open(fnCachedLst + '.csv.bz2', 'wt', encoding='utf-8') as f:
            __writeCsv(f, lstSHZ)
    
    return lstSHZ

    # csvNoneST = SYMBOL_LIST_HEADERSEQ + EOL
    # for i in noneST:
    #     csvNoneST += ','.join([str(i[k]) for k in SYMBOL_LIST_HEADERSEQ.split(',')]) +EOL

    # csvSTs = SYMBOL_LIST_HEADERSEQ + EOL
    # for i in STs:
    #     csvSTs += ','.join([str(i[k]) for k in SYMBOL_LIST_HEADERSEQ.split(',')]) +EOL

    # return csvNoneST, csvSTs

# def commitToday(self, login, symbol, asofYYMMDD, fnJsons, fnSnapshot, fnTcsv) :
@shared_task(bind=True, base=Retryable)
def commitToday(self, dictArgs) : # urgly at the parameter list
    '''
    in order to chain:
    import celery
    import dapps.sinaMaster.tasks as mt
    import dapps.sinaCrawler.tasks_Dayend as ct
    s3 = celery.chain(ct.downloadToday.s('SZ000005'), mt.commitToday.s())
    s3().get()
    '''
    if not dictArgs or not isinstance(dictArgs, dict) or len(dictArgs) <=0:
        thePROG.error('commitToday() invalid dictArgs: %s' % str(dictArgs))
        return

    login, asofYYMMDD = 'hpx01', datetime.now().strftime('%Y%m%d')
    login = dictArgs.get('login', login)
    asofYYMMDD = dictArgs.get('asofYYMMDD', asofYYMMDD)

    symbol = dictArgs.get('symbol', None)
    fnJsons = dictArgs.get('fnJsons', [])
    fnSnapshot = dictArgs.get('fnSnapshot', None)
    fnTcsv = dictArgs.get('fnTcsv', None)
    ''' sample value:
    fnJsons = ['SZ000002_KL1d20201202.json', 'SZ000002_MF1d20201202.json', 'SZ000002_KL5m20201202.json', 'SZ000002_MF1m20201202.json']
    fnSnapshot = 'SZ000002_sns.h5';
    ~{HOME}
    |-- archived -> ../archived
    `-- hpx_template -> /home/wkspaces/hpx_template
    '''

    if not symbol:
        thePROG.error('commitToday() invalid dictArgs: %s' % str(dictArgs))
        return

    if '@' in login : login = login[:login.index('@')]
    if ':' in login : login = login[:login.index(':')]

    pubDir = os.path.join(SINA_USERS_ROOT, login, 'hpx_publish')
    archDir = os.path.join(MAPPED_HOME, 'archived', 'sina')

    # archDir = '/tmp/arch_test' # test hardcode
    # pubDir = '/mnt/s/hpx_publish' # test hardcode

    thePROG.debug('commitToday() %s_%s dictArgs: %s from %s to %s' % (symbol, asofYYMMDD, str(dictArgs), pubDir, archDir))

    # step 1. zip the JSON files
    for fn in fnJsons:
        srcpath = os.path.join(pubDir, fn)
        fn = os.path.basename(fn)
        m = re.match(r'%s_([A-Za-z0-9]*)%s.json' %(symbol, asofYYMMDD), fn)
        if not m : continue
        evtShort = m.group(1)

        destpath = os.path.join(archDir, 'Sina%s_%s.h5t' % (evtShort, asofYYMMDD) )
        if h5tar.tar_utf8(destpath, srcpath) :
            thePROG.info('archived %s into %s' %(srcpath, destpath))

    # step 1. zip the Tcsv file
    srcpath = os.path.join(pubDir, fnTcsv)
    destpath = os.path.join(archDir, 'SinaDay_%s.h5t' % asofYYMMDD )
    if h5tar.tar_utf8(destpath, srcpath, baseNameAsKey=True) :
        thePROG.info('archived %s into %s' %(srcpath, destpath))

    # step 3. append the snapshots
    srcpath = os.path.join(pubDir, fnSnapshot)
    destpath = os.path.join(archDir, 'SNS_%s.h5' % (symbol) )
    gns = []
    with h5py.File(srcpath, 'r') as h5r:
        with h5py.File(destpath, 'a') as h5w:
            for gn in h5r.keys():
                if not symbol in gn: continue
                g = h5r[gn]
                if not 'desc' in g.attrs or not 'pickled market state' in g.attrs['desc'] : continue

                if gn in h5w.keys(): del h5w[gn]
                # Note that this is not a copy of the dataset! Like hard links in a UNIX file system, objects in an HDF5 file can be stored in multiple groups
                # So, h5w[gn] = g doesn't work because across different files
                go = h5w.create_group(gn)
                h5w.copy(g, go)
                # h5w[gn] = g()
                # h5py.Group.copy(g, h5w[gn])
                gns.append(gn)
    thePROG.info('added snapshot[%s] of %s into %s' % (','.join(gns), srcpath, destpath))

def __listAllSymbols():
    result ={}
    EOL ="\n"

    md = sina.SinaCrawler(thePROG, None)

    httperr, retryNo =100, 0
    for i in range(50):
        httperr, lstSH = md.GET_AllSymbols('SH')
        if 2 == int(httperr/100): break

        thePROG.warn('SH-resp(%d) len=%d' %(httperr, len(lstSH)))
        if 456 == httperr :
            retryNo += 1
            sleep(prod.defaultNextYield(retryNo))
            continue

    if len(lstSH) <=0:
        raise RetryableError(httperr, "no SH fetched")
    thePROG.info('SH-resp(%d) len=%d' %(httperr, len(lstSH)))

    for i in range(50):
        httperr, lstSZ = md.GET_AllSymbols('SZ')
        if 2 == int(httperr/100): break

        thePROG.warn('SZ-resp(%d) len=%d' %(httperr, len(lstSZ)))
        if 456 == httperr :
            retryNo += 1
            sleep(prod.defaultNextYield(retryNo))
            continue

    if len(lstSZ) <=0:
        raise RetryableError(httperr, "no SZ fetched")
    thePROG.info('SZ-resp(%d) len=%d' %(httperr, len(lstSZ)))
    return lstSH, lstSZ

'''
@worker.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Calls test('hello') every 10 seconds.
    sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
 
    # Calls test('world') every 30 seconds
    sender.add_periodic_task(30.0, test.s('world'), expires=10)
 
    # Executes every Monday morning at 7:30 a.m.
    sender.add_periodic_task(
        crontab(hour=7, minute=30, day_of_week=1),
        test.s('Happy Mondays!'),
    )
'''
 
@shared_task(bind=True, base=Retryable)
def topActives(self, topNum = 500):
    lstTops = listAllSymbols() [:topNum]
    return lstTops

__asyncResult_downloadToday = {}

@shared_task(bind=True, base=Retryable)
def schOn_Every5min(self):
    global __asyncResult_downloadToday
    todels = []
    for k, v in __asyncResult_downloadToday.items():
        if not v: 
            todels.append(k)
            continue

        if v.ready():
            todels.append(k)
            thePROG.info('schOn_Every5min() downloadToday[%s]%s done: succ[%s] and will clear' %(k, v.task_id, v.successful()))
            continue

        thePROG.info('schOn_Every5min() downloadToday[%s]%s still working' %(k, v.task_id))

    if len(todels) >0:
        thePROG.info('schOn_Every5min() clearing keys: %s' % ','.join(todels))
        for k in todels:
            del __asyncResult_downloadToday[k]
        if len(__asyncResult_downloadToday) <=0:
            thePROG.info('schOn_Every5min() downloadToday all done')

@shared_task(bind=True, base=Retryable)
def schOn_TradeDayClose(self):
    lstSHZ = listAllSymbols()

    thePROG.info('schOn_TradeDayClose() listAllSymbols got %d symbols' %len(lstSHZ))
    if len(lstSHZ) <=2000:
        raise RetryableError(401, 'incompleted symbol list')

    global __asyncResult_downloadToday
    __asyncResult_downloadToday = {}

    for i in lstSHZ[:10]: # should be the complete lstSHZ
        symbol = i['symbol']
        if symbol in __asyncResult_downloadToday.keys():
            continue

        thePROG.debug('schOn_TradeDayClose() adding subtask to download %s %s' % (symbol, i['name']))
        wflow = CTDayend.downloadToday.s(symbol) | commitToday.s()
        __asyncResult_downloadToday[symbol] = wflow()


####################################
if __name__ == '__main__':
    schOn_TradeDayClose()
    for i in range(20):
        schOn_Every5min()
        sleep(10)

    # nTop = 1000
    # lstSHZ = topActives(nTop)
    # with open(os.path.join(MAPPED_HOME, 'hpx_publish', 'top%s_%s' % (nTop, datetime.now().strftime('%Y%m%d'))) + '.csv', 'wb') as f:
    #     __writeCsv(f, lstSHZ)
    # print(lstSHZ)

    # symbol = 'SZ000002'
    # login = 'root@tc2.syscheme.com'
    # asofYYMMDD = '20201202'
    # fnJsons = ['SZ000002_KL1d20201202.json', 'SZ000002_MF1d20201202.json', 'SZ000002_KL5m20201202.json', 'SZ000002_MF1m20201202.json']
    # fnSnapshot = 'SZ000002_sns.h5';
    # fnTcsv = 'SZ000002_day20201202.tcsv';

    # commitToday(login, symbol, asofYYMMDD, fnJsons, fnSnapshot, fnTcsv)

''' A test
import dapps.sinaCrawler.tasks_Dayend as ct
import dapps.sinaMaster.tasks as mt
c1 = ct.downloadToday.s('SZ000002') | mt.commitToday.s()
c1().get()
'''