# encoding: UTF-8

from __future__ import absolute_import, unicode_literals
from celery import shared_task

from dapps.celeryCommon import RetryableError, Retryable, getMappedAs
from dapps.sinaMaster.worker import thePROG
import dapps.sinaCrawler.tasks_Dayend as CTDayend

import crawler.crawlSina as sina
import crawler.producesSina as prod

from MarketData import MARKETDATE_EVENT_PREFIX, EVENT_KLINE_1DAY
import HistoryData as hist

import h5tar, h5py, pickle, bz2
from urllib.parse import quote, unquote
import sys, os, re, glob, stat, shutil, fnmatch
from datetime import datetime, timedelta

SYMBOL_LIST_HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,open,high,low,close,volume"
EOL = "\r\n"
SINA_USERS_ROOT = '/mnt/data/hpwkspace/users'
MAPPED_USER, MAPPED_HOME = getMappedAs(homeDir = '/mnt/s') # master certainly take the local volume /mnt/s
if MAPPED_USER in [ None, 'nobody'] :  MAPPED_USER = 'hpx'
SUBDIR_Reqs = 'reqs'
DIR_ARCHED_HOME = os.path.join(MAPPED_HOME, 'archived', 'sina')

IDXs_to_COLLECT=[ # http://vip.stock.finance.sina.com.cn/mkt/#dpzs
'SH000001',	# 上证指数
'SZ399001',	# 深证成指
'SZ399005',	# 中小板指
'SZ399006',	# 创业板指
'SH000011',	# 基金指数
]

ETFs_to_COLLECT=[   # asof 2020-12-08 top actives: http://vip.stock.finance.sina.com.cn/fund_center/index.html#jjhqetf
'SH510300','SH512880','SH510050','SH510900','SH518880','SZ159919','SH510500','SZ159934','SZ159949','SH512000',
'SH511660','SZ159920','SZ159995','SH588000','SH510330','SZ159915','SH515030','SH512760','SH512800','SZ159937',
'SH512660','SH512480','SH512690','SH515700','SH515050','SH515380','SH518800','SH512400','SZ159922','SH588080',
'SH512500','SZ159001','SH588050','SZ159003','SH510310','SH515000','SH513050','SH588090','SZ159992','SH510880',
'SH513090','SH512290','SZ159928','SZ159901','SZ159806','SH511260','SH512010','SH515220','SZ159952','SH511810',
'SH512710','SH510850','SH510510','SH512900','SZ159966','SH512170','SZ159994','SH511010','SH510180','SZ159996',
'SZ159801','SZ159967','SH510230','SH515210','SZ159993','SH515880','SZ159997','SH513100','SZ159807','SH512070',
'SZ159941','SH515330','SH511380','SH515260','SH512200','SH513500','SZ159905','SH512720','SZ159820','SH512980',
'SH515650','SH515800','SH515560','SH511690','SH515770','SH510760','SH515750','SZ159819','SZ159948','SH512100',
'SH512670','SZ159813','SH512700','SZ159977','SH510710','SH510630','SZ159939','SH510580','SH510350','SZ159968',
'SZ159902','SH512680','SH512910','SZ159998','SH513300','SZ159816','SH512090','SH510100','SZ159972','SH512160',
'SZ159980','SH515530','SH512580','SH515630','SZ159938','SZ159811','SZ159985','SH515390','SZ159929','SH515580',
'SH515070','SH510800','SH510600','SH511180','SH515980','SZ159808','SH512510','SH510390','SH510150','SH512730'
]

SYMBOLS_WithNoMF = IDXs_to_COLLECT + ETFs_to_COLLECT

TASK_TIMEOUT_DownloadToday = timedelta(minutes=60)
BATCHSIZE_DownloadToday    = 500

TODAY_YYMMDD = None

@shared_task
def add(x, y):
    sleep(30)
    return x + y

@shared_task
def mul(x, y):
    sleep(30)
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

def __rmfile(fn) :
    try :
        os.remove(fn)
    except:
        pass

# ===================================================
@shared_task(bind=True, base=Retryable)
def listAllSymbols(self):

    lstSHZ = []
    fnCachedLst = os.path.join(MAPPED_HOME, 'hpx_publish', 'lstSHZ_%s.pkl.bz2' % datetime.now().strftime('%Y%m%d'))
    try :
        st = os.stat(fnCachedLst)
        ctime = datetime.fromtimestamp(st.st_ctime)
        if st.st_size >1000 and (ctime.isoweekday() >5 or ctime.hour >=16): 
            with bz2.open(fnCachedLst, 'rb') as f:
                lstSHZ = pickle.load(f)
    except Exception as ex:
        pass

    if len(lstSHZ) <=2000:
        lstSH, lstSZ = prod.listAllSymbols(thePROG)
        if len(lstSH) <= 0 or len(lstSZ) <= 0:
            raise RetryableError(456, "empty SH[%s] or empty SH[%s] fetched" %(len(lstSH), len(lstSZ)))

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

        try:
            with bz2.open(fnCachedLst, 'wb') as f:
                f.write(pickle.dumps(lstSHZ))
        except :
            pass

        try:
            lstArch = os.path.join(MAPPED_HOME, 'hpx_archived', 'sina', 'lstSHZ_%s.csv.bz2' % datetime.now().strftime('%Y%m%d'))
            with bz2.open(lstArch, 'wt', encoding='utf-8') as f:
                __writeCsv(f, lstSHZ)
        except :
            pass
    
    return lstSHZ

    # csvNoneST = SYMBOL_LIST_HEADERSEQ + EOL
    # for i in noneST:
    #     csvNoneST += ','.join([str(i[k]) for k in SYMBOL_LIST_HEADERSEQ.split(',')]) +EOL

    # csvSTs = SYMBOL_LIST_HEADERSEQ + EOL
    # for i in STs:
    #     csvSTs += ','.join([str(i[k]) for k in SYMBOL_LIST_HEADERSEQ.split(',')]) +EOL

    # return csvNoneST, csvSTs

# ===================================================
@shared_task(bind=True, base=Retryable, max_retries=5)
def commitToday(self, dictArgs) : # urgly at the parameter list
    '''
    in order to chain:
    import celery
    import dapps.sinaMaster.tasks as mt
    import dapps.sinaCrawler.tasks_Dayend as ct
    s3 = celery.chain(ct.downloadToday.s('SZ000005'), mt.commitToday.s())
    s3().get()
    '''
    if dictArgs is None:
        thePROG.warn('commitToday() None dictArgs, prev-req might be cancelled')
        return

    if not isinstance(dictArgs, dict) or len(dictArgs) <=0:
        thePROG.error('commitToday() invalid dictArgs: %s' % str(dictArgs))
        return

    login, asofYYMMDD = 'hpx01', datetime.now().strftime('%Y%m%d')
    login = dictArgs.get('login', login)
    asofYYMMDD = dictArgs.get('asofYYMMDD', asofYYMMDD)

    symbol = dictArgs.get('symbol', None)
    fnJsons = dictArgs.get('fnJsons', [])
    fnSnapshot = dictArgs.get('fnSnapshot', None)
    fnTcsv = dictArgs.get('fnTcsv', None)
    lastDays = dictArgs.get('lastDays', [])
    ''' sample value:
    fnJsons = ['SZ000002_KL1d20201202.json', 'SZ000002_MF1d20201202.json', 'SZ000002_KL5m20201202.json', 'SZ000002_MF1m20201202.json']
    fnSnapshot = 'SZ000002_sns.h5';
    ~{HOME}
    |-- archived -> ../archived
    `-- hpx_template -> /home/wkspaces/hpx_template
    2021-01-03 10:05:03,683: DEBUG/ForkPoolWorker-1] commitToday() archived /mnt/data/hpwkspace/users/hpx/hpx_publish/SZ300422_day20201228.tcsv by[hpx] into /mnt/data/hpwkspace/users/master/archived/sina/SinaMDay_20201228.h5t
    '''


    if not symbol:
        thePROG.error('commitToday() invalid dictArgs: %s' % str(dictArgs))
        return

    if '@' in login : login = login[:login.index('@')]
    if ':' in login : login = login[:login.index(':')]

    pubDir = os.path.join(SINA_USERS_ROOT, login, 'hpx_publish')

    # pubDir = '/mnt/s/hpx_publish' # test hardcode
    # DIR_ARCHED_HOME = '/tmp/arch_test' # test hardcode

    try:
        os.mkdir(os.path.join(DIR_ARCHED_HOME, 'snapshots'))
        os.chmod(dirReqs, stat.S_IRWXU | stat.S_IRWXG |stat.S_IROTH )
    except: pass

    if TODAY_YYMMDD and asofYYMMDD < TODAY_YYMMDD:
        # this symbol must be frozen today
        thePROG.warn('commitToday() archiving %s_%s sounds not open, dictArgs: %s, cleaning %s' % (symbol, asofYYMMDD, str(dictArgs), pubDir))
        for fn in fnJsons + [fnTcsv, fnSnapshot]:
            srcpath = os.path.join(pubDir, fn)
            __rmfile(srcpath)
        asofYYMMDD = TODAY_YYMMDD # to clear the req of today
    else:

        thePROG.debug('commitToday() archiving %s_%s dictArgs: %s from %s to %s' % (symbol, asofYYMMDD, str(dictArgs), pubDir, DIR_ARCHED_HOME))

        # step 1. zip the JSON files
        for fn in fnJsons:
            srcpath = os.path.join(pubDir, fn)
            m = re.match(r'%s_([A-Za-z0-9]*)%s.json' %(symbol, asofYYMMDD), os.path.basename(srcpath))
            if not m : continue
            evtShort = m.group(1)

            try :
                destpath = os.path.join(DIR_ARCHED_HOME, 'Sina%s_%s.h5t' % (evtShort, asofYYMMDD) )
                if h5tar.tar_utf8(destpath, srcpath, baseNameAsKey=True) :
                    thePROG.debug('commitToday() archived %s into %s' %(srcpath, destpath))
                    __rmfile(srcpath)
                else:
                    thePROG.error('commitToday() failed to archived %s into %s' %(srcpath, destpath))
            except Exception as ex:
                thePROG.logexception(ex, 'commitToday() archiving[%s->%s] error' % (srcpath, destpath))

        # step 2. zip the Tcsv file
        srcpath = os.path.join(pubDir, fnTcsv)
        destpath = os.path.join(DIR_ARCHED_HOME, 'SinaMDay_%s.h5t' % asofYYMMDD )
        if h5tar.tar_utf8(destpath, srcpath, baseNameAsKey=True) :
            thePROG.debug('commitToday() archived %s by[%s] into %s' %(srcpath, login, destpath))
            __rmfile(srcpath)
        else:
            thePROG.error('commitToday() failed to archived %s by[%s] into %s' %(srcpath, login, destpath))

        # step 3. append the snapshots
        srcpath = os.path.join(pubDir, fnSnapshot)
        destpath = os.path.join(DIR_ARCHED_HOME, 'snapshots', 'SNS_%s.h5' % (symbol) )
        gns = []

        lastDates = [ x[0] for x in lastDays] if lastDays and len(lastDays)>0 else []
        try :
            with h5py.File(destpath, 'a') as h5w:
                # step 3.1, copy the new SNS into the dest h5f
                with h5py.File(srcpath, 'r') as h5r:
                    for gn in h5r.keys():
                        if not symbol in gn: continue
                        g = h5r[gn]
                        if not 'desc' in g.attrs.keys() or not 'pickled market state' in g.attrs['desc'] : continue
                        gdesc = g.attrs['desc']

                        if gn in h5w.keys(): del h5w[gn]
                        # Note that this is not a copy of the dataset! Like hard links in a UNIX file system, objects in an HDF5 file can be stored in multiple groups
                        # So, h5w[gn] = g doesn't work because across different files
                        # go = h5w.create_group(gn)
                        h5r.copy(g.name, h5w) # note the destGroup is the parent where the group want to copy under-to
                        go = h5w[gn]
                        gns.append(gn)
                        # m1, m2 = list(g.keys()), list(go.keys())
                        # a1, a2 = list(g.attrs.keys()), list(go.attrs.keys())

                # step 3.2, determine the grainRate_X based on lastDays
                if len(lastDates) >0:
                    start, end = '%sT000000' % lastDays[-1][0], '%sT235959' % lastDays[0][0]
                    for k in h5w.keys() :
                        if not symbol +'@' in k : continue
                        strAsOf = k[1 + k.index('@'):]
                        if strAsOf < start or strAsOf > end: continue
                        go = h5w[k]
                        if not 'price' in go.attrs.keys() or float(go.attrs['price']) <=0.0:
                            continue

                        try :
                            strYYMMDD = strAsOf[:strAsOf.index('T'):] if 'T' in strAsOf else strAsOf
                            nDaysAgo = lastDates.index(strYYMMDD)
                            if nDaysAgo <0: continue

                            grk, grv = 'grainRate_%d' % nDaysAgo, lastDays[nDaysAgo][4] / go.attrs['price'] -1
                            go.attrs[grk] = grv
                        except Exception as ex:
                            thePROG.warn('commitToday() failed to determine grainRate for: %s' % k)
                        
            thePROG.debug('commitToday() added snapshot[%s] of %s into %s' % (','.join(gns), srcpath, destpath))
            __rmfile(srcpath)
        except Exception as ex:
            thePROG.logexception(ex, 'commitToday() snapshot[%s->%s] error' % (srcpath, destpath))

    # step 4, delete the request file and record
    dirReqs = os.path.join(DIR_ARCHED_HOME, SUBDIR_Reqs)
    # fnReq = os.path.join(dirReqs, '%s_%s.tcsv.bz2' % (asofYYMMDD, symbol))
    # __rmfile(fnReq)
    # thePROG.debug('commitToday() removed %s' % fnReq)

    dictDownloadReqs = _loadDownloadReqs(dirReqs)
    if asofYYMMDD in dictDownloadReqs.keys():
        dictToday = dictDownloadReqs[asofYYMMDD]
        if symbol in dictToday.keys():
            reqNode = dictToday[symbol]
            stampNow = datetime.now()
            taskId, stampIssued, tn = reqNode['taskId'], reqNode['stampIssued'], reqNode['taskFn']
            reqNode['stampCommitted'] = stampNow
            __rmfile(tn)
            thePROG.info('commitToday() dictDownloadReqs[%s][%s] task[%s] took %s by[%s], deleted %s' % (asofYYMMDD, symbol, taskId, stampNow - stampIssued, login, tn))
        
        nleft = len(dictToday)
        c = sum([1 if not v['stampCommitted'] else 0 for v in dictToday.values() ])
        thePROG.debug('commitToday() dictDownloadReqs[%s] has %d/%d onging' % (asofYYMMDD, c, nleft))
        
        _saveDownloadReqs(dirReqs)

# ===================================================
@shared_task(bind=True, base=Retryable)
def topActives(self, topNum = 500):
    lstTops = listAllSymbols() [:topNum]
    return lstTops


# RETRY_DOWNLOAD_INTERVAL = timedelta(hours=1)
RETRY_DOWNLOAD_INTERVAL = timedelta(minutes=30)
# ===================================================
@shared_task(bind=True)
def schChkRes_Crawlers(self, asofYYMMDD =None): # asofYYMMDD ='20201231'):
    global MAPPED_HOME, TODAY_YYMMDD

    if asofYYMMDD:
        TODAY_YYMMDD = asofYYMMDD

    stampNow = datetime.now()
    if not TODAY_YYMMDD:
        TODAY_YYMMDD = (stampNow-timedelta(hours=9)).strftime('%Y%m%d')
    
    dirReqs = os.path.join(DIR_ARCHED_HOME, SUBDIR_Reqs)

    thePROG.debug('schChkRes_Crawlers() refreshing tasks of downloadTodays[%s]' % TODAY_YYMMDD)
    __refreshBatch_DownloadToday(dirReqs, TODAY_YYMMDD)

    from dapps.sinaCrawler.worker import worker as crawler
    crawlers = crawler.control.ping(timeout=2.0, queue='crawler')
    crawlers = [ list(c.keys())[0] for c in crawlers ]
    thePROG.info('schChkRes_Crawlers() found %d crawlers: %s' % (len(crawlers), ','.join(crawlers) ) )
    '''
    cacheFiles = [ 'SinaMF1m_%s.h5t' %i for i in yymmddToCache]

    for c in crawlers:
        q = c.split('@')[0]
        if not q or len(q) <=0: continue
        r = CTDayend.fetchArchivedFiles.apply_async(args=[cacheFiles], queue=q)
        thePROG.info('schDo_pitchArchiedFiles() called crawler[%s].fetchArchivedFiles: %s' % (q, ','.join(cacheFiles)))
    '''


# ===================================================
__dictDownloadReqs = None
def _loadDownloadReqs(dirReqs) :
    global __dictDownloadReqs
    if not __dictDownloadReqs:
        fn = os.path.join(dirReqs, 'dictDownloadReqs.pkl.bz2')
        try:
            with bz2.open(fn, 'rb') as f:
                __dictDownloadReqs = pickle.load(f)
        except Exception as ex:
            __dictDownloadReqs ={}
            __rmfile(fn)

    return __dictDownloadReqs

def _saveDownloadReqs(dirReqs):
    global __dictDownloadReqs
    if not __dictDownloadReqs: return
    fn = os.path.join(dirReqs, 'dictDownloadReqs.pkl.bz2')
    try:
        with bz2.open(fn, 'wb') as f:
            f.write(pickle.dumps(__dictDownloadReqs))
    except Exception as ex:
        pass

# ===================================================
def __refreshBatch_DownloadToday(dirReqs, TODAY_YYMMDD):

    dictDownloadReqs = _loadDownloadReqs(dirReqs)
    if not dictDownloadReqs or not TODAY_YYMMDD or not TODAY_YYMMDD in dictDownloadReqs.keys():
        thePROG.debug('__refreshBatch_DownloadToday() no active downloadToday[%s]' %TODAY_YYMMDD)
        return

    dictToday = dictDownloadReqs[TODAY_YYMMDD]
    thePROG.debug('__refreshBatch_DownloadToday() %d actives in downloadToday[%s]' %(len(dictToday), TODAY_YYMMDD))

    todels, bDirty = [], False
    reqsPending = []
    stampNow = datetime.now()

    for k, v in dictToday.items():
        if not v or not 'task' in v.keys() or not v['task']: 
            todels.append(k)
            continue

        task = v['task']
        try :
            timelive = stampNow - v['stampIssued']
            if v['stampCommitted']:
                todels.append(k)
                thePROG.info('__refreshBatch_DownloadToday() downloadToday[%s]%s committed, duration %s, removed from dictToday' %(k, task.id, v['stampCommitted']-v['stampIssued']))
                continue

            if not v['stampReady'] and task.ready():
                v['stampReady'] = stampNow
                thePROG.debug('__refreshBatch_DownloadToday() downloadToday[%s]%s:%s succ[%s], took %s' %(k, task.id, task.state, task.successful(), timelive))
                continue

            if timelive > TASK_TIMEOUT_DownloadToday and task.state in ['PENDING', 'REVOKED']:
                todels.append(k)
                thePROG.warn('__refreshBatch_DownloadToday() downloadToday[%s]%s:%s took %s timeout, revoking[%s] and retry' %(k, task.id, task.state, timelive, task.parent.id))
                task.parent.revoke() # we only revoke the first in the chain here, always let commitToday go if its prev steps have been completed
                continue

            reqsPending.append(v['taskFn'])

        except Exception as ex:
            thePROG.logexception(ex, '__refreshBatch_DownloadToday() checking task of %s' % (k))

    if len(todels) >0:
        bDirty = True
        thePROG.info('__refreshBatch_DownloadToday() clearing %s keys: %s' % (len(todels), ','.join(todels)))
        for k in todels:
            del dictToday[k]

    cTasksToAdd = BATCHSIZE_DownloadToday - len(reqsPending)

    if cTasksToAdd <=0:
        thePROG.debug('__refreshBatch_DownloadToday() %d pendings[%s ~ %s] hit max %d, no more add-in' % (len(reqsPending), reqsPending[0], reqsPending[-1], BATCHSIZE_DownloadToday))
        return

    Tname_batchStart = os.path.basename(max(reqsPending)) if len(reqsPending) >0 else ''

    allfiles = hist.listAllFiles(dirReqs, depthAllowed=1)
    taskfiles, potentialRetries = [], []
    for fn in allfiles:
        bn = os.path.basename(fn)
        if not fnmatch.fnmatch(bn, 'T%s.*.tcsv.bz2' % TODAY_YYMMDD) :
            continue

        if bn <= Tname_batchStart and (len(potentialRetries) + len(taskfiles)) < cTasksToAdd and not fn in reqsPending:
            potentialRetries.append(fn)
            continue

        taskfiles.append(fn)
    
    taskfiles.sort()
    potentialRetries.sort()
    newissued = []

    prefix2cut = DIR_ARCHED_HOME +'/'
    prefixlen = len(prefix2cut)

    for tn in taskfiles + potentialRetries:
        bn = os.path.basename(tn)
        symbol = bn.split('.')[2]
        exclMF = symbol in SYMBOLS_WithNoMF
        fnTask = tn[prefixlen:] if prefix2cut == tn[: prefixlen] else tn
        wflow = CTDayend.downloadToday.s(symbol, fnPrevTcsv = fnTask, excludeMoneyFlow=exclMF) | commitToday.s()
        task = wflow()
        dictToday[symbol] = {
            'symbol': symbol,
            'taskFn': tn,
            'task': task,
            'taskId': task.id,
            'stampIssued': datetime.now(),
            'stampReady': None,
            'stampCommitted': None
            }

        newissued.append(symbol)
        if len(newissued) >= cTasksToAdd: break

    thePROG.info('__refreshBatch_DownloadToday() fired %d/%d new requests: %s' % (len(newissued), len(taskfiles), ','.join(newissued)))
    if len(newissued) >0 : 
        bDirty = True
    elif len(dictToday) <=0:
        del dictDownloadReqs[TODAY_YYMMDD]
        bDirty = True
        thePROG.info('__refreshBatch_DownloadToday() all DownloadReqs[%s] done, removed' % (TODAY_YYMMDD))

    if bDirty:
        _saveDownloadReqs(dirReqs)

# ===================================================
@shared_task(bind=True, base=Retryable)
def schKickOff_DownloadToday(self):
    lastYYMMDDs = prod.determineLastDays(thePROG, nLastDays =7)
    if len(lastYYMMDDs) <=0:
        return

    global TODAY_YYMMDD
    TODAY_YYMMDD = lastYYMMDDs[0]
    # DIR_ARCHED_HOME = '/mnt/e/AShareSample/hpx_archived/sina'  # TEST CODE
    dirReqs = os.path.join(DIR_ARCHED_HOME, SUBDIR_Reqs)

    try:
        os.mkdir(dirReqs)
        os.chmod(dirReqs, stat.S_IRWXU | stat.S_IRWXG |stat.S_IRWXO )
        shutil.chown(dirReqs, group ='hpx')
    except: pass

    dictDownloadReqs = _loadDownloadReqs(dirReqs)

    lstSHZ = listAllSymbols()
    thePROG.info('schKickOff_DownloadToday() listAllSymbols got %d symbols and last trade-days: %s' % (len(lstSHZ), ','.join(lastYYMMDDs)))
    if len(lstSHZ) <=2000:
        raise RetryableError(401, 'incompleted symbol list')

    if not TODAY_YYMMDD in __dictDownloadReqs.keys():
        # TODO cancel dictDownloadReqs[TODAY_YYMMDD]
        dictDownloadReqs[TODAY_YYMMDD] = {}
    else:
        for v in dictDownloadReqs[TODAY_YYMMDD].values():
            try :
                if 'task' in v.keys() or not v['task']: continue
                task = v['task']
                task.parent.revoke() # we only revoke the first in the chain here, always let commitToday go if its prev steps have been completed
            except: pass

    _saveDownloadReqs(dirReqs)

    lstStocks = [ x['symbol'] for x in lstSHZ ]

    cTasks =0
    for symbol in IDXs_to_COLLECT + ETFs_to_COLLECT + lstStocks:
        cTasks += 1
        rfnRequest = os.path.join(SUBDIR_Reqs, 'T%s.%04d.%s.tcsv.bz2' % (TODAY_YYMMDD, cTasks, symbol))
        fullfnRequest = os.path.join(DIR_ARCHED_HOME, rfnRequest)
        excludeMoneyFlow = symbol in SYMBOLS_WithNoMF
        try:
            st = os.stat(fullfnRequest)
            thePROG.debug('schKickOff_DownloadToday() %s already exists' % rfnRequest)
            continue
        except: pass

        thePROG.debug('schKickOff_DownloadToday() generating request-file %s' % rfnRequest)
        alllines = prod.readArchivedDays(thePROG, DIR_ARCHED_HOME, symbol, lastYYMMDDs[1:])
        # no tcsv data in the nLastDays doesn't mean it has no trades today:
        # if len(alllines) <= 100:
        #     thePROG.debug('schKickOff_DownloadToday() skip empty request %s size %d' % (rfnRequest, len(alllines))
        #     continue
        with bz2.open(fullfnRequest, 'wt', encoding='utf-8') as f:
            f.write(alllines)
            try:
                shutil.chown(fullfnRequest, group ='hpx')
                os.chmod(fullfnRequest, stat.S_IREAD|stat.S_IWRITE|stat.S_IRGRP|stat.S_IWGRP|stat.S_IROTH )
            except: pass
            thePROG.debug('schKickOff_DownloadToday() generated task-file %s' % rfnRequest)

    __refreshBatch_DownloadToday(dirReqs, TODAY_YYMMDD)

'''
# ===================================================
@shared_task(bind=True, base=Retryable)
def schDo_pitchArchiedFiles(self):

    listAllSymbols()

    nLastDays, lastDays = 7, []
    yymmddToday = (stampNow-timedelta(hours=9)).strftime('%Y%m%d')
    yymmddToday = datetime.now().strftime('%Y%m%d')

    playback = prod.SinaMux(thePROG)
    httperr, _, lastDays = playback.loadOnline(EVENT_KLINE_1DAY, IDXs_to_COLLECT[0], nLastDays+3)
    lastDays.reverse()
    yymmddToCache = []
    for i in lastDays:
        yymmdd = i.asof.strftime('%Y%m%d')
        if yymmdd >= yymmddToday:
            continue
        yymmddToCache.append(yymmdd)
        if len(yymmddToCache) >= nLastDays:
            break
    
    if len(yymmddToCache) <=0:
        return

    from dapps.sinaMaster.worker import worker as wkr
    crawlers = wkr.control.ping(timeout=2.0, queue='crawler')
    crawlers = [ list(c.keys())[0] for c in crawlers ]
    cacheFiles = [ 'SinaMF1m_%s.h5t' %i for i in yymmddToCache]

    for c in crawlers:
        q = c.split('@')[0]
        if not q or len(q) <=0: continue
        r = CTDayend.fetchArchivedFiles.apply_async(args=[cacheFiles], queue=q)
        thePROG.info('schDo_pitchArchiedFiles() called crawler[%s].fetchArchivedFiles: %s' % (q, ','.join(cacheFiles)))
'''

# ===================================================
@shared_task(bind=True, max_retries=0, compression='bzip2')
def readArchivedDays(self, symbol, YYYYMMDDs):
    if isinstance(YYYYMMDDs, str):
        YYYYMMDDs = [YYYYMMDDs]

    YYYYMMDDs.sort()

    all_lines=''
    readtxn = ''
    for yymmdd in YYYYMMDDs:
        fnArch = os.path.join(MAPPED_HOME, 'archived', 'sina', 'SinaMDay_%s.h5t' % yymmdd)
        # fnArch = '/mnt/e/AShareSample/arch/SinaMDay_%s.h5t' % yymmdd
        memName = '%s_day%s.tcsv' %(symbol, yymmdd)
        try :
            lines = ''
            lines = h5tar.read_utf8(fnArch, memName)
            if lines and len(lines) >0 :
                all_lines += '\n' + lines
            readtxn += '%s(%dB)@%s, ' % (memName, len(lines), fnArch)
        except:
            thePROG.error('readArchivedDays() failed to read %s from %s' % (memName, fnArch))

    thePROG.info('readArchivedDays() read %s' % readtxn) 
    return all_lines # take celery's compression instead of return bz2.compress(all_lines.encode('utf8'))

# ===================================================
@shared_task(bind=True, base=Retryable)
def readArchivedH5t(self, h5tFileName, memberNode):
    if '.h5t' != h5tFileName[-4:]: h5tFileName+='.h5t'
    pathname = os.path.join(MAPPED_HOME, 'archived', 'sina', h5tFileName)
    pathname = '/tmp/sina_cache/' + h5tFileName

    k = h5tar.quote(memberNode)
    ret = None
    try :
        with h5py.File(pathname, 'r') as h5r:
            if k in h5r.keys():
                ret = h5r[k][()].tobytes()

            if h5tar.GNAME_TEXT_utf8 in h5r.keys():
                g = h5r[h5tar.GNAME_TEXT_utf8]
                if k in g.keys():
                    ret = g[k][()].tobytes()

    except Exception as ex:
        thePROG.logexception(ex, 'readArchivedH5t() %s[%s]'% (h5tFileName, memberNode), ex)

    if ret and len(ret) > 0:
        #typical compress-rate 1/8: ret = bz2.decompress(ret).decode('utf8')
        thePROG.info('readArchivedH5t() read %s[%s] %dB'% (h5tFileName, memberNode, len(ret)))
    else :
        thePROG.error('readArchivedH5t() read %s[%s] failed: %s'% (h5tFileName, memberNode, ret))
    return ret

# ===================================================
@shared_task(bind=True, base=Retryable)
def schDo_ZipWeek(self, asofYYMMDD =None):
    global DIR_ARCHED_HOME

    dtInWeek = None
    try :
        if isinstance(asofYYMMDD, str):
            dtInWeek = datetime.strptime(asofYYMMDD, '%Y-%m-%d')
    except:
        dtInWeek = None

    if not dtInWeek:
        dtInWeek = datetime.now() - timedelta(days=5)

    thePROG.debug('schDo_ZipWeek() start archiving the week of %s under %s' % (dtInWeek.strftime('%Y-%m-%d'), DIR_ARCHED_HOME))
    fn, lst = prod.archiveWeek(DIR_ARCHED_HOME, None, dtInWeek, thePROG)
    thePROG.info('schDo_ZipWeek() %s archived %s symbols'% (fn, len(lst)))

####################################
from time import sleep
if __name__ == '__main__':
    thePROG.setLogLevel('debug')
    # schKickOff_DownloadToday()
    # exit(0)

    # readArchivedDays('SZ300913', ['20201221', '20201222'])
    # readArchivedH5t('SinaMF1m_20201222.h5t', 'SZ300913_MF1m20201222.json')

    listAllSymbols()
    # schKickOff_DownloadToday()
    for i in range(20):
        schChkRes_Crawlers('20201231')
        sleep(10)

    # nTop = 1000
    # lstSHZ = topActives(nTop)
    # with open(os.path.join(MAPPED_HOME, 'hpx_publish', 'top%s_%s' % (nTop, datetime.now().strftime('%Y%m%d'))) + '.csv', 'wb') as f:
    #     __writeCsv(f, lstSHZ)
    # print(lstSHZ)

    '''
    symbol, asofYYMMDD = 'SZ002670', '20201204'
    
    login = 'root@tc2.syscheme.com'
    fnJsons = []
    for evt in ['KL1d', 'MF1d', 'KL5m', 'MF1m']:
        fnJsons.append('%s_%s%s.json' % (symbol, evt, asofYYMMDD))
    
    today = {
        'symbol': symbol,
        'login': 'hxp01@test',
        'asofYYMMDD': asofYYMMDD,
        'fnSnapshot': '%s_sns%s.h5' % (symbol, asofYYMMDD), 
        'fnJsons': fnJsons,
        'fnTcsv': '%s_day%s.tcsv' % (symbol, asofYYMMDD),
        'lastDays': [
            ['20201204', 15.31, 17.5, 15.0, 15.5, 222133283.0], 
            ['20201203', 15.98, 16.48, 15.5, 15.97, 176615259.0], 
            ['20201202', 14.38, 14.98, 14.26, 14.98, 113319552.0], 
            ['20201201', 12.41, 13.62, 11.77, 13.62, 163043226.0], 
            ['20201130', 12.17, 12.72, 12.02, 12.38, 166906351.0]
            ]
    }

    commitToday(today)
    '''
    

''' A test
import dapps.sinaCrawler.tasks_Dayend as ct
import dapps.sinaMaster.tasks_Archive as mt
c1 = ct.downloadToday.s('SZ000002') | mt.commitToday.s()
c1().get()
'''

