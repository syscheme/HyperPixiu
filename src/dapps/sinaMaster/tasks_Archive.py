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

SYMBOL_LIST_HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,open,high,low,close,volume"
EOL = "\r\n"
SINA_USERS_ROOT = '/mnt/data/hpwkspace/users'
MAPPED_USER, MAPPED_HOME = getMappedAs()

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

def __rmfile(fn) :
    try :
        os.remove(fn)
    except:
        pass

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
    lastDays = dictArgs.get('lastDays', [])
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

    try:
        os.mkdir(os.path.join(archDir, 'snapshots'))
    except: pass
    
    thePROG.debug('commitToday() archiving %s_%s dictArgs: %s from %s to %s' % (symbol, asofYYMMDD, str(dictArgs), pubDir, archDir))

    # step 1. zip the JSON files
    for fn in fnJsons:
        srcpath = os.path.join(pubDir, fn)
        m = re.match(r'%s_([A-Za-z0-9]*)%s.json' %(symbol, asofYYMMDD), os.path.basename(srcpath))
        if not m : continue
        evtShort = m.group(1)

        destpath = os.path.join(archDir, 'Sina%s_%s.h5t' % (evtShort, asofYYMMDD) )
        if h5tar.tar_utf8(destpath, srcpath) :
            thePROG.info('commitToday() archived %s into %s' %(srcpath, destpath))
            __rmfile(srcpath)
        else:
            thePROG.error('commitToday() failed to archived %s into %s' %(srcpath, destpath))

    # step 1. zip the Tcsv file
    srcpath = os.path.join(pubDir, fnTcsv)
    destpath = os.path.join(archDir, 'SinaMDay_%s.h5t' % asofYYMMDD )
    if h5tar.tar_utf8(destpath, srcpath, baseNameAsKey=True) :
        thePROG.info('commitToday() archived %s into %s' %(srcpath, destpath))
        __rmfile(srcpath)
    else:
        thePROG.error('commitToday() failed to archived %s into %s' %(srcpath, destpath))

    # step 3. append the snapshots
    srcpath = os.path.join(pubDir, fnSnapshot)
    destpath = os.path.join(archDir, 'snapshots', 'SNS_%s.h5' % (symbol) )
    gns = []

    lastDates = [ x[0] for x in lastDays] if lastDays and len(lastDays)>0 else []
    try :
        with h5py.File(destpath, 'a') as h5w:
            # step 1, copy the new SNS into the dest h5f
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

            # step 2, determine the grainRate_X based on lastDays
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
                    
        thePROG.info('commitToday() added snapshot[%s] of %s into %s' % (','.join(gns), srcpath, destpath))
        __rmfile(srcpath)
    except Exception as ex:
        thePROG.logexception(ex, 'commitToday() snapshot[%s->%s] error' % (srcpath, destpath))

@shared_task(bind=True, base=Retryable)
def topActives(self, topNum = 500):
    lstTops = listAllSymbols() [:topNum]
    return lstTops

__asyncResult_downloadToday = {}

@shared_task(bind=True, base=Retryable)
def schOn_Every5min(self):
    global __asyncResult_downloadToday
    todels = []
    cWorking =0
    for k, v in __asyncResult_downloadToday.items():
        if not v: 
            todels.append(k)
            continue

        if v.ready():
            todels.append(k)
            thePROG.info('schOn_Every5min() downloadToday[%s]%s done: succ[%s] and will clear' %(k, v.task_id, v.successful()))
            continue

        cWorking += 1
        thePROG.debug('schOn_Every5min() downloadToday[%s]%s still working' %(k, v.task_id))

    thePROG.info('schOn_Every5min() downloadToday has %d-working and %d-done tasks' %(cWorking, len(todels)))
    if len(todels) >0:
        thePROG.info('schOn_Every5min() clearing keys: %s' % ','.join(todels))
        for k in todels:
            del __asyncResult_downloadToday[k]
        if len(__asyncResult_downloadToday) <=0:
            thePROG.info('schOn_Every5min() downloadToday all done')

@shared_task(bind=True, base=Retryable)
def schOn_TradeDayClose(self):
    global __asyncResult_downloadToday
    __asyncResult_downloadToday = {}
    for s in IDXs_to_COLLECT + ETFs_to_COLLECT:
        if s in __asyncResult_downloadToday.keys():
            continue

        thePROG.debug('schOn_TradeDayClose() adding subtask to download ETF[%s]' % s)
        wflow = CTDayend.downloadToday.s(s, excludeMoneyFlow=True) | commitToday.s()
        __asyncResult_downloadToday[s] = wflow()

    lstSHZ = listAllSymbols()
    thePROG.info('schOn_TradeDayClose() listAllSymbols got %d symbols' %len(lstSHZ))
    if len(lstSHZ) <=2000:
        raise RetryableError(401, 'incompleted symbol list')

    # del lstSHZ[5:] # should be the complete lstSHZ
    for i in lstSHZ : # the full lstSHZ
        symbol = i['symbol']
        if symbol in __asyncResult_downloadToday.keys():
            continue

        thePROG.debug('schOn_TradeDayClose() adding subtask to download %s %s' % (symbol, i['name']))
        wflow = CTDayend.downloadToday.s(symbol) | commitToday.s()
        __asyncResult_downloadToday[symbol] = wflow()


####################################
if __name__ == '__main__':
    thePROG.setLogLevel('debug')

    schOn_TradeDayClose()
    for i in range(20):
        schOn_Every5min()
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
import dapps.sinaMaster.tasks as mt
c1 = ct.downloadToday.s('SZ000002') | mt.commitToday.s()
c1().get()
'''