# encoding: UTF-8

from __future__ import absolute_import, unicode_literals
from celery import shared_task

import sys, os, re
if __name__ == '__main__':
    sys.path.append(".")
    from worker import thePROG, getLogin
else:
    from .worker import thePROG, getLogin

from dapps.CeleryDefs import RetryableError, Retryable
import crawler.crawlSina as sina
import crawler.producesSina as prod

from time import sleep

HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,close,volume"
EOL = "\r\n"

__accLogin, __accHome, __workersRoot= None, '/mnt/s', '/mnt/data/hpwkspace/workers'

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
    ret = 10* math.sqrt(item['amount'] / __totalAmt1W)
    ret += item['turnoverratio']  
    '''
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

@shared_task(bind=True, base=Retryable)
def listAllSymbols(self):
    noneST, STs = __listAllSymbols()
    csvNoneST = HEADERSEQ + EOL
    for i in noneST:
        csvNoneST += ','.join([str(i[k]) for k in HEADERSEQ.split(',')]) +EOL

    csvSTs = HEADERSEQ + EOL
    for i in STs:
        csvSTs += ','.join([str(i[k]) for k in HEADERSEQ.split(',')]) +EOL

    return csvNoneST, csvSTs

@shared_task(bind=True, base=Retryable)
def commitToday(self, login, symbol, asofYYMMDD, fnJsons, fnSnapshot) :
    '''
    fnJsons = ['SZ000002_KL1d20201127.json', 'SZ000002_MF1d20201127.json', 'SZ000002_KL5m20201127.json', 'SZ000002_MF1m20201127.json']
    fnSnapshot = 'SZ000002_sns.h5';
    '''

    global __accLogin, __accHome, __workersRoot
    if '@' in login : login = login[:login.index('@')]
    if ':' in login : login = login[:login.index(':')]

    pubHome = os.path.join(__workersRoot, login, to_publish)

    # step 1. zip the JSON files
    for fn in fnJsons:
        srcpath = os.path.join(pubHome, fn)
        fn = os.path.basename(fn)
        evt = 
        destpath = os.path.join(__accHome, 'archived', 'sina', 'Sina%s_%s.h5t' % (evt, asofYYMMDD) )
        h5tar.tar_utf8(destpath, srcpath)

    # step 2. zip the snapshots
    srcpath = os.path.join(pubHome, fnSnapshot)
    destpath = os.path.join(__accHome, 'archived', 'sina', 'SNS_%s.h5' % (symbol) )
    with h5.open(srcpath, 'r') as h5f:
        pass


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

    for i in lstSH + lstSZ:
        result[i['symbol']] =i
    result = list(result.values())

    totalAmt1W=0
    for i in result:
        totalAmt1W += i['amount'] /10000.0

    if totalAmt1W >1.0: # for activityOf() 
        global __totalAmt1W
        __totalAmt1W = totalAmt1W

    noneST = list(filter(lambda x: not 'ST' in x['name'], result))
    noneST.sort(key=activityOf)
    noneST.reverse()

    STs = list(filter(lambda x: 'ST' in x['name'], result))
    STs.sort(key=activityOf)
    STs.reverse()
    
    return noneST, STs

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
 

####################################
if __name__ == '__main__':
    # csvNoneST, csvSTs = listAllSymbols()
    # print(csvNoneST)
    # print(csvSTs)
    login = 'hpx01@tc2.syscheme.com'
    asofYYMMDD = '20201127'
    fnJsons = ['SZ000002_KL1d20201127.json', 'SZ000002_MF1d20201127.json', 'SZ000002_KL5m20201127.json', 'SZ000002_MF1m20201127.json']
    fnSnapshot = 'SZ000002_sns.h5';

    commitToday(login, symbol, asofYYMMDD, fnJsons, fnSnapshot)

