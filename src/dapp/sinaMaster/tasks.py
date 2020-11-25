# encoding: UTF-8

from __future__ import absolute_import, unicode_literals
from celery import shared_task
from .celery import theProg

import crawler.crawlSina as sina

from time import sleep

@shared_task
def add(x, y):
    return x + y

@shared_task
def mul(x, y):
    return x * y

@shared_task
def xsum(numbers):
    return sum(numbers)

__totalAmt1W=0
import math
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

@shared_task
def listAllSymbols():
    result ={}
    EOL ="\n"

    md = sina.SinaCrawler(theProg, None)

    httperr =100
    while 2 != int(httperr /100):
        httperr, lstSH = md.GET_AllSymbols('SH')
        theProg.warn('SH-resp(%d) len=%d' %(httperr, len(lstSH)))
        if 456 == httperr : sleep(30)

    httperr =100
    while 2 != int(httperr/100):
        httperr, lstSZ = md.GET_AllSymbols('SZ')
        theProg.warn('SZ-resp(%d) len=%d' %(httperr, len(lstSZ)))
        if 456 == httperr : sleep(30)

    for i in lstSH + lstSZ:
        result[i['symbol']] =i
    result = list(result.values())

    global __totalAmt1W
    __totalAmt1W=0
    HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,close,volume"
    csvAll = HEADERSEQ + EOL
    for i in result:
        __totalAmt1W += i['amount'] /10000.0
        csvAll += ','.join([str(i[k]) for k in HEADERSEQ.split(',')]) +EOL

    # filter the top active 1000
    topXXX = list(filter(lambda x: not 'ST' in x['name'], result))
    topNum = min(500, int(len(topXXX)/50) *10)

    topXXX.sort(key=activityOf)
    topXXX= topXXX[-topNum:]
    topXXX.reverse()
    csvTop = HEADERSEQ + EOL
    for i in topXXX:
        csvTop += ','.join([str(i[k]) for k in HEADERSEQ.split(',')]) +EOL
    
    return csvTop, csvAll

