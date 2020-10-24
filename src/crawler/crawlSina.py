# encoding: UTF-8

from __future__ import division

from MarketCrawler import *
from EventData import Event, datetime2float, DT_EPOCH
from MarketData import KLineData, TickData, MoneyflowData, MARKETDATE_EVENT_PREFIX, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY
from Account import Account_AShare, DAYCLOSE_TIME_ERR
import crawler.disguise as dsg

import requests # pip3 install requests
from copy import copy
from datetime import datetime, timedelta
import demjson # pip3 install demjson

import os, sys, fnmatch, tarfile, re
import threading # for __step_pollTicks
from time import sleep

EXECLUDE_LIST = ["SH600005"]


'''
分类-中国银行: http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_gupiao_fenlei?daima=SH601988
[{cate_type:"2",cate_name:"银行业",category:"hangye_ZI01"}]

http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=40&sort=mkcap&asc=0&node=sw2_420100

sw2_[0-9]{4}00 - 申万二级行业
http://www.daquant.com/help/2-html/html4/2.1.3.html

https://baike.baidu.com/item/%E9%BE%99%E5%A4%B4%E8%82%A1/2268306

'''

CLOCK_ERROR_SEC   = 2*60.0  # 2min
OFFHOUR_ERROR_SEC = 45*60.0 # 45min is necessary to warm up every morning, X DAYCLOSE_TIME_ERR.seconds
TICK_INTERVAL_DEFAULT_SEC = 0.7 # 0.7sec

def toFloatVal(val, defaultval=0.0) :
    try :
        return float(val) if val else defaultval
    except:
        pass
    return defaultval

########################################################################
class SinaCrawler(MarketCrawler):
    '''MarketData BackEnd API
    cron: 2 9 * * 1,2,3,4,5 mkdir -p /tmp/data; cd /root/wkspace/HyperPixiu ; ./run.sh src/launch/test_Crawler.py &
    '''
    # 常量定义
    #----------------------------------------------------------------------
    DEFAULT_GET_HEADERS = { # in order to protent like a real browser
        "Content-type": "application/x-www-form-urlencoded",
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Cache-Control': 'max-age=2',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        # 'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }

    TIMEOUT = 5
    TICK_BATCH_SIZE = 200
    NEXTSTAMP_KLINE_5MIN = 'nstamp.'+EVENT_KLINE_5MIN
    NEXTSTAMP_KLINE_1DAY = 'nstamp.'+EVENT_KLINE_1DAY
    IDX_KLINE_5MIN = 'idx.'+EVENT_KLINE_5MIN
    IDX_KLINE_1DAY = 'idx.'+EVENT_KLINE_1DAY
    
    INTV_MINs_OF_EVENTS = {
            EVENT_KLINE_5MIN: 5,
            EVENT_KLINE_1DAY: 240,
            EVENT_MONEYFLOW_1MIN : 10,  # avoid 456 if query too frequently although the events are 1min-based
            EVENT_MONEYFLOW_1DAY : 240,
        }

    def __init__(self, program, marketState=None, recorder=None, objectives=[], **kwargs):
        """Constructor"""
        super(SinaCrawler, self).__init__(program, marketState, recorder, **kwargs)
        
        # SinaCrawler take multiple HTTP requests to collect data, each of them may take different duration to complete.
        # Among them the Ticks are expected more timely, so run the collection of Ticks in a seperate thread
        self.__trdTick = threading.Thread(target=self.__loopTick)
        self.__trdQuit = False

        self._eventsToPost = [EVENT_TICK, EVENT_KLINE_1MIN, EVENT_MONEYFLOW_1MIN]
        self._steps = [self.__step_poll1st, self.__step_pollKline, self.__step_pollMoneyflow] # __step_pollTicks() will be executed in a separated thread __trdTick
        self.__urlProxy = None

        symbols              = self.getConfig('securities', objectives) # just to be different with config "objectives" of Trader/TraderAdvisors
        if len(symbols) >0 and not isinstance(symbols[0], str):
            symbols = [s('') for s in symbols]

        self._secYield456    = self.getConfig('yield456',    230)
        self.__excludeAt404  = self.getConfig('excludeAt404',True)

        # COVERED inside of __step_pollMoneyflow()
        # self._excludeMoneyFlow = self.getConfig('excludeMoneyFlow', False)
        # if not self._excludeMoneyFlow:
        #     self._steps.append(self.__step_pollMoneyflow)

        self.__idxTickBatch, self.__idxKL, self.__idxMF = 0,0,0
        self.__tickBatches = None

        self.__tickToKL1m = {} # dict of symbol to SinaTickToKL1m
        self.__mfMerger   = {} # dict of symbol to SinaMF1mToXm
        self.__stampNextPolls = {} # dict of symbol_event to float stamp

        if len(symbols) >0:
            self.subscribe(symbols)

        self.__step_poll1st() # perform an init-step

    def __stampOfNext(self, symbol, eventType):
        key = '%s_%s' % (symbol, eventType)
        if key in self.__stampNextPolls.keys():
            return self.__stampNextPolls[key]
        return 0

    def __scheduleNext(self, symbol, eventType, seconds):
        key = '%s_%s' % (symbol, eventType)
        ftime = datetime2float(datetime.now()) + seconds
        self.__stampNextPolls[key]= ftime
        return ftime

    def __loopTick(self):
        '''a separated thread to capture EVENT_TICK more timely'''
        nextSleep = 0.2 # 200msec
        while not self.__trdQuit:

            if nextSleep >0.01 :
                sleep(nextSleep)

            if self.__trdQuit: break

            try :
                nextSleep = 0.2 # 200msec
                if self.__step_pollTicks() >0:
                    nextSleep =0
            except Exception as ex:
                self.error('__step_pollTicks() exception: %s' % ex)

    def connect(self):
        if self.__trdTick:
            self.__trdTick.start()
        return True

    def stop(self):
        self.__trdQuit = True
        if self.__trdTick:
            self.__trdTick.join()
            self.__trdTick =None

        super(SinaCrawler, self).stop()

    #------------------------------------------------
    # sub-steps
    def __step_poll1st(self):
        self.__BEGIN_OF_TODAY = datetime2float(Account_AShare.tradeBeginOfDay()) -CLOCK_ERROR_SEC
        self.__END_OF_TODAY = datetime2float(Account_AShare.tradeEndOfDay()) + CLOCK_ERROR_SEC
        return 0

    def __step_pollTicks(self):
        '''
        @return cMerged how many new tickets merged as the hits to indicate if the poll is busy
        '''
        if self._stepAsOf < (self.__BEGIN_OF_TODAY - OFFHOUR_ERROR_SEC/2) or self._stepAsOf > (self.__END_OF_TODAY + OFFHOUR_ERROR_SEC/2):
            return 0 # well off-trade hours

        #step 1. build up self.__tickBatches if necessary
        if (not self.__tickBatches or len(self.__tickBatches)<=0) and len(self._symbolsToPoll) >0:
            self.__tickBatches = []
            self.__idxTickBatch = 0
            i =0
            while True:
                bth = self._symbolsToPoll[i*SinaCrawler.TICK_BATCH_SIZE: (i+1)*SinaCrawler.TICK_BATCH_SIZE]
                if len(bth) <=0: break
                self.__tickBatches.append(bth)
                i +=1
            self.debug("step_pollTicks() %d symbols are divided into %d batches" %(len(self._symbolsToPoll), len(self.__tickBatches)))

        batches = len(self.__tickBatches)
        if batches <=0:
            return 0

        if self._stepAsOf > self.__END_OF_TODAY :
            for k, v in self.__tickToKL1m.items():
                v.flushAtClose()
            
        #step 2. check if to yield time
        self.__idxTickBatch = self.__idxTickBatch % batches
        if 0 == self.__idxTickBatch : # this is a new round
            # yield some time in order not to poll SiNA too frequently
            nextStamp = self.__stampOfNext('all', 'tick')
            if self._stepAsOf < nextStamp  or nextStamp > self.__END_OF_TODAY :
                return 0

            self.__scheduleNext('all', 'tick', TICK_INTERVAL_DEFAULT_SEC)

        idxBtch = self.__idxTickBatch
        self.__idxTickBatch +=1
        
        stampStart = datetime2float(datetime.now())

        updated=[]
        bth = self.__tickBatches[idxBtch]
        httperr, result = self.GET_RecentTicks(bth)
        if 200 != httperr:
            self.error("step_pollTicks() GET_RecentTicks failed, err(%s) bth:%s" %(httperr, bth))
            return 0
            
        stampResp = datetime2float(datetime.now())
        # succ at previous batch here

        if len(result) <=0 : # market idle??
            self.__scheduleNext('all', 'tick', 1.0 + max(1, 10 - len(bth)/10)) # likely after a trade-day closed, yield 10sec

        cMerged =0
        for tk in result:
            s = tk.symbol

            if not s in self.__tickToKL1m.keys():
                self.__tickToKL1m[s] = SinaTickToKL1m(self.__onKL1mMerged)
                self.__mfMerger[s]   = SinaMF1mToXm(self.__onMF5mMerged)
            
            tkstate=self.__tickToKL1m[s]
            if tk.datetime <= tkstate.lastTickAsOf :
                continue

            tkstate.pushTick(tk)

            ev = Event(EVENT_TICK)
            ev.setData(tk)

            # if not s in self.marketState.keys():
            #     self.marketState[s] = Perspective('AShare', symbol =s) # , KLDepth_1min=self._depth_1min, KLDepth_5min=self._depth_5min, KLDepth_1day=self._depth_1day, tickDepth=self._depth_ticks)
            # psp = self.marketState[s]
            ev = self.marketState.updateByEvent(ev) # so to filter duplicated event out if updateByEvent() returns None
            self.debug("step_pollTicks() pushed tick %s into psp, now: %s" %(tk.desc, self.marketState.descOf(s)))

            if not ev: continue
            cMerged +=1
            updated.append(s)

            if self._recorder:
                self.OnEventCaptured(ev)
            
        stampNow = datetime2float(datetime.now())
        if cMerged >0:
            self.info("step_pollTicks() btch[%d/%d] cached %d new-tick of %d/%d symbols, took %.3f/%.3fs: %s" %(idxBtch +1, batches, cMerged, len(result), len(self.__tickBatches[idxBtch]), (stampResp-stampStart), (stampNow-stampStart), ','.join(updated)))
            if cMerged >= len(self.__tickBatches[idxBtch]):
                self.__scheduleNext('all', 'tick', 0.01) # to speed up the next poll
        else :
            if not Account_AShare.duringTradeHours():
                self.__scheduleNext('all', 'tick', (61 - int(stampNow) %60))
                self.info("step_pollTicks() btch[%d/%d] no new ticks during off-market time, took %.3f/%.3fs, extended sleep time" %(idxBtch +1, batches, (stampResp-stampStart), (stampNow-stampStart)))
            else :
                self.debug("step_pollTicks() btch[%d/%d] no new ticks updated, took %.3f/%.3fs" %(idxBtch +1, batches, (stampResp-stampStart), (stampNow-stampStart)))

        return cMerged

    def __step_pollKline(self):
        cBusy =0       
        if self._stepAsOf < self.__stampOfNext('all', 'KL'):
            return cBusy

        s = None
        cSyms = len(self._symbolsToPoll)
        if cSyms >0:
            self.__idxKL = self.__idxKL % cSyms
            s = self._symbolsToPoll[self.__idxKL]

        self.__idxKL += 1

        if not s or len(s) <=0:
            del self._symbolsToPoll[s]
            return 1 # return as busy for this error case

        if self._stepAsOf < (self.__BEGIN_OF_TODAY - OFFHOUR_ERROR_SEC) or self._stepAsOf > (self.__END_OF_TODAY + OFFHOUR_ERROR_SEC*2):
            return cBusy # well off-trade hours

        urlProxy = self.__urlProxy if self._stepAsOf < self.__stampOfNext('all', 'KL_yield456') else None

        # if not s in self.marketState.keys():
        #     self.marketState[s] = Perspective('AShare', symbol=s) # , KLDepth_1min=self._depth_1min, KLDepth_5min=self._depth_5min, KLDepth_1day=self._depth_1day, tickDepth=self._depth_ticks)
        # psp = self.marketState[s]
        minIntvKL = 30*60 # initialize with a large 30min

        stampNow = datetime2float(datetime.now())
        for evType in [EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] :

            if self._stepAsOf < self.__stampOfNext(s, evType):
                continue

            stampStart = stampNow
            minutes = SinaCrawler.INTV_MINs_OF_EVENTS[evType]

            if (minutes < (OFFHOUR_ERROR_SEC/60) and self._stepAsOf > (self.__END_OF_TODAY + OFFHOUR_ERROR_SEC)):
                continue # skip polling in-day KLs shorter after hours

            self.__scheduleNext(s, evType, min(30, minutes)*60*0.7)

            # if minutes < 240:
            #     # in-day events
            #     etimatedNext = datetime2float(self.marketState.getAsOf(s, evType)) + 60*minutes -1
            # else :
            #     etimatedNext = datetime2float(self.marketState.getAsOf(s, evType)) + 60*(int((minutes) /240)*60*24)  -1
            #     sz, esz = self.marketState.sizesOf(s, evType)
            #     tmpStamp = datetime2float(self.marketState.stampUpdatedOf(s, evType))
            #     if sz >=esz and tmpStamp and tmpStamp >= self.__BEGIN_OF_TODAY:
            #         etimatedNext = tmpStamp + 60*60 # one hr later

            # if etimatedNext > self.__END_OF_TODAY or stampStart < etimatedNext:
            #     continue

            size, lines = self.marketState.sizesOf(s, evType)
            if size >0:
                lines -= size
            lines = (lines+10) if lines >0 else 10

            httperr, result = self.GET_RecentKLines(s, minutes, lines, urlProxy)
            if 200 != httperr:
                self.error("step_pollKline(%s:%s) failed, err(%s)" %(s, evType, httperr))
                if 456 == httperr:
                    self.__urlProxy = None # self.__urlProxy = dsg.nextProxy()
                    self.__scheduleNext('all', 'KL_yield456', self._secYield456) # self.__scheduleNext('all', 'KL', self._secYield456)
                    self.warn("step_pollKline(%s:%s) [%d/%d]sym SINA complained err(%s), yielding %ssec, nextProxy[%s]" %(s, evType, self.__idxKL, cSyms, httperr, self._secYield456, self.__urlProxy))
                    return cBusy
           
                if 404 == httperr and self.__excludeAt404:
                    del self._symbolsToPoll[s]
                    self.warn("step_pollKline(%s:%s) [%d/%d]sym excluded symbol per err(%s)" %(s, evType, self.__idxKL, cSyms, httperr))

                continue

            # succ at query
            self.__scheduleNext(s, evType, min(60, minutes)*60*0.7)

            cMerged =0
            for i in result:
                cBusy +=1
                ev = Event(evType)
                ev.setData(i)
                ev = self.marketState.updateByEvent(ev) # so to filter duplicated event out if updateByEvent() returns None
                if not ev: continue
                cMerged +=1

                if self._recorder:
                    self.OnEventCaptured(ev)

            stampNow = datetime2float(datetime.now())
            if not urlProxy: dsg.stampGoodProxy(stampNow-stampStart) # taking the duration as the priority of good proxy

            if cMerged >0:
                self.info("step_pollKline(%s:%s) [%d/%d]sym merged %d/%d KLs into stack, took %.3fs, psp: %s" % (s, evType, self.__idxKL, cSyms, cMerged, len(result), (stampNow-stampStart), self.marketState.descOf(s)))
            elif not Account_AShare.duringTradeHours():
                self.__scheduleNext('all', 'KL', 60*60)
                self.info("step_pollKline(%s:%s) [%d/%d]sym no new KLs during off-hours" %(s, evType, self.__idxKL, cSyms))

        return cBusy

    def __onKL1mMerged(self, kl1m):
        ev = Event(EVENT_KLINE_1MIN)
        ev.setData(kl1m)
        ev = self.marketState.updateByEvent(ev)
        if ev: 
            self.OnEventCaptured(ev)
            self.debug("onKL1mMerged() merged from ticks: %s ->psp: %s" % (kl1m.desc, self.marketState.descOf(kl1m.symbol)))

    def __onMF5mMerged(self, mf5m):
        ev = Event(EVENT_MONEYFLOW_5MIN)
        ev.setData(mf5m)
        ev = self.marketState.updateByEvent(ev)
        if ev: 
            self.OnEventCaptured(ev)
            self.debug("onMF5mMerged() merged: %s ->psp: %s" % (mf5m.desc, self.marketState.descOf(mf5m.symbol)))

    def __step_pollMoneyflow(self):
        cBusy =0       
        if self._stepAsOf < self.__stampOfNext('all', 'MF'):
            return cBusy

        s = None
        cSyms = len(self._symbolsToPoll)
        if cSyms >0:
            self.__idxMF = self.__idxMF % cSyms
            s = self._symbolsToPoll[self.__idxMF]

        self.__idxMF += 1
        
        # SAD: sina doesn't support moneyflow on ETFs, so skip it to avoid 456s
        if 'SH51' in s or 'SZ15' in s:
            return cBusy

        # SAD: exclude the index
        if 'SH000' in s or 'SZ399' in s:
            return cBusy

        if self._stepAsOf < (self.__BEGIN_OF_TODAY - OFFHOUR_ERROR_SEC) or self._stepAsOf > (self.__END_OF_TODAY + OFFHOUR_ERROR_SEC*2):
            return cBusy # well off-trade hours

        urlProxy = self.__urlProxy if self._stepAsOf < self.__stampOfNext('all', 'MF_yield456') else None

        stampNow = datetime2float(datetime.now())
        for evType in [EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY] :

            if self._stepAsOf < self.__stampOfNext(s, evType):
                continue

            stampStart = stampNow
            minutes = SinaCrawler.INTV_MINs_OF_EVENTS[evType]
            if (minutes < (OFFHOUR_ERROR_SEC/60) and self._stepAsOf > (self.__END_OF_TODAY + OFFHOUR_ERROR_SEC)):
                continue # skip polling in-day KLs shorter after hours

            self.__scheduleNext(s, evType, min(30, minutes)*60*0.7)

            size, lines = self.marketState.sizesOf(s, evType)
            if size >0:
                lines -= size
            lines = (lines+10) if lines >0 else 10

            httperr, result = self.GET_MoneyFlow(s, lines, EVENT_MONEYFLOW_1MIN== evType)
            if 200 != httperr:
                self.error("step_pollMoneyflow(%s:%s) failed, err(%s)" %(s, evType, httperr))
                if 456 == httperr:
                    self.__urlProxy = dsg.nextProxy()
                    self.__scheduleNext('all', 'MF_yield456', self._secYield456) # self.__scheduleNext('all', 'MF', self._secYield456)
                    self.warn("step_pollMoneyflow(%s:%s) [%d/%d]sym SINA complained err(%s), yielding %ssec, nextProxy[%s]" %(s, evType, self.__idxKL, cSyms, httperr, self._secYield456, self.__urlProxy))
                    return cBusy
           
                if 404 == httperr and self.__excludeAt404:
                    self.warn("step_pollMoneyflow(%s:%s) [%d/%d]sym excluded symbol per err(%s)" %(s, evType, self.__idxKL, cSyms, httperr))

                continue

            # succ at query
            self.__scheduleNext(s, evType, min(60, minutes)*60*0.7)

            cMerged =0
            for i in result:
                cBusy +=1
                ev = Event(evType)
                ev.setData(i)
                ev = self.marketState.updateByEvent(ev) # so to filter duplicated event out if updateByEvent() returns None
                if not ev : continue

                cMerged +=1
                if self._recorder:
                    self.OnEventCaptured(ev)

                if EVENT_MONEYFLOW_1MIN == evType:
                    if not s in self.__mfMerger.keys():
                        self.__mfMerger[s] = SinaMF1mToXm(self.__onMF5mMerged, 5)
            
                    self.__mfMerger[s].pushMF1m(i)

            stampNow = datetime2float(datetime.now())
            if not urlProxy: dsg.stampGoodProxy(stampNow-stampStart) # taking the duration as the priority of good proxy

            if cMerged >0:
                self.info("step_pollMoneyflow(%s:%s) [%d/%d]sym merged %d/%d MFs into stack, took %.3fs, psp: %s" % (s, evType, self.__idxKL, cSyms, cMerged, len(result), (stampNow-stampStart), self.marketState.descOf(s)))
            elif not Account_AShare.duringTradeHours():
                self.__scheduleNext('all', 'MF', 60*60)
                self.info("step_pollMoneyflow(%s:%s) [%d/%d]sym no new MFs during off-hours" %(s, evType, self.__idxKL, cSyms))

        return cBusy

    # end of sub-steps
    #------------------------------------------------

    def subscribe(self, symbols):
        if isinstance(symbols, str):
            symbols = symbols.split(',')

        ret = super(SinaCrawler, self).subscribe([SinaCrawler.fixupSymbolPrefix(s) for s in symbols])

        # reset the intermedia vars
        if ret >0:
            self.__tickBatches = None

        return ret

    def unsubscribe(self, symbols):
        ret = super(SinaCrawler, self).unsubscribe([SinaCrawler.fixupSymbolPrefix(s) for s in symbols])

        # reset the intermedia vars
        self.__tickBatches = None
        return ret

    #------------------------------------------------
    # private methods
    def __sinaGET(self, url, apiName, urlProxy=None):
        errmsg = '%s GET ' % (apiName)
        httperr = 400
        strThru =''
        try:
            self.debug("%s GET %s" %(apiName, url))
            headers = copy(self.DEFAULT_GET_HEADERS)
            headers['User-Agent'] = dsg.nextUserAgent()
            proxies, connectTimeout = {}, self.TIMEOUT
            if urlProxy and len(urlProxy) >3:
                proxies['http']  = urlProxy
                proxies['https'] = urlProxy
                strThru = ' thru[%s]' % urlProxy
                connectTimeout = max(3, self.TIMEOUT/2)

            response = requests.get(url, headers=headers, proxies=proxies, timeout=connectTimeout)
            httperr = response.status_code
            if httperr == 200:
                return httperr, response.text

            errmsg += 'err(%s)%s' % (httperr, response.reason)
        except Exception as e:
            errmsg += 'exception: %s' % e
        
        self.info('%s <-%s%s'% (errmsg, url, strThru)) # lower this loglevel
        return httperr, errmsg

    def convertToKLineDatas(symbol, text) :
        klineseq =[]
        jsonData = demjson.decode(text)
        if not jsonData:
            return klineseq

        ''' sample response
        [{day:"2019-09-23 14:15:00",open:"15.280",high:"15.290",low:"15.260",close:"15.270",volume:"892600",ma_price5:15.274,ma_volume5:1645033,ma_price10:15.272,ma_volume10:1524623,ma_price30:15.296,ma_volume30:2081080},
         {day:"2019-09-23 14:20:00",open:"15.270",high:"15.280",low:"15.240",close:"15.240",volume:"1591705",ma_price5:15.266,ma_volume5:1676498,ma_price10:15.27,ma_volume10:1593887,ma_price30:15.292,ma_volume30:1955370},
         ...]
        '''
        for kl in jsonData :
            kldata = KLineData("AShare", symbol)
            kldata.open   = toFloatVal(kl['open'])             # OHLC
            kldata.high   = toFloatVal(kl['high'])             # OHLC
            kldata.low    = toFloatVal(kl['low'])             # OHLC
            kldata.close  = toFloatVal(kl['close'])             # OHLC
            kldata.volume = toFloatVal(kl['volume'])
            kldata.close  = toFloatVal(kl['close'])
            try :
                kldata.datetime = datetime.strptime(kl['day'][:10], '%Y-%m-%d').replace(hour=15, minute=0, second=0, microsecond=0)
                kldata.datetime = datetime.strptime(kl['day'], '%Y-%m-%d %H:%M:%S')
            except :
                pass
            kldata.date = kldata.datetime.strftime('%Y-%m-%d')
            kldata.time = kldata.datetime.strftime('%H:%M:%S')
            klineseq.append(kldata)

        klineseq.sort(key=SinaCrawler.sortKeyOfMD)
        # klineseq.reverse() # SINA returns from oldest to newest
        return klineseq

    def GET_RecentKLines(self, symbol, minutes=1200, lines=10, urlProxy=None, saveAs=None): # deltaDays=2)
        '''"查询 KLINE
        will call cortResp.send(csvline) when the result comes
        '''

        if minutes <5: minutes=5 # sina only allow minimal 5min-KLines
        # if deltaDays <1: deltaDays=1
        # symbol = SinaCrawler.fixupSymbolPrefix(symbol)
        # now_datetime = datetime.now()
        # lines = deltaDays * 4 * 60 / minutes

        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=%s&scale=%s&datalen=%d" % (symbol, minutes, lines)
        httperr, text = self.__sinaGET(url, 'GET_RecentKLines()', urlProxy)
        if 200 != httperr:
            return httperr, text

        # maximal return: scale=5->around 1mon, scale=15->around 1.5mon, scale=30->2mon, scale=60->3mon, scale=240->a bit longer than 1yr
        # js = response.json()
        # result = demjson.decode(response.text)
        # result.decode('utf-8')
        klineseq =[]
        jsonData = demjson.decode(text)
        if not jsonData:
            return httperr, klineseq

        klineseq = self.__class__.convertToKLineDatas(symbol, text)

        try:
            if saveAs and len(klineseq) >0:
                with open(saveAs, 'w') as f:
                    f.write(text)
        except: pass

        return httperr, klineseq

    def sortKeyOfMD(md) : # for sort
        return md.datetime

    def fixupSymbolPrefix(symbol):
        if symbol.isdigit() :
            if symbol.startswith('0') :
                return "SZ%s" % symbol
            elif symbol.startswith('3') :
                return "SZ%s" % symbol
            elif symbol.startswith('6') :
                return "SH%s" % symbol
        return symbol.upper()

    #------------------------------------------------    
    def convertToTickDatas(text) :
        tickseq = []

        HEADERSEQ="name,open,prevClose,price,high,low,bid,ask,volume,total,bid1v,bid1,bid2v,bid2,bid3v,bid3,bid4v,bid4,bid5v,bid5,ask1v,ask1,ask2v,ask2,ask3v,ask3,ask4v,ask4,ask5v,ask5,date,time"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^var hq_str_([^=]*)="([^"]*).*')

        for line in text.split('\n') :
            m = SYNTAX.match(line)
            if not m :
                # LOG ERR
                continue

            s = m.group(1).upper()
            row = m.group(2).split(',')

            if len(row) <=1: continue # likely end of the trade-day

            ncols = min([len(HEADERS), len(row)])
            d = {HEADERS[i]:row[i] for i in range(ncols)}
            tickdata = TickData("AShare", s)
            tickdata.price  = toFloatVal(d['price'])
            tickdata.volume = toFloatVal(d['volume'])
            tickdata.total  = toFloatVal(d['total'])

            tickdata.datetime = datetime.strptime(d['date'] + 'T' + d['time'], '%Y-%m-%dT%H:%M:%S')
            tickdata.date = tickdata.datetime.strftime('%Y-%m-%d')
            tickdata.time = tickdata.datetime.strftime('%H:%M:%S')
            
            tickdata.open      = toFloatVal(d['open'])
            tickdata.high      = toFloatVal(d['high'])
            tickdata.low       = toFloatVal(d['low'])
            tickdata.prevClose = toFloatVal(d['prevClose'])

            # tickdata.upperLimit = EventData.EMPTY_FLOAT           # 涨停价
            # tickdata.lowerLimit = EventData.EMPTY_FLOAT           # 跌停价
            
            # 五档行情
            tickdata.b1P = toFloatVal(d['bid1'])
            tickdata.b1V = toFloatVal(d['bid1v'])
            tickdata.b2P = toFloatVal(d['bid2'])
            tickdata.b2V = toFloatVal(d['bid2v'])
            tickdata.b3P = toFloatVal(d['bid3'])
            tickdata.b3V = toFloatVal(d['bid3v'])
            tickdata.b4P = toFloatVal(d['bid4'])
            tickdata.b4V = toFloatVal(d['bid4v'])
            tickdata.b5P = toFloatVal(d['bid5'])
            tickdata.b5V = toFloatVal(d['bid5v'])
            
            # ask to sell: price and volume
            tickdata.a1P = toFloatVal(d['ask1'])
            tickdata.a1V = toFloatVal(d['ask1v'])
            tickdata.a2P = toFloatVal(d['ask2'])
            tickdata.a2V = toFloatVal(d['ask2v'])
            tickdata.a3P = toFloatVal(d['ask3'])
            tickdata.a3V = toFloatVal(d['ask3v'])
            tickdata.a4P = toFloatVal(d['ask4'])
            tickdata.a4V = toFloatVal(d['ask4v'])
            tickdata.a5P = toFloatVal(d['ask5'])
            tickdata.a5V = toFloatVal(d['ask5v'])

            tickseq.append(tickdata)

        return tickseq

    def GET_RecentTicks(self, symbols):
        ''' 查询Tick
        will call cortResp.send(csvline) when the result comes
        '''
        if not isinstance(symbols, list) :
            symbols = symbols.split(',')
        qsymbols = [SinaCrawler.fixupSymbolPrefix(s) for s in symbols]
        url = 'http://hq.sinajs.cn/list=%s' % (','.join(qsymbols).lower())

        httperr, text = self.__sinaGET(url, 'GET_RecentTicks()')
        if 200 != httperr:
            return httperr, text

        tickseq = self.__class__.convertToTickDatas(text)
        
        self.debug("GET_RecentTicks() resp(%d) got %d ticks" %(httperr, len(tickseq)))
        return httperr, tickseq

    #------------------------------------------------    
    def convertToMoneyFlow(symbol, text, byMinutes=False):
        '''
        will call cortResp.send(csvline) when the result comes
        ({r0_in:"0.0000",r0_out:"0.0000",r0:"0.0000",r1_in:"3851639.0000",r1_out:"4794409.0000",r1:"9333936.0000",r2_in:"8667212.0000",r2_out:"10001938.0000",r2:"18924494.0000",r3_in:"7037186.0000",r3_out:"7239931.2400",r3:"15039741.2400",curr_capital:"9098",name:"朗科智能",trade:"24.4200",changeratio:"0.000819672",volume:"1783866.0000",turnover:"196.083",r0x_ratio:"0",netamount:"-2480241.2400"})

        TICK:    http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssx_bkzj_fszs?daima=SH601988
        1MIN!!:  http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssx_ggzj_fszs?daima=SH601988
                ["240",[{opendate:"2020-03-20",ticktime:"15:00:00",trade:"3.5000",changeratio:"0.0115607",inamount:"173952934.3200",outamount:"144059186.4000",netamount:"29893747.9200",ratioamount:"0.0839833",r0_ratio:"0.0767544",r3_ratio:"0.010566"},
                ticktime时间15:00:00,trade价格3.50,涨跌幅+1.156%,inamount流入资金/万17395.29,outamount流出资金/万14405.92,净流入/万2989.37,netamount净流入率8.40%,r0_ratio主力流入率7.68%,r3_ratio散户流入率1.06%
                {opendate:"2020-03-20",ticktime:"14:58:00",trade:"3.5200",changeratio:"0.017341",inamount:"173952934.3200",outamount:"144059186.4000",netamount:"29893747.9200",ratioamount:"0.093927",r0_ratio:"0.0858422",r3_ratio:"0.011817"},
        DAILY: http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_zjlrqs?daima=SH601988
              [{opendate:"2020-03-20",trade:"3.5000",changeratio:"0.0115607",turnover:"4.84651",netamount:"29893747.9200",ratioamount:"0.0839833",r0_net:"27320619.5800",r0_ratio:"0.07675441",r0x_ratio:"81.4345",cnt_r0x_ratio:"1",cate_ra:"0.103445",cate_na:"1648659621.3400"},
              trade收盘价3.50,changeratio涨跌幅+1.156%,turnover换手率0.0485%,netamount净流入/万2989.37,ratioamount净流入率8.40%,r0_net主力净流入/万2732.06,r0_ratio主力净流入率7.68%,r0x_ratio主力罗盘81.43°,cate_ra行业净流入率10.34%
              {opendate:"2020-03-19",trade:"3.4600",changeratio:"-0.0114286",turnover:"5.71814",netamount:"-6206568.4600",ratioamount:"-0.0148799",r0_net:"-21194529.9100",r0_ratio:"-0.05081268",r0x_ratio:"-102.676",cnt_r0x_ratio:"-2",cate_ra:"-0.0122277",cate_na:"-253623190.4100"},
        '''

        mfseq =[]
        if byMinutes:
            pbeg, pend = text.find('[{'), text.rfind('}]')
            if pbeg<0 or pbeg>=pend:
                return mfseq
            text = text[pbeg:pend] + '}]'

        if len(text)>80*1024: # maximal 80KB is enough to cover 1Yr
            # EOM = text[text.find('}'):]
            text = text[:80*1024]
            pend = text.rfind('},{')
            if pend<=0:
                return mfseq
            text = text[:pend] + "}]"

        jsonData = demjson.decode(text)
        if not jsonData:
            return mfseq

        for mf in jsonData :
            mfdata = MoneyflowData("AShare", symbol)
            mfdata.price        = toFloatVal(mf['trade'])
            mfdata.netamount    = toFloatVal(mf['netamount'])
            mfdata.ratioNet     = toFloatVal(mf['ratioamount'])
            mfdata.ratioR0      = toFloatVal(mf['r0_ratio'])
            mfdata.ratioR3cate  = toFloatVal(mf['r3_ratio']) if byMinutes else toFloatVal(mf['cate_ra'])
            mfdata.datetime     = datetime.strptime(mf['opendate'], '%Y-%m-%d').replace(hour=15, minute=0, second=0, microsecond=0)
            if byMinutes:
                mfdata.datetime = datetime.strptime(mf['opendate'] + ' ' + mf['ticktime'], '%Y-%m-%d %H:%M:%S')
            elif len(mfseq) >260 :
                break
            mfdata.date = mfdata.datetime.strftime('%Y-%m-%d')
            mfdata.time = mfdata.datetime.strftime('%H:%M:%S')
            mfseq.append(mfdata)

        mfseq.sort(key=SinaCrawler.sortKeyOfMD)
        return mfseq

    def GET_MoneyFlow(self, symbol, lines=260, byMinutes=False, urlProxy=None, saveAs=None) :
        ''' 查询现金流
        '''
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_zjlrqs?sort=opendate&page=1&num=%d&daima=%s' % (lines, SinaCrawler.fixupSymbolPrefix(symbol)) # page 1 of 300lines is enough to cover days of a year
        if byMinutes:
            url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssx_ggzj_fszs?sort=time&num=%d&page=1&daima=%s' % (lines, SinaCrawler.fixupSymbolPrefix(symbol)) # page 1 of 300lines is enough to cover 4hr of a whole day

        httperr, text = self.__sinaGET(url, 'GET_MoneyFlow()', urlProxy)
        if 200 != httperr:
            return httperr, text

        mfseq = self.__class__.convertToMoneyFlow(symbol, text, byMinutes)
        try:
            if saveAs and len(mfseq) >0:
                with open(saveAs, 'w') as f:
                    f.write(text)
        except: pass

        return httperr, mfseq

    #------------------------------------------------    
    def GET_Transactions(self, symbol):
        ''' 查询逐笔交易明细
        will call cortResp.send(csvline) when the result comes
        var trade_item_list = new Array();
        trade_item_list[0] = new Array('15:00:00', '608400', '36.640', 'UP');
        trade_item_list[1] = new Array('14:57:02', '1000', '36.610', 'DOWN');
        ...
        trade_item_list[4563] = new Array('09:25:00', '537604', '37.220', 'EQUAL');
        '''
        HEADERSEQ="time,volume,price,type"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^.*trade_item_list.*Array\(([^\)]*)\).*')
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/view/CN_TransListV2.php?symbol=%s' % (SinaCrawler.fixupSymbolPrefix(symbol))
        httperr, text = self.__sinaGET(url, 'GET_Transactions()')
        if 200 != httperr:
            return httperr, text

        txns = []

        for line in text.split('\n') :
            m = SYNTAX.match(line)
            if not m :
                # LOG ERR
                continue

            row = m.group(1).split(',')
            ncols = min([len(HEADERS), len(row)])
            d = {HEADERS[i]:row[i].strip(" \t\r\n,\'\"") for i in range(ncols)}
            txns.append(d)

        return True, txns

    #------------------------------------------------    
    def GET_AllSymbols(self, ex_node='SH', urlProxy=None): # ex_node={SH|SZ}
        '''
        resp body would be like
        [{"symbol":"sh600238","code":"600238","name":"ST\u6930\u5c9b","trade":"5.390","pricechange":"0.000","changepercent":"0.000","buy":"0.000","sell":"0.000","settlement":"5.390","open":"0.000","high":"0.000","low":"0.000","volume":0,"amount":0,"ticktime":"15:29:59","per":-8.983,"pb":4.784,"mktcap":241579.8,"nmc":239855.85162,"turnoverratio":0},
         {"symbol":"sh600239","code":"600239","name":"\u4e91\u5357\u57ce\u6295","trade":"4.330","pricechange":"-0.070","changepercent":"-1.591","buy":"4.330","sell":"4.340","settlement":"4.400","open":"4.330","high":"4.490","low":"4.290","volume":41860323,"amount":182987159,"ticktime":"15:00:00","per":-2.474,"pb":3.322,"mktcap":695262.431597,"nmc":695262.431597,"turnoverratio":2.607},
         {"symbol":"sh600241","code":"600241","name":"*ST\u65f6\u4e07","trade":"3.390","pricechange":"-0.010","changepercent":"-0.294","buy":"3.380","sell":"3.390","settlement":"3.400","open":"3.400","high":"3.400","low":"3.370","volume":1572918,"amount":5328448,"ticktime":"15:00:00","per":-3.606,"pb":1.025,"mktcap":99768.416985,"nmc":85281.237408,"turnoverratio":0.62525},
        ]
        '''
        HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,trade,open,high,low,volume,amount"
        HEADERS=HEADERSEQ.split(',')
        MAX_SYM_COUNT=6000
        ex_node = ex_node.lower()

        ret =[]
        url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?&sort=symbol&asc=1&node=%s_a&_s_r_a=init&num=100" % ex_node
        for page in range(1, int(MAX_SYM_COUNT/100)) : # 100 per page
            httperr, text = self.__sinaGET(url + '&page=%d' % page, 'GET_AllSymbols(%d)' %page, urlProxy)
            if 4 == int(httperr/100) or len(text) <20 :
                break

            if 200 != httperr:
                self.warn("GET_AllSymbols(%s) page %d got httperr(%d)" %(ex_node, page, httperr))
                continue

            jsonData = demjson.decode(text)
            if not jsonData:
                continue

            if len(jsonData) <=0:
                break

            for i in jsonData :
                item = {}
                for h in HEADERS:
                    v = i[h]
                    if isinstance(v, str):
                        v = v.upper()
                    if 'trade' == h:
                        h ='close'
                    item[h] = v
                     # utf-8 .decode()
                ret.append(item)
        
        self.info("GET_AllSymbols(%s) found %d symbols" %(ex_node, len(ret)))
        return httperr, ret

    #------------------------------------------------    
    def GET_SplitRate(self, symbol):
        ''' 查询前复权因子
        will call cortResp.send(csvline) when the result comes
        var sz002604hfq=[{total:1893,data:{_2019_05_14:"5.3599",...,2015_06_05:"52.5870",_2015_06_04:"53.8027",..._2011_07_29:"27.8500",_2011_07_28:"25.3200"}}]
        '''
        url = 'http://finance.sina.com.cn/realstock/newcompany/%s/phfq.js' % (SinaCrawler.fixupSymbolPrefix(symbol))
        httperr, text = self.__sinaGET(url, 'GET_SplitRate')
        if 200 != httperr:
            return httperr, text

        spliterates = []
        str=text[text.find('{'):text.rfind('}')+1]

        ret = []
        jsonData = demjson.decode(re.sub('_([0-9]{4})_([0-9]{2})_([0-9]{2})', r'\1-\2-\3', str)) # convert _2019_05_14 to 2019-05-14
        if not jsonData:
            return httperr, ret

        for k,v in jsonData['data'].items() : 
            ret.append({k, v.toFloatVal()})
        return httperr, ret

########################################################################
class SinaTickToKL1m(object):
    """
    SINA Tick合成1分钟K线
    """

    #----------------------------------------------------------------------
    def __init__(self, onKLine1min):
        """Constructor"""
        self.__lastTick = None
        self.__onKline1min = onKLine1min      # 1分钟K线回调函数
        self.__kline = None
        self.__lastVol = 0

    @property
    def lastTickAsOf(self):
        return self.__lastTick.datetime if self.__lastTick else DT_EPOCH
        
    #----------------------------------------------------------------------
    def pushTick(self, tick):
        """TICK更新"""

        if self.__kline and self.__kline.datetime.minute != tick.datetime.minute and Account_AShare.duringTradeHours(tick.datetime):
            # 生成一分钟K线的时间戳
            self.__kline.datetime = self.__kline.datetime.replace(second=0, microsecond=0) +timedelta(minutes=1)  # align to the next minute:00.000
            self.__kline.date = self.__kline.datetime.strftime('%Y-%m-%d')
            self.__kline.time = self.__kline.datetime.strftime('%H:%M:%S.%f')
        
            # 推送已经结束的上一分钟K线
            if self.__onKline1min :
                while self.__kline.datetime <= tick.datetime : # the new tick time might stepped more than 1min, so make a loop here
                    kl = copy(self.__kline)
                    kl.volume -= self.__lastVol
                    self.__lastVol = self.__kline.volume
                    self.__onKline1min(kl)
                    self.__kline.datetime += timedelta(minutes=1)
            
            self.__kline = None # 创建新的K线对象
            
        # 初始化新一分钟的K线数据
        if not self.__kline:
            # 创建新的K线对象
            self.__kline = KLineData(tick.exchange + '_t2k', tick.symbol)
            self.__kline.open = tick.price
            self.__kline.high = tick.price
            self.__kline.low = tick.price
            if self.__lastVol <=0:
                self.__lastVol = tick.volume

        # 累加更新老一分钟的K线数据
        else:                                   
            self.__kline.high = max(self.__kline.high, tick.price)
            self.__kline.low = min(self.__kline.low, tick.price)

        # 通用更新部分
        self.__kline.close = tick.price        
        self.__kline.datetime = tick.datetime  
        self.__kline.openInterest = tick.openInterest
   
        volumeChange = tick.volume - self.__kline.volume   # 当前K线内的成交量
        self.__kline.volume += max(volumeChange, 0)             # 避免夜盘开盘lastTick.volume为昨日收盘数据，导致成交量变化为负的情况
            
        # 缓存Tick
        self.__lastTick = tick

    def flushAtClose(self):
        if not self.__lastTick or not self.__kline:
            return

        self.__kline.datetime = (self.__lastTick.datetime +timedelta(seconds=30)).replace(second=0, microsecond=0)
        self.__kline.date = self.__kline.datetime.strftime('%Y-%m-%d')
        self.__kline.time = self.__kline.datetime.strftime('%H:%M:%S.%f')
    
        # 推送已经结束的上一分钟K线
        if self.__onKline1min :
            kl = copy(self.__kline)
            kl.volume -= self.__lastVol

            self.__lastVol = self.__kline.volume
            self.__onKline1min(kl)
        
        self.__kline = None # 创建新的K线对象

########################################################################
class SinaMF1mToXm(object):
    """ SINA MF1m合成X分钟MF    """

    #----------------------------------------------------------------------
    def __init__(self, onMFlowXm, X=5):
        """Constructor"""
        self.__mfLatest = None
        self.__onMFlowXm = onMFlowXm      # callback
        self.__X = int(X)
        if self.__X <1: self.__X =1

    @property
    def asof(self):
        return self.__mfLatest.asof if self.__mfLatest else DT_EPOCH
        
    #----------------------------------------------------------------------
    def pushMF1m(self, mf1m):
        if not mf1m or self.asof >= mf1m.asof:
            return

        if 0 == ((mf1m.asof.hour *60 + mf1m.asof.minute) % self.__X):
            d = copy(mf1m)
            if not '_m2x' in d.exchange :
                d.exchange = '%s_m2x' % d.exchange

            self.__mfLatest = copy(d)
            if self.__onMFlowXm :
                self.__onMFlowXm(d)

########################################################################
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

def listSymbols(program, mdSina):
    # 3869 symbols as of 2020-06-20
    result ={}
    thruPrxy = False

    httperr =100
    while 2 != int(httperr/100):
        urlProxy = None # urlProxy= dsg.nextProxy() if thruPrxy else None
        httperr, lstSH = md.GET_AllSymbols('SH', urlProxy)
        print('SH-resp(%d) thru[%s] len=%d' %(httperr, urlProxy, len(lstSH)))
        if 456 == httperr and urlProxy is None: sleep(30)

    if 2 == int(httperr/100): dsg.stampGoodProxy()

    httperr =100
    while 2 != int(httperr/100):
        urlProxy = None # urlProxy= dsg.nextProxy() if thruPrxy else None
        httperr, lstSZ = md.GET_AllSymbols('SZ', urlProxy)
        print('SZ-resp(%d) thru[%s] len=%d' %(httperr, urlProxy, len(lstSZ)))
        if 456 == httperr and urlProxy is None: sleep(30)

    for i in lstSH + lstSZ:
        result[i['symbol']] =i
    result = list(result.values())

    global __totalAmt1W
    __totalAmt1W=0
    print('-'*10 + ' All %d symbols '%len(result) + '-'*10)
    HEADERSEQ="symbol,name,mktcap,nmc,turnoverratio,close,volume"
    print(HEADERSEQ)
    for i in result:
        __totalAmt1W += i['amount'] /10000.0
        print(','.join([str(i[k]) for k in HEADERSEQ.split(',')]))

    # filter the top active 1000
    topXXX = list(filter(lambda x: not 'ST' in x['name'], result))
    topNum = min(500, int(len(topXXX)/50) *10)

    print('-'*10 + ' TopAct %s ' %topNum + '-'*10)
    topXXX.sort(key=activityOf)
    topXXX= topXXX[-topNum:]
    topXXX.reverse()
    print(HEADERSEQ)
    for i in topXXX: print(','.join([str(i[k]) for k in HEADERSEQ.split(',')]))

########################################################################
if __name__ == '__main__':
    from Application import Program

    p = Program()
    # mc = p.createApp(SinaCrawler, configNode ='crawler', marketState = gymtdr._marketState, recorder=rec) # md = SinaCrawler(p, None);
    md = SinaCrawler(p, None)
    # _, result = md.GET_MoneyFlow("SH601005")
    listSymbols(p, md)
    # _, result = md.searchKLines("000002", EVENT_KLINE_5MIN)
    # _, result = md.GET_RecentTicks('sh601006,sh601005,sh000001,sz000001')
    # _, result = md.GET_SplitRate('sh601006')
    # print(result)
    # md.subscribe(['601006','sh601005','sh000001','000001'])
    # while True:
    #     md.doAppStep()
