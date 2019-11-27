# encoding: UTF-8

from __future__ import division

from MarketCrawler import *
from EventData import Event, datetime2float
from MarketData import KLineData, TickData, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY
from Perspective import Perspective

import requests # pip3 install requests
from copy import copy
from datetime import datetime , timedelta
import demjson # pip3 install demjso

import re

########################################################################
class SinaCrawler(MarketCrawler):
    """MarketData BackEnd API"""
    # 常量定义
    #----------------------------------------------------------------------
    DEFAULT_GET_HEADERS = {
        "Content-type": "application/x-www-form-urlencoded",
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'
    }

    TIMEOUT = 5
    TICK_BATCH_SIZE = 100
    NEXTSTAMP_KLINE_5MIN = 'nstamp.'+EVENT_KLINE_5MIN
    NEXTSTAMP_KLINE_1DAY = 'nstamp.'+EVENT_KLINE_1DAY
    IDX_KLINE_5MIN = 'idx.'+EVENT_KLINE_5MIN
    IDX_KLINE_1DAY = 'idx.'+EVENT_KLINE_1DAY
    
    MINs_OF_EVENT = {
            EVENT_KLINE_5MIN: 5,
            EVENT_KLINE_1DAY: 240,
        }

    def __init__(self, program, **kwargs):
        """Constructor"""
        super(SinaCrawler, self).__init__(program, **kwargs)
        
        # SinaCrawler take multiple HTTP requests to collect data, each of them may take different
        # duration to complete, so this crawler should be threaded
        # self._threadless = False

        self._steps = [self.__step_poll1st, self.__step_pollTicks, self.__step_pollKline]

        self._proxies = {}

        self._depth_ticks  = self.getConfig('depth/ticks', 120)
        self._depth_5min   = self.getConfig('depth/5min',  96)
        self._depth_1day  = self.getConfig('depth/1day',   220)

        self.__tickBatches = None
        self.__idxTickBatch = 0
        self.__nextStamp_PollTick = None

        self.__cacheKLs = {} # dict of symbol to Perspective
        self.__idxKL = 0

        self.__step_poll1st() # perform an init-step

    #------------------------------------------------
    # sub-steps
    def __step_poll1st(self):
        self.__END_OF_TODAY = datetime2float(datetime.now().replace(hour=15, minute=1))
        return 0

    def __step_pollTicks(self):
        
        cBusy =0 
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
            return cBusy
            
        #step 2. check if to yield time
        self.__idxTickBatch = self.__idxTickBatch % batches
        if 0 == self.__idxTickBatch : # this is a new round
            # yield some time in order not to poll SiNA too frequently
            if self.__nextStamp_PollTick and (self._stepAsOf < self.__nextStamp_PollTick 
                or self.__nextStamp_PollTick > self.__END_OF_TODAY):
                return cBusy

            self.__nextStamp_PollTick = self._stepAsOf +0.7

        idxBtch = self.__idxTickBatch
        self.__idxTickBatch +=1
        
        updated=[]
        httperr, result = self.GET_RecentTicks(self.__tickBatches[idxBtch])
        if httperr !=200:
            self.error("step_pollTicks() GET_RecentTicks failed, err(%s) bth:%s" %(httperr, bth))
            return cBusy
            
        # succ at previous batch here
        if len(result) <=0 : 
            self.__nextStamp_PollTick + 60*10 # likely after a trade-day closed 10min

        for tk in result:
            cBusy +=1
            s = tk.symbol
            if not s in self.__cacheKLs.keys():
                self.__cacheKLs[s] = Perspective('AShare', symbol =s, KLDepth_1min=0, KLDepth_5min=self._depth_5min, KLDepth_1day=self._depth_1day, tickDepth=self._depth_ticks)
            psp = self.__cacheKLs[s]
            ev = Event(EVENT_TICK)
            ev.setData(tk)
            psp.push(ev)
            self.debug("step_pollTicks() pushed tick %s into psp" %(tk.desc))
            updated.append(s)
            
        self.info("step_pollTicks() btch[%d/%d] cached %s into psp" %(idxBtch, batches, updated))
        return cBusy

    def __step_pollKline(self):
        cBusy =0       
        s = None
        if len(self._symbolsToPoll) >0:
            self.__idxKL = self.__idxKL % len(self._symbolsToPoll)
            s = self._symbolsToPoll[self.__idxKL]
        self.__idxKL += 1

        if not s or len(s) <=0:
            return cBusy

        if not s in self.__cacheKLs.keys():
            self.__cacheKLs[s] = Perspective('AShare', symbol=s, KLDepth_1min=0, KLDepth_5min=self._depth_5min, KLDepth_1day=self._depth_1day, tickDepth=self._depth_ticks)

        psp = self.__cacheKLs[s]
        for evType in [EVENT_KLINE_5MIN, EVENT_KLINE_1DAY] :
            minutes = SinaCrawler.MINs_OF_EVENT[evType]
            etimatedNext = datetime2float(psp.getAsOf(evType)) + minutes*60 -1
            self.__END_OF_TODAY = datetime2float(datetime.now().replace(hour=15, minute=1))
            if etimatedNext > self.__END_OF_TODAY or self._stepAsOf < etimatedNext:
                continue

            size, lines = psp.sizesOf(evType)
            if size >0:
                lines = 0
            lines +=10

            httperr, result = self.GET_RecentKLines(s, minutes, lines)
            if httperr !=200:
                self.error("step_pollKline(%s:%s) failed, err(%s)" %(s, evType, httperr))
                continue

            # succ at query
            for i in result:
                cBusy +=1
                ev = Event(evType)
                ev.setData(i)
                psp.push(ev)

            self.info("step_pollKline(%s:%s) merged %s-KLs into stack, psp now: %s" %(s, evType, len(result), psp.desc))

        return cBusy

    # end of sub-steps
    #------------------------------------------------

    def subscribe(self, symbols):
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
    def __sinaGET(self, url, apiName):
        errmsg = '%s() GET ' % apiName
        httperr = 400
        try:
            self.debug("%s() GET %s" %(apiName, url))
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            httperr = response.status_code
            if httperr == 200:
                return httperr, response.text

            errmsg += 'err(%s)' % httperr
        except Exception as e:
            errmsg += 'exception：%s' % e
        
        self.error(errmsg)
        return httperr, errmsg

    def GET_RecentKLines(self, symbol, minutes=1200, lines=10): # deltaDays=2)
        '''"查询 KLINE
        will call cortResp.send(csvline) when the result comes
        '''

        if minutes <5: minutes=5 # sina only allow minimal 5min-KLines
        # if deltaDays <1: deltaDays=1
        # symbol = SinaCrawler.fixupSymbolPrefix(symbol)
        # now_datetime = datetime.now()
        # lines = deltaDays * 4 * 60 / minutes

        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=%s&scale=%s&datalen=%d" % (symbol, minutes, lines)
        httperr, text = self.__sinaGET(url, 'GET_RecentKLines')
        if httperr != 200:
            return httperr, text

        # [{day:"2019-09-23 14:15:00",open:"15.280",high:"15.290",low:"15.260",close:"15.270",volume:"892600",ma_price5:15.274,ma_volume5:1645033,ma_price10:15.272,ma_volume10:1524623,ma_price30:15.296,ma_volume30:2081080},
        # {day:"2019-09-23 14:20:00",open:"15.270",high:"15.280",low:"15.240",close:"15.240",volume:"1591705",ma_price5:15.266,ma_volume5:1676498,ma_price10:15.27,ma_volume10:1593887,ma_price30:15.292,ma_volume30:1955370},
        # ...]
        # js = response.json()
        # result = demjson.decode(response.text)
        # result.decode('utf-8')
        klineseq =[]
        jsonData = demjson.decode(text)

        for kl in jsonData :
            kldata = KLineData("ASHARE", symbol)
            kldata.open = kl['open']             # OHLC
            kldata.high = kl['high']             # OHLC
            kldata.low = kl['low']             # OHLC
            kldata.close = kl['close']             # OHLC
            kldata.volume = kl['volume']
            kldata.close = kl['close']
            try :
                kldata.datetime = datetime.strptime(kl['day'][:10], '%Y-%m-%d').replace(hour=15, minute=0, second=0, microsecond=0)
                kldata.datetime = datetime.strptime(kl['day'], '%Y-%m-%d %H:%M:%S')
            except :
                pass
            kldata.date = kldata.datetime.strftime('%Y-%m-%d')
            kldata.time = kldata.datetime.strftime('%H:%M:%S')
            klineseq.append(kldata)

        klineseq.sort(key=SinaCrawler.sortKeyOfKL)
        # klineseq.reverse() # SINA returns from oldest to newest
        return httperr, klineseq

    def sortKeyOfKL(KL) : # for sort
        return KL.datetime

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
    def GET_RecentTicks(self, symbols):
        ''' 查询Tick
        will call cortResp.send(csvline) when the result comes
        '''
        if not isinstance(symbols, list) :
            symbols = symbols.split(',')
        qsymbols = [SinaCrawler.fixupSymbolPrefix(s) for s in symbols]
        url = 'http://hq.sinajs.cn/list=%s' % (','.join(qsymbols).lower())

        httperr, text = self.__sinaGET(url, 'GET_RecentTicks')
        if httperr != 200:
            return httperr, text

        HEADERSEQ="name,open,prevClose,price,high,low,bid,ask,volume,total,bid1v,bid1,bid2v,bid2,bid3v,bid3,bid4v,bid4,bid5v,bid5,ask1v,ask1,ask2v,ask2,ask3v,ask3,ask4v,ask4,ask5v,ask5,date,time"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^var hq_str_([^=]*)="([^"]*).*')
        tickseq = []

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
            tickdata = TickData("ASHARE", s)
            tickdata.price = float(d['price'])
            tickdata.volume = float(d['volume'])
            tickdata.total = float(d['total'])

            tickdata.datetime = datetime.strptime(d['date'] + 'T' + d['time'], '%Y-%m-%dT%H:%M:%S')
            tickdata.date = tickdata.datetime.strftime('%Y-%m-%d')
            tickdata.time = tickdata.datetime.strftime('%H:%M:%S')
            
            tickdata.open = float(d['open'])
            tickdata.high = float(d['high'])
            tickdata.low = float(d['low'])
            tickdata.prevClose = float(d['prevClose'])

            # tickdata.upperLimit = EventData.EMPTY_FLOAT           # 涨停价
            # tickdata.lowerLimit = EventData.EMPTY_FLOAT           # 跌停价
            
            # 五档行情
            tickdata.b1P = float(d['bid1'])
            tickdata.b1V = float(d['bid1v'])
            tickdata.b2P = float(d['bid2'])
            tickdata.b2V = float(d['bid2v'])
            tickdata.b3P = float(d['bid3'])
            tickdata.b3V = float(d['bid3v'])
            tickdata.b4P = float(d['bid4'])
            tickdata.b4V = float(d['bid4v'])
            tickdata.b5P = float(d['bid5'])
            tickdata.b5V = float(d['bid5v'])
            
            # ask to sell: price and volume
            tickdata.a1P = float(d['ask1'])
            tickdata.a1V = float(d['ask1v'])
            tickdata.a2P = float(d['ask2'])
            tickdata.a2V = float(d['ask2v'])
            tickdata.a3P = float(d['ask3'])
            tickdata.a3V = float(d['ask3v'])
            tickdata.a4P = float(d['ask4'])
            tickdata.a4V = float(d['ask4v'])
            tickdata.a5P = float(d['ask5'])
            tickdata.a5V = float(d['ask5v'])

            tickseq.append(tickdata)
        
        self.debug("GET_RecentTicks() resp(%d) got %d ticks" %(httperr, len(tickseq)))
        return httperr, tickseq

    #------------------------------------------------    
    def GET_MoneyFlow(self, symbol):
        ''' 查询现金流
        will call cortResp.send(csvline) when the result comes
        ({r0_in:"0.0000",r0_out:"0.0000",r0:"0.0000",r1_in:"3851639.0000",r1_out:"4794409.0000",r1:"9333936.0000",r2_in:"8667212.0000",r2_out:"10001938.0000",r2:"18924494.0000",r3_in:"7037186.0000",r3_out:"7239931.2400",r3:"15039741.2400",curr_capital:"9098",name:"朗科智能",trade:"24.4200",changeratio:"0.000819672",volume:"1783866.0000",turnover:"196.083",r0x_ratio:"0",netamount:"-2480241.2400"})
        '''
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj?daima=%s' % (mdSina.fixupSymbolPrefix(symbol))
        httperr, text = self.__sinaGET(url, 'GET_MoneyFlow')
        if httperr != 200:
            return httperr, text

        str=text[text.find('{'):text.rfind('}')+1]
        jsonData = demjson.decode(str)

        return True, jsonData

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
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/view/CN_TransListV2.php?symbol=%s' % (mdSina.fixupSymbolPrefix(symbol))
        httperr, text = self.__sinaGET(url, 'GET_Transactions')
        if httperr != 200:
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
    def GET_SplitRate(self, symbol):
        ''' 查询前复权因子
        will call cortResp.send(csvline) when the result comes
        var sz002604hfq=[{total:1893,data:{_2019_05_14:"5.3599",...,2015_06_05:"52.5870",_2015_06_04:"53.8027",..._2011_07_29:"27.8500",_2011_07_28:"25.3200"}}]
        '''
        url = 'http://finance.sina.com.cn/realstock/newcompany/%s/phfq.js' % (SinaCrawler.fixupSymbolPrefix(symbol))
        httperr, text = self.__sinaGET(url, 'GET_SplitRate')
        if httperr != 200:
            return httperr, text

        spliterates = []
        str=text[text.find('{'):text.rfind('}')+1]
        jsonData = demjson.decode(re.sub('_([0-9]{4})_([0-9]{2})_([0-9]{2})', r'\1-\2-\3', str)) # convert _2019_05_14 to 2019-05-14
        ret = []
        for k,v in jsonData['data'].items() : 
            ret.append({k, v.float()})
        return True, ret

if __name__ == '__main__':
    from Application import Program

    p = Program()
    md = SinaCrawler(p, None);
    # _, result = md.searchKLines("000002", EVENT_KLINE_5MIN)
    # _, result = md.GET_RecentTicks('sh601006,sh601005,sh000001,sz000001')
    # _, result = md.GET_SplitRate('sh601006')
    # print(result)
    md.subscribe(['601006','sh601005','sh000001','000001'])
    while True:
        md.step()
