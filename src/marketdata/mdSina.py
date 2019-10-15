# encoding: UTF-8

from __future__ import division

from mdBackEnd import MarketData, KLineData, TickData, DataToEvent
from event.ecBasic import Event, datetime2float

import requests # pip3 install requests
from copy import copy
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from multiprocessing.dummy import Pool
from datetime import datetime , timedelta
from abc import ABCMeta, abstractmethod
import demjson # pip3 install demjso

import zlib
import re

from threading import Thread
from multiprocessing import Pool

########################################################################
class mdSina(MarketData):
    """MarketData BackEnd API"""
    # 常量定义
    #----------------------------------------------------------------------
    DEFAULT_GET_HEADERS = {
        "Content-type": "application/x-www-form-urlencoded",
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'
    }

    TIMEOUT = 5

    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(mdSina, self).__init__(mainRoutine, settings)

        self._proxies = {}

        # self._mode        = Account.BROKER_API_ASYNC
        # self._queRequests = Queue()        # queue of request ids
        # self._dictRequests = {}            # dict from request Id to request

    # @property
    # def cashSymbol(self): # overwrite Account's cash to usdt
    #     return 'usdt'

    # @property
    # def accessKey(self) :
    #     return self._settings.accessKey('')

    # @property
    # def secretKey(self) :
    #     return self._settings.secretKey('')

    #----------------------------------------------------------------------
    # if the MarketData has background thread, connect() will not start the thread
    # but start() will
    @abstractmethod
    def connect(self):
        """连接"""
        raise NotImplementedError
#        return self.active

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def start(self):
        """连接"""
        self.connect()
        
    @abstractmethod
    def step(self):
        """连接"""
        return 0 # raise NotImplementedError

    @abstractmethod
    def stop(self):
        """停止"""
        if self._active:
            self._active = False
            self.close()

    #------------------------------------------------
    # overwrite of BackEnd
    #------------------------------------------------    
    def searchKLines(self, symbol, eventType, since=None, cb=None):
        """查询请求""" 
        '''
        will call cortResp.send(csvline) when the result comes
        '''
        
        symbol = fixupSymbolPrefix(symbol)

        if (eventType == MarketData.EVENT_TICK) :
            return self.searchTicks(symbol, since, cb)

        scale =1200
        lines =1000

        if   MarketData.EVENT_KLINE_1MIN == eventType :
            scale =5
        if   MarketData.EVENT_KLINE_5MIN == eventType :
            scale =5
        elif MarketData.EVENT_KLINE_15MIN == eventType :
            scale =15
        elif MarketData.EVENT_KLINE_30MIN == eventType :
            scale =30
        elif MarketData.EVENT_KLINE_1HOUR == eventType :
            scale =60
        elif MarketData.EVENT_KLINE_4HOUR == eventType :
            scale =240
        elif MarketData.EVENT_KLINE_1DAY == eventType :
            scale =240

        now_datetime = datetime.now()
        if since:
            tdelta = datetime2float(now_datetime) - datetime2float(since)
            lines = (tdelta + 60*scale -1) / 60/scale

        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=%s&scale=%s&datalen=%d" % (symbol, scale, lines)
        klineseq =[]
        try:
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code == 200:
                # [{day:"2019-09-23 14:15:00",open:"15.280",high:"15.290",low:"15.260",close:"15.270",volume:"892600",ma_price5:15.274,ma_volume5:1645033,ma_price10:15.272,ma_volume10:1524623,ma_price30:15.296,ma_volume30:2081080},
                # {day:"2019-09-23 14:20:00",open:"15.270",high:"15.280",low:"15.240",close:"15.240",volume:"1591705",ma_price5:15.266,ma_volume5:1676498,ma_price10:15.27,ma_volume10:1593887,ma_price30:15.292,ma_volume30:1955370},
                # ...]
                # js = response.json()
                # result = demjson.decode(response.text)
                # result.decode('utf-8')
                jsonData = demjson.decode(response.text)

                for kl in jsonData :
                    kldata = KLineData("ASHARE", symbol)
                    kldata.open = kl['open']             # OHLC
                    kldata.high = kl['high']             # OHLC
                    kldata.low = kl['low']             # OHLC
                    kldata.close = kl['close']             # OHLC
                    kldata.volume = kl['volume']
                    kldata.close = kl['close']
                    kldata.date = kl['day'][0:10]
                    kldata.time = kl['day'][11:]
                    klineseq.append(kldata)

                return True, klineseq
            else:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

    def fixupSymbolPrefix(symbol):
        if symbol.isdigit() :
            if symbol.startswith('0') :
                return "SZ%s" % symbol
            elif symbol.startswith('3') :
                return "SZ%s" % symbol
            elif symbol.startswith('6') :
                return "SH%s" % symbol
        return symbol

    #------------------------------------------------    
    def getRecentTicks(self, symbols):
        """查询请求""" 
        '''
        will call cortResp.send(csvline) when the result comes
        '''
        if not isinstance(symbols, list) :
            symbols = symbols.split(',')
        qsymbols = [mdSina.fixupSymbolPrefix(s) for s in symbols]
        url = 'http://hq.sinajs.cn/list=%s' % (','.join(qsymbols))
        HEADERSEQ="name,open,prevClose,price,high,low,bid,ask,volume,total,bid1v,bid1,bid2v,bid2,bid3v,bid3,bid4v,bid4,bid5v,bid5,ask1v,ask1,ask2v,ask2,ask3v,ask3,ask4v,ask4,ask5v,ask5,date,time"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^var hq_str_([^=]*)="([^"]*).*')
        tickseq = []
        try:
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code != 200:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

        for line in response.text.split('\n') :
            m = SYNTAX.match(line)
            if not m :
                # LOG ERR
                continue

            s = m.group(1).upper()
            row = m.group(2).split(',')
            ncols = min([len(HEADERS), len(row)])
            d = {HEADERS[i]:row[i] for i in range(ncols)}
            tickdata = TickData("ASHARE", s)
            tickdata.price = float(d['price'])
            tickdata.volume = float(d['volume'])
            tickdata.total = float(d['total'])

            # tickdata.time = EventData.EMPTY_STRING                # 时间 11:20:56.5
            # tickdata.date = EventData.EMPTY_STRING                # 日期 20151009
            # tickdata.datetime = None                    # python的datetime时间对象
            
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
        
        return True, tickseq

    #------------------------------------------------    
    def getMoneyFlow(self, symbol):
        """查询请求""" 
        '''
        will call cortResp.send(csvline) when the result comes
        ({r0_in:"0.0000",r0_out:"0.0000",r0:"0.0000",r1_in:"3851639.0000",r1_out:"4794409.0000",r1:"9333936.0000",r2_in:"8667212.0000",r2_out:"10001938.0000",r2:"18924494.0000",r3_in:"7037186.0000",r3_out:"7239931.2400",r3:"15039741.2400",curr_capital:"9098",name:"朗科智能",trade:"24.4200",changeratio:"0.000819672",volume:"1783866.0000",turnover:"196.083",r0x_ratio:"0",netamount:"-2480241.2400"})
        '''
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj?daima=%s' % (mdSina.fixupSymbolPrefix(symbol))
        try:
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code != 200:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

        str=response.text[response.text.find('{'):response.text.rfind('}')+1]
        jsonData = demjson.decode(str)

        return True, jsonData

    #------------------------------------------------    
    def getTransactions(self, symbol):
        """查询逐笔交易明细""" 
        '''
        will call cortResp.send(csvline) when the result comes
        var trade_item_list = new Array();
        trade_item_list[0] = new Array('15:00:00', '608400', '36.640', 'UP');
        trade_item_list[1] = new Array('14:57:02', '1000', '36.610', 'DOWN');
        ...
        trade_item_list[4563] = new Array('09:25:00', '537604', '37.220', 'EQUAL');
        '''
        HEADERSEQ="time,volume,price,type"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^.*trade_item_list\[.*Array\(([^\)]*)\).*')
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/view/CN_TransListV2.php?symbol=%s' % (mdSina.fixupSymbolPrefix(symbol))
        try:
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code != 200:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

        txns = []

        for line in response.text.split('\n') :
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
    def getSplitRate(self, symbol):
        """查询前复权因子""" 
        '''
        will call cortResp.send(csvline) when the result comes
        var sz002604hfq=[{total:1893,data:{_2019_05_14:"5.3599",...,2015_06_05:"52.5870",_2015_06_04:"53.8027",..._2011_07_29:"27.8500",_2011_07_28:"25.3200"}}]
        '''
        HEADERSEQ="time,volume,price,type"
        HEADERS=HEADERSEQ.split(',')
        SYNTAX = re.compile('^.*trade_item_list\[.*Array\(([^\)]*)\).*')
        url = 'http://finance.sina.com.cn/realstock/newcompany/%s/phfq.js' % (mdSina.fixupSymbolPrefix(symbol))
        try:
            response = requests.get(url, headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code != 200:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

        spliterates = []
        str=response.text[response.text.find('{'):response.text.rfind('}')+1]
        jsonData = demjson.decode(str)
        jsonData['data']

        return True, jsonData['data']

if __name__ == '__main__':
    md = mdSina(None, None);
    # result = md.searchKLines("000002", MarketData.EVENT_KLINE_5MIN)
    # _, result = md.getRecentTicks('sh601006,sh601005,sh000001,sz000001')
    _, result = md.getSplitRate('sh601006')
    print(result)
