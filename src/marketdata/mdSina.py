# encoding: UTF-8

from __future__ import division

from .mdBackEnd import MarketData, KLineData, TickData, DataToEvent
#from ..event.EventChannel import Event

import requests 
from copy import copy
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from multiprocessing.dummy import Pool
from datetime import datetime , timedelta
from abc import ABCMeta, abstractmethod
import demjson

import zlib

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
        
        if symbol.isdigit() :
            if symbol.startswith('0') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('3') :
                symbol = "sz%s" % symbol
            elif symbol.startswith('6') :
                symbol = "sh%s" % symbol

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
            tdelta = now_datetime - since
            tslide = timedelta(minuts=scale)
            lines = (tdelta + tslide -1) / tslide

        url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=%s&scale=%s&datalen=%s" % (symbol, scale, lines)
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


if __name__ == '__main__':
    md = mdSina(None, None);
    result = md.searchKLines("000002", MarketData.EVENT_KLINE_5MIN)
    result


