# encoding: UTF-8

from __future__ import division

from vnApp.MarketData import *

from vnpy.trader.vtConstant import *
from vnpy.event import Event

import urllib
import hmac
import base64
import hashlib
import requests 
import traceback
import socket
from copy import copy
from datetime import datetime
from threading import Thread
from Queue import Queue, Empty
from multiprocessing.dummy import Pool
from time import sleep

import json
import zlib

# retrieve package: sudo pip install websocket websocket-client pathlib
from websocket import create_connection, _exceptions

# 常量定义
TIMEOUT = 5
HUOBI_API_HOST = "api.huobi.pro"
HADAX_API_HOST = "api.hadax.com"
LANG = 'zh-CN'

#----------------------------------------------------------------------
def createSign(params, method, host, path, secretKey):
    """创建签名"""
    sortedParams = sorted(params.items(), key=lambda d: d[0], reverse=False)
    encodeParams = urllib.urlencode(sortedParams)
    
    payload = [method, host, path, encodeParams]
    payload = '\n'.join(payload)
    payload = payload.encode(encoding='UTF8')

    secretKey = secretKey.encode(encoding='UTF8')

    digest = hmac.new(secretKey, payload, digestmod=hashlib.sha256).digest()

    signature = base64.b64encode(digest)
    signature = signature.decode()
    return signature    



########################################################################
class mdHuobi(MarketData):
    """行情接口
    https://github.com/huobiapi/API_Docs/wiki/WS_request
    """
    className = 'Hadax'
    displayName = 'Market data subscriber from HADAX'
    typeName = 'Market data subscriber from HADAX'

    HUOBI = 'huobi'
    HADAX = 'hadax'

    #----------------------------------------------------------------------
    # setting schema
    # {'exchange':HADAX, 'proxies':{'host': 'url' }} } 
    def __init__(self, mainRoutine, settings):
        """Constructor"""

        super(mdHuobi, self).__init__(mainRoutine, settings, MarketData.DATA_SRCTYPE_REALTIME)

        self.ws = None
        self.url = ''
        self._dictSubscriptions = {}
        self._dictTicks = {}
        
        # huobi will repeat message for a same KLine, so to merge and only post the last
        self._dictKLineLatest = {}
        
        self._reqid = 0
        # self.thread = Thread(target=self._run)
        self._exchange = settings.exchange('')
        if self._exchange == self.HADAX:
            hostname = HADAX_API_HOST
        else:
            hostname = HUOBI_API_HOST
            self._exchange = self.HUOBI
        
        self.url = 'wss://%s/ws' % hostname
        self._proxy = settings.proxy('')

    # #----------------------------------------------------------------------
    # def _run(self):
    #     """执行连接 and receive"""
    #     while self._active:
    #         try :
    #             if self.step() <0:
    #                 self.onError(u'等待3秒后再次重连')
    #                 sleep(3)
    #         except Exception as ex:
    #             self.onError(u'行情服务器step err: %s' % ex)

    
    # #----------------------------------------------------------------------
    # def start(self):
    #     """连接"""
    #     self.connect()

    #     self._active = True
    #     self.thread.start()
            
    #     return self.active

    def connect(self):
        """连接"""
        if not self._doConnect() :
            return False

        self._resubscribe()

    def step(self):
        if not self.ws :
            if self._doConnect() :
                self.onError(u'行情服务器重连成功')
                self._resubscribe()
            else:
                self.onError(u'等待3秒后再次重连')
                return -1

        c = 0
        while self.ws:
            data = None
            try:
                stream = self.ws.recv() # (timeout=0.1)
                if not stream or len(stream) <=0:
                    break
                result = zlib.decompress(stream, 47).decode('utf-8')
                data = json.loads(result)
                c +=1
            except socket.timeout:
                break
            except zlib.error:
                self.onError(u'数据解压出错：%s' %stream)
            except Exception as ex:
                self.onError(u'行情服务器recv err %s' % ex)
                self.ws = None # disconnect it 

            if not data:
                break

            try:
                self._onData(data)
                return 1
            except Exception as ex:
                self.onError(u'数据分发错误：%s'  %ex)

        return c

    #----------------------------------------------------------------------
    def close(self):
        """停止"""
        if self._active:
            self._active = False
            # self.thread.join()
            self.ws.close()
        
    #----------------------------------------------------------------------
    def subscribe(self, symbol, eventType):
        """订阅成交细节"""
        if eventType != MarketData.EVENT_TICK:
            topic =""
            if eventType == MarketData.EVENT_MARKET_DEPTH0:
                topic = 'market.%s.depth.step0' % symbol
            elif eventType == MarketData.EVENT_KLINE_1MIN:
                topic = 'market.%s.kline.1min' % symbol
            elif eventType == MarketData.EVENT_KLINE_5MIN:
                topic = 'market.%s.kline.5min' % symbol
            elif eventType == MarketData.EVENT_KLINE_15MIN:
                topic = 'market.%s.kline.15min' % symbol
            elif eventType == MarketData.EVENT_KLINE_30MIN:
                topic = 'market.%s.kline.30min' % symbol
            elif eventType == MarketData.EVENT_KLINE_1HOUR:
                topic = 'market.%s.kline.60min' % symbol
            elif eventType == MarketData.EVENT_KLINE_4HOUR:
                topic = 'market.%s.kline.4hour' % symbol
            elif eventType == MarketData.EVENT_KLINE_1DAY:
                topic = 'market.%s.kline.1day' % symbol            
        
            if len(topic) <=0 or topic in self._dictSubscriptions.keys():
                return

            return self._subTopic(topic)

        # MarketData.EVENT_TICK here
        if symbol in self._dictTicks:
            return

        tick = mdTickData(self, symbol)
        tick.gatewayName = self.ident
        self._dictTicks[symbol] = tick

        self.subscribeMarketDepth(symbol)
        self.subscribeMarketDetail(symbol)

    #----------------------------------------------------------------------
    def subscribeMarketDepth(self, symbol,step=0):
        """订阅行情深度"""
        topic = 'market.%s.depth.step%d' %(symbol, step%6) # allowed step 0~5
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeTradeDetail(self, symbol):
        """订阅成交细节"""
        topic = 'market.%s.detail' %symbol
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeMarketDetail(self, symbol):
        """订阅市场细节"""
        topic = 'market.%s.detail' %symbol
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeKline(self, symbol,minutes=1):
        """订阅K线数据"""

        eventType = MarketData.EVENT_KLINE_1MIN

        minutes /=5
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_5MIN

        minutes /=3
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_15MIN

        minutes /=2
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_30MIN

        minutes /=2
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_1HOUR

        minutes /=4
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_4HOUR

        minutes /=6
        if minutes >0:
            eventType = MarketData.EVENT_KLINE_1DAY

        # minutes /=7
        # if minutes >0:
        #     eventType = MarketData.EVENT_KLINE_1Week

        # minutes /=4
        # if minutes >0:
        #     eventType = MarketData.EVENT_KLINE_1Mon

        # minutes /=12
        # if minutes >0:
        #     eventType = MarketData.EVENT_KLINE_1Year

        self._subTopic(symbol, eventType)

    #----------------------------------------------------------------------
    def pong(self, data):
        """响应心跳"""
        req = {'pong': data['ping']}
        self._sendReq(req)
    
    #----------------------------------------------------------------------
    def _sendReq(self, req):
        """发送请求"""
        stream = json.dumps(req)
        self.ws.send(stream)
        self.debug('SENT:%s' % stream)
        
    #----------------------------------------------------------------------
    def _doConnect(self):
        """重连"""
        self.ws = None
        proxyHost=None
        proxyPort=0
        if len(self._proxy) > 0:
            pos = self._proxy.find(':')
            if pos>=0:
                proxyHost=self._proxy[:pos]
                proxyPort=int(self._proxy[pos+1:])
            if len(proxyHost) <=0:
                proxyHost='localhost'

        try:
            if proxyHost :
                self.ws = create_connection(self.url, http_proxy_host =proxyHost, http_proxy_port=proxyPort)
                self.debug('CONN:%s thru {%s:%s}' % (self.url, proxyHost, proxyPort))
            else :
                self.ws = create_connection(self.url)
                self.debug('CONN:%s' % (self.url))

            return True

        except Exception as ex:
            msg = traceback.format_exc()
            self.onError(u'行情服务器重连失败：%s,%s' %(ex, msg))            
            return False
        
    #----------------------------------------------------------------------
    def _resubscribe(self):
        """重新订阅"""
        d = self._dictSubscriptions
        self._dictSubscriptions = {}
        for topic in d.keys():
            self._subTopic(topic)
        
    #----------------------------------------------------------------------
    @property
    def nextReqId(self) :
        self._reqid += 1
        return 'cseq-' + str(self._reqid)

    def _subTopic(self, topic):
        """订阅主题"""

        if topic in self._dictSubscriptions:
            return

        reqId = self.nextReqId
        req = {
            'sub': topic,
            'id' : reqId
        }

        if self.ws :
            self._sendReq(req)
        self._dictSubscriptions[topic] = reqId
    
    #----------------------------------------------------------------------
    def _doUnsubscribe(self, topic):
        """取消订阅主题"""

        reqId = self._dictSubscriptions[topic]

        req = {
            'unsub': topic,
            'id': reqId
        }

        self._sendReq(req)
        del self._dictSubscriptions[topic]
    
    #----------------------------------------------------------------------
    def onError(self, msg):
        """错误推送"""
        self.error(msg)
        
    #----------------------------------------------------------------------
    #

    def _onData(self, data):
        """数据推送"""
        self.debug('RECV:%s' % data)
        if 'ping' in data:
            self.pong(data)
            return

        if 'err-code' in data:
            self.onError(u'错误代码：%s, 信息：%s' %(data['err-code'], data['err-msg']))
            return

        if not 'ch' in data:
            return

        event = None
        ch = data['ch']
        if '.kline.' in ch:
            """K线数据
            RECV:{u'tick': {u'count': 49, u'vol': 37120.18320073684, u'high': 8123.28, u'amount': 4.569773683996716, u'low': 8122.46, u'close': 8123.16, u'open': 8122.46, u'id': 1533014820}, u'ch': u'market.btcusdt.kline.1min', u'ts': 1533014873302}            """
            pos = ch.find('.kline.')
            symbol =ch[len('market.'): pos]
            ch = ch[pos+7:]

            eventType = MarketData.EVENT_KLINE_1MIN
            if   '1min' == ch:
                eventType = MarketData.EVENT_KLINE_1MIN
            elif '5min' == ch:
                eventType = MarketData.EVENT_KLINE_5MIN
            elif '15min' == ch:
                eventType = MarketData.EVENT_KLINE_15MIN
            elif '30min' == ch:
                eventType = MarketData.EVENT_KLINE_30MIN
            elif '60min' == ch:
                eventType = MarketData.EVENT_KLINE_1HOUR
            elif '4hour' == ch:
                eventType = MarketData.EVENT_KLINE_4HOUR
            elif '1day' == ch:
                eventType = MarketData.EVENT_KLINE_1DAY

            tick = data['tick']
            asof = datetime.fromtimestamp(tick['id'])

            d =  {
                'stamp' : {
                    'id' : asof,
                    'count' : int(tick['count'])
                    },
                'tick' : tick
            }

            k = self.subscribeKey(symbol, eventType)

            if k in self._dictKLineLatest:
                latest = self._dictKLineLatest[k]
                if d['stamp']['id'] > latest['stamp']['id'] :
                    t = latest['tick']

                    edata = mdKLineData(self, symbol)
                    edata.open = t['open']
                    edata.close = t['close']
                    edata.high = t['high']
                    edata.low = t['low']
                    edata.volume = t['vol']

                    ts = latest['stamp']['id']
                    edata.date = ts.date().strftime('%Y%m%d')
                    edata.time = ts.time().strftime('%H:%M:%S')

                    event = Event(type_=eventType)
                    event.dict_['data'] = edata

            self._dictKLineLatest[k] = d
            if event:
                self.postMarketEvent(event)
            return
        # end of Kline

        # now combine marketDepth and marketDetail into tick
        tick = None
        tickReady = False
        if 'depth.step' in ch:
            symbol = data['ch'].split('.')[1]
            tick = self._dictTicks.get(symbol, None)
            if not tick:
                return
            tick.datetime = datetime.fromtimestamp(data['ts']/1000)
            tick.date = tick.datetime.strftime('%Y%m%d')
            tick.time = tick.datetime.strftime('%H:%M:%S.%f')

            bids = data['tick']['bids']
            for n in range(5):
                l = bids[n]
                tick.__setattr__('bidPrice' + str(n+1), l[0])
                tick.__setattr__('bidVolume' + str(n+1), l[1])

            asks = data['tick']['asks']
            for n in range(5):
                l = asks[n]
                tick.__setattr__('askPrice' + str(n+1), l[0])
                tick.__setattr__('askVolume' + str(n+1), l[1])

            if tick.lastPrice:
                 tickReady = True

        elif 'trade.detail' in ch:
            """成交细节推送
            {u'tick': {u'data': [{u'price': 481.93, u'amount': 0.1499, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405480484L}, {u'price': 481.94, u'amount': 0.2475, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405466973L}, {u'price': 481.97, u'amount': 6.3635, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405475106L}, {u'price': 481.98, u'amount': 0.109, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405468495L}, {u'price': 481.98, u'amount': 0.109, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405468818L}, {u'price': 481.99, u'amount': 6.3844, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405471868L}, {u'price': 482.0, u'amount': 0.6367, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405439802L}], u'id': 11837877646, u'ts': 1531119914439}, u'ch': u'market.ethusdt.trade.detail', u'ts': 1531119914494}
            {u'tick': {u'data': [{u'price': 481.96, u'amount': 0.109, u'direction': u'sell', u'ts': 1531119918505, u'id': 118378822907405482834L}], u'id': 11837882290, u'ts': 1531119918505}, u'ch': u'market.ethusdt.trade.detail', u'ts': 1531119918651}
            """
            pass
            # event = Event(type_=MarketData.EVENT_TICK)
            # event.dict_['data'] = data # TODO: covert the event format
        elif '.detail' in ch:
            """市场细节推送, 最近24小时成交量、成交额、开盘价、收盘价、最高价、最低价、成交笔数等
            RECV:{u'tick': {u'count': 124159, u'vol': 69271108.31560345, u'high': 465.03, u'amount': 151833.21684737998, u'version': 14324514537, u'low': 446.41, u'close': 451.07, u'open': 463.97, u'id': 14324514537}, u'ch': u'market.ethusdt.detail', u'ts': 1533015571033}
            """
            symbol = data['ch'].split('.')[1]
            tick = self._dictTicks.get(symbol, None)
            if not tick:
                return

            tick.datetime = datetime.fromtimestamp(data['ts']/1000)
            tick.date = tick.datetime.strftime('%Y%m%d')
            tick.time = tick.datetime.strftime('%H:%M:%S.%f')

            t = data['tick']
            tick.openPrice = t['open']
            tick.highPrice = t['high']
            tick.lowPrice = t['low']
            tick.lastPrice = t['close']
            tick.volume = t['vol']
            tick.preClosePrice = tick.openPrice

            if tick.bidPrice1:
                 tickReady = True

        if tick and tickReady :
            event = Event(type_=MarketData.EVENT_TICK)
            event.dict_['data'] = copy(tick)
            self.postMarketEvent(event)

