# encoding: UTF-8

from __future__ import division

from vnpy.trader.vtObject import VtBarData
from vnpy.trader.vtConstant import *
from ..DataSubscriber import *

########################################################################
class dsHadax(DataSubscriber):
    
    """行情接口
    https://github.com/huobiapi/API_Docs/wiki/WS_request
    """
    HUOBI = 'huobi'
    HADAX = 'hadax'

    #----------------------------------------------------------------------
    # setting schema
    # {'exchange':HADAX, 'proxies':{'host': 'url' }} } 
    def __init__(self, settings):
        """Constructor"""

        super(dsHadax, self).__init__(settings)

        self.ws = None
        self.url = ''
        
        self._reqid = 0
        self.thread = Thread(target=self._run)

        if exchHost == self.HUOBI:
            hostname = HUOBI_API_HOST
        else:
            hostname = HADAX_API_HOST
        self.url = 'wss://%s/ws' % hostname

    #----------------------------------------------------------------------
    def _run(self):
        """执行连接 and receive"""
        while self._active:
            try:
                stream = self.ws.recv()
                result = zlib.decompress(stream, 47).decode('utf-8')
                data = json.loads(result)
                self._onData(data)
            except zlib.error:
                self.onError(u'数据解压出错：%s' %stream)
            except:
                self.onError('行情服务器连接断开')
                if self._doConnect() :
                    self.onError(u'行情服务器重连成功')
                    self._resubscribe()
                else:
                    self.onError(u'等待3秒后再次重连')
                    sleep(3)
    
    #----------------------------------------------------------------------
    def connect(self):
        """连接"""
        if not self._doConnect() :
            return False
            
        self._active = True
        self.thread.start()
            
        return self.active
        
    #----------------------------------------------------------------------
    def close(self):
        """停止"""
        if self._active:
            self._active = False
            self.thread.join()
            self.ws.close()
        
    #----------------------------------------------------------------------
    def subscribeMarketDepth(self, symbol,step=0):
        """订阅行情深度"""
        topic = 'market.%s.depth.step%d' %(symbol, step%6) # allowed step 0~5
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeTradeDetail(self, symbol):
        """订阅成交细节"""
        topic = 'market.%s.trade.detail' %symbol
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeMarketDetail(self, symbol):
        """订阅市场细节"""
        topic = 'market.%s.detail' %symbol
        self._subTopic(topic)
        
    #----------------------------------------------------------------------
    def subscribeKline(self, symbol,minutes=1):
        """订阅K线数据"""
        period ="1min"
        minutes /=5
        if minutes >0:
            period ="5min"

        minutes /=3
        if minutes >0:
            period ="15min"

        minutes /=2
        if minutes >0:
            period ="30min"

        minutes /=2
        if minutes >0:
            period ="60min"

        minutes /=4
        if minutes >0:
            period ="4hour"

        minutes /=6
        if minutes >0:
            period ="1day"

        minutes /=7
        if minutes >0:
            period ="1week"

        minutes /=4
        if minutes >0:
            period ="1mon"

        minutes /=12
        if minutes >0:
            period ="1year"

        topic = 'market.%s.kline.%s' %(symbol, period) # allowed { 1min, 5min, 15min, 30min, 60min, 4hour,1day, 1mon, 1week, 1year }
        self._subTopic(topic)

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
        
    #----------------------------------------------------------------------
    def _doConnect(self):
        """重连"""
        self.ws = None
        try:
            if self.proxyHost :
                self.ws = create_connection(self.url, http_proxy_host=self.proxyHost, http_proxy_port=self.proxyPort)
            else :
                self.ws = create_connection(self.url)

            return True

        except:
            msg = traceback.format_exc()
            self.onError(u'行情服务器重连失败：%s' %msg)            
            return False
        
    #----------------------------------------------------------------------
    def _resubscribe(self):
        """重新订阅"""
        d = self.subDict
        self.subDict = {}
        for key in d.keys():
            eventType, symbol= self.chopSubscribeKey(key)
            self._subTopic(symbol, eventType)
        
    #----------------------------------------------------------------------
    def _subTopic(self, symbol, eventType):
        """订阅主题"""

        key = self.subscribeKey(symbol, eventType)
        if key in self.subscriptions():
            return

        topic = 'market.%s.trade.detail' %symbol
        if   eventType == EVENT_TICK:
            topic = 'market.%s.trade.detail' %symbol
        elif eventType == EVENT_MARKET_DEPTH0:
            topic = 'market.%s.depth.step0' % symbol
        elif eventType == EVENT_KLINE_1min:
            topic = 'market.%s.kline.1min' % symbol
        elif eventType == EVENT_KLINE_5min:
            topic = 'market.%s.kline.5min' % symbol
        elif eventType == EVENT_KLINE_15min:
            topic = 'market.%s.kline.15min' % symbol
        elif eventType == EVENT_KLINE_30min:
            topic = 'market.%s.kline.30min' % symbol
        elif eventType == EVENT_KLINE_60min:
            topic = 'market.%s.kline.60min' % symbol
        elif eventType == EVENT_KLINE_4hour:
            topic = 'market.%s.kline.4hour' % symbol
        elif eventType == EVENT_KLINE_1day:
            topic = 'market.%s.kline.1day' % symbol

        self._reqid += 1
        req = {
            'sub': topic,
            'id': str(self._reqid)
        }
        self._sendReq(req)
        
        self.subDict[key] = str(self._reqid)+'>' + topic
    
    #----------------------------------------------------------------------
    def _doUnsubscribe(self, key):
        """取消订阅主题"""

        pos = self.subDict[key].find('>')
        self.subDict[key]
        id, topic = self.subDict[key][:pos], self.subDict[key][pos+1:]

        req = {
            'unsub': topic,
            'id': id
        }

        self._sendReq(req)
    
    #----------------------------------------------------------------------
    def onError(self, msg):
        """错误推送"""
        print (msg)
        
    #----------------------------------------------------------------------
    def _onData(self, data):
        """数据推送"""
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
        if 'depth.step' in ch:
            event = Event(type_=EVENT_MARKET_DEPTH)
            event.dict_['data'] = data # TODO: covert the event format
        elif 'kline' in ch:
            """K线数据
            {u'tick': {u'count': 103831, u'vol': 39127093.56251132, u'high': 494.99, u'amount': 80614.73642133494, u'version': 11838075621, u'low': 478.0, u'close': 482.08, u'open': 484.56, u'id': 11838075621}, u'ch': u'market.ethusdt.detail', u'ts': 1531120097994}
            """
            event = Event(type_=EVENT_KLINE)
            event.dict_['data'] = data # TODO: covert the event format, seperate the EVENT_KLINE_Xmin
        elif 'trade.detail' in ch:
            """成交细节推送
            {u'tick': {u'data': [{u'price': 481.93, u'amount': 0.1499, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405480484L}, {u'price': 481.94, u'amount': 0.2475, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405466973L}, {u'price': 481.97, u'amount': 6.3635, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405475106L}, {u'price': 481.98, u'amount': 0.109, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405468495L}, {u'price': 481.98, u'amount': 0.109, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405468818L}, {u'price': 481.99, u'amount': 6.3844, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405471868L}, {u'price': 482.0, u'amount': 0.6367, u'direction': u'buy', u'ts': 1531119914439, u'id': 118378776467405439802L}], u'id': 11837877646, u'ts': 1531119914439}, u'ch': u'market.ethusdt.trade.detail', u'ts': 1531119914494}
            {u'tick': {u'data': [{u'price': 481.96, u'amount': 0.109, u'direction': u'sell', u'ts': 1531119918505, u'id': 118378822907405482834L}], u'id': 11837882290, u'ts': 1531119918505}, u'ch': u'market.ethusdt.trade.detail', u'ts': 1531119918651}
            """
            event = Event(type_=EVENT_TICK)
            event.dict_['data'] = data # TODO: covert the event format
        elif 'detail' in ch:
            """市场细节推送, 最近24小时成交量、成交额、开盘价、收盘价、最高价、最低价、成交笔数等
            {u'tick': {u'count': 103831, u'vol': 39127093.56251132, u'high': 494.99, u'amount': 80614.73642133494, u'version': 11838075621, u'low': 478.0, u'close': 482.08, u'open': 484.56, u'id': 11838075621}, u'ch': u'market.ethusdt.detail', u'ts': 1531120097994}
            """
            event = Event(type_=EVENT_TICK)
            event.dict_['data'] = data # TODO: covert the event format

        # post the event if valid
        if event:
            self.postMarketEvent(event)
    
