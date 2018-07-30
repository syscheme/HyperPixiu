########################################################################
class DataSubscriber(object):

    #----------------------------------------------------------------------
    def __init__(self, eventChannel, settings):
        """Constructor"""

        self._eventCh = eventChannel

        self._active = False
        self.subDict = {}
        
        self.proxies = {}
    
    #----------------------------------------------------------------------
    @property
    def active(self):
        return self._active

    @property
    def subscriptions(self):
        return self.subDict.keys()
        
    #----------------------------------------------------------------------
    @abstractmethod
    def connect(self):
        """连接"""
        raise NotImplementedError
        return self.active
        
    #----------------------------------------------------------------------
    @abstractmethod
    def close(self):
        """停止"""
        if self._active:
            self._active = False

        raise NotImplementedError
        
    #----------------------------------------------------------------------
    def subscribeKey(self, symbol, eventType):
        key = '%s>%s' %(eventType, symbol)
        return key

    #----------------------------------------------------------------------
    # return eventType, symbol
    def chopSubscribeKey(self, key):
        pos = key.find('>')
        return key[:pos], key[pos+1:]

    #----------------------------------------------------------------------
    @abstractmethod
    def subscribe(self, symbol, eventType =EVENT_TICK):
        """订阅成交细节"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    def unsubscribe(self, symbol, eventType):
        """取消订阅主题"""
        key = self.subscribeKey(symbol, eventType)
        if key not in self.subDict:
            return

        self._doUnsubscribe(self, key)
        del self.subDict[topic]

    #----------------------------------------------------------------------
    @abstractmethod
    def subscribeTick(self, symbol):
        """订阅成交细节"""
        topic = 'market.%s.trade.detail' %symbol
        self.subTopic(topic)

    #----------------------------------------------------------------------
    @abstractmethod
    def subscribeMarketDepth(self, symbol,step=0):
        """订阅行情深度"""
        raise NotImplementedError
        
    #----------------------------------------------------------------------
    @abstractmethod
    def subscribeKline(self, symbol, minutes=1):
        """订阅K线数据"""
        raise NotImplementedError
        
    #----------------------------------------------------------------------
    @abstractmethod
    def onError(self, msg):
        """错误推送"""
        print (msg)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def onMarketDepth(self, data):
        """行情深度推送 
        sample:
        {u'ch': u'market.ethusdt.depth.step0', u'ts': 1531119204038, u'tick': {	u'version': 11837097362, u'bids': [
            [481.2, 6.9387], [481.18, 1.901], [481.17, 5.0], [481.02, 0.96], [481.0, 4.9474], [480.94, 9.537], [480.93, 3.5159], [480.89, 1.0], [480.81, 2.06], [480.8, 30.2504], [480.72, 0.109], [480.64, 0.06], 		[480.63, 1.0], [480.61, 0.109], [480.6, 0.4899], [480.58, 1.9059], [480.56, 0.06], [480.5, 21.241], [480.49, 1.1444], [480.46, 1.0], [480.44, 2.4982], [480.43, 1.0], [480.41, 0.1875], [480.35, 1.0637], 		...		
            [494.01, 0.05], [494.05, 0.231], [494.26, 50.3659], [494.45, 0.2889]
        ]}}
        """
        if self._eventCh ==None:
            return

        event = Event(type_=EVENT_MARKET_DEPTH)

        # TODO: covert the event format
        event.dict_['data'] = data
        self._eventCh.put(event)
    
    #----------------------------------------------------------------------
    @abstractmethod
    def postMarketEvent(self, event):
        if self._eventCh ==None:
            return

        self._eventCh.put(event)
    