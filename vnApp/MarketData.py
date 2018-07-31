# encoding: UTF-8

from __future__ import division

from vnpy.trader.vtConstant import *
from vnpy.trader.vtObject import VtBarData, VtTickData

########################################################################
class MarketData(object):
    # Market相关events
    EVENT_TICK = 'eTick.'                   # TICK行情事件，可后接具体的vtSymbol
    EVENT_MARKET_DEPTH0 = 'eMD0.'           # Market depth0
    EVENT_MARKET_DEPTH2 = 'eMD2.'           # Market depth2
    EVENT_KLINE_1MIN    = 'eKL1m.'
    EVENT_KLINE_5MIN    = 'eKL5m.'
    EVENT_KLINE_15MIN   = 'eKL15m.'
    EVENT_KLINE_30MIN   = 'eKL30m.'
    EVENT_KLINE_1HOUR   = 'eKL1h.'
    EVENT_KLINE_4HOUR   = 'eKL4h.'
    EVENT_KLINE_1DAY    = 'eKL1d.'

    DATA_SRCTYPE_REALTIME   = 'market'
    DATA_SRCTYPE_IMPORT     = 'import'
    DATA_SRCTYPE_BACKTEST   = 'backtest'

    from abc import ABCMeta, abstractmethod

    #----------------------------------------------------------------------
    def __init__(self, eventChannel, settings, srcType=DATA_SRCTYPE_REALTIME):
        """Constructor"""

        self._eventCh = eventChannel
        self._sourceType = srcType

        self._active = False
        self.subDict = {}
        
        self.proxies = {}
    
    #----------------------------------------------------------------------
    @property
    def active(self):
        return self._active

    @property
    def subscriptions(self):
        return self.subDict
        
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

        self.doUnsubscribe(key)
        del self.subDict[key]

    #----------------------------------------------------------------------
    @abstractmethod
    def doUnsubscribe(self, key):
        """取消订阅主题"""
        raise NotImplementedError

    #----------------------------------------------------------------------
    @abstractmethod
    def onError(self, msg):
        """错误推送"""
        print (msg)
        
    #----------------------------------------------------------------------
    @abstractmethod
    def postMarketEvent(self, event):
        if self._eventCh ==None:
            return

        self._eventCh.put(event)
    
 
########################################################################
class mdTickData(VtTickData):
    """Tick行情数据类"""

    #----------------------------------------------------------------------
    def __init__(self, md):
        """Constructor"""
        super(mdTickData, self).__init__()
        
        self.exchange   = md._exchange
        self.sourceType = md._sourceType          # 数据来源类型
    
########################################################################
class mdKLineData(VtBarData):
    """K线数据"""

    #----------------------------------------------------------------------
    def __init__(self, md):
        """Constructor"""
        super(mdKLineData, self).__init__()
        
        self.exchange   = md._exchange
        self.sourceType = md._sourceType          # 数据来源类型


########################################################################
class BackTestData(MarketData):

    #----------------------------------------------------------------------
    def __init__(self, eventChannel, settings):
        """Constructor"""

        super(mdHuobi, self).__init__(eventChannel, dbConn, settings, DATA_SRCTYPE_BACKTEST)

        self._dbConn = dbConn
        self._dbName = setting.dbNamePrefix + "Tick or 1min"
        self._symbol = ???
        self._dbCursor= ???
        self._initData =[]

        self.ws = None
        self.url = ''
        self._dictCh = {}
        
        self._reqid = 0
        self.thread = Thread(target=self._run)
        self._exchange = settings.exchange('')
        if self._exchange == self.HADAX:
            hostname = HADAX_API_HOST
        else:
            hostname = HUOBI_API_HOST
            self._exchange = self.HUOBI
        
        self.url = 'wss://%s/ws' % hostname
        self._proxy = settings.proxy('')

    #----------------------------------------------------------------------
    def start(self):
        """载入历史数据"""
        if self
        dbName = _dbNamePrefix
        collection = self._dbConn[self._dbName][self._symbol]          
        self.stdout(u'开始载入数据 %s on %s/%s' % (self.symbol, self.dbHost, self.dbName))
      
        # 首先根据回测模式，确认要使用的数据类
        if self.mode == self.BAR_MODE:
            dataClass = VtBarData
            func = self.OnNewBar
        else:
            dataClass = VtTickData
            func = self.OnNewTick

        # 载入初始化需要用的数据
        flt = {'datetime':{'$gte':self.dataStartDate,
                           '$lt':self.strategyStartDate}}        
        initCursor = collection.find(flt).sort('datetime')
        
        # 将数据从查询指针中读取出，并生成列表
        self.initData = []              # 清空initData列表
        for d in initCursor:
            data = dataClass()
            data.__dict__ = d
            self._initData.append(data)      
        
        # 载入回测数据
        if not self.dataEndDate:
            flt = {'datetime':{'$gte':self.strategyStartDate}}   # 数据过滤条件
        else:
            flt = {'datetime':{'$gte':self.strategyStartDate,
                               '$lte':self.dataEndDate}}  
        self._dbCursor = collection.find(flt).sort('datetime')
        
        cRows = initCursor.count() + self._dbCursor.count()
        self.stdout(u'载入完成，数据量：%s' %cRows)
        return cRows
        
    #----------------------------------------------------------------------
    def _run(self):

        while self._active and self._eventCh:
            if self._eventCh.pendingSize >100:
                sleep(1)
                continue
            
        for i in range(1,10) :
            if not self._dbCursor.hasNext() :
                self._active = False
                return

            d = self._dbCursor.next()
            # for d in self.dbCursor:
            data = dataClass()
            data.__dict__ = d

            event = None

            if 'Tick' in self._dbName :
                edata = mdTickData(self)
                edata.vtSymbol = edata.symbol = symbol
                edata.lastPrice = tick['close']
                edata.lastVolume = tick['vol']
                edata.openPrice = tick['open']
                edata.highPrice = tick['high']
                edata.lowPrice = tick['low']

                edata.date = ts.date().strftime('%Y%m%d')
                edata.time = ts.time().strftime('%H:%M:%S.%3f')[:-3]

                event = Event(type_=MarketData.EVENT_TICK)
                event.dict_['data'] = edata
            else:
                    edata = mdKLineData(self)
                    edata.vtSymbol   = edata.symbol = symbol
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

            # post the event if valid
            if event:
                event['data'].sourceType = self._sourceType  # 数据来源类型
                self.postMarketEvent(event)
