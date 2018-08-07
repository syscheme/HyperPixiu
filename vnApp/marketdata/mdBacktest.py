# encoding: UTF-8

from __future__ import division

from vnApp.MarketData import *

from vnpy.trader.vtConstant import *
from vnpy.event import Event

from copy import copy
from datetime import datetime
from threading import Thread
from Queue import Queue, Empty
from multiprocessing.dummy import Pool
from time import sleep

import json

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

from vnpy.trader.vtGlobal import globalSetting
from vnpy.trader.vtObject import VtTickData, VtBarData
from vnpy.trader.vtConstant import *
from vnpy.trader.vtGateway import VtOrderData, VtTradeData

from ..MainRoutine import *
from ..Account import *

########################################################################
class mdBacktest(MarketData):

    className = 'Backtest'
    displayName = 'Playback history data as MarketData'
    typeName = 'Market data subscriber from HHistDB'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor
            setting schema:
                {
                    "id": "backtest",
                    "source": "Backtest",

                    // the following is up to the MarketData class
                    "sourceDBPreffix": "dr1min",
                    "exchange": "huobi", // the original exchange
                    "events": {
                        "tick": "True", // to trigger tick events
                    }
                },
        """

        super(mdBacktest, self).__init__(mainRoutine, settings, MarketData.DATA_SRCTYPE_BACKTEST)

        # self._btApp = btApp
        self._main = mainRoutine
        self._dbConn = self._main._dbConn
        self._dbPreffix = settings.sourceDBPreffix    # Prefix + "Tick or 1min"
        self._exchange =  settings.exchange('')
        
        self._mode = self.TICK_MODE
        self._t2k1min = False
        mode = settings.mode('') # "mode": "tick,t2k1"
        if 'kl1min' == mode[:4] :
            self._mode = self.BAR_MODE
            self._dbName = self._dbPreffix + '1min'
        else:
            self._mode = self.TICK_MODE
            self._dbName = self._dbPreffix + 'Tick'
            if 't2k1min' in mode:
                self._t2k1min = True 

#        self._symbol = settings.symbol(A601005)
        self._dbCursor = None
        self._initData =[]

    #----------------------------------------------------------------------
    def subscribe(self, symbol, eventType=None) :
        """订阅成交细节"""

        self._collectionName = symbol +'.' + self._exchange

    #----------------------------------------------------------------------
    def connect(self):
        """载入历史数据"""

        self._collection = self._dbConn[self._dbName][self._collectionName]          
      
        # 首先根据回测模式，确认要使用的数据类
        if self.mode == self.TICK_MODE:
            dataClass = VtTickData
            func = self.OnNewTick
        else:
            dataClass = VtBarData
            func = self.OnNewBar

        self.log('reading %s from %s' % (self._collectionName, self._dbName))
        return cRows
        
    #----------------------------------------------------------------------
    def step(self):

        # 载入回测数据
        flt = {'datetime':{'$gte':self.strategyStartDate}}   # 数据过滤条件
        if self.dataEndDate:
            flt = {'datetime':{'$gte':self.strategyStartDate,
                               '$lte':self.dataEndDate}}  

        self._dbCursor = collection.find(flt).sort('datetime')
        reachedEnd = False

        eventyType_= MarketData.EVENT_TICK
        if not 'Tick' in self._dbName :
            eventyType_=  MarketData.EVENT_1MIN
            #TODO: more

        while self._active and self._eventCh: # check if there are too many event pending on eventChannel
            if self._eventCh.pendingSize >100:
                sleep(1)
                continue
            
            # read a batch then post to the event channel
            for i in range(1, 10) :
                try :
                    if not self._dbCursor.hasNext() :
                        self._active = False
                        reachedEnd = True
                        break

                    d = self._dbCursor.next()

                    event = None
                    if eventType == MarketData.EVENT_TICK :
                        edata = mdTickData(self)
                        edata.__dict__ = d
                        edata.vtSymbol  = edata.symbol = self._symbol
                        event = Event(eventType)
                        event.dict_['data'] = edata
                    else: # as Kline
                        edata = mdKLineData(self)
                        edata.__dict__ = d
                        edata.vtSymbol  = edata.symbol = self._symbol
                        event = Event(eventType)
                        event.dict_['data'] = edata

                    # post the event if valid
                    if event:
                        event['data'].sourceType = MarketData.DATA_SRCTYPE_BACKTEST  # 数据来源类型
                        self.postMarketEvent(event)

                except Exception as ex:
                    pass

        #fill a STOP event into the event channel
        if reachedEnd:
            if eventType == MarketData.EVENT_TICK :
                edata = mdTickData(self)
                edata.date ='39991231'
                edata.vtSymbol  = edata.symbol = self._symbol
                event = Event(eventType)
                event.dict_['data'] = edata
            else: # as Kline
                edata = mdKLineData(self)
                edata.date ='39991231'
                edata.vtSymbol  = edata.symbol = self._symbol
                event = Event(eventType)
                event.dict_['data'] = edata
            
            event['data'].sourceType = MarketData.DATA_SRCTYPE_BACKTEST  # 数据来源类型
            self.postMarketEvent(event)

