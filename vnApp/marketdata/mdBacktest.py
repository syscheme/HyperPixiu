# encoding: UTF-8

from __future__ import division

from ..MainRoutine import *
from ..Account import *
from ..MarketData import *
from ..EventChannel import *


from copy import copy
from datetime import datetime
from threading import Thread
from Queue import Queue, Empty
from multiprocessing.dummy import Pool
from time import sleep
from datetime import datetime

import json

# 如果安装了seaborn则设置为白色风格
# try:
#     import seaborn as sns       
#    sns.set_style('whitegrid')  
# except ImportError:
#    pass

# from vnpy.trader.vtConstant import *
# from vnpy.event import Event
# from vnpy.trader.vtGlobal import globalSetting
# from vnpy.trader.vtObject import VtTickData, VtBarData
# from vnpy.trader.vtConstant import *
# from vnpy.trader.vtGateway import VtOrderData, VtTradeData


########################################################################
class mdBacktest(MarketData):

    TICK_MODE = 'tick'
    BAR_MODE = 'bar'

    className = 'Backtest'
    displayName = 'Playback history data as MarketData'
    typeName = 'Market data subscriber from HHistDB'

    DUMMY_DT_EOS = datetime(2999, 12, 31, 23,59,59)
    DUMMY_DATE_EOS = DUMMY_DT_EOS.strftime('%Y%m%d')
    DUMMY_TIME_EOS = DUMMY_DT_EOS.strftime('%H%M%S')

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

        super(mdBacktest, self).__init__(mainRoutine, settings)

        # self._btApp = btApp
        self._main = mainRoutine
        self._dbPreffix = settings.sourceDBPreffix('')    # Prefix + "Tick or 1min"
        self._timerStep =  settings.timerStep(0)
        self._dateStart = settings.startDate('2000-01-01')
        self._dateEnd   = settings.endDate('2999-12-31')
        
        self._mode        = self.TICK_MODE
        self._t2k1min     = False
        mode = settings.mode('') # "mode": "tick,t2k1min"
        if 'kl1min' == mode[:4] :
            self._mode        = self.BAR_MODE
        elif 't2k1min' in mode:
            self._t2k1min = True
                
        # self._symbol = settings.symbol(A601005)
        self._stampLastTimer = 0
        self._timerStep = 10
        self._dictCursors = {} # symbol to dbCusor
        # self._dictCollectionNames = {} # symbol to CollectionNames

        self._eventEndOfData = []

    @property
    def exchange(self) :
        # for Backtest, we take self._exchange as the source exchange to read
        # and (self._exchange + MarketData.TAG_BACKTEST) as output exhcnage to post into eventChannel
        return self._exchange + MarketData.TAG_BACKTEST

    #----------------------------------------------------------------------
    def subscribe(self, symbol, eventType=None) :
        """订阅成交细节"""

        dataCategory = 'Tick'
        if len(eventType) >len(EVENT_NAME_PREFIX):
            dataCategory = eventType[len(EVENT_NAME_PREFIX):]

        collectionName = symbol +'.' + self._exchange
        key = dataCategory + '/' +collectionName
        if key in self._dictCursors:
            self.warn('subscribe() %s already exists' % key)
            return

        d = {
            'collectionName' : collectionName,
            'category': dataCategory,
            'cursor': None,
            'currentData': None
        }

        self._dictCursors[key] =d
        self.info('subscribe() %s added' % key)

    #----------------------------------------------------------------------
    def connect(self):
        """载入历史数据"""
        self._doConnect()

    #----------------------------------------------------------------------
    def _doConnect(self):
        """载入历史数据"""
        flt = {'date':{'$gte':self._dateStart}}   # 数据过滤条件
        if self._dateEnd:
            flt = {'date':{'$gte':self._dateStart, '$lte':self._dateEnd}}  

        for c in self._dictCursors.values() :
            if c['cursor']:
                continue # already connected

            dbName = self._dbPreffix + c['category']
            cltName = c['collectionName']

            collection = self._main.dbConn[dbName][cltName]
            self.debug('reading %s from %s with %s' % (cltName, dbName, flt))
            c['cursor'] = collection.find(flt).sort('datetime')

    #----------------------------------------------------------------------
    def step(self):

#        if not self._active :
#            return 0
        nleft = 20 - self._main._eventChannel.pendingSize
        if nleft <= 0 or len(self._dictCursors) <=0:
            return -3
        
        self._doConnect()

        cursorFocus = None
        c= 0
        while nleft >0 and len(self._dictCursors) >0:
            # scan the _dictCursors for the eariest data
            cursorsEnd = []
            focusContainer = []
            for d in self._dictCursors.values():

                if not d['cursor']: # ignore those collection disconnected
                    continue

                try :
                    if not d['currentData'] :
                        d['currentData'] = next(d['cursor'], None)
                except :
                    pass
                
                if not d['currentData']: # end of cursor
                    cursorsEnd.append(d)
                    continue
                
                # value = d['currentData']
                if not d['currentData']['datetime']:
                    dtstr = ' '.join([d['currentData']['date'], d['currentData']['time']])
                    d['currentData']['datetime'] = datetime.strptime(dtstr, '%Y%m%d %H:%M:%S.')

                # if not cursorFocus or not cursorFocus['currentData'] or cursorFocus['currentData']['datetime'] > d['currentData']['datetime']:
                #     cursorFocus = d
                if len(focusContainer)<=0 or not focusContainer[0]['currentData'] or focusContainer[0]['currentData']['datetime'] > d['currentData']['datetime']:
                    focusContainer = [d]
                    cursorFocus = focusContainer[0]
            
            for d in cursorsEnd:
                #fill a dummy END event into the event channel
                symbol = d['collectionName'].split('.')[0]
                newSymbol = '%s.%s' % (symbol, self.exchange)
                if d['category'] == 'Tick' :
                    edata = TickData(self.exchange, symbol)
                    event = Event(MarketData.EVENT_TICK)
                else: # as Kline
                    edata = KLineData(self.exchange, symbol)
                    event = Event(EVENT_NAME_PREFIX +d['category'])
                
                edata.date = self.DUMMY_DATE_EOS
                edata.time = self.DUMMY_TIME_EOS
                edata.datetime = self.DUMMY_DT_EOS
                edata.exchange  = self.exchange # output exchange name 
                edata.vtSymbol  = newSymbol

                event.dict_['data'] = edata
                self._eventEndOfData.append(event)
                cusrKey = d['category'] + '/' + d['collectionName']
                self.info('%s reached end, queued dummy Event(End), deleting from _dictCursors' % cusrKey)
                del self._dictCursors[cusrKey]
                nleft -=1

            if not cursorFocus or not cursorFocus['currentData'] : # none of the read found
                self.info('all data sequence reached end, flushing %d Event(End) to event channel' % len(self._eventEndOfData))
                for eos in self._eventEndOfData :
                    self.postMarketEvent(eos); c+=1

                self._eventEndOfData = []
                self._active = False
                break

            # cursorFocus['currentData'] is the most earilest record
            event = None
            symbol = cursorFocus['collectionName'].split('.')[0]
            newSymbol = '%s.%s' % (symbol, self.exchange)
            if cursorFocus['category'] == 'Tick' :
                edata = TickData(self.exchange, symbol)
                edata.__dict__ = copy(cursorFocus['currentData'])
                edata.vtSymbol  = newSymbol
                event = Event(MarketData.EVENT_TICK)
                event.dict_['data'] = edata
                if self._t2k1min :
                    pass # TODO: ...

            else: # as Kline
                edata = KLineData(self.exchange, symbol)
                edata.__dict__ = copy(cursorFocus['currentData'])
                edata.vtSymbol  = newSymbol
                event = Event(EVENT_NAME_PREFIX +cursorFocus['category'])
                event.dict_['data'] = edata

            # cursorFocus['currentData'] sent, force to None to read next
            dtData = cursorFocus['currentData']['datetime']
            cursorFocus['currentData'] = None
            nleft -=1 # decrease anyway each loop

            # post the event if valid
            if event:
                # event['data'].sourceType = MarketData.DATA_SRCTYPE_BACKTEST  # 数据来源类型
                self.postMarketEvent(event); c +=1

            # 向队列中模拟计时器事件
            stampData = datetime2float(dtData)
            if self._timerStep >0 and (self._stampLastTimer + self._timerStep) < stampData:
                edata = edTimer(dtData, self.exchange)
                event = Event(type_= EventChannel.EVENT_TIMER)
                event.dict_['data'] = edata
                self._stampLastTimer = stampData
                self._eventCh.put(event)

        return c
