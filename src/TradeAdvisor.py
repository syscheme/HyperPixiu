# encoding: UTF-8
'''
Trader maps to the agent in OpenAI/Gym
'''
from __future__ import division

from EventData    import EventData, datetime2float, EVENT_NAME_PREFIX
from MarketData   import *
from Perspective  import PerspectiveState
from Application  import BaseApplication, BOOL_STRVAL_TRUE
from Account      import OrderData

# event type
EVENT_ADVICE      = EVENT_NAME_PREFIX + 'TAdv'  # 交易建议事件
ADVICE_DIRECTIONS = [OrderData.DIRECTION_NONE, OrderData.DIRECTION_LONG, OrderData.DIRECTION_SHORT]

import os
import logging
import json   # to save params
from collections import OrderedDict
from datetime import datetime, timedelta
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
import traceback
import numpy as np

########################################################################
class TradeAdvisor(BaseApplication):
    '''TradeAdvisor
        TradeAdvisor observe the marketState and perform evaluation and prediction
        then publish its AdviceData into the EventChannel, which could be referenced
        by the Traders who subscribe the AdviceData. The latter may issue orders thru
        Accounts
    '''
    def __init__(self, program, recorder =None, objectives =None, **kwargs) :
        super(TradeAdvisor, self).__init__(program, **kwargs)

        self._recorder = recorder
        self.__dictAdvices = {} # dict of symbol to recent AdviceData
        self.__dictPerf = {} # dict of symbol to performance {'Rdaily':,'Rdstd':}

        self._minimalAdvIntv = self.getConfig('minimalInterval', 5) # minimal interval in seconds between two advice s
        self._exchange = self.getConfig('exchange', 'AShare')
        self.__recMarketEvent = self.getConfig('recMarketEvent', 'False').lower() in BOOL_STRVAL_TRUE
        self._enableMStateSS  = self.getConfig('enableMarketStateSafeStore', 'False').lower() in BOOL_STRVAL_TRUE # False to skip saving/restore the safestore of marketState, which is expected by BackTest/OfflineSimulator

        if not objectives or not isinstance(objectives, list) or len(objectives) <=0:
            objectives = self.getConfig('objectives', [])

        if isinstance(objectives, list):
            for o in objectives:
                self.__dictAdvices[o]=None

        self._marketState = PerspectiveState(self._exchange) # take PerspectiveState by default
        self.__stampMStateRestored, self.__stampMStateSaved = None, None
        # try :
        #     shutil.rmtree(self.__wkTrader.outdir)
        # except:
        #     pass

        # try :
        #     os.makedirs(self.__wkTrader.outdir)
        # except:
        #     pass
        self.program.setShelveFilename('%s%s.sobj' % (self.dataRoot, self.ident))

    @property
    def marketState(self): return self._marketState # the default account
    
    @property
    def objectives(self): return list(self.__dictAdvices.keys())

    @property
    def recorder(self): return self._recorder

    def __saveMarketState(self) :
        if not self._enableMStateSS : return
        try :
            self.program.saveObject(self.marketState, '%s/marketState' % 'OnlineSimulator')
        except Exception as ex:
            self.logexception(ex)

    def __restoreMarketState(self) :
        if not self._enableMStateSS : return None
        try :
            return self.program.loadObject('%s/marketState' % 'OnlineSimulator') # '%s/marketState' % self.__class__)
        except Exception as ex:
            self.logexception(ex)
        return None

    # @abstractmethod
    # def onDayOpen(self, symbol, date): raise NotImplementedError

    @abstractmethod
    def generateAdviceOnMarketEvent(self, mdEvent):
        '''processing an incoming MarketEvent and generate an advice'''
        return None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(TradeAdvisor, self).doAppInit() :
            return False

        # step 2. associate the marketstate
        prevState = self.__restoreMarketState()
        if prevState:
            self._marketState = prevState
            self.__stampMStateRestored = datetime.now()
            self.info('doAppInit() previous market state restored: %s' % self._marketState.descOf(None))

        if not self._marketState :
            for obsId in self._program.listByType(MarketState) :
                marketstate = self._program.getObj(obsId)
                if marketstate and marketstate.exchange == self._exchange:
                    self._marketState = marketstate
                    break

            if not self._marketState :
                self.error('no MarketState found')
                return False

            self.info('taking MarketState[%s]' % self._marketState.ident)

        if len(self.__dictAdvices) <=0:
            sl = self._marketState.listOberserves()
            # for symbol in sl:
            #     self.__dictAdvices[symbol] = AdviceData(self.ident, symbol, self._marketState.exchange)

        if self._marketState :
            for symbol in self.objectives:
                self._marketState.addMonitor(symbol)

        self.subscribeEvent(EVENT_TICK)
        self.subscribeEvent(EVENT_KLINE_1MIN)
        self.subscribeEvent(EVENT_KLINE_5MIN)
        self.subscribeEvent(EVENT_KLINE_1DAY)
        self.subscribeEvent(EVENT_MONEYFLOW_1MIN)
        self.subscribeEvent(EVENT_MONEYFLOW_1DAY)

        if self._recorder :
            self._recorder.registerCategory(EVENT_ADVICE,         params= {'columns' : AdviceData.COLUMNS})

            self._recorder.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
            self._recorder.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

        return True

    def doAppStep(self):
        if not super(TradeAdvisor, self).doAppInit() :
            return False

        saveInterval = timedelta(minutes=10)
        #TODO: increase the saveInterval if it is off-hours
        stampNow = datetime.now()
        today = stampNow.strftime('%Y-%m-%d')
            
        if not self.__stampMStateSaved:
            self.__stampMStateSaved = stampNow
            
        if stampNow - self.__stampMStateSaved > saveInterval:
            self.__stampMStateSaved = stampNow
            self.__saveMarketState()

    def OnEvent(self, ev):
        '''
        dispatch the event
        '''
        if MARKETDATE_EVENT_PREFIX == ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            if self._marketState:
                self._marketState.updateByEvent(ev)

            d = ev.data
            tokens = (d.vtSymbol.split('.'))
            symbol = tokens[0]
            ds = tokens[1] if len(tokens) >1 else d.exchange
            if not symbol in self.objectives :
                return # ignore those not interested

            if d.asof > (datetime.now() + timedelta(days=7)):
                self.warn('Trade-End signal received: %s' % d.desc)
                self.eventHdl_TradeEnd(ev)
                return

            latestAdvc = self.__dictAdvices[symbol] if symbol in self.objectives else None
            if latestAdvc :
                elapsed = datetime2float(d.datetime) - datetime2float(latestAdvc.asof)
                if (elapsed < self._minimalAdvIntv) :
                    return # too frequently

            # if not latestAdvc.asof['date'] or d.date > objective['date'] :
            #     self.onDayOpen(symbol, d.date)
            #     objective['date'] = d.date
            #     # objective['ohlc'] = self.updateOHLC(None, d.open, d.high, d.low, d.close)

            # step 2. # call each registed procedure to handle the incoming MarketEvent
            newAdvice = None
            try:
                newAdvice = self.generateAdviceOnMarketEvent(ev)
            except Exception as ex:
                self.error('call generateAdviceOnMarketEvent %s caught %s: %s' % (ev.desc, ex, traceback.format_exc()))

            if not newAdvice:
                return

            newAdvice.dirString() # generate the dirString to ease reading
            
            newAdvice.advisorId = '%s@%s' %(self.ident, self.program.hostname)
            newAdvice.datetime  = d.asof
            if not newAdvice.exchange or len(newAdvice.exchange)<=0 or '_k2x' == newAdvice.exchange:
                newAdvice.exchange = self._marketState.exchange
            
            if symbol in self.__dictPerf.keys():
                perf = self.__dictPerf[symbol]
                newAdvice.Rdaily = perf['Rdaily']
                newAdvice.Rdstd  = perf['Rdstd']
            
            if latestAdvc:
                newAdvice.pdirLONG  = latestAdvc.dirLONG
                newAdvice.pdirSHORT = latestAdvc.dirSHORT
                newAdvice.pdirNONE  = latestAdvc.dirNONE
                newAdvice.pdirPrice = latestAdvc.price
                newAdvice.pdirAsOf  = latestAdvc.datetime
            
            evAdv = Event(EVENT_ADVICE)
            evAdv.setData(newAdvice)
            self.postEvent(evAdv)
            self.__dictAdvices[symbol] = newAdvice

            if self.__recMarketEvent :
                self._recorder.pushRow(ev.type, d)
            self._recorder.pushRow(EVENT_ADVICE, newAdvice)

            return

    # end of BaseApplication routine
    #----------------------------------------------------------------------


########################################################################
class AdviceData(EventData):
    '''交易建议'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime,symbol,exchange,advisorId,price,dirNONE,dirLONG,dirSHORT,strDir,Rdaily,Rdstd'
    DIRSTR = ['NONE','LONG','SHORT']

    def __init__(self, advisorId, symbol, exchange):
        """Constructor"""
        self.advisorId   = EventData.EMPTY_STRING  # who issued this advice
        self.symbol      = symbol
        self.exchange    = exchange
        self.price       = EventData.EMPTY_FLOAT   # 最新价格
        self.dirNONE     = EventData.EMPTY_FLOAT   # 不操作权重
        self.dirLONG     = EventData.EMPTY_FLOAT   # 买入向权重
        self.dirSHORT    = EventData.EMPTY_FLOAT   # 卖出向权重
        self.strDir      = EventData.EMPTY_STRING  # to ease reading
        self.Rdaily      = EventData.EMPTY_FLOAT   # advisor的历史日均收益率
        self.Rdstd       = EventData.EMPTY_FLOAT   # advisor的历史日均收益率标准差，用以衡量advice可信度/风险

        # 前次advice, 与self.dirXXX比较可用于识别advice变化
        self.pdirNONE     = EventData.EMPTY_FLOAT   # 前次不操作权重
        self.pdirLONG     = EventData.EMPTY_FLOAT   # 前次买入向权重
        self.pdirSHORT    = EventData.EMPTY_FLOAT   # 前次卖出向权重
        self.pdirPrice    = EventData.EMPTY_FLOAT   # 前次价格
        self.pdirAsOf     = None                    # 前次Advice datetime

    def dirString(self) :
        if not self.strDir or len(self.strDir) <=0 :
            self.dirNONE, self.dirLONG, self.dirSHORT = round(self.dirNONE,4), round(self.dirLONG,4), round(self.dirSHORT,4)
            dirIdx = np.argmax([self.dirNONE,self.dirLONG,self.dirSHORT])
            self.strDir = AdviceData.DIRSTR[dirIdx]
        return self.strDir

    @property
    def desc(self) :
        dirIdx = np.argmax([self.dirNONE,self.dirLONG,self.dirSHORT])
        return 'tadv.%s@%s>%s@%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S'), self.strDir, round(self.price,2))

