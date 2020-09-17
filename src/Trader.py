# encoding: UTF-8
'''
Trader maps to the agent in OpenAI/Gym
'''
from __future__ import division

from EventData    import EventData, datetime2float
from MarketData   import *
from Application  import BaseApplication, configToStrList
from Account      import *
from TradeAdvisor import *

'''
from .MarketData   import MarketData
from .strategies   import STRATEGY_CLASS
from .language     import text
'''

import os
import logging
import json   # to save params
from collections import OrderedDict
from datetime import datetime, timedelta
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
import traceback

# from pymongo import MongoClient, ASCENDING
# from pymongo.errors import ConnectionFailure

########################################################################
class MetaTrader(BaseApplication):
    '''defines the common interace of a Trader'''
    FINISHED_STATUS = [OrderData.STATUS_ALLTRADED, OrderData.STATUS_REJECTED, OrderData.STATUS_CANCELLED]
    RUNTIME_TAG_TODAY = '$today'

    def __init__(self, program, account=None, recorder =None, **kwargs) :
        super(MetaTrader, self).__init__(program, **kwargs)
        self._account   = account
        self._accountId = account._id if account else None
        self._dtData = None # datetime of data
        self._dictObjectives = {}
        self._marketState = None
        self._recorder =recorder
        self._latestCash, self._latestPosValue =0.0, 0.0
        self._maxBalance =0.0
        self.__outDir  = self.getConfig('outDir', os.path.join(super(MetaTrader, self).outdir, 'Tdr.P%s/' % self.program.pid) )
        if self.__outDir and '/' != self.__outDir[-1]: self.__outDir +='/'

    def __deepcopy__(self, other):
        result = object.__new__(type(self))
        result.__dict__ = copy(self.__dict__)
        result._dictObjectives = deepcopy(self._dictObjectives)
        return result

    @property
    def outdir(self) : return self.__outDir # replace that of BaseApplication 

    @property
    def account(self): return self._account # the default account
    @property
    def marketState(self): return self._marketState # the default account
    @property
    def recorder(self): return self._recorder
    @property
    def objectives(self): return list(self._dictObjectives.keys())

    @abstractmethod
    def eventHdl_Order(self, event): raise NotImplementedError
    @abstractmethod
    def eventHdl_Trade(self, event): raise NotImplementedError
    @abstractmethod
    def eventHdl_DayOpen(self, symbol, date): raise NotImplementedError

    # start from BaseTrader, trader becomes advice-driven
    # # @abstractmethod
    # def OnAdvice(self, evAdvice): raise NotImplementedError

    def openObjective(self, symbol):
        if not symbol in self._dictObjectives.keys() :
            self._dictObjectives[symbol] = {
                'date' : None
            }

        return self._dictObjectives[symbol]

########################################################################
class BaseTrader(MetaTrader):
    '''BaseTrader Application'''

     #----------------------------------------------------------------------
    def __init__(self, program, objectives=[], account=None, recorder =None, **kwargs):
        """Constructor"""

        super(BaseTrader, self).__init__(program, account, recorder, **kwargs)

        # 引擎类型为实盘
        # self._tradeType = TRADER_TYPE_TRADING
        if not self._accountId:
            self._accountId      = self.getConfig('accountId', self._accountId)
        self._annualCostRatePcnt = self.getConfig('annualCostRatePcnt', 10) # the annual cost rate of capital time, 10% by default
        self._maxValuePerOrder   = self.getConfig('maxValuePerOrder', 0) # the max value limitation of a single order
        self._minBuyPerOrder     = self.getConfig('minBuyPerOrder', 1000.0) # the min value limitation of a single buy
        if not objectives or len(objectives) <=0 :
            objectives  = self.getConfig('objectives', [])
        
        objectives = configToStrList(objectives)

        self._minBuyPerOrder   = self.getConfig('minBuyPerOrder', 1000.0) # the min value limitation of a single buy
        
        for symbol in objectives:
            self.openObjective(symbol)

        # 持仓细节相关
        # self._lstTdPenalty = settings.tdPenalty       # 平今手续费惩罚的产品代码列表

        # 读取保存在硬盘的合约数据
        # TODO self.loadContracts()
        
        # 风控引擎实例（特殊独立对象）
        self._riskMgm = None

        self.debug('local data cache initialized')

        #------from old ctaEngine--------------
        self._pathContracts = self.dataRoot + 'contracts'

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(BaseTrader, self).doAppInit() :
            return False

        if len(self.objectives) <=0:
            return False

        # step 1. find and adopt the account
        if not self._account :
            # find the Account from the program
            if not self._accountId :
                self._accountId = ''

            searchKey = '.%s' % self._accountId
            for appId in self._program.listApps(Account) :
                pos = appId.find(searchKey)
                if self._accountId == appId or pos >0 and appId[pos:] == searchKey:
                    self._account = self._program.getApp(appId)
                    if self._account : 
                        self._accountId = self._account.ident

        if not self._account :
            self.error('no account adopted')
            return False

        self._account.hostTrader(self)

        # TODO part 2. scan those in the positions of Accounts, and also put into _dictObjectives

        # step 2. associate the marketstate
        # if not self._marketState :
        #     self._marketState = self._account.marketState

        if not self._marketState :
            for obsId in self._program.listByType(MarketState) :
                marketstate = self._program.getObj(obsId)
                if marketstate and marketstate.exchange == self._account.exchange:
                    self._marketState = marketstate
                    break
        elif not self._marketState.exchange:
            self._marketState._exchange = self._account.exchange
                
        if not self._marketState :
            self.warn('no existing MarketState found, creating a new PerspectiveState instead')
            self._marketState = PerspectiveState(self._account.exchange)

        self._account._marketState = self._marketState
        self.info('taking MarketState[%s], registering objectives' % self._marketState.ident)
        for symbol in self._dictObjectives.keys():
            self._marketState.addMonitor(symbol)

        # step 3.1 subscribe the TradeAdvices
        self.subscribeEvents([EVENT_ADVICE, EVENT_TICK_OF_ADVICE])
        # in order to receive EVENT_TICK_OF_ADVICE, the placehold of EVENT_TICK is MUST in self._marketState
        enlarged = []
        for s in self._marketState.listOberserves():
            _, maxTicks = self._marketState.sizesOf(s, EVENT_TICK)
            if maxTicks <=0:
                self._marketState.resize(s, EVENT_TICK, 60)
                enlarged.append(s)

        if len(enlarged) >0:
            self.warn('reserved EVENT_TICK(60) in MarketState[%s]: %s' % (self._marketState.ident, ','.join(enlarged)))
        
        # step 3.2 subscribe the account and market events
        self.subscribeEvents([Account.EVENT_ORDER,Account.EVENT_TRADE])
        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_MONEYFLOW_1MIN])
        self.subscribeEvents([EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, EVENT_MONEYFLOW_1DAY])

        if self._recorder :
            self._recorder.registerCategory(Account.RECCATE_ORDER,       params= {'columns' : OrderData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_TRADE,       params= {'columns' : TradeData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYPOS,    params= {'columns' : DailyPosition.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYRESULT, params= {'columns' : DailyResult.COLUMNS})

        return True

    def doAppStep(self):
        super(BaseTrader, self).doAppStep()

    def OnEvent(self, ev):
        '''
        dispatch the event
        '''
        if Account.EVENT_ORDER == ev.type:
            return self.eventHdl_Order(ev)
        if Account.EVENT_TRADE == ev.type:
            return self.eventHdl_Trade(ev)

        if EVENT_SYS_CLOCK == ev.type :
            if self._account:
                self._account.OnEvent(ev)
            return

        if EVENT_TICK_OF_ADVICE == ev.type :
            d = copy(ev.data)
            ev = Event(EVENT_TICK)
            ev.setData(d)
            self.debug('OnEvent(%s) treating as: %s' % (EVENT_TICK_OF_ADVICE, ev.desc))

        d = ev.data

        if MARKETDATE_EVENT_PREFIX == ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            tokens = (d.vtSymbol.split('.'))
            symbol = tokens[0]
            ds = tokens[1] if len(tokens) >1 else d.exchange

            if self._marketState and self._marketState.updateByEvent(ev) :
                self.debug('updated marketState by ev[%s]-> %s' % (ev.desc, self._marketState.descOf(symbol)))

            if not symbol in self._dictObjectives.keys() : # or ds != self._dictObjectives[symbol]['ds1min']:
                return # ignore those not interested

            if d.asof > (datetime.now() + timedelta(days=7)):
                self.warn('Trade-End signal received: %s' % d.desc)
                self.eventHdl_TradeEnd(ev)
                return

            objective = self._dictObjectives[symbol]
            #  objective['ohlc'] = self.updateOHLC(objective['ohlc'] if 'ohlc' in objective.keys() else None, kline.open, kline.high, kline.low, kline.close)

            if not objective['date'] or d.date > objective['date'] :
                try :
                    self.eventHdl_DayOpen(symbol, d.date)
                except Exception as ex:
                    self.logexception(ex)
                objective['date'] = d.date
                # objective['ohlc'] = self.updateOHLC(None, d.open, d.high, d.low, d.close)

            # step 1. cache into the latest, lnf DataEngine
            if not self._dtData or d.asof > self._dtData:
                self._dtData = d.asof # datetime of data
            
            return

        if EVENT_ADVICE == ev.type :
            # step 2. # call each registed procedure to handle the incoming MarketEvent
            try:
                self.OnAdvice(ev)
            except Exception as ex:
                self.error('call OnAdvice %s caught %s: %s' % (ev.desc, ex, traceback.format_exc()))

            return

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # about the event handling
    # --- eventOrder from Account ------------
    def eventHdl_Order(self, ev):
        """处理委托事件"""
        pass
            
    def eventHdl_Trade(self, ev):
        """处理成交事件"""
        pass

    #----------------------------------------------------------------------
    # usually back test will overwrite this
    def eventHdl_DayOpen(self, symbol, date):
        # step1. notify accounts
        self.debug('eventHdl_DayOpen(%s:%s) dispatching to account' % (symbol, date))
        if self._account :
            self._account.onDayOpen(date)

    def OnAdvice(self, evAdvice):
        '''
        processing an TradeAdvice, this basic DnnTrader takes whatever the advice told

        Take an action (buy/sell/hold) and computes the immediate reward.
        @param action (numpy.array): Action to be taken, one-hot encoded.
        @returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        if not evAdvice or EVENT_ADVICE != evAdvice.type :
            return

        adv = evAdvice.data
        symbol = adv.symbol
        if not self.marketState.exchange in adv.exchange or not symbol in self.objectives:
            self.debug('OnAdvice() ignored that unattended of exchange[%s]: %s' % (adv.exchange, adv.desc))
            return

        if not self._account.executable:
            self.debug('OnAdvice() ignored[%s] per account not yet executable on %s: %s' % (adv.desc, symbol, self.marketState.descOf(symbol)))
            return

        # TODO: validate the advice's datetime
        # TODO: perform the risk management of advice

        dirToExec = ADVICE_DIRECTIONS[np.argmax([adv.dirNONE, adv.dirLONG, adv.dirSHORT])]

        strExec =''
        self.debug('OnAdvice() processing advDir[%s] %s' % (dirToExec, adv.desc))
        if OrderData.DIRECTION_NONE == dirToExec:
            return

        prevCap = self._latestCash + self._latestPosValue

        # step 1. collected information from the account
        cashAvail, cashTotal, positions = self._account.positionState()
        _, posvalue = self._account.summrizeBalance(positions, cashTotal)
        capitalBeforeStep = cashTotal + posvalue

        # TODO: the first version only support one symbol to play, so simply take the first symbol in the positions        
        latestPrice, asofP = self._marketState.latestPrice(symbol)
        buyPrice  = self._account.roundByPriceTick(latestPrice, OrderData.DIRECTION_LONG)
        sellPrice = self._account.roundByPriceTick(latestPrice, OrderData.DIRECTION_SHORT)

        maxBuy, maxSell = self._account.maxOrderVolume(symbol, buyPrice)
        if self._maxValuePerOrder >0:
            if self._maxValuePerOrder < (maxBuy * buyPrice*100):
                maxBuy = int(maxBuy * self._maxValuePerOrder / (maxBuy* buyPrice*100))
            if self._maxValuePerOrder < (maxSell*1.5 * sellPrice*100):
                maxSell = int(maxSell * self._maxValuePerOrder / (maxSell * sellPrice*100))
        if self._minBuyPerOrder >0 and (maxBuy * buyPrice*100) < self._minBuyPerOrder :
            maxBuy =0

        # TODO: the first version only support FULL-BUY and FULL-SELL
        if OrderData.DIRECTION_LONG == dirToExec :
            strExec = '%s:%dx%so%s' %(dirToExec, maxBuy, buyPrice, latestPrice)
            if maxBuy <=0 :
                dirExeced = OrderData.DIRECTION_NONE
                self.debug('OnAdvice() advice[%s] dropped per no buy-power: %s' % (adv.desc, strExec))
            else:
                self.info('OnAdvice() issuing max%s on advice[%s]' % (strExec, adv.desc))
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_BUY, buyPrice, maxBuy, reason=adv.desc)
                dirExeced = OrderData.DIRECTION_LONG

        elif OrderData.DIRECTION_SHORT == dirToExec :
            strExec = '%s:%dx%so%s' %(dirToExec, maxSell, sellPrice, latestPrice)
            if maxSell <=0:
                dirExeced = OrderData.DIRECTION_NONE
                self.debug('OnAdvice() advice[%s] dropped per no sell-power: %s' % (adv.desc, strExec))
            else:
                self.info('OnAdvice() issuing max%s on advice[%s]' % (strExec, adv.desc))
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_SELL, sellPrice, maxSell, reason=adv.desc)
                dirExeced = OrderData.DIRECTION_SHORT

        # step 3. calculate the rewards
        self._latestCash, self._latestPosValue = self._account.summrizeBalance() # most likely the cashAmount changed due to comission
        capitalAfterStep = self._latestCash + self._latestPosValue

        # instant_pnl = capitalAfterStep - capitalBeforeStep
        # self._total_pnl += instant_pnl

    # end of event handling
    #----------------------------------------------------------------------

    # --- eventTick from MarketData ----------------
    def updateOHLC(self, OHLC, open, high, low, close):
        if not OHLC:
            return (open, high, low, close)
        
        oopen, ohigh, olow, oclose = OHLC
        return (oopen, high if high>ohigh else ohigh, low if low<olow else olow, close)



