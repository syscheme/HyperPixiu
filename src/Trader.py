# encoding: UTF-8
'''
Trader maps to the agent in OpenAI/Gym
'''
from __future__ import division

from EventData    import EventData, datetime2float
from MarketData   import *
from Application  import BaseApplication
from Account      import Account, Account_AShare, PositionData, TradeData, OrderData
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

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

########################################################################
class MetaTrader(BaseApplication):
    '''defines the common interace of a Trader'''
    FINISHED_STATUS = [OrderData.STATUS_ALLTRADED, OrderData.STATUS_REJECTED, OrderData.STATUS_CANCELLED]
    RUNTIME_TAG_TODAY = '$today'

    def __init__(self, program, **kwargs) :
        super(MetaTrader, self).__init__(program, **kwargs)
        self._account = None
        self._accountId = None
        self._dtData = None # datetime of data
        self._dictObjectives = {}

    def __deepcopy__(self, other):
        result = object.__new__(type(self))
        result.__dict__ = copy(self.__dict__)
        result._dictObjectives = deepcopy(self._dictObjectives)
        return result

    @property
    def account(self): return self._account # the default account
    @abstractmethod
    def eventHdl_Order(self, event): raise NotImplementedError
    @abstractmethod
    def eventHdl_Trade(self, event): raise NotImplementedError
    @abstractmethod
    def onDayOpen(self, symbol, date): raise NotImplementedError
    @abstractmethod
    def proc_MarketEvent(self, ev): raise NotImplementedError

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
    def __init__(self, program, **kwargs):
        """Constructor"""

        super(BaseTrader, self).__init__(program, **kwargs)

        # 引擎类型为实盘
        # self._tradeType = TRADER_TYPE_TRADING

        self._marketstate = None
        self._accountId      = self.getConfig('accountId', self._accountId)

        #--------------------
        # from old 数据引擎

        # moved to MarketState: self._dictLatestTick = {}         # the latest tick of each symbol
        # moved to MarketState: self._dictLatestKline1min = {}    #SSS the latest kline1min of each symbol
        # self._dictLatestContract = {}

        # inside of Account self._dictTrade = {}
        # inside of Account self._dictPositions= {}

        self.debug('local data cache initialized')
        
        # 持仓细节相关
        # inside of Account self._dictDetails = {}                        # vtSymbol:PositionDetail
        # self._lstTdPenalty = settings.tdPenalty       # 平今手续费惩罚的产品代码列表

        # 读取保存在硬盘的合约数据
        # TODO self.loadContracts()
        
        # 风控引擎实例（特殊独立对象）
        self._riskMgm = None

        #------from old ctaEngine--------------
        self._pathContracts = self.dataRoot + 'contracts'

        # # 本地停止单字典
        # # key为stopOrderID，value为stopOrder对象
        # self.stopOrderDict = {}             # 停止单撤销后不会从本字典中删除
        # self.workingStopOrderDict = {}      # 停止单撤销后会从本字典中删除
        
        # 成交号集合，用来过滤已经收到过的成交推送
        # inside of Account self.tradeSet = set()

        # 本地停止单编号计数
        self.stopOrderCount = 0
        # stopOrderID = STOPORDERPREFIX + str(stopOrderCount)

        # #   part 1. those specified in the configuration
        # for s in jsoncfg.expect_array(self._settings.objectives):
        #     try :
        #         symbol   = s.symbol('')
        #         dsTick   = s.dsTick('')
        #         ds1min   = s.ds1min('')
        #         if len(symbol) <=0 or (len(dsTick) <=0 and len(ds1min) <=0):
        #             continue
        #         d = {
        #             "dsTick"  : dsTick,
        #             "ds1min"  : ds1min,
        #             Trader.RUNTIME_TAG_TODAY : None,
        #         }

        #         self._dictObjectives[symbol] = d
        #     except:
        #         pass
        
        
    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(BaseTrader, self).doAppInit() :
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

        # TODO part 2. scan those in the positions of Accounts, and also put into _dictObjectives

        # step 2. associate the marketstate
        if not self._marketstate :
            self._marketstate = self._account.marketState

        if not self._marketstate :
            for obsId in self._program.listByType(MarketState) :
                marketstate = self._program.getObj(obsId)
                if marketstate and marketstate.exchange == self._account.exchange:
                    self._marketstate = marketstate
                    break
                
        if not self._marketstate :
            self.error('no MarketState found')
            return False

        self.info('taking MarketState[%s]' % self._marketstate.ident)

        # step 3. subscribe the market events
        self.subscribeEvent(EVENT_TICK)
        self.subscribeEvent(EVENT_KLINE_1MIN)

        if self._marketstate :
            for symbol in self._dictObjectives.keys():
                self._marketstate.addMonitor(self, symbol)

        # step 4. subscribe account events
        self.subscribeEvent(Account.EVENT_ORDER)
        self.subscribeEvent(Account.EVENT_TRADE)

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

        if MARKETDATE_EVENT_PREFIX == ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            if self._marketstate:
                self._marketstate.updateByEvent(ev)

            d = ev.data
            tokens = (d.vtSymbol.split('.'))
            symbol = tokens[0]
            ds = tokens[1] if len(tokens) >1 else d.exchange
            if not symbol in self._dictObjectives.keys() : # or ds != self._dictObjectives[symbol]['ds1min']:
                return # ignore those not interested

            if d.asof > (datetime.now() + timedelta(days=7)):
                self.warn('Trade-End signal received: %s' % d.desc)
                self.eventHdl_TradeEnd(ev)
                return

            objective = self._dictObjectives[symbol]
            #  objective['ohlc'] = self.updateOHLC(objective['ohlc'] if 'ohlc' in objective.keys() else None, kline.open, kline.high, kline.low, kline.close)

            if not objective['date'] or d.date > objective['date'] :
                self.onDayOpen(symbol, d.date)
                objective['date'] = d.date
                # objective['ohlc'] = self.updateOHLC(None, d.open, d.high, d.low, d.close)

            # step 1. cache into the latest, lnf DataEngine
            if not self._dtData or d.asof > self._dtData:
                self._dtData = d.asof # datetime of data

            # step 2. 收到行情后，在启动策略前的处理
            # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
            try:
                self.proc_MarketEvent(ev)
            except Exception as ex:
                self.error('proc_MarketEvent %s caught %s: %s' % (ev.desc, ex, traceback.format_exc()))
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

    # # --- eventContract from ??? ----------------
    # @abstractmethod
    # def processContractEvent(self, event):
    #     """处理合约事件"""
    #     contract = event.data

    #     # step 1. cache lnf DataEngine
    #     self._dictLatestContract[contract.vtSymbol] = contract
    #     self._dictLatestContract[contract.symbol] = contract       # 使用常规代码（不包括交易所）可能导致重复

    #----------------------------------------------------------------------
    # usually back test will overwrite this
    def onDayOpen(self, symbol, date):
        # step1. notify accounts
        self.debug('onDayOpen(%s) dispatching to account' % symbol)
        if self._account :
            try :
                self._account.onDayOpen(date)
            except Exception as ex:
                self.logexception(ex)

    def proc_MarketEvent(self, ev):
        '''processing an incoming MarketEvent'''
        pass

    # end of event handling
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # local the recent data by symbol
    #----------------------------------------------------------------------
    # def getTick(self, vtSymbol):
    #     """查询行情对象"""
    #     try:
    #         return self._dictLatestTick[vtSymbol]
    #     except KeyError:
    #         return None        
    
    # def getContract(self, vtSymbol):
    #     """查询合约对象"""
    #     try:
    #         return self._dictLatestContract[vtSymbol]
    #     except KeyError:
    #         return None
        
    # def getAllContracts(self):
    #     """查询所有合约对象（返回列表）"""
    #     return self._dictLatestContract.values()
    
    # def saveContracts(self):
    #     """保存所有合约对象到硬盘"""
    #     f = shelve.open(self._pathContracts)
    #     f['data'] = self._dictLatestContract
    #     f.close()
    
    # def loadContracts(self):
    #     """从硬盘读取合约对象"""
    #     f = shelve.open(self._pathContracts)
    #     if 'data' in f:
    #         d = f['data']
    #         for key, value in d.items():
    #             self._dictLatestContract[key] = value
    #     f.close()
        
    #----------------------------------------------------------------------
    # about the orders
    #----------------------------------------------------------------------
    # def getOrder(self, vtOrderID):
    #     """查询委托"""
    #     try:
    #         return self._dictLatestOrder[vtOrderID]
    #     except KeyError:
    #         return None
    
    # def getAllWorkingOrders(self):
    #     """查询所有活动委托（返回列表）"""
    #     return self._account.getAllWorkingOrders()

    # def getAllOrders(self):
    #     """获取所有委托"""
    #     orders = []
    #     for acc in self._dictAccounts.values():
    #         orders.append(acc.getAllOrders())
    #     return orders
    
    # def getAllTrades(self):
    #     """获取所有成交"""
    #     traders = []
    #     for acc in self._dictAccounts.values():
    #         traders.append(acc.getAllTrades())
    #     return traders
    
    #----------------------------------------------------------------------
    # about the positions
    #----------------------------------------------------------------------
    # def getAllPositions(self):
    #     """获取所有持仓"""
    #     poslist = []
    #     for acc in self._dictAccounts.values():
    #         poslist.append(acc.getAllPositions())
    #     return poslist

    # def getPositionDetail(self, symbol):
    #     """查询持仓细节"""
    #     poslist = []
    #     for acc in self._dictAccounts.values():
    #         poslist.append(acc.getPositionDetail(symbol))
    #     return poslist

    # def getAllPositionDetails(self):
    #     """查询所有本地持仓缓存细节"""
    #     poslist = []
    #     for acc in self._dictAccounts.values():
    #         poslist.append(acc.getAllPositionDetails())
    #     return poslist
    
    # def getOHLC(self, symbol):
    #     """查询所有本地持仓缓存细节"""
    #     if symbol in self._dictObjectives.keys():
    #         return self._dictObjectives[symbol]['ohlc']
        
    #     return None
    #----------------------------------------------------------------------
    # --- eventTick from MarketData ----------------
    def updateOHLC(self, OHLC, open, high, low, close):
        if not OHLC:
            return (open, high, low, close)
        
        oopen, ohigh, olow, oclose = OHLC
        return (oopen, high if high>ohigh else ohigh, low if low<olow else olow, close)

    # def latestPrice(self, symbol) :
    #     kline = self._dictLatestKline1min.get(symbol, None)
    #     tick  = self._dictLatestTick.get(symbol, None)

    #     if kline and tick:
    #         if kline.datetime > tick.datetime:
    #             return kline.close
    #         else:
    #             return tick.price
    #     elif kline:
    #         return kline.close
    #     elif tick:
    #         return tick.price
    #     return 0

