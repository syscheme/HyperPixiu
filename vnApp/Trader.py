# encoding: UTF-8

from __future__ import division

from .EventChannel import EventChannel, EventData, datetime2float
from .MainRoutine  import BaseApplication
from .MarketData   import MarketData
from .Account      import Account, Account_AShare, PositionData, TradeData, OrderData
from .strategies   import STRATEGY_CLASS
from .language     import text

import os
import logging
import json   # to save params
from collections import OrderedDict
from datetime import datetime, timedelta
from copy import copy
from abc import ABCMeta, abstractmethod
import traceback
import jsoncfg # pip install json-cfg
import shelve

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

########################################################################
class Trader(BaseApplication):
    """Trader Application"""

    FINISHED_STATUS = [OrderData.STATUS_ALLTRADED, OrderData.STATUS_REJECTED, OrderData.STATUS_CANCELLED]

    RUNTIME_TAG_TODAY = '$today'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""

        super(Trader, self).__init__(mainRoutine, settings)

        # 引擎类型为实盘
        # self._tradeType = TRADER_TYPE_TRADING
        self._settingfilePath = './temp/stgdata.dat'
        
        #--------------------
        # from old 数据引擎
        # 保存数据的字典和列表

        self._dictLatestTick = {}         # the latest tick of each symbol
        self._dictLatestKline1min = {}    #SSS the latest kline1min of each symbol
        self._dictLatestContract = {}
        self._dictLatestOrder = {}

        # inside of Account self._dictTrade = {}
        self._dictAccounts = {}
        self._defaultAccId = None
        self._dtData = None # datetime of data
        # inside of Account self._dictPositions= {}
        self._lstErrors = []

        self.debug('local data cache initialized')
        
        # 持仓细节相关
        # inside of Account self._dictDetails = {}                        # vtSymbol:PositionDetail
        self._lstTdPenalty = settings.tdPenalty       # 平今手续费惩罚的产品代码列表

        # 读取保存在硬盘的合约数据
        # TODO self.loadContracts()
        
        # 风控引擎实例（特殊独立对象）
        self._riskMgm = None

        #------from old ctaEngine--------------
        self._pathContracts = settings.pathContracts('./contracts')
        # 保存策略实例的字典
        # key为策略名称，value为策略实例，注意策略名称不允许重复
        self._dictStrategies = {}

        # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # value为包含所有相关strategy对象的list
        self._idxSymbolToStrategy = {}
        
        # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # key为vtOrderID，value为strategy对象
        self._idxOrderToStategy = {}
        self._idxStrategyToOrder = {}

        # # 本地停止单字典
        # # key为stopOrderID，value为stopOrder对象
        # self.stopOrderDict = {}             # 停止单撤销后不会从本字典中删除
        # self.workingStopOrderDict = {}      # 停止单撤销后会从本字典中删除
        
        # 成交号集合，用来过滤已经收到过的成交推送
        # inside of Account self.tradeSet = set()

        #------end of old ctaEngine--------------

        # 本地停止单编号计数
        self.stopOrderCount = 0
        # stopOrderID = STOPORDERPREFIX + str(stopOrderCount)

        # test hardcoding
        self.debug('adopting account')
        # TODO: instantiaze different account by type: accountClass = self._settings.account.type('Account')
        account = Account_AShare(self, self._settings.account)
        if account:
            self.adoptAccount(account)

        # step 1. collect all interested symbol of market data into _dictObjectives
        self.debug('building up security objectives')
        self._dictObjectives = {}

        #   part 1. those specified in the configuration
        for s in jsoncfg.expect_array(self._settings.objectives):
            try :
                symbol   = s.symbol('')
                dsTick   = s.dsTick('')
                ds1min   = s.ds1min('')
                if len(symbol) <=0 or (len(dsTick) <=0 and len(ds1min) <=0):
                    continue
                d = {
                    "dsTick"  : dsTick,
                    "ds1min"  : ds1min,
                    Trader.RUNTIME_TAG_TODAY : None,
                }

                self._dictObjectives[symbol] = d
            except:
                pass
        
        # TODO part 2. scan those in the positions of Accounts, and also put into _dictObjectives

        
    #----------------------------------------------------------------------
    # access to the Account
    #----------------------------------------------------------------------
    def adoptAccount(self, account, default=False):
        if not account:
            return
        
        self._dictAccounts[account.ident] = account
        if default or self._defaultAccId ==None:
            self._defaultAccId = account.ident

    @property
    def account(self): # the default account
        return self._dictAccounts[self._defaultAccId]

    @property
    def allAccounts(self):
        """获取所有帐号"""
        return self._dictAccounts.values()
        
    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ
        return super(Trader, self).init()

    @abstractmethod
    def start(self):

        self.debug('collected %s interested symbols, adopting strategies' % len(self._dictObjectives))
        self.strategies_LoadAll(self._settings.strategies)

        # step 1. subscribe all interested market data
        self.subscribeSymbols()

        self.account.onStart()

        # step 2. subscribe account events
        self.subscribeEvent(Account.EVENT_ORDER, self.eventHdl_Order)
        self.subscribeEvent(Account.EVENT_TRADE, self.eventHdl_Trade)

        self.subscribeEvent(EventChannel.EVENT_TIMER, self.eventHdl_OnTimer)

        # step 3. call allstrategy.onInit()
        self.strategies_Start()

    @abstractmethod
    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # TODO: subscribe all interested market data

        self.strategies_Stop()
        pass

    @abstractmethod
    def step(self):
        for a in self._dictAccounts.values():
            a.step()
    
    #----------------------------------------------------------------------
    # local the recent data by symbol
    #----------------------------------------------------------------------
    def getTick(self, vtSymbol):
        """查询行情对象"""
        try:
            return self._dictLatestTick[vtSymbol]
        except KeyError:
            return None        
    
    def getContract(self, vtSymbol):
        """查询合约对象"""
        try:
            return self._dictLatestContract[vtSymbol]
        except KeyError:
            return None
        
    def getAllContracts(self):
        """查询所有合约对象（返回列表）"""
        return self._dictLatestContract.values()
    
    def saveContracts(self):
        """保存所有合约对象到硬盘"""
        f = shelve.open(self._pathContracts)
        f['data'] = self._dictLatestContract
        f.close()
    
    def loadContracts(self):
        """从硬盘读取合约对象"""
        f = shelve.open(self._pathContracts)
        if 'data' in f:
            d = f['data']
            for key, value in d.items():
                self._dictLatestContract[key] = value
        f.close()
        
    #----------------------------------------------------------------------
    # about the orders
    #----------------------------------------------------------------------
    def getOrder(self, vtOrderID):
        """查询委托"""
        try:
            return self._dictLatestOrder[vtOrderID]
        except KeyError:
            return None
    
    def getAllWorkingOrders(self):
        """查询所有活动委托（返回列表）"""
        orders = []
        for acc in self._dictAccounts.values():
            orders.append(acc.getAllWorkingOrders())
        return orders

    def getAllOrders(self):
        """获取所有委托"""
        orders = []
        for acc in self._dictAccounts.values():
            orders.append(acc.getAllOrders())
        return orders
    
    def getAllTrades(self):
        """获取所有成交"""
        traders = []
        for acc in self._dictAccounts.values():
            traders.append(acc.getAllTrades())
        return traders
    
    #----------------------------------------------------------------------
    # about the positions
    #----------------------------------------------------------------------
    def getAllPositions(self):
        """获取所有持仓"""
        poslist = []
        for acc in self._dictAccounts.values():
            poslist.append(acc.getAllPositions())
        return poslist

    def getPositionDetail(self, symbol):
        """查询持仓细节"""
        poslist = []
        for acc in self._dictAccounts.values():
            poslist.append(acc.getPositionDetail(symbol))
        return poslist

    def getAllPositionDetails(self):
        """查询所有本地持仓缓存细节"""
        poslist = []
        for acc in self._dictAccounts.values():
            poslist.append(acc.getAllPositionDetails())
        return poslist
    
    #----------------------------------------------------------------------
    # Interested Events from EventChannel
    #----------------------------------------------------------------------
    def subscribeSymbols(self) :

        # subscribe the symbols
        self.subscribeEvent(MarketData.EVENT_TICK,       self.eventHdl_Tick)
        self.subscribeEvent(MarketData.EVENT_KLINE_1MIN, self.eventHdl_KLine1min)
        for k in self._dictObjectives.keys():
            s = self._dictObjectives[k]
            if len(s['dsTick']) >0:
                ds = self._engine.getMarketData(s['dsTick'])
                if ds:
                    self.debug('calling local subcribers for EVENT_TICK: %s' % s)
                    ds.subscribe(k, MarketData.EVENT_TICK)
                
            if len(s['ds1min']) >0:
                ds = self._engine.getMarketData(s['ds1min'])
                if ds:
                    self.debug('calling local subcribers for EVENT_KLINE_1MIN: %s' % s)
                    ds.subscribe(k, MarketData.EVENT_KLINE_1MIN)

    ### eventTick from MarketData ----------------
    @abstractmethod
    def eventHdl_KLine1min(self, event):
        """TODO: 处理行情推送"""
        d = event.dict_['data']
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        ds = tokens[1] if len(tokens) >1 else d.exchange
        if not symbol in self._dictObjectives or ds != self._dictObjectives[symbol]['ds1min']:
            return # ignore those not interested

        kline = copy(d)
        try:
            # 添加datetime字段
            if not kline.datetime:
                tmpstr = ' '.join([kline.date, kline.time])
                kline.datetime = datetime.strptime(tmpstr, '%Y%m%d %H:%M:%S')
                kline.datetime = datetime.strptime(tmpstr, '%Y%m%d %H:%M:%S.%f')
        except ValueError:
            self.error(traceback.format_exc())
            return

        if kline.datetime > (datetime.now() + timedelta(days=7)):
            self.warn('Trade-End signal kline(%s) received' % kline.desc)
            self.eventHdl_TradeEnd(event)
            return

        if self._dictObjectives[symbol][Trader.RUNTIME_TAG_TODAY] != kline.date:
            self.onDayOpen(symbol, kline.date)
            self._dictObjectives[symbol][Trader.RUNTIME_TAG_TODAY] = kline.date

        # step 1. cache into the latest, lnf DataEngine
        self._dtData = kline.datetime # datetime of data
        self._dictLatestKline1min[symbol] = kline

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        try:
            self.preStrategyByKLine(kline)
        except Exception, ex:
            self.error('eventHdl_KLine1min(%s) pre-strategy processing caught %s: %s' % (kline.desc, ex, traceback.format_exc()))
            return

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        execStgList = []
        if symbol in self._idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            for strategy in l:
                try:
                    self._stg_call(strategy, strategy.onBar, kline)
                    execStgList.append(strategy.id)
                except Exception, ex:
                    self.error('eventHdl_KLine1min(%s) [%s].onBar() caught %s: %s' % (kline.desc, strategy.id, ex, traceback.format_exc()))

        # step 4. 执行完策略后的的处理，通常为综合决策
        try:
            self.postStrategy(symbol)
        except Exception, ex:
            self.error('eventHdl_KLine1min(%s) post-strategy processing caught %s: %s' % (kline.desc, ex, traceback.format_exc()))

        self.debug('eventHdl_KLine1min(%s) processed: %s' % (kline.desc, execStgList))

    @abstractmethod
    def eventHdl_Tick(self, event):
        """处理行情推送"""
        d = event.dict_['data']
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        ds = tokens[1] if len(tokens) >1 else d.exchange
        if not symbol in self._dictObjectives or ds != self._dictObjectives[symbol]['ds1min']:
            return # ignore those not interested

        tick = copy(d)
        try:
            # 添加datetime字段
            if not tick.datetime:
                tmpstr = ' '.join([tick.date, tick.time])
                tick.datetime = datetime.strptime(tmpstr, '%Y%m%d %H:%M:%S')
                tick.datetime = datetime.strptime(tmpstr, '%Y%m%d %H:%M:%S.%f')
        except ValueError:
            self.error(traceback.format_exc())

        if tick.datetime > (datetime.now() + timedelta(days=7)):
            self.warn('Trade-End signal tick(%s) received' % tick.desc)
            self.eventHdl_TradeEnd(event)
            return

        if self._dictObjectives[symbol][Trader.RUNTIME_TAG_TODAY] != tick.date:
            self._dictObjectives[symbol][Trader.RUNTIME_TAG_TODAY] = tick.date
            self.onDayOpen(symbol, tick.date)

        # step 1. cache into the latest, lnf DataEngine
        self._dtData = tick.datetime # datetime of data
        self._dictLatestTick[symbol] = tick

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        self.debug('eventHdl_Tick(%s) pre-strategy processing' % tick.desc)
        self.preStrategyByTick(tick)

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        if symbol in self._idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            self.debug('eventHdl_Tick(%s) dispatching to %d strategies' % (tick.desc, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onTick, tick)
    
        # step 4. 执行完策略后的的处理，通常为综合决策
        self.debug('eventHdl_Tick(%s) post-strategy processing' % tick.desc)
        self.postStrategy(symbol)

        self.debug('eventHdl_Tick(%s) done' % tick.desc)

    def latestPrice(self, symbol) :
        kline = self._dictLatestKline1min.get(symbol, None)
        tick  = self._dictLatestTick.get(symbol, None)

        if kline and tick:
            if kline.datetime > tick.datetime:
                return kline.close
            else:
                return tick.price
        elif kline:
            return kline.close
        elif tick:
            return tick.price
        return 0

    @abstractmethod
    def eventHdl_TradeEnd(self, event):
        self.info('eventHdl_TradeEnd() quitting')
        self.mainRoutine.stop()
        # exit(0) # usualy exit() would be called in it to quit the program

    ### eventOrder from Account ----------------
    @abstractmethod
    def eventHdl_Order(self, event):
        """处理委托事件"""
        order = event.dict_['data']        

        # step 3. 逐个推送到策略实例中 lnf DataEngine
        if not order.brokerOrderId in self._idxOrderToStategy:
            return

        strategy = self._idxOrderToStategy[order.brokerOrderId]            
            
        # 如果委托已经完成（拒单、撤销、全成），则从活动委托集合中移除
        if order.status == self.FINISHED_STATUS:
            s = self._idxStrategyToOrder[strategy.id]
            if order.brokerOrderId in s:
                s.remove(order.brokerOrderId)
            
        self._stg_call(strategy, strategy.onOrder, order)
            
    ### eventTrade from Account ----------------
    @abstractmethod
    def eventHdl_Trade(self, event):
        """处理成交事件"""
        trade = event.dict_['data']
        
        # step 3. 将成交推送到策略对象中 lnf ctaEngine
        if trade.orderID in self._idxOrderToStategy:
            strategy = self._idxOrderToStategy[trade.orderID]
            self._stg_call(strategy, strategy.onTrade, trade)
            # 保存策略持仓到数据库
            # goes to Account now : self._stg_flushPos(strategy)

    ### generic events ??? ----------------
    def eventHdl_OnTimer(self, event):
        # edata = event.dict_['data']
        # self.debug('OnTimer() src[%s] %s' % (edata.sourceType, edata.datetime.strftime('%Y%m%dT%H:%M:%S')))
        # TODO: forward to account.onTimer()
        pass                      

    ### eventContract from ??? ----------------
    @abstractmethod
    def processContractEvent(self, event):
        """处理合约事件"""
        contract = event.dict_['data']

        # step 1. cache lnf DataEngine
        self._dictLatestContract[contract.vtSymbol] = contract
        self._dictLatestContract[contract.symbol] = contract       # 使用常规代码（不包括交易所）可能导致重复

    #----------------------------------------------------------------------
    @abstractmethod    # usually back test will overwrite this
    def onDayOpen(self, symbol, date):

        # step1. notify accounts
        # TODO: to support multiaccount: for acc in self._dictAccounts.values():
        self.debug('onDayOpen(%s) dispatching to account' % symbol)
        for acc in self._dictAccounts.values():
            try :
                acc.onDayOpen(date)
            except Exception as ex:
                self.logexception(ex)

        # step1. notify stategies
        if symbol in self._idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            self.debug('onDayOpen(%s) dispatching to %d strategies' % (symbol, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onDayOpen, date)

    @abstractmethod    # usually back test will overwrite this
    def preStrategyByKLine(self, kline):
        """收到行情后，在启动策略前的处理
        通常处理本地停止单（检查是否要立即发出）"""

        self.processStopOrdersByKLine(kline)
        pass

    @abstractmethod    # usually back test will overwrite this
    def postStrategy(self, symbol) :
        """执行完策略后的的处理，通常为综合决策"""
        pass

    @abstractmethod    # usually back test will overwrite this
    def preStrategyByTick(self, tick):
        """收到行情后，在启动策略前的处理
        通常处理本地停止单（检查是否要立即发出）"""

        self.processStopOrdersByTick(tick)
        pass

    # normal Trader cares StopOrders
    @abstractmethod
    def processStopOrdersByTick(self, tick):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        pass

    # normal Trader cares StopOrders
    @abstractmethod
    def processStopOrdersByKLine(self, kline):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        pass

    #----------------------------------------------------------------------
    #  Strategy methods
    #----------------------------------------------------------------------
    def strategies_LoadAll(self, settingList):
        """读取策略配置"""
        self.debug('loading all strategies')
        for s in jsoncfg.expect_array(settingList):
            self._stg_load(s)
            
        self.debug('loaded strategies: %s' % self._dictStrategies.keys())

    def strategies_List(self):
        """查询所有策略名称"""
        return self._dictStrategies.keys()        

    def strategies_Start(self):
        """全部初始化"""
        for n in self._dictStrategies.values():
            self._stg_start(n['strategy'])    

    def strategies_Stop(self):
        """全部停止"""
        for n in self._dictStrategies.values():
            self._stg_stop(n['strategy'])

    def strategies_Save(self):
        """保存策略配置"""
        with open(self._settingfilePath, 'w') as f:
            l = []
            
            for strategy in self._dictStrategies.values():
                setting = {}
                for param in strategy.paramList:
                    setting[param] = strategy.__getattribute__(param)
                l.append(setting)
            
            jsonL = json.dumps(l, indent=4)
            f.write(jsonL)

    def _stg_start(self, strategy):
        """启动策略"""
        if strategy.inited and not strategy.trading:
            self._stg_call(strategy, strategy.onStart)
            strategy.trading = True
    
    def _stg_stop(self, strategy):
        """停止策略"""
        if strategy.trading:
            self._stg_call(strategy, strategy.onStop)
            self._stg_call(strategy, strategy.cancellAll)
            strategy.trading = False

    def _stg_load(self, setting):
        """载入策略, setting schema:
            {
                "name" : "BBand", // strategy name equals to class name
                "symbols": ["ethusdt"],
                "weights": { // weights to affect decisions, in range of [0-100] each
                    "long" : 100, // optimisti
                    "short": 100, // pessimistic
                },

                // the following is up to the stategy class
            },
        """
        className = setting.name()

        # 获取策略类
        strategyClass = STRATEGY_CLASS.get(className, None)
        if not strategyClass:
            self.error(u'找不到策略类：%s' %className)
            return
        
        # 创建策略实例
        symbols = []
        for s in jsoncfg.expect_array(setting.symbols):
            symbol = s('')
            if len(symbol) <=0:
                continue
            if '*' == symbol:
                symbols = []
            symbols.append(symbol)
        if len(symbols) <=0:
            symbols = self._dictObjectives.keys()
        
        for s in symbols:
            strategy = strategyClass(self, s, self.account, setting)
            if strategy.id in self._dictStrategies:  # 防止策略重名
                self.error(u'策略实例重名：%s' %id)
                continue

            self._dictStrategies[strategy.id] = {
                'weights' : setting.weights({}),
                'strategy' : strategy
            }

            # 创建委托号列表
            self._idxStrategyToOrder[strategy.id] = set()

            # 保存Tick映射关系
            if s in self._idxSymbolToStrategy:
                l = self._idxSymbolToStrategy[s]
            else:
                l = []
                self._idxSymbolToStrategy[s] = l
            l.append(strategy)

            self._stg_call(strategy, strategy.onInit)
            strategy.inited = True
            self.info('initialized strategy[%s]' %strategy.id)

    def _stg_allVars(self, name):
        """获取策略当前的变量字典"""
        if name in self._dictStrategies:
            strategy = self._dictStrategies[name]
            varDict = OrderedDict()
            
            for key in strategy.varList:
                varDict[key] = strategy.__getattribute__(key)
            
            return varDict
        else:
            self.error(u'策略实例不存在：' + name)    
            return None
    
    def _stg_allParams(self, name):
        """获取策略的参数字典"""
        if name in self._dictStrategies:
            strategy = self._dictStrategies[name]
            paramDict = OrderedDict()
            
            for key in strategy.paramList:  
                paramDict[key] = strategy.__getattribute__(key)
            
            return paramDict
        else:
            self.error(u'策略实例不存在：' + name)    
            return None

    def _stg_call(self, strategy, func, params=None):
        """调用策略的函数，若触发异常则捕捉"""
        try:
            if params:
                func(params)
            else:
                func()
        except Exception:
            # 停止策略，修改状态为未初始化
            strategy.trading = False
            strategy.inited = False
            
            # 发出日志
            content = '\n'.join([u'策略%s触发异常已停止' %strategy.name,
                                traceback.format_exc()])
            self.error(content)
    #----------------------------------------------------------------------
    def ordersOfStrategy(self, strategyId, symbol=None):
        if not strategyId in self._idxStrategyToOrder :
            return []
        l = self._idxStrategyToOrder[strategyId]
        if not symbol:
            return l
        
        ret = []
        for o in l:
            if o.symbol == symbol:
                ret.append(o)
        return ret

    def postStrategyEvent(self, strategyId) :
        pass

