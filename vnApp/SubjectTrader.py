# encoding: UTF-8

from __future__ import division

from .EventChannel import EventChannel, EventData, datetime2float
from .MainRoutine  import BaseApplication
from .MarketData   import MarketData, TickToKLineMerger
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
class Position(Object):
    """持仓数据类"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        # 持仓相关
        self.direction      = EventData.EMPTY_STRING      # 持仓方向
        self.position       = EventData.EMPTY_INT         # 持仓量
        self.posAvail       = EventData.EMPTY_INT         # 冻结数量
        self.price          = EventData.EMPTY_FLOAT       # 持仓最新交易价
        self.avgPrice       = EventData.EMPTY_FLOAT       # 持仓均价
        self.stampByTrader  = EventData.EMPTY_INT         # 该持仓数是基于Trader的计算
        self.stampByBroker  = EventData.EMPTY_INT        # 该持仓数是基于与broker的数据同步

########################################################################
class OHLCV(Object):
    """MarketData"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        # 持仓相关
        self.open         = EventData.EMPTY_FLOAT
        self.high         = EventData.EMPTY_FLOAT
        self.low          = EventData.EMPTY_FLOAT
        self.close        = EventData.EMPTY_FLOAT
        self.volume       = EventData.EMPTY_FLOAT

########################################################################
class SubjectTrader(BaseApplication):
    """Trader Application based on
        1) one objective symbol
        2) one cash and quota
        3) one strategy (may be an aggregated strategy consists of mutiple stratgies but the aggregated strategy must be respond to weighting and deciding)
             maps to the configuration nodes under account/subjects

        "account": { // account settings
            "id": "3733421",
            "broker": "huobi",

            // the following is up to the broker driver
            "httpproxy": "localhost:8118",
            "accessKey": "867bedde-0f59abba-cb645ccf-25714",
            "secretKey": "c8694e94-7289bc81-6466c65e-87af9",

            "subjects" : {
                "eosusdt" : {
                    "initialCapital": 10000, // the initial capital
                    "strategy": {
                        "name": BBand", // take the BollingBand stategy
                        }
                    "marketEvent": "KL1min", // which market data to subscribe or monitor: tick or KL1min
                }
            }
        },

    """

    FINISHED_STATUS = [OrderData.STATUS_ALLTRADED, OrderData.STATUS_REJECTED, OrderData.STATUS_CANCELLED]

    RUNTIME_TAG_TODAY = '$today'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, brokerAccount, symbol, settings):
        """Constructor"""

        super(SubjectTrader, self).__init__(mainRoutine, settings)

        # 
        self._brokerAccount   = brokerAccount
        self._symbol          = symbol
        self._id              = '%s/%s' % (brokerAccount.ident, self._symbol)
        self._settings        = settings

        self._latestTick      = None         # the latest tick of each symbol
        self._latestKline1min = None    #SSS the latest kline1min of each symbol
        self._dtMarketData    = None # datetime of data
        self._latestContract  = None
        self._strategy        = None  # the strategy
        self._stampLastSave   = None  # the s

        self._cashTotal       = 0
        self._cashAvailable   = 0

        # about the subject position according to the symbol
        self._position       = Position()       # 持仓量
        self._dayOHLCV       = OHLCV()          # jiao

        self._paramfilePath = '%s/st_%s' % (self._brokerAccount.dataPath, self._id)

    #----------------------------------------------------------------------
    # properties
    #----------------------------------------------------------------------
    @property
    def account(self): # the broker account
        return self._brokerAccount

    @property
    def symbol(self): return self._symbol

    @property
    def strategy(self): return self._strategy

    """查询行情对象"""
    @property
    def latestTick(self): return self._latestTick

    @property
    def latestKline1min(self): return self._latestKline1min

    @property
    def latestKline1min(self): return self._latestKline1min

    @property
    def getOHLCV(self): return self._dayOpen, self._dayHigh, self._dayLow, self._price, self._volume

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ

        savedParams = self.loadParameters()
        if savedParams == None :
            self.debug('parameters[%s] not pre-exist, initializing the trader' % self._paramfilePath)
            self._cashTotal = self._cashAvailable = self._settings.initialCapital(0)

        # instance the strategy
        strategySettings = self._settings.strategy
        strategyClass   = SubjectStrategies.get(strategySettings.name, None)
        strategyParams  = savedParams.strategy if savedParams else {}
        strategyConf    = strategySettings.params({})
        for k in savedParams.strategyConf.keys():
            strategyParams[k] = strategyConf[k]

        self._strategy  = strategyClass(self, strategyParams)

        marketEvent = self._settings.marketEvent('KL1min')
        self.debug('subscribe market event %s' % marketEvent)
        if marketEvent == 'tick' :
            self.subscribeEvent(MarketData.EVENT_TICK, self.eventHdl_Tick)
            self._mergerTick2KL1min = TickToKLineMerger(self.eventHdl_KLine1min)
        else :
            self.subscribeEvent(MarketData.EVENT_KLINE_1MIN, self.eventHdl_KLine1min)

        return super(SubjectTrader, self).init()

    @abstractmethod
    def start(self):

        # step 3. call allstrategy.onInit()
        self.strategies_Start()

    @abstractmethod
    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # TODO: subscribe all interested market data

        self.strategies_Stop()
        self.saveParameters()
        pass

    @abstractmethod
    def step(self):
        for a in self._dictAccounts.values():
            a.step()
    
    #----------------------------------------------------------------------
    # 
    #----------------------------------------------------------------------
    def loadParameters(paramfilePath) :
        
        self.debug('loading the saved parameters from %s' % self._paramfilePath)
        params = None
        with shelve.open(self._paramfilePath) as f:
            if not 'params' in f:
                return params

            params = f['params']
            if 'stamp' in f:
                params['stamp'] =f['stamp']

        return params

    def saveParameters() :

        self.debug('saving the parameters to %s' % self._paramfilePath)
        params = {
            'position' : self._position,
            'strategy' : {
                'name' : self._strategy.__class__.__name__,
            }
        }

        for k in self._strategy.paramDict.keys() :
            params['strategy'][k] = self._strategy.paramDict[k]

        with shelve.open(self._paramfilePath) as f:
            f['params'] = params
            f['stamp'] = datetime.now()


    #----------------------------------------------------------------------
    # local the recent data by symbol
    #----------------------------------------------------------------------
    
    # def getContract(self, vtSymbol):
    #     """查询合约对象"""
    #     try:
    #         return self._latestContract[vtSymbol]
    #     except KeyError:
    #         return None
        
    # def getAllContracts(self):
    #     """查询所有合约对象（返回列表）"""
    #     return self._latestContract.values()
    
    # def saveContracts(self):
    #     """保存所有合约对象到硬盘"""
    #     f = shelve.open(self._pathContracts)
    #     f['data'] = self._latestContract
    #     f.close()
    
    # def loadContracts(self):
    #     """从硬盘读取合约对象"""
    #     f = shelve.open(self._pathContracts)
    #     if 'data' in f:
    #         d = f['data']
    #         for key, value in d.items():
    #             self._latestContract[key] = value
    #     f.close()
        
    ### eventTick from MarketData ----------------
    @abstractmethod
    def eventHdl_KLine1min(self, event):
        """TODO: 处理行情推送"""
        d = event.dict_['data']
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        ds =""
        if len(tokens) >1:
            ds = tokens[1]
        if symbol != self.symbol or ds != self.account.exchange:
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

        if self._dtMarketData.strftime('%Y%m%d') != kline.date:
            self.onDayOpen(symbol, kline.date)

        # step 1. cache into the latest, lnf DataEngine
        self._dtMarketData = kline.datetime # datetime of data
        self._latestKline1min = kline

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        try:
            self.preStrategyByKLine(kline)
        except Exception, ex:
            self.error('eventHdl_KLine1min(%s) pre-strategy processing caught %s: %s' % (kline.desc, ex, traceback.format_exc()))
            return

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        try:
            self._stg_call(strategy, strategy.onBar, kline)
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
        ds =""
        if len(tokens) >1:
            ds = tokens[1]
        if symbol != self.symbol or ds != self.account.exchange:
            return # ignore those not interested

        tick = copy(d)

        # step 0. if Tick2KL1min is presented
        if self._mergerTick2KL1min :
            self._mergerTick2KL1min.pushTick(tick)
            return

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

        self._position.position = 

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

        if self._dtMarketData.strftime('%Y%m%d') != tick.date:
            self.onDayOpen(symbol, tick.date)

        # step 1. cache into the latest, lnf DataEngine
        self._dtMarketData = tick.datetime # datetime of data
        self._latestTick = tick

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        self.debug('eventHdl_Tick(%s) pre-strategy processing' % tick.desc)
        try:
            self.preStrategyByTick(tick)
        except Exception, ex:
            self.error('eventHdl_Tick(%s) pre-strategy processing caught %s: %s' % (kline.desc, ex, traceback.format_exc()))
            return

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        try:
            self._stg_call(strategy, strategy.onTick, tick)
        except Exception, ex:
            self.error('eventHdl_Tick(%s) [%s].onTick() caught %s: %s' % (tick.desc, strategy.id, ex, traceback.format_exc()))
    
        # step 5. 执行完策略后的的处理，通常为综合决策
        self.debug('eventHdl_Tick(%s) post-strategy processing' % tick.desc)
        self.postStrategy(symbol)

        self.debug('eventHdl_Tick(%s) done' % tick.desc)

    def latestPrice(self, symbol) :
        kline = self._latestKline1min.get(symbol, None)
        tick  = self._latestTick.get(symbol, None)

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
        self._latestContract[contract.vtSymbol] = contract
        self._latestContract[contract.symbol] = contract       # 使用常规代码（不包括交易所）可能导致重复

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

