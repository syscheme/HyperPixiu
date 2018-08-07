# encoding: UTF-8

from __future__ import division

import os
import shelve
import logging
from collections import OrderedDict
from datetime import datetime
from copy import copy

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from vnApp.MainRoutine import *
from vnApp.BrokerDriver import *
import vnApp.strategies as tg
from vnApp.brokerdrivers import tdHuobi as td

from vnpy.event import Event
from vnpy.trader.vtGlobal import globalSetting
from vnpy.trader.vtEvent import *
from vnpy.trader.vtGateway import *
from vnpy.trader.language import text
from vnpy.trader.vtFunction import getTempPath

import jsoncfg # pip install json-cfg

# 引擎类型，用于区分当前策略的运行环境
TRADER_TYPE_BACKTESTING = 'backtesting'  # 回测
TRADER_TYPE_TRADING = 'trading'          # 实盘

########################################################################
class Trader(BaseApplication):
    """Trader Application"""

    FINISHED_STATUS = [STATUS_ALLTRADED, STATUS_REJECTED, STATUS_CANCELLED]

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""

        super(Trader, self).__init__(mainRoutine, settings)

        # 引擎类型为实盘
        self._tradeType = TRADER_TYPE_TRADING
        
        #--------------------
        # from old 数据引擎
        # 保存数据的字典和列表

        self._dictLatestTick = {}         # the latest tick of each symbol
        self._dictLatestKline1min = {}    #SSS the latest kline1min of each symbol
        self._dictLatestContract = {}
        self._dictLatestOrder = {}
        self._dictWorkingOrder = {} # 可撤销委托
        # inside of Account self._dictTrade = {}
        self._dictAccounts = {}
        self._defaultAccId = None
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
        account = Account(self, td.tdHuobi, self._settings.account)
        if account:
            self.adoptAccount(account)

        # step 1. collect all interested symbol of market data into _dictObjectives
        self.debug('building up security objectives')
        self._dictObjectives = {}

        #   part 1. those specified in the configuration
        for s in jsoncfg.expect_array(self._settings.objectives):
            try :
                symbol = s.symbol('')
                dsTick = s.dsTick('')
                ds1min = s.ds1min('')
                if len(symbol) <=0 or (len(dsTick) <=0 and len(ds1min) <=0):
                    continue
                d = {
                    "dsTick" : dsTick,
                    "ds1min" : ds1min,
                }

                self._dictObjectives[symbol] = d
            except:
                pass
        
        # TODO part 2. scan those in the positions of Accounts, and also put into _dictObjectives

        self.debug('collected %s interested symbols, adopting strategies' % len(self._dictObjectives))

        self.strategies_LoadAll(self._settings.strategies)
        
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
    def start(self):
        # TODO: subscribe all interested market data
        self.subscribeSymbols()

        # subscribe account events
        self.subscribeEvent(BrokerDriver.EVENT_ORDER, self.eventHdl_Order)
        self.subscribeEvent(BrokerDriver.EVENT_TRADE, self.eventHdl_Trade)

        self.subscribeEvent(EventChannel.EVENT_TIMER, self.eventHdl_OnTimer)

        """注册事件监听"""

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # TODO: subscribe all interested market data
        pass
    
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
        f = shelve.open(self.contractFilePath)
        f['data'] = self._dictLatestContract
        f.close()
    
    def loadContracts(self):
        """从硬盘读取合约对象"""
        f = shelve.open(self.contractFilePath)
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
        return self._dictWorkingOrder.values()

    def getAllOrders(self):
        """获取所有委托"""
        return self._dictLatestOrder.values()
    
    def getAllTrades(self):
        """获取所有成交"""
        return self._dictTrade.values()
    
    #----------------------------------------------------------------------
    def convertOrderReq(self, req):
        """根据规则转换委托请求"""
        detail = self._dictDetails.get(req.vtSymbol, None)
        if not detail:
            return [req]
        else:
            return detail.convertOrderReq(req)

    #----------------------------------------------------------------------
    # about the position
    #----------------------------------------------------------------------
    def getAllPositions(self):
        """获取所有持仓"""
        return self._dictPositions.values()

    def getPositionDetail(self, vtSymbol):
        """查询持仓细节"""
        if vtSymbol in self._dictDetails:
            detail = self._dictDetails[vtSymbol]
        else:
            contract = self.getContract(vtSymbol)
            detail = PositionDetail(vtSymbol, contract)
            self._dictDetails[vtSymbol] = detail
            
            # 设置持仓细节的委托转换模式
            contract = self.getContract(vtSymbol)
            
            if contract:
                detail.exchange = contract.exchange
                
                # 上期所合约
                if contract.exchange == EXCHANGE_SHFE:
                    detail.mode = detail.MODE_SHFE
                
                # 检查是否有平今惩罚
                for productID in self._lstTdPenalty:
                    if str(productID) in contract.symbol:
                        detail.mode = detail.MODE_TDPENALTY
                
        return detail
    
    def getAllPositionDetails(self):
        """查询所有本地持仓缓存细节"""
        return self._dictDetails.values()
    
    def updateOrderReq(self, req, vtOrderID):
        """委托请求更新"""
        vtSymbol = req.vtSymbol
            
        detail = self.getPositionDetail(vtSymbol)
        detail.updateOrderReq(req, vtOrderID)
    
    #----------------------------------------------------------------------
    # logging
    #----------------------------------------------------------------------
    @abstractmethod
    def logEvent(self, event):
        """记录交易日志事件"""
        data = event.dict_['data']
        self._lstLogs.append(event)
        self._engine.logEvent(EVENT_LOG, data)

    @abstractmethod
    def logError(self, event):
        """处理错误事件"""
        error = event.dict_['data']
        self._lstErrors.append(error)
        self._engine.logError(data)

    def getLog(self):
        """获取日志"""
        return self._lstLogs
    
    def getError(self):
        """获取错误"""
        return self._lstErrors

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
                ds = self._engine.getMarketData(s['ds1min'])
                if ds:
                    self.debug('calling local subcribers for EVENT_KLINE_1MIN: %s' % s)
                    ds.subscribe(k, MarketData.EVENT_KLINE_1MIN)

    ### eventTick from MarketData ----------------
    @abstractmethod
    def eventHdl_KLine1min(self, event):
        """TODO: 处理行情推送"""
        self.debug('eventHdl_KLine1min')
        d = event.dict_['data']
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        ds =""
        if len(tokens) >1:
            ds = tokens[1]
        if not symbol in self._dictObjectives or ds != self._dictObjectives[symbol]['ds1min']:
            return # ignore those not interested

        kline = copy.copy(d)

        # step 1. cache into the latest, lnf DataEngine
        self.debug('eventHdl_KLine1min(%s)' % kline.vtSymbol)
        self._dictLatestKline1min[symbol] = kline

        # step 2. 收到tick行情后，先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        self.debug('eventHdl_KLine1min(%s) processing stop orders' % kline.vtSymbol)
        self._procOrdersByKLine(kline)

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        if symbol in self._idxSymbolToStrategy:
            # tick时间可能出现异常数据，使用try...except实现捕捉和过滤
            try:
                # 添加datetime字段
                if not kline.datetime:
                    kline.datetime = datetime.strptime(' '.join([tick.date, tick.time]), '%Y%m%d %H:%M:%S.%f')
            except ValueError:
                self.error(traceback.format_exc())
                return
                
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            self.debug('eventHdl_KLine1min(%s) dispatching to %d strategies' % (kline.vtSymbol, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onKline, kline)

        self.debug('eventHdl_KLine1min(%s) done' % kline.vtSymbol)

    @abstractmethod
    def eventHdl_Tick(self, event):
        """处理行情推送"""
        d = event.dict_['data']
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        ds =""
        if len(tokens) >1:
            ds = tokens[1]
        if not symbol in self._dictObjectives or ds != self._dictObjectives[symbol]['ds1min']:
            return # ignore those not interested

        tick = copy.copy(d)

        self.debug('eventHdl_Tick(%s)' % tick.vtSymbol)
        # step 1. cache into the latest, lnf DataEngine
        self._dictLatestTick[symbol] = tick

        # step 2. 收到tick行情后，先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        self.debug('eventHdl_Tick(%s) processing stop orders' % tick.vtSymbol)
        self._procOrdersByTick(tick)

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        if symbol in self._idxSymbolToStrategy:
            # tick时间可能出现异常数据，使用try...except实现捕捉和过滤
            try:
                # 添加datetime字段
                if not tick.datetime:
                    tick.datetime = datetime.strptime(' '.join([tick.date, tick.time]), '%Y%m%d %H:%M:%S.%f')
            except ValueError:
                self.error(traceback.format_exc())
                return
                
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            self.debug('eventHdl_Tick(%s) dispatching to %d strategies' % (tick.vtSymbol, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onTick, tick)
    
        self.debug('eventHdl_Tick(%s) done' % tick.vtSymbol)

    ### eventTick from Account ----------------
    @abstractmethod
    def eventHdl_Order(self, event):
        """处理委托事件"""
        order = event.dict_['data']        
        self._dictLatestOrder[order.vtOrderID] = order
        
        # step 1. 如果订单的状态是全部成交或者撤销，则需要从workingOrderDict中移除  lnf DataEngine
        if order.status in self.FINISHED_STATUS:
            if order.vtOrderID in self._dictWorkingOrder:
                del self._dictWorkingOrder[order.vtOrderID]
        # 否则则更新字典中的数据        
        else:
            self._dictWorkingOrder[order.vtOrderID] = order
            
        # step 2. 更新到持仓细节中 lnf DataEngine
        detail = self.getPositionDetail(order.vtSymbol)
        detail.updateOrder(order)            

        # step 3. 逐个推送到策略实例中 lnf DataEngine
        if vtOrderID in self._idxOrderToStategy:
            strategy = self._idxOrderToStategy[vtOrderID]            
            
            # 如果委托已经完成（拒单、撤销、全成），则从活动委托集合中移除
            if order.status == self.STATUS_FINISHED:
                s = self._idxStrategyToOrder[strategy.name]
                if vtOrderID in s:
                    s.remove(vtOrderID)
            
            self._stg_call(strategy, strategy.onOrder, order)
            
    ### eventTrade from Account ----------------
    @abstractmethod
    def eventHdl_Trade(self, event):
        """处理成交事件"""
        trade = event.dict_['data']
        
        # step 1. cache, lnf DataEngine
        if trade.vtTradeID in self._dictTrade: # 过滤已经收到过的成交回报
            return
        self._dictTrade[trade.vtTradeID] = trade
    
        # step 2. 更新到持仓细节中 lnf DataEngine
        detail = self.getPositionDetail(trade.vtSymbol)
        detail.updateTrade(trade)
        
        # step 3. 将成交推送到策略对象中 lnf ctaEngine
        if trade.vtOrderID in self._idxOrderToStategy:
            strategy = self._idxOrderToStategy[trade.vtOrderID]
            
            # 计算策略持仓
            if trade.direction == DIRECTION_LONG:
                strategy.pos += trade.volume
            else:
                strategy.pos -= trade.volume
            
            self._stg_call(strategy, strategy.onTrade, trade)
            
            # 保存策略持仓到数据库
            self._stg_flushPos(strategy)

    ### generic events ??? ----------------
    def eventHdl_OnTimer(self, event):
        edata = event.dict_['data']
        self.debug('OnTimer() src[%s] %s' % (edata.sourceType, edata.datetime.strftime('%Y%m%dT%H:%M:%S.%f')))
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

    ### eventPosition from Account ----------------
    def processPositionEvent(self, event):
        """处理持仓事件"""
        pos = event.dict_['data']

        self._dictPositions[pos.symbol] = pos
    
        # 更新到持仓细节中 lnf ctaEngine
        detail = self.getPositionDetail(pos.vtSymbol)
        detail.updatePosition(pos)                
        
    #----------------------------------------------------------------------
    @abstractmethod    # usually back test will overwrite this
    def _procOrdersByKLine(self, kline):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        self.processStopOrdersByKLine(kline)
        pass

    @abstractmethod    # usually back test will overwrite this
    def _procOrdersByTick(self, tick):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
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

    def strategies_Init(self):
        """全部初始化"""
        for name in self._dictStrategies.keys():
            self._stg_init(name)    

    def strategies_Stop(self):
        """全部停止"""
        for name in self._dictStrategies.keys():
            self._stg_stop(name)

    def strategies_Save(self):
        """保存策略配置"""
        with open(self.settingfilePath, 'w') as f:
            l = []
            
            for strategy in self._dictStrategies.values():
                setting = {}
                for param in strategy.paramList:
                    setting[param] = strategy.__getattribute__(param)
                l.append(setting)
            
            jsonL = json.dumps(l, indent=4)
            f.write(jsonL)

    def _stg_init(self, name):
        """初始化策略"""
        if name in self._dictStrategies:
            self.error(u'策略实例不存在：%s' %name)    

        strategy = self._dictStrategies[name]
        if not strategy.inited:
            strategy.inited = True
            self._stg_call(strategy, strategy.onInit)
            self._stg_loadPos(strategy)                             # 初始化完成后加载同步数据
            self.subscribeMarketData(strategy)                      # 加载同步数据后再订阅行情

    def _stg_start(self, name):
        """启动策略"""
        if name in self._dictStrategies:
            self.error(u'策略实例不存在：%s' %name)    

        strategy = self._dictStrategies[name]
        if strategy.inited and not strategy.trading:
            strategy.trading = True
            self._stg_call(strategy, strategy.onStart)
    
    def _stg_stop(self, name):
        """停止策略"""
        if name in self._dictStrategies:
            self.error(u'策略实例不存在：%s' %name)    

        strategy = self._dictStrategies[name]
            
        if strategy.trading:
            strategy.trading = False
            self._stg_call(strategy, strategy.onStop)
                
            # 对该策略发出的所有限价单进行撤单
            for vtOrderID, s in self.orderStrategyDict.items():
                if s is strategy:
                    self.cancelOrder(vtOrderID)
                
            # 对该策略发出的所有本地停止单撤单
            for stopOrderID, so in self.workingStopOrderDict.items():
                if so.strategy is strategy:
                    self.cancelStopOrder(stopOrderID)   

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
        strategyClass = tg.STRATEGY_CLASS.get(className, None)
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




########################################################################
class PositionDetail(object):
    """本地维护的持仓信息"""
    WORKING_STATUS = [STATUS_UNKNOWN, STATUS_NOTTRADED, STATUS_PARTTRADED]
    
    MODE_NORMAL = 'normal'          # 普通模式
    MODE_SHFE = 'shfe'              # 上期所今昨分别平仓
    MODE_TDPENALTY = 'tdpenalty'    # 平今惩罚

    #----------------------------------------------------------------------
    def __init__(self, vtSymbol, contract=None):
        """Constructor"""
        self.vtSymbol = vtSymbol
        self.symbol = EMPTY_STRING
        self.exchange = EMPTY_STRING
        self.name = EMPTY_UNICODE    
        self.size = 1
        
        if contract:
            self.symbol = contract.symbol
            self.exchange = contract.exchange
            self.name = contract.name
            self.size = contract.size
        
        self.longPos = EMPTY_INT
        self.longYd = EMPTY_INT
        self.longTd = EMPTY_INT
        self.longPosFrozen = EMPTY_INT
        self.longYdFrozen = EMPTY_INT
        self.longTdFrozen = EMPTY_INT
        self.longPnl = EMPTY_FLOAT
        self.longPrice = EMPTY_FLOAT
        
        self.shortPos = EMPTY_INT
        self.shortYd = EMPTY_INT
        self.shortTd = EMPTY_INT
        self.shortPosFrozen = EMPTY_INT
        self.shortYdFrozen = EMPTY_INT
        self.shortTdFrozen = EMPTY_INT
        self.shortPnl = EMPTY_FLOAT
        self.shortPrice = EMPTY_FLOAT
        
        self.lastPrice = EMPTY_FLOAT
        
        self.mode = self.MODE_NORMAL
        self.exchange = EMPTY_STRING
        
        self._dictWorkingOrder = {}
        
    #----------------------------------------------------------------------
    def updateTrade(self, trade):
        """成交更新"""
        # 多头
        if trade.direction is DIRECTION_LONG:
            # 开仓
            if trade.offset is OFFSET_OPEN:
                self.longTd += trade.volume
            # 平今
            elif trade.offset is OFFSET_CLOSETODAY:
                self.shortTd -= trade.volume
            # 平昨
            elif trade.offset is OFFSET_CLOSEYESTERDAY:
                self.shortYd -= trade.volume
            # 平仓
            elif trade.offset is OFFSET_CLOSE:
                # 上期所等同于平昨
                if self.exchange is EXCHANGE_SHFE:
                    self.shortYd -= trade.volume
                # 非上期所，优先平今
                else:
                    self.shortTd -= trade.volume
                    
                    if self.shortTd < 0:
                        self.shortYd += self.shortTd
                        self.shortTd = 0    
        # 空头
        elif trade.direction is DIRECTION_SHORT:
            # 开仓
            if trade.offset is OFFSET_OPEN:
                self.shortTd += trade.volume
            # 平今
            elif trade.offset is OFFSET_CLOSETODAY:
                self.longTd -= trade.volume
            # 平昨
            elif trade.offset is OFFSET_CLOSEYESTERDAY:
                self.longYd -= trade.volume
            # 平仓
            elif trade.offset is OFFSET_CLOSE:
                # 上期所等同于平昨
                if self.exchange is EXCHANGE_SHFE:
                    self.longYd -= trade.volume
                # 非上期所，优先平今
                else:
                    self.longTd -= trade.volume
                    
                    if self.longTd < 0:
                        self.longYd += self.longTd
                        self.longTd = 0
                    
        # 汇总
        self.calculatePrice(trade)
        self.calculatePosition()
        self.calculatePnl()
    
    #----------------------------------------------------------------------
    def updateOrder(self, order):
        """委托更新"""
        # 将活动委托缓存下来
        if order.status in self.WORKING_STATUS:
            self._dictWorkingOrder[order.vtOrderID] = order
            
        # 移除缓存中已经完成的委托
        else:
            if order.vtOrderID in self._dictWorkingOrder:
                del self._dictWorkingOrder[order.vtOrderID]
                
        # 计算冻结
        self.calculateFrozen()
    
    #----------------------------------------------------------------------
    def updatePosition(self, pos):
        """持仓更新"""
        if pos.direction is DIRECTION_LONG:
            self.longPos = pos.position
            self.longYd = pos.ydPosition
            self.longTd = self.longPos - self.longYd
            self.longPnl = pos.positionProfit
            self.longPrice = pos.price
        elif pos.direction is DIRECTION_SHORT:
            self.shortPos = pos.position
            self.shortYd = pos.ydPosition
            self.shortTd = self.shortPos - self.shortYd
            self.shortPnl = pos.positionProfit
            self.shortPrice = pos.price
            
        #self.output()
    
    #----------------------------------------------------------------------
    def updateOrderReq(self, req, vtOrderID):
        """发单更新"""
        vtSymbol = req.vtSymbol        
            
        # 基于请求生成委托对象
        order = VtOrderData()
        order.vtSymbol = vtSymbol
        order.symbol = req.symbol
        order.exchange = req.exchange
        order.offset = req.offset
        order.direction = req.direction
        order.totalVolume = req.volume
        order.status = STATUS_UNKNOWN
        
        # 缓存到字典中
        self._dictWorkingOrder[vtOrderID] = order
        
        # 计算冻结量
        self.calculateFrozen()
        
    #----------------------------------------------------------------------
    def updateTick(self, tick):
        """行情更新"""
        self.lastPrice = tick.lastPrice
        self.calculatePnl()
        
    #----------------------------------------------------------------------
    def calculatePnl(self):
        """计算持仓盈亏"""
        self.longPnl = self.longPos * (self.lastPrice - self.longPrice) * self.size
        self.shortPnl = self.shortPos * (self.shortPrice - self.lastPrice) * self.size
        
    #----------------------------------------------------------------------
    def calculatePrice(self, trade):
        """计算持仓均价（基于成交数据）"""
        # 只有开仓会影响持仓均价
        if trade.offset == OFFSET_OPEN:
            if trade.direction == DIRECTION_LONG:
                cost = self.longPrice * self.longPos
                cost += trade.volume * trade.price
                newPos = self.longPos + trade.volume
                if newPos:
                    self.longPrice = cost / newPos
                else:
                    self.longPrice = 0
            else:
                cost = self.shortPrice * self.shortPos
                cost += trade.volume * trade.price
                newPos = self.shortPos + trade.volume
                if newPos:
                    self.shortPrice = cost / newPos
                else:
                    self.shortPrice = 0
    
    #----------------------------------------------------------------------
    def calculatePosition(self):
        """计算持仓情况"""
        self.longPos = self.longTd + self.longYd
        self.shortPos = self.shortTd + self.shortYd      
        
    #----------------------------------------------------------------------
    def calculateFrozen(self):
        """计算冻结情况"""
        # 清空冻结数据
        self.longPosFrozen = EMPTY_INT
        self.longYdFrozen = EMPTY_INT
        self.longTdFrozen = EMPTY_INT
        self.shortPosFrozen = EMPTY_INT
        self.shortYdFrozen = EMPTY_INT
        self.shortTdFrozen = EMPTY_INT     
        
        # 遍历统计
        for order in self._dictWorkingOrder.values():
            # 计算剩余冻结量
            frozenVolume = order.totalVolume - order.tradedVolume
            
            # 多头委托
            if order.direction is DIRECTION_LONG:
                # 平今
                if order.offset is OFFSET_CLOSETODAY:
                    self.shortTdFrozen += frozenVolume
                # 平昨
                elif order.offset is OFFSET_CLOSEYESTERDAY:
                    self.shortYdFrozen += frozenVolume
                # 平仓
                elif order.offset is OFFSET_CLOSE:
                    self.shortTdFrozen += frozenVolume
                    
                    if self.shortTdFrozen > self.shortTd:
                        self.shortYdFrozen += (self.shortTdFrozen - self.shortTd)
                        self.shortTdFrozen = self.shortTd
            # 空头委托
            elif order.direction is DIRECTION_SHORT:
                # 平今
                if order.offset is OFFSET_CLOSETODAY:
                    self.longTdFrozen += frozenVolume
                # 平昨
                elif order.offset is OFFSET_CLOSEYESTERDAY:
                    self.longYdFrozen += frozenVolume
                # 平仓
                elif order.offset is OFFSET_CLOSE:
                    self.longTdFrozen += frozenVolume
                    
                    if self.longTdFrozen > self.longTd:
                        self.longYdFrozen += (self.longTdFrozen - self.longTd)
                        self.longTdFrozen = self.longTd
                        
            # 汇总今昨冻结
            self.longPosFrozen = self.longYdFrozen + self.longTdFrozen
            self.shortPosFrozen = self.shortYdFrozen + self.shortTdFrozen
            
    #----------------------------------------------------------------------
    def output(self):
        """"""
        print self.vtSymbol, '-'*30
        print 'long, total:%s, td:%s, yd:%s' %(self.longPos, self.longTd, self.longYd)
        print 'long frozen, total:%s, td:%s, yd:%s' %(self.longPosFrozen, self.longTdFrozen, self.longYdFrozen)
        print 'short, total:%s, td:%s, yd:%s' %(self.shortPos, self.shortTd, self.shortYd)
        print 'short frozen, total:%s, td:%s, yd:%s' %(self.shortPosFrozen, self.shortTdFrozen, self.shortYdFrozen)        
    
    #----------------------------------------------------------------------
    def convertOrderReq(self, req):
        """转换委托请求"""
        # 普通模式无需转换
        if self.mode is self.MODE_NORMAL:
            return [req]
        
        # 上期所模式拆分今昨，优先平今
        elif self.mode is self.MODE_SHFE:
            # 开仓无需转换
            if req.offset is OFFSET_OPEN:
                return [req]
            
            # 多头
            if req.direction is DIRECTION_LONG:
                posAvailable = self.shortPos - self.shortPosFrozen
                tdAvailable = self.shortTd- self.shortTdFrozen
                ydAvailable = self.shortYd - self.shortYdFrozen            
            # 空头
            else:
                posAvailable = self.longPos - self.longPosFrozen
                tdAvailable = self.longTd - self.longTdFrozen
                ydAvailable = self.longYd - self.longYdFrozen
                
            # 平仓量超过总可用，拒绝，返回空列表
            if req.volume > posAvailable:
                return []
            # 平仓量小于今可用，全部平今
            elif req.volume <= tdAvailable:
                req.offset = OFFSET_CLOSETODAY
                return [req]
            # 平仓量大于今可用，平今再平昨
            else:
                l = []
                
                if tdAvailable > 0:
                    reqTd = copy.copy(req)
                    reqTd.offset = OFFSET_CLOSETODAY
                    reqTd.volume = tdAvailable
                    l.append(reqTd)
                    
                reqYd = copy.copy(req)
                reqYd.offset = OFFSET_CLOSEYESTERDAY
                reqYd.volume = req.volume - tdAvailable
                l.append(reqYd)
                
                return l
            
        # 平今惩罚模式，没有今仓则平昨，否则锁仓
        elif self.mode is self.MODE_TDPENALTY:
            # 多头
            if req.direction is DIRECTION_LONG:
                td = self.shortTd
                ydAvailable = self.shortYd - self.shortYdFrozen
            # 空头
            else:
                td = self.longTd
                ydAvailable = self.longYd - self.longYdFrozen
                
            # 这里针对开仓和平仓委托均使用一套逻辑
            
            # 如果有今仓，则只能开仓（或锁仓）
            if td:
                req.offset = OFFSET_OPEN
                return [req]
            # 如果平仓量小于昨可用，全部平昨
            elif req.volume <= ydAvailable:
                if self.exchange is EXCHANGE_SHFE:
                    req.offset = OFFSET_CLOSEYESTERDAY
                else:
                    req.offset = OFFSET_CLOSE
                return [req]
            # 平仓量大于昨可用，平仓再反向开仓
            else:
                l = []
                
                if ydAvailable > 0:
                    reqClose = copy.copy(req)
                    if self.exchange is EXCHANGE_SHFE:
                        reqClose.offset = OFFSET_CLOSEYESTERDAY
                    else:
                        reqClose.offset = OFFSET_CLOSE
                    reqClose.volume = ydAvailable
                    
                    l.append(reqClose)
                    
                reqOpen = copy.copy(req)
                reqOpen.offset = OFFSET_OPEN
                reqOpen.volume = req.volume - ydAvailable
                l.append(reqOpen)
                
                return l
        
        # 其他情况则直接返回空
        return []