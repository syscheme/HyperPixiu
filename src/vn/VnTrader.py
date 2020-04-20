# encoding: UTF-8
'''
'''

from Trader import BaseTrader
from MarketData import *

########################################################################
class VnTrader(BaseTrader):
    '''vnpy-like trader driven by strategies'''

    def __init__(self, program, **kwargs):
        """Constructor"""

        super(VnTrader, self).__init__(program, **kwargs)
        self._lstMarketEventProc.append(self.__onMarketEvent)

        # 保存策略实例的字典
        # key为策略名称，value为策略实例，注意策略名称不允许重复
        self.__dictStrategies = {}
        self.__strategieCfgs = self.getConfig('strategies', [])

        # 保存数据的字典和列表
        self._settingfilePath = self.dataRoot + 'stgdata.dat'

        # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # value为包含所有相关strategy对象的list
        self.__idxSymbolToStrategy = {}
        
        # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # key为vtOrderID，value为strategy对象
        self.__idxOrderToStategy = {}
        self.__idxStrategyToOrder = {}


    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(VnTrader, self).doAppInit() :
            return False

        self.debug('collected %s interested symbols, adopting strategies' % len(self._dictObjectives))
        self.strategies_LoadAll()

        # step 1. subscribe all interested market data
        # self._account.onStart()

        # step 2. call allstrategy.onInit()
        self.strategies_Start()
        return True

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # TODO: subscribe all interested market data

        self.strategies_Stop()
        super(VnTrader, self).stop()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

   #----------------------------------------------------------------------
    # about the event handling
    def eventHdl_DayOpen(self, symbol, date):

        super(VnTrader, self).eventHdl_DayOpen(symbol, date)

        # step1. notify stategies
        if symbol in self.__idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self.__idxSymbolToStrategy[symbol]
            self.debug('eventHdl_DayOpen(%s) dispatching to %d strategies' % (symbol, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onDayOpen, date)

    def eventHdl_Order(self, ev):
        """处理委托事件"""
        order = ev.data        

        # step 3. 逐个推送到策略实例中 lnf DataEngine
        if not order.brokerOrderId in self.__idxOrderToStategy:
            return

        strategy = self.__idxOrderToStategy[order.brokerOrderId]            
            
        # 如果委托已经完成（拒单、撤销、全成），则从活动委托集合中移除
        if order.status == MetaTrader.FINISHED_STATUS:
            s = self.__idxStrategyToOrder[strategy.id]
            if order.brokerOrderId in s:
                s.remove(order.brokerOrderId)
            
        self._stg_call(strategy, strategy.onOrder, order)

    def eventHdl_Trade(self, ev):
        """处理成交事件"""
        trade = ev.data
        
        # step 3. 将成交推送到策略对象中 lnf ctaEngine
        if trade.orderID in self.__idxOrderToStategy:
            strategy = self.__idxOrderToStategy[trade.orderID]
            self._stg_call(strategy, strategy.onTrade, trade)
            # 保存策略持仓到数据库
            # goes to Account now : self._stg_flushPos(strategy)

    def __onMarketEvent(self, ev):
        '''processing an incoming MarketEvent'''

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        if EVENT_TICK == ev.type:
            self.processStopOrdersByTick(ev.data)
        elif EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            self.processStopOrdersByKLine(ev.data)
        else: return

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        execStgList = []
        if ev.data.symbol in self.__idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self.__idxSymbolToStrategy[ev.data.symbol]
            for strategy in l:
                try:
                    f = strategy.onTick if md.EVENT_TICK == ev.type else strategy.onBar
                    self._stg_call(strategy, f, ev.data)
                    execStgList.append(strategy.id)
                except Exception as ex:
                    self.error('proc_MarketData(%s) [%s] caught %s: %s' % (d.desc, strategy.id, ex, traceback.format_exc()))

        # step 4. 执行完策略后的的处理，通常为综合决策
        self.OnMarketEventProcessed(ev)

        self.debug('__onMarketEvent(%s) processed: %s' % (ev.desc, execStgList))

    # end of event handling
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    #  Strategy methods
    def strategies_LoadAll(self):
        """读取策略配置"""
        self.debug('loading all strategies: %s' % self.__strategieCfgs)
        for s in self.__strategieCfgs :
            self._stg_load(s)
            
        self.info('loaded strategies: %s' % self.__dictStrategies.keys())

    def strategies_List(self):
        """查询所有策略名称"""
        return self.__dictStrategies.keys()        

    def strategies_Start(self):
        """全部初始化"""
        for n in self.__dictStrategies.values():
            self._stg_start(n['strategy'])    

    def strategies_Stop(self):
        """全部停止"""
        for n in self.__dictStrategies.values():
            self._stg_stop(n['strategy'])

    def strategies_Save(self):
        """保存策略配置"""
        with open(self._settingfilePath, 'w') as f:
            l = []
            
            for strategy in self.__dictStrategies.values():
                setting = {}
                for param in strategy.paramList:
                    setting[param] = strategy.__getattribute__(param)
                l.append(setting)
            
            jsonL = json.dumps(l, indent=4)
            f.write(jsonL)

    #----------------------------------------------------------------------
    # private methods
    def _stg_start(self, strategy):
        """启动策略"""
        if strategy.inited and not strategy.trading:
            self._stg_call(strategy, strategy.onStart)
            strategy.trading = True
    
    def _stg_stop(self, strategy):
        """停止策略"""
        if strategy.trading:
            self._stg_call(strategy, strategy.onStop)
            # self._stg_call(strategy, strategy.cancellAll)
            strategy.trading = False

    def _stg_load(self, setting):
        """载入策略, setting schema:
            {
                "name" : "BBand", // strategy name equals to class name
                "weights": { // weights to affect decisions, in range of [0-100] each
                    "long" : 100, // optimisti
                    "short": 100, // pessimistic
                },

                // the following is up to the stategy class
            },
        """

        # 获取策略类
        strategyClass =None
        className = setting['name']
        try :
            strategyClass = STRATEGY_CLASS.get(className, None)
        except:
            self.error('failed to find strategy-class：%s' %className)
            return

        if not strategyClass:
            self.error('failed to find strategy-class：%s' %className)
            return
        
        # 创建策略实例
        for s in self._dictObjectives.keys():
            strategy = None
            try :
                strategy = strategyClass(self, s, self._account, setting)
                sid = strategy.id
                if strategy.id in self.__dictStrategies:  # 防止策略重名
                    self.error('strategy-instance[%s] exists' % strategy.id)
                    continue

                self.__dictStrategies[strategy.id] = {
                    'weights' : setting['weights'],
                    'strategy' : strategy
                }

                # 创建委托号列表
                self.__idxStrategyToOrder[strategy.id] = set()

                # 保存Tick映射关系
                if s in self.__idxSymbolToStrategy:
                    l = self.__idxSymbolToStrategy[s]
                else:
                    l = []
                    self.__idxSymbolToStrategy[s] = l
                l.append(strategy)
            except:
                continue

            try :
                self._stg_call(strategy, strategy.onInit)
                strategy.inited = True
                self.info('initialized strategy[%s]' % strategy.id)
            except:
                del(self.__dictStrategies[strategy.id])
                del(self.__idxStrategyToOrder[strategy.id])

    def _stg_allVars(self, name):
        """获取策略当前的变量字典"""
        if name in self.__dictStrategies:
            strategy = self.__dictStrategies[name]
            varDict = OrderedDict()
            
            for key in strategy.varList:
                varDict[key] = strategy.__getattribute__(key)
            
            return varDict
        else:
            self.error(u'策略实例不存在：' + name)    
            return None
    
    def _stg_allParams(self, name):
        """获取策略的参数字典"""
        if name in self.__dictStrategies:
            strategy = self.__dictStrategies[name]
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
        if not strategyId in self.__idxStrategyToOrder :
            return []
        l = self.__idxStrategyToOrder[strategyId]
        if not symbol:
            return l
        
        ret = []
        for o in l:
            if o.symbol == symbol:
                ret.append(o)
        return ret

    def postStrategyEvent(self, strategyId) :
        pass

    def OnMarketEventProcessed(self, ev) :
        """执行完策略后的的处理，通常为综合决策"""
        pass

    # normal Trader cares StopOrders
    def processStopOrdersByTick(self, tick):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        pass

    # normal Trader cares StopOrders
    def processStopOrdersByKLine(self, kline):
        """收到行情后处理本地停止单（检查是否要立即发出）"""
        pass



