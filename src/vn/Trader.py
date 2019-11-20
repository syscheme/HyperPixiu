# encoding: UTF-8
'''
Trader maps to the agent in OpenAI/Gym
'''

from ../Trader import BaseTrader

########################################################################
class VnTrader(BaseTrader):
    '''vnpy-like trader driven by strategies'''

    def __init__(self, program, settings):
        """Constructor"""

        super(VnTrader, self).__init__(program, settings)

        # 保存策略实例的字典
        # key为策略名称，value为策略实例，注意策略名称不允许重复
        self._dictStrategies = {}

        # 保存数据的字典和列表
        self._settingfilePath = self.dataRoot + 'stgdata.dat'

        # 保存vtSymbol和策略实例映射的字典（用于推送tick数据）
        # 由于可能多个strategy交易同一个vtSymbol，因此key为vtSymbol
        # value为包含所有相关strategy对象的list
        self._idxSymbolToStrategy = {}
        
        # 保存vtOrderID和strategy对象映射的字典（用于推送order和trade数据）
        # key为vtOrderID，value为strategy对象
        self._idxOrderToStategy = {}
        self._idxStrategyToOrder = {}

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(VnTrader, self).doAppInit() :
            return False

        self.debug('collected %s interested symbols, adopting strategies' % len(self._dictObjectives))
        self.strategies_LoadAll(self._settings.strategies)

        # step 1. subscribe all interested market data

        self._account.onStart()

        # step 2. call allstrategy.onInit()
        self.strategies_Start()

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        # TODO: subscribe all interested market data

        self.strategies_Stop()
        super(VnTrader, self).stop()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

   #----------------------------------------------------------------------
    # about the event handling
    #----------------------------------------------------------------------
    @abstractmethod    # usually back test will overwrite this
    def onDayOpen(self, symbol, date):

        super(VnTrader, self).onDayOpen(symbol, date)

        # step1. notify stategies
        if symbol in self._idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            self.debug('onDayOpen(%s) dispatching to %d strategies' % (symbol, len(l)))
            for strategy in l:
                self._stg_call(strategy, strategy.onDayOpen, date)

    def eventHdl_Order(self, event):
        """处理委托事件"""
        order = event.data        

        # step 3. 逐个推送到策略实例中 lnf DataEngine
        if not order.brokerOrderId in self._idxOrderToStategy:
            return

        strategy = self._idxOrderToStategy[order.brokerOrderId]            
            
        # 如果委托已经完成（拒单、撤销、全成），则从活动委托集合中移除
        if order.status == MetaTrader.FINISHED_STATUS:
            s = self._idxStrategyToOrder[strategy.id]
            if order.brokerOrderId in s:
                s.remove(order.brokerOrderId)
            
        self._stg_call(strategy, strategy.onOrder, order)

    def eventHdl_Trade(self, event):
        """处理成交事件"""
        trade = event.data
        
        # step 3. 将成交推送到策略对象中 lnf ctaEngine
        if trade.orderID in self._idxOrderToStategy:
            strategy = self._idxOrderToStategy[trade.orderID]
            self._stg_call(strategy, strategy.onTrade, trade)
            # 保存策略持仓到数据库
            # goes to Account now : self._stg_flushPos(strategy)

    def proc_MarketEvent(self, ev):
        '''processing an incoming MarketEvent'''

        # step 2. 收到行情后，在启动策略前的处理
        # 先处理本地停止单（检查是否要立即发出） lnf ctaEngine
        if md.EVENT_TICK == evtype:
            self.processStopOrdersByTick(d)
        elif EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)] :
            self.processStopOrdersByKLine(d)
        else: return

        # step 3. 推送tick到对应的策略实例进行处理 lnf ctaEngine
        execStgList = []
        if symbol in self._idxSymbolToStrategy:
            # 逐个推送到策略实例中
            l = self._idxSymbolToStrategy[symbol]
            for strategy in l:
                try:
                    f = strategy.onTick if md.EVENT_TICK == evtype else strategy.onBar
                    self._stg_call(strategy, f, d)
                    execStgList.append(strategy.id)
                except Exception as ex:
                    self.error('proc_MarketData(%s) [%s] caught %s: %s' % (d.desc, strategy.id, ex, traceback.format_exc()))

        # step 4. 执行完策略后的的处理，通常为综合决策
        self.OnMarketEventProcessed(evtype, symbol, data)

        self.debug('proc_MarketEvent(%s) processed: %s' % (ev.desc, execStgList))

    # end of event handling
    #----------------------------------------------------------------------

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
            # self._stg_call(strategy, strategy.cancellAll)
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
            strategy = strategyClass(self, s, self._account, setting)
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

    @abstractmethod    # usually back test will overwrite this
    def OnMarketEventProcessed(self, evtype, symbol, data) :
        """执行完策略后的的处理，通常为综合决策"""
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



