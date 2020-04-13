# encoding: UTF-8
'''
EventProxy to remote ZeroMQ
'''
from __future__ import division

from EventData    import *
from Application  import BaseApplication

import os
import json   # to save params
from datetime import datetime, timedelta
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
import traceback

########################################################################
class EventProxy(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(EventProxy, self).__init__(program, **kwargs)

        # 事件队列
        self.__queue = Queue()

        if isinstance(objectives, list):
            for o in objectives:
                self._dictAdvices[o]=None

        self._marketState = PerspectiveState(self._exchange) # take PerspectiveState by default
        self.__stampLastSaveState = None
        try :
            shutil.rmtree(self.__wkTrader._outDir)
        except:
            pass

        try :
            os.makedirs(self.__wkTrader._outDir)
        except:
            pass
        self.program.setShelveFilename('%s%s.sobj' % (self.dataRoot, self.ident))

    def publish(self, event):
        '''向事件队列中存入事件'''
        self.__queue.put(event)

    @property
    def recorder(self): return self._recorder

    def __saveMarketState(self) :
        try :
            self.program.saveObject(self.marketState, '%s/marketState' % 'OnlineSimulator')
        except Exception as ex:
            self.logexception(ex)

    def __restoreMarketState(self) :
        try :
            return self.program.loadObject('%s/marketState' % 'OnlineSimulator') # '%s/marketState' % self.__class__)
        except Exception as ex:
            self.logexception(ex)
        return False

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

        if len(self._dictAdvices) <=0:
            sl = self._marketState.listOberserves()
            # for symbol in sl:
            #     self._dictAdvices[symbol] = AdviceData(self.ident, symbol, self._marketState.exchange)

        if self._marketState :
            for symbol in self._dictAdvices.keys():
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

            # pop the event to dispatch
            bEmpty = False
            while self._bRun and not bEmpty:
                event = None
                try :
                    event = self.__queue.get(block = enabledHB, timeout = timeout)  # 获取事件的阻塞时间设为0.1秒
                    bEmpty = False
                except Empty:
                    bEmpty = True
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    self._bRun = False
                    break

                # do the step only when there is no event
                if not event :
                    # if blocking: # ????
                    #     continue
                    cApps =0
                    for appId in self.__activeApps :
                        app = self.getObj(appId)
                        # if threaded, it has its own trigger to step()
                        # if isinstance(app, ThreadedAppWrapper)
                        #   continue
                        if not isinstance(app, MetaApp):
                            continue

                        if not app.isActive :
                            continue
                        cApps +=1

                        if not isinstance(app, BaseApplication):
                            continue
                        
                        try:
                            app.doAppStep()
                        except KeyboardInterrupt:
                            self.error("quit per KeyboardInterrupt")
                            self._bRun = False
                            break
                        except Exception as ex:
                            self.error("app[%s] step exception %s %s" % (appId, ex, traceback.format_exc()))

                    if cApps <=0:
                        self.info("Program has no more active apps running, update running state")
                        self._bRun = False
                        break
                            
                    continue

                # 检查是否存在对该事件进行监听的处理函数
                if not event.type in self.__subscribers.keys():
                    continue

                # 若存在，则按顺序将事件传递给处理函数执行
                for appId in self.__subscribers[event.type] :
                    app = self.getApp(appId)
                    if not app or not app.isActive:
                        continue

                    try:
                        app._procEvent(event)
                    except KeyboardInterrupt:
                        self.error("quit per KeyboardInterrupt")
                        self._bRun = False
                        break
                    except Exception as ex:
                        self.error("app step exception %s %s" % (ex, traceback.format_exc()))

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
            if not symbol in self._dictAdvices.keys() :
                return # ignore those not interested

            if d.asof > (datetime.now() + timedelta(days=7)):
                self.warn('Trade-End signal received: %s' % d.desc)
                self.eventHdl_TradeEnd(ev)
                return

            latestAdvc = self._dictAdvices[symbol] if symbol in self._dictAdvices.keys() else None
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
            if not newAdvice.exchange or len(newAdvice.exchange)<=0:
                newAdvice.exchange = self._marketState.exchange

            if symbol in self._dictPerf.keys():
                perf = self._dictPerf[symbol]
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
            self._dictAdvices[symbol] = newAdvice

            # self._recorder.pushRow(ev.type, d)
            self._recorder.pushRow(EVENT_ADVICE, newAdvice)

            return

    # end of BaseApplication routine
    #----------------------------------------------------------------------


########################################################################
class AdviceData(EventData):
    '''交易建议'''

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime,symbol,exchange,advisorId,price,dirLONG,dirSHORT,dirNONE,strDir,Rdaily,Rdstd'
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
        dirIdx = np.argmax([self.dirNONE,self.dirLONG,self.dirSHORT])
        self.strDir = AdviceData.DIRSTR[dirIdx]
        return self.strDir

    @property
    def desc(self) :
        dirIdx = np.argmax([self.dirNONE,self.dirLONG,self.dirSHORT])
        return 'tadv.%s@%s>%s@%s' % (self.symbol, self.asof.strftime('%Y%m%dT%H%M%S'), self.strDir, round(self.price,2))

########################################################################
from tensorflow.keras.models import model_from_json

class NeuralNetAdvisor(TradeAdvisor):
    '''
    NeuralNetAdvisor impls TradeAdvisor by ev
    '''
    def __init__(self, program, **kwargs) :
        self._brainId = None
        super(NeuralNetAdvisor, self).__init__(program, **kwargs)

        self._brainId = self.getConfig('brainId', "default")
        self._brainDir = '%s%s/' % (self.dataRoot, self._brainId)
        self.__stateSize, self.__actionSize = 1548, 3 #TODO

    @property
    def ident(self) :
        return 'NNAdv.%s.%s' % (self._brainId, self._id) if self._brainId else super(NeuralNetAdvisor,self).ident

    def __loadBrain(self, brainDir) :
        ''' load the previous saved brain
        @param brainDir must be given, in which there are model.json definition and weights.h5 parameters
        '''
        brainDir = '%s%s.S%dI%dA%d/' % (self.dataRoot, self._brainId, self.__stateSize, EXPORT_FLOATS_DIMS, self.__actionSize)
        try : 
            # step 1. read the model file in json
            self.debug('loading saved brain from %s' % brainDir)
            with open('%smodel.json' % brainDir, 'r') as mjson:
                model_json = mjson.read()
            brain = model_from_json(model_json)

            # step 2. read the weights of the model
            self.debug('loading saved brain weights from %s' %brainDir)
            brain.load_weights('%sweights.h5' % brainDir)

            self.info('loaded brain from %s' % (brainDir))
            return brain

        except Exception as ex:
            self.logexception(ex)

        return None

    def generateAdviceOnMarketEvent(self, ev):
        '''processing an incoming MarketEvent and generate an advice'''

        if MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            return None

        d = ev.data
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]

        market_state = self._marketState.exportKLFloats(symbol)
        market_state = np.array(market_state).astype(NN_FLOAT).reshape(1, self.__stateSize)

        act_values = self._brain.predict(market_state)
        action = np.zeros(self.__actionSize)
        action[np.argmax(act_values[0])] = 1
        advice = AdviceData(self.ident, symbol, d.exchange)
        advice.dirNONE, advice.dirLONG, advice.dirSHORT = act_values[0][0], act_values[0][1], act_values[0][2]
        advice.price = d.price if EVENT_TICK == ev.type else d.close

        return advice

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        self._brain = self.__loadBrain(self._brainId)
        if not self._brain:
            return False

        return super(NeuralNetAdvisor, self).doAppInit()
