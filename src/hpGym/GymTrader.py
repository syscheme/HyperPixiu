# encoding: UTF-8
'''
GymTrader impls BaseTrader and represent itself as a Gym Environment
'''
from __future__ import division

# from gym import GymEnv
from Account import Account, OrderData, Account_AShare
from Application import MetaObj, BOOL_STRVAL_TRUE
from Trader import MetaTrader, BaseTrader
from BackTest import BackTestApp, RECCATE_ESPSUMMARY
from Perspective import PerspectiveState, EXPORT_SIGNATURE
from MarketData import EVENT_TICK, EVENT_KLINE_PREFIX, EXPORT_FLOATS_DIMS, NORMALIZE_ID
from HistoryData import listAllFiles

import hpGym

from abc import abstractmethod
import matplotlib as mpl # pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
import copy
import h5py, tarfile, numpy

RFGROUP_PREFIX = 'ReplayFrame:'

plt.style.use('dark_background')
mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "lines.linewidth": 1,
        "lines.markersize": 8
    }
)

DUMMY_BIG_VAL = 999999.9

########################################################################
class MetaAgent(MetaObj): # TODO:
    def __init__(self, gymTrader, **kwargs):

        super(MetaAgent, self).__init__()

        self.__kwargs = kwargs
        self.__jsettings = None
        if 'jsettings' in self.__kwargs.keys():
            self.__jsettings = self.__kwargs.pop('jsettings', None)

        self._gymTrader = gymTrader
        self._stateSize = len(self._gymTrader.gymReset())
        self._actionSize = len(type(gymTrader).ACTIONS)

        self._learningRate = self.getConfig('learningRate', 0.001)
        self._batchSize = self.getConfig('batchSize', 128)

        self._gamma = self.getConfig('gamma', 0.95)
        self._epsilon = self.getConfig('epsilon', 1.0) # rand()[0,1) <= self._epsilon will trigger a random explore
        self._epsilonMin = self.getConfig('epsilonMin', 0.02) #  self._epsilon < self._epsilonMin means at playmode and no more explorations

        self._wkBrainId = self.getConfig('brainId', None)

        self._trainInterval = self._batchSize /2
        if self._trainInterval < 10:
            self._trainInterval =10

        self._outDir = self.getConfig('outDir', self._gymTrader.dataRoot)
        if '/' != self._outDir[-1]: self._outDir +='/'

        self._statusAttrs = {}

        self._brain = self.buildBrain(self._wkBrainId)

    @property
    def brainId(self):
        return self._wkBrainId

    @abstractmethod
    def isReady(self) : return True

    def getConfig(self, configName, defaultVal) :
        try :
            if configName in self.__kwargs.keys() :
                return self.__kwargs[configName]

            if self.__jsettings:
                jn = self.__jsettings
                for i in configName.split('/') :
                    jn = jn[i]

                if defaultVal :
                    if isinstance(defaultVal, list):
                        return jsoncfg.expect_array(jn(defaultVal))
                    if isinstance(defaultVal, dict):
                        return jsoncfg.expect_object(jn(defaultVal))

                return jn(defaultVal)
        except:
            pass

        return defaultVal

    @abstractmethod
    def buildBrain(self, brainId =None):
        '''
        @return the brain built to set to self._brain
        '''
        raise NotImplementedError

    @abstractmethod
    def gymAct(self, state, bObserveOnly=False):
        '''
        @return one of self.__gymTrader.ACTIONS
        '''
        raise NotImplementedError

    @abstractmethod
    def gymObserve(self, state, action, reward, next_state, done, bObserveOnly=False, **feedbacks):
        '''Memory Management and training of the agent
        @param bObserveOnly True if the account is not executable, so only observing, no predicting and/or training
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        raise NotImplementedError

    @abstractmethod
    def saveBrain(self, brainId =None) :
        ''' save the current brain into the dataRoot
        @param a unique brainId must be given
        '''
        raise NotImplementedError
        
    @abstractmethod
    def loadBrain(self, brainId) :
        ''' load the previous saved brain
        @param a unique brainId must be given
        '''
        raise NotImplementedError

########################################################################
class GymTrader(BaseTrader):
    '''
    GymTrader impls BaseTrader and represent itself as a Gym Environment
    '''
    NN_FLOAT = 'float32'
    
    ACTION_BUY  = OrderData.ORDER_BUY
    ACTION_SELL = OrderData.ORDER_SELL
    ACTION_HOLD = 'HOLD'

    ACTIONS = {
        ACTION_HOLD: np.array([1, 0, 0]).astype(NN_FLOAT),
        ACTION_BUY:  np.array([0, 1, 0]).astype(NN_FLOAT),
        ACTION_SELL: np.array([0, 0, 1]).astype(NN_FLOAT)
    }

    POS_DIRECTIONS = {
        OrderData.DIRECTION_NONE:  np.array([1, 0, 0]).astype(NN_FLOAT),
        OrderData.DIRECTION_LONG:  np.array([0, 1, 0]).astype(NN_FLOAT),
        OrderData.DIRECTION_SHORT: np.array([0, 0, 1]).astype(NN_FLOAT)
    }

    def __init__(self, program, agentClass=None, **kwargs):
        '''Constructor
        '''
        super(GymTrader, self).__init__(program, **kwargs) # redirect to BaseTrader, who will adopt account and so on
        self._lstMarketEventProc.append(self.__processMarketEvent)

        self._agent = None
        self._timeCostYrRate = self.getConfig('timeCostYrRate', 0)
        self._tradeSymbol    = self.getConfig('tradeSymbol', '000001')
        self.openObjective(self._tradeSymbol)

        #TODO: the separate the Trader for real account and training account
        
        self.__1stRender = True
        self._AGENTCLASS = None
        agentType = self.getConfig('agent/type', 'DQN') # 'DoubleDQN')
        if agentType and agentType in hpGym.GYMAGENT_CLASS.keys():
            self._AGENTCLASS = hpGym.GYMAGENT_CLASS[agentType]

        # GymTrader always take PerspectiveState as the market state
        self._marketState = PerspectiveState(None)
        self._gymState = None
        self._total_pnl = 0.0

        self._dailyCapCost = 0.0 # just to ease calc
        self.__deposittedReward = 0.0
        # self.n_actions = 3
        # self._prices_history = []

    @property
    def withdrawReward(self) : # any read will reset the self.__deposittedReward
        ret = self.__deposittedReward
        self.__deposittedReward = 0.0
        return ret

    def depositReward(self, reward) :
        self.__deposittedReward += round(reward, 4)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(GymTrader, self).doAppInit() :
            self.error('doAppInit() super-of[GymTrader] failed')
            return False

        if not self._AGENTCLASS :
            self.error('doAppInit() no AGENTCLASS associated')
            return False

        # the self._account should be good here

        # step 1. GymTrader always take PerspectiveState as the market state
        self._marketState._exchange = self._account.exchange
        self._account._marketState = self._marketState

        agentKwArgs = self.getConfig('agent', {})
        agentKwArgs = {**agentKwArgs, 'outDir': self._outDir }
        self._agent = self._AGENTCLASS(self, jsettings=self.subConfig('agent'), **agentKwArgs)
        # gymReset() will be called by agent above, self._gymState = self.gymReset() # will perform self._action = ACTIONS[ACTION_HOLD]
        if not self._agent:
            self.error('doAppInit() failed to instantize agent')
            return False

        # self.__stampActStart = datetime.now()
        # self.__stampActEnd = self.__stampActStart

        # self.debug('doAppInit() dummy gymStep to initialize cache')
        # while not self._agent.isReady() :
        #     action = self._agent.gymAct(self._gymState)
        #     next_state, reward, done, _ = self.gymStep(action)
        #     self._agent.gymObserve(self._gymState, action, reward, next_state, done) # regardless state-stepping, rewards and loss here

        self.debug('doAppInit() done')
        return True

    def doAppStep(self):
        super(GymTrader, self).doAppStep()
        # perform some dummy steps in order to fill agent._memory[]

    def __processMarketEvent(self, ev):
        '''processing an incoming MarketEvent'''
        action, bObserveOnly, strAdj = self.determinActionByMarketEvent(ev)
        next_state, reward, done, info = self.gymStep(action, bObserveOnly)

        self._gymState = next_state

    def determinActionByMarketEvent(self, ev):
        '''processing an incoming MarketEvent'''
        bObserveOnly = False
        if not self._account.executable:
            bObserveOnly = True

        action = self._agent.gymAct(self._gymState, bObserveOnly)
        strActionAdj =''

        # the gymAct only determin the direction, adjust the action to execute per current balance
        if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]) and self._latestCash < 1000:
            action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
            strActionAdj += '<BC'
        elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]) and self._latestPosValue < 1.0:
            action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
            strActionAdj += '<SP'

        if not all(action == GymTrader.ACTIONS[GymTrader.ACTION_HOLD]):
            dtAsOf = self.marketState.getAsOf()
            if (dtAsOf.hour in [14, 23] and dtAsOf.minute in [58,59]) or (dtAsOf.hour in [0, 15] and dtAsOf.minute in [0,1]) :
                action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
                strActionAdj += '<EoD'

        self._action = action

        return self._action, bObserveOnly, strActionAdj

    # end of impl/overwrite of BaseApplication
    #----------------------------------------------------------------------

    #------------------------------------------------
    # GymEnv related entries
    def gymReset(self) :
        '''
        reset the gym environment, will be called when each episode starts
        reset the trading environment / rewards / data generator...
        @return:
            observation (numpy.array): observation of the state
        '''
        self.__closed_plot = False
        self.__stepNo = 0
        self._total_pnl = 0.0
        self._total_reward = 0.0
        self._latestCash, self._latestPosValue = self._account.summrizeBalance()
        balance = self._latestCash + self._latestPosValue
        if balance > self._maxBalance :
            self._maxBalance = balance

        observation = self.makeupGymObservation()
        self._shapeOfState = observation.shape
        self._action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
        self._gymState = observation
        self.debug('gymReset() returning observation')
        return observation

    def gymStep(self, action, bObserveOnly=False) :
        '''Take an action (buy/sell/hold) and computes the immediate reward.
        @param action (numpy.array): Action to be taken, one-hot encoded.
        @returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        assert any([(action == x).all() for x in self.ACTIONS.values()])
        self._action = action
        self.__stepNo += 1
        done = False
        instant_pnl = 0.0
        reward =0.0
        info = {'execAction': GymTrader.ACTION_HOLD }

        prevCap = self._latestCash + self._latestPosValue

        if bObserveOnly:
            info['status'] = 'observe only'
        else :
            reward += - round(self._dailyCapCost /240, 4) # this is supposed the capital timecost every minute as there are 4 open hours every day

            # step 1. collected information from the account
            cashAvail, cashTotal, positions = self._account.positionState()
            _, posvalue = self._account.summrizeBalance(positions, cashTotal)
            capitalBeforeStep = cashTotal + posvalue

            # TODO: the first version only support one symbol to play, so simply take the first symbol in the positions        
            symbol = self._tradeSymbol # TODO: should take the __dictOberserves
            latestPrice = self._marketState.latestPrice(symbol)

            maxBuy, maxSell = self._account.maxOrderVolume(symbol, latestPrice)
            if self._maxValuePerOrder >0:
                if self._maxValuePerOrder < (maxBuy * latestPrice*100):
                    maxBuy = int(maxBuy * self._maxValuePerOrder / (maxBuy* latestPrice*100))
                if self._maxValuePerOrder < (maxSell*1.5 * latestPrice*100):
                    maxSell = int(maxSell * self._maxValuePerOrder / (maxSell * latestPrice*100))
            if self._minBuyPerOrder >0 and (maxBuy * latestPrice*100) < self._minBuyPerOrder :
                maxBuy =0

            # TODO: the first version only support FULL-BUY and FULL-SELL
            if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]) :
                if maxBuy >0 :
                    info['execAction'] = '%s:%sx%s' %(GymTrader.ACTION_BUY, latestPrice, maxBuy)
                    self.debug('gymStep() issuing max%s' % info['execAction'])
                    self._account.cancelAllOrders()
                    vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_BUY, latestPrice, maxBuy, strategy=None)
                    info['status'] = 'buy issued'
                else: reward -= 1 # penalty: is the agent blind to buy with no cash? :)
            elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]):
                if  maxSell >0:
                    info['execAction'] = '%s:%sx%s' %(GymTrader.ACTION_SELL, latestPrice, maxBuy)
                    self.debug('gymStep() issuing max%s' % info['execAction'])
                    self._account.cancelAllOrders()
                    vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_SELL, latestPrice, maxSell, strategy=None)
                    info['status'] = 'sell issued'
                else: reward -= 1 # penalty: is the agent blind to sell with no position? :)
            else : reward += self.withdrawReward # only allow withdraw the depositted reward when action=HOLD

            # step 3. calculate the rewards
            self._latestCash, self._latestPosValue = self._account.summrizeBalance() # most likely the cashAmount changed due to comission
            capitalAfterStep = self._latestCash + self._latestPosValue
            if capitalAfterStep > self._maxBalance :
                self.depositReward(round((capitalAfterStep - self._maxBalance) *5, 4)) # exceeding maxBalance should be encouraged
                self._maxBalance = capitalAfterStep

            reward += round(capitalAfterStep - prevCap, 4)

            instant_pnl = capitalAfterStep - capitalBeforeStep
            self._total_pnl += instant_pnl
        
        # step 5. combine account and market observations as final observations, then return
        observation = self.makeupGymObservation()
        if prevCap >1:
            reward = reward *10000 /prevCap

        return observation, reward, done, info
    
    def gymRender(self, savefig=False, filename='myfig'):
        """Matlplotlib gymRendering of each step.

        @param savefig (bool): Whether to save the figure as an image or not.
        @param filename (str): Name of the image file.
        """
        if self.__1stRender:
            self._f, self._ax = plt.subplots(
                len(self._spread_coefficients) + int(len(self._spread_coefficients) > 1),
                sharex=True
            )

            if len(self._spread_coefficients) == 1:
                self._ax = [self._ax]

            self._f.set_size_inches(12, 6)
            self.__1stRender = False
            self._f.canvas.mpl_connect('close_event', self.__OnRenderClosed)

        if len(self._spread_coefficients) > 1:
            # TODO: To be checked
            for prod_i in range(len(self._spread_coefficients)):
                bid = self._prices_history[-1][2 * prod_i]
                ask = self._prices_history[-1][2 * prod_i + 1]
                self._ax[prod_i].plot([self.__stepNo, self.__stepNo + 1],
                                      [bid, bid], color='white')
                self._ax[prod_i].plot([self.__stepNo, self.__stepNo + 1],
                                      [ask, ask], color='white')
                self._ax[prod_i].set_title('Product {} (spread coef {})'.format(
                    prod_i, str(self._spread_coefficients[prod_i])))

        # Spread price
        prices = self._prices_history[-1]
        bid, ask = calc_spread(prices, self._spread_coefficients)
        self._ax[-1].plot([self.__stepNo, self.__stepNo + 1],
                          [bid, bid], color='white')
        self._ax[-1].plot([self.__stepNo, self.__stepNo + 1],
                          [ask, ask], color='white')
        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        if (self._action == self.ACTIONS[ACTION_SELL]).all():
            self._ax[-1].scatter(self.__stepNo + 0.5, bid + 0.03 *
                                 yrange, color='orangered', marker='v')
        elif (self._action == self.ACTIONS[ACTION_BUY]).all():
            self._ax[-1].scatter(self.__stepNo + 0.5, ask - 0.03 *
                                 yrange, color='lawngreen', marker='^')
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + [OrderData.DIRECTION_NONE, OrderData.DIRECTION_LONG, OrderData.DIRECTION_SHORT][list(self._position).index(1)] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price)
        self._f.tight_layout()
        plt.xticks(range(self.__stepNo)[::5])
        plt.xlim([max(0, self.__stepNo - 80.5), self.__stepNo + 0.5])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename)

    # end of GymEnv entries
    #------------------------------------------------

    def makeupGymObservation(self):
        market_state = self._marketState.exportKLFloats(self._tradeSymbol)
        return np.array(market_state).astype(GymTrader.NN_FLOAT)

    def makeupGymObservation_0(self):
        '''Concatenate all necessary elements to create the observation.

        Returns:
            numpy.array: observation array with each element dim=EXPORT_FLOATS_DIMS
        '''
        # part 1. build up the account_state
        cashAvail, cashTotal, positions = self._account.positionState()
        _, posvalue = self._account.summrizeBalance(positions, cashTotal)
        stateCapital = [0.0] * EXPORT_FLOATS_DIMS
        stateCapital[0] = cashAvail
        stateCapital[1] = cashTotal
        stateCapital[2] = posvalue

        # POS_COLS = PositionData.COLUMNS.split(',')
        # del(POS_COLS['exchange', 'stampByTrader', 'stampByBroker'])
        # del(POS_COLS['symbol']) # TODO: this version only support single Symbol, so regardless field symbol

        # [price,avgPrice,posAvail,position,...] 
        statePOS = [0.0] * EXPORT_FLOATS_DIMS

        # supposed to be [[price,avgPrice,posAvail,position,...],...] when mutliple-symbols
        for s, pos in positions.items() :
            # row = []
            # for c in POS_COLS:
            #     row.append(pos.__dict__[c])
            # statePOS.append(row)
            statePOS[0] = pos.price
            statePOS[1] = pos.avgPrice
            statePOS[2] = pos.posAvail
            statePOS[3] = pos.position
            break

        account_state = np.concatenate([stateCapital + statePOS], axis=0)

        # part 2. build up the market_state
        market_state = self._marketState.exportKLFloats(self._tradeSymbol)

        # TODO: more observations in the future could be:
        #  - money flow
        #  - market index

        # return the concatenation of account_state and market_state as gymEnv sate
        envState = np.concatenate((account_state, market_state))
        return envState.astype(GymTrader.NN_FLOAT)

    #----------------------------------------------------------------------
    # access to the account observed

    def __OnRenderClosed(self, evt):
        self.__closed_plot = True

########################################################################
class OfflineSimulator(BackTestApp):
    '''
    OfflineSimulator extends BackTestApp by reading history and perform training
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(OfflineSimulator, self).__init__(program, trader, histdata, **kwargs)
        self._iterationsPerEpisode = self.getConfig('iterationsPerEpisode', 1)

        self._masterExportHomeDir = self.getConfig('master/homeDir', None) # this agent work as the master when configured, usually point to a dir under webroot
        if self._masterExportHomeDir and '/' != self._masterExportHomeDir[-1]: self._masterExportHomeDir +='/'
        
        # the base URL of local web for the slaves to GET/POST the tasks
        # current OfflineSimulator works as slave if this masterExportURL presents but masterExportHomeDir abendons
        self._masterExportURL = self.getConfig('master/exportURL', self._masterExportHomeDir)

        self.__lastEpisode_loss = DUMMY_BIG_VAL

        self.__savedEpisode_Id = -1
        self.__savedEpisode_opendays = 0
        self.__savedEpisode_loss = DUMMY_BIG_VAL
        self.__savedEpisode_reward = -DUMMY_BIG_VAL
        self.__stampLastSaveBrain = '0000'
        self.__maxKnownOpenDays =0
        self.__prevMaxBalance =0

        self.__recentLoss = None
        self._total_reward = 0.0
        self._feedbackToAgent = {
            'dailyReward': 0.0, 
            'bestRewardTotal': 0.0,
            'bestRewardDays': 0,
        }

        # we encourage the train to reach end of history, so give some reward every week for its survive
        self.__rewardDayStepped = float(self._startBalance) * self._initTrader._annualCostRatePcnt /220 / 100

    @property
    def loss(self) :
        return round(self.__recentLoss.history["loss"][0], 6) if self.__recentLoss else DUMMY_BIG_VAL

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        # make sure OfflineSimulator is ONLY wrappering GymTrader
        if not self._initTrader or not isinstance(self._initTrader, GymTrader) :
            self.error('doAppInit() invalid initTrader')
            return False

        if not super(OfflineSimulator, self).doAppInit() :
            return False

        self.wkTrader.gymReset()
        self.wkTrader._agent.enableMaster(self._masterExportHomeDir)
        self._account.account._skipSavingByEvent = True
        return True

    def OnEvent(self, ev): # this overwrite BackTest's because there are some different needs
        symbol  = None
        try :
            symbol = ev.data.symbol
        except:
            pass

        if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            self._account.matchTrades(ev)

        self.wkTrader.OnEvent(ev) # to perform the gym step

        asOf = self.wkTrader.marketState.getAsOf(symbol)

        # get some additional reward when survived for one more day
        if self._dataEnd_date and asOf.day > self._dataEnd_date.day :
            self.wkTrader.depositReward(self.__rewardDayStepped)

        self._dataEnd_date = asOf
        self._dataEnd_closeprice = self.wkTrader.marketState.latestPrice(symbol)

        if not self._dataBegin_date:
            self._dataBegin_date = self._dataEnd_date
            self._dataBegin_openprice = self._dataEnd_closeprice
            self.debug('OnEvent() taking dataBegin(%s @%s)' % (self._dataBegin_openprice, self._dataBegin_date))

        if (self.wkTrader._latestCash  + self.wkTrader._latestPosValue) < (self.wkTrader._maxBalance*(100.0 -self._pctMaxDrawDown)/100):
            self._bGameOver = True
            self._episodeSummary['reason'] = '%s cash +%s pv drewdown %s%% of maxBalance[%s]' % (self.wkTrader._latestCash, self.wkTrader._latestPosValue, self._pctMaxDrawDown, self.wkTrader._maxBalance)
            self.error('episode[%s] has been KO-ed: %s' % (self.episodeId, self._episodeSummary['reason']))
        
    # end of BaseApplication routine
    #----------------------------------------------------------------------

    # to replace GymTrader's __actPerMarketEvent
    def __trainPerMarketEvent(self, ev):
        '''processing an incoming MarketEvent'''

        action, bObserveOnly, strActionAdj = self.wkTrader.determinActionByMarketEvent(ev)

        next_state, reward, done, info = self.wkTrader.gymStep(action, bObserveOnly)
    
        loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, action, reward, next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
        if loss: self.__recentLoss =loss

        self.wkTrader._gymState = next_state

        self._total_reward += reward
        self.debug('__trainPerMarketEvent(%s) performed gymAct(%s%s) got reward[%s/%s] done[%s], agent ack-ed loss[%s]'% (ev.desc, action, strActionAdj, reward, self._total_reward, done, self.loss))

    #------------------------------------------------
    # BackTest related entries
    def OnEpisodeDone(self, reachedEnd=True):
        super(OfflineSimulator, self).OnEpisodeDone(reachedEnd)

        # determin whether it is a best episode
        lstImproved=[]

        opendays = self._episodeSummary['openDays']
        if opendays > self.__maxKnownOpenDays:
            self.__maxKnownOpenDays = opendays
            lstImproved.append('openDays')

        if self.wkTrader._maxBalance > self.__prevMaxBalance :
            lstImproved.append('maxBalance')
            self.debug('OnEpisodeDone() maxBalance improved from %s to %s' % (self.__prevMaxBalance, self.wkTrader._maxBalance))
            self.__prevMaxBalance = self.wkTrader._maxBalance

        if reachedEnd or (opendays>2 and opendays > (self.__maxKnownOpenDays *3/4)): # at least stepped most of known days
            # determin best by reward
            rewardMean = self._total_reward /opendays if opendays>0 else 0
            self._episodeSummary['dailyReward'] = rewardMean
            self._feedbackToAgent['dailyReward'] = rewardMean
            rewardMeanBest = self.__savedEpisode_reward
            if self.__savedEpisode_opendays>0 :
                rewardMeanBest /= self.__savedEpisode_opendays

            if rewardMean > rewardMeanBest or (rewardMean > rewardMeanBest/2 and opendays > self.__savedEpisode_opendays *1.2):
                lstImproved.append('meanReward')
                self.debug('OnEpisodeDone() meanReward improved from %s/%s to %s/%s' % (self.__savedEpisode_reward, self.__savedEpisode_opendays, self._total_reward, opendays))

            # determin best by loss
            if self.loss < self.__savedEpisode_loss and opendays >= self.__savedEpisode_opendays:
                lstImproved.append('loss')

        # save brain and decrease epsilon if improved
        if len(lstImproved) >0 :
            self._feedbackToAgent['bestRewardTotal'] = self._total_reward
            self._feedbackToAgent['bestRewardDays'] = opendays
            if self.__savedEpisode_loss < DUMMY_BIG_VAL: # do not save for the first episode
                self._feedbackToAgent['improved'] = lstImproved
                self.wkTrader._agent.saveBrain(**self._feedbackToAgent)
                self.__stampLastSaveBrain = datetime.now()
                self.info('OnEpisodeDone() brain saved per improvements: %s' % lstImproved )

            self.__savedEpisode_opendays = opendays
            self.__savedEpisode_loss = self.loss
            self.__savedEpisode_Id = self.episodeId
            self.__savedEpisode_reward = self._total_reward

        mySummary = {
            'totalReward' : round(self._total_reward, 2),
            'epsilon'     : round(self.wkTrader._agent._epsilon, 6),
            'learningRate': self.wkTrader._agent._learningRate,
            'loss'        : self.loss,
            'lastLoss'    : self.__lastEpisode_loss,
            'savedLoss'   : self.__savedEpisode_loss,
            'savedEId'    : self.__savedEpisode_Id,
            'savedReward' : round(self.__savedEpisode_reward,2),
            'savedODays'  : self.__savedEpisode_opendays,
            'savedTime'   : self.__stampLastSaveBrain.strftime('%Y%m%dT%H%M%S') if isinstance(self.__stampLastSaveBrain, datetime) else self.__stampLastSaveBrain,
            'frameNum'    : self.wkTrader._agent.frameNum
        }

        self._episodeSummary = {**self._episodeSummary, **mySummary}
        self.__lastEpisode_loss = self.loss

        # decrease agent's learningRate and epsilon if reward improved
        if 'meanReward' in lstImproved :
            self.wkTrader._agent._learningRate *=0.8
            if self.wkTrader._agent._learningRate < 0.0001 : 
                self.wkTrader._agent._learningRate = 0.0001

            self.wkTrader._agent._epsilon -= self.wkTrader._agent._epsilon/8
            if self.wkTrader._agent._epsilon < self.wkTrader._agent._epsilonMin :
                self.wkTrader._agent._epsilon = self.wkTrader._agent._epsilonMin

            self.debug('OnEpisodeDone() reward improved, decreased to learningRate[%s] epsilon[%s]' % (self.wkTrader._agent._learningRate, self.wkTrader._agent._epsilon))

    def resetEpisode(self) :
        '''
        reset the gym environment, will be called when each episode starts
        reset the trading environment / rewards / data generator...
        @return:
            observation (numpy.array): observation of the state
        '''
        super(OfflineSimulator, self).resetEpisode()
        self.wkTrader._lstMarketEventProc =[self.__trainPerMarketEvent] # replace GymTrader's with training method
        self.__recentLoss = None
        self._total_reward = 0.0

        return self.wkTrader.gymReset()

    def formatSummary(self, summary=None):
        strReport = super(OfflineSimulator, self).formatSummary(summary)
        if not isinstance(summary, dict) :
            summary = self._episodeSummary

        strReport += '\n totalReward: %s'  % summary['totalReward']
        strReport += '\n     epsilon: %s'  % summary['epsilon']
        strReport += '\nlearningRate: %s'  % summary['learningRate']
        strReport += '\n        loss: %s from %s' % (summary['loss'], summary['lastLoss'])
        strReport += '\n   savedLoss: %s <-(%s: %sd reward=%s @%s)' % (summary['savedLoss'], summary['savedEId'], summary['savedODays'], summary['savedReward'], summary['savedTime'])
        # durInner, durOuter = self.wkTrader.durMeasure_sum() 
        # strReport += '\n   durations: %s, %s'  % (durInner, durOuter)
        # durInner, durOuter = self.durMeasure_sum() 
        # strReport += '\n   durations: %s, %s'  % (durInner, durOuter)

        return strReport

    def __updateSimulatorTask(self, inputFiles):
        
        if not self._masterExportHomeDir or len(self._masterExportHomeDir) <=0:
            return # not as the master
        
        # collect all needed files into a tmpdir
        simTaskId = '%s_%s' % (self.wkTrader._agent.brainId, datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        tmpdir = os.path.join(self.wkTrader._outDir, 'sim_tmp.%s' % simTaskId)
        try :
            os.makedirs(tmpdir)
        except :
            pass

        for fn in listAllFiles(self.wkTrader._outDir) :
            if '/model.' in fn:
                os.system('cp -f %s %s' % (fn, tmpdir))

        if not 'model.json.h5' in os.listdir(tmpdir) :
            os.system('cp -f %s/model.* %s' % (os.path.join(self.dataRoot, self.wkTrader._agent.brainId), tmpdir))
        
        # generate the URL list where the slave is able to download the input csv files
        exportedCsvHome = '%s/' % os.path.join(self._masterExportHomeDir, 'csv')
        try :
            os.makedirs(exportedCsvHome)
        except :
            pass

        urlList= []
        for fn in inputFiles:
            basename = os.path.basename(fn)
            exportedFn = '%s%s' % (exportedCsvHome, basename)
            try :
                os.stat(exportedFn)
            except:
                os.system('cp -f %s %s' % (fn, exportedFn))

            urlList.append(os.path.join(exportedCsvHome, basename))
        
        with open(os.path.join(tmpdir, 'source.lst'), 'wt') as lstfile:
            lstfile.write('\r\n'.join(urlList))
        
        # now make a tar ball as the task file
        # this is a tar.bz2 including a) model.json, b) current weight h5 file, c) version-number, d) the frame exported as hdf5 file
        fn_task = os.path.join(self._masterExportHomeDir, 'tasks', 'sim_%s.tak' % simTaskId)
        try :
            os.makedirs(os.path.dirname(fn_task))
        except :
            pass

        with tarfile.open(fn_task, "w:bz2") as tar:
            files = os.listdir(tmpdir)
            for f in files:
                tar.add(os.path.join(tmpdir, f), f)

        self.debug('__updateSimulatorTask() prepared task-file %s, activating it' % (fn_task))
        os.system('rm -rf %s' % tmpdir)

        # swap into the active task
        target_task_file = os.path.join(self._masterExportHomeDir, 'tasks', 'sim.tak')
        os.system('rm -rf $(realpath %s) %s' % (target_task_file, target_task_file))
        os.system('ln -sf %s %s' % (fn_task, target_task_file))
        self.info('__updateSimulatorTask() task updated: %s->%s' % (target_task_file, fn_task))

########################################################################
class IdealDayTrader(OfflineSimulator):
    '''
    IdealTrader extends OfflineSimulator by scanning the MarketEvents occurs in a day, determining
    the ideal actions then pass the events down to the models
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(IdealDayTrader, self).__init__(program, trader, histdata, **kwargs)
        self._dayPercentToCatch           = self.getConfig('constraints/dayPercentToCatch',          1.0) # pecentage of daychange to catch, otherwise will keep position empty
        self._constraintBuy_closeOverOpen = self.getConfig('constraints/buy_closeOverOpen',          0.5) #pecentage price-close more than price-open - indicate buy
        self._constraintBuy_closeOverRecovery = self.getConfig('constraint/buy_closeOverRecovery',   2.0) #pecentage price-close more than price-low at the recovery edge - indicate buy
        self._constraintSell_lossBelowHigh = self.getConfig('constraint/sell_lossBelowHigh',         2.0) #pecentage price-close less than price-high at the loss edge - indicate sell
        self._constraintSell_downHillOverClose = self.getConfig('constraint/sell_downHillOverClose', 0.5) #pecentage price more than price-close triggers sell during a downhill-day to reduce loss
        self._generateReplayFrames  = self.getConfig('generateReplayFrames', 'directionOnly').lower()

        self._pctMaxDrawDown =99.0 # IdealTrader will not be constrainted by max drawndown, so overwrite it with 99%
        self._warmupDays =0 # IdealTrader will not be constrainted by warmupDays

        self.__cOpenDays =0

        self.__ordersToPlace = [] # list of faked OrderData, the OrderData only tells the direction withno amount

        self.__dtToday = None
        self.__mdEventsToday = [] # list of the datetime of open, high, low, close price occured today

        self.__dtTomrrow = None
        self.__mdEventsTomrrow = [] # list of the datetime of open, high, low, close price occured 'tomorrow'


    # to replace OfflineSimulator's __trainPerMarketEvent
    def __idealActionPerMarketEvent(self, ev):
        '''processing an incoming MarketEvent'''

        bObserveOnly = False
        if not self.wkTrader._account.executable:
            bObserveOnly = True

        # see if need to perform the next order pre-determined
        action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
        if len(self.__ordersToPlace) >0 :
            nextOrder = self.__ordersToPlace[0]
            if nextOrder.datetime <= ev.data.datetime :
                action = GymTrader.ACTIONS[GymTrader.ACTION_BUY] if (OrderData.DIRECTION_LONG == nextOrder.direction) else GymTrader.ACTIONS[GymTrader.ACTION_SELL]
                del self.__ordersToPlace[0]

        next_state, reward, done, info = self.wkTrader.gymStep(action, bObserveOnly)
        loss = None
    
        # fake reward here and make every possible as an obervation into the replaybuffer
        # fakedRewards = {
        #     GymTrader.ACTION_HOLD: -round(self.wkTrader._dailyCapCost/240, 2),
        #     GymTrader.ACTION_BUY: -1,
        #     GymTrader.ACTION_SELL: -1,
        # }
        # for a in [GymTrader.ACTION_HOLD, GymTrader.ACTION_BUY, GymTrader.ACTION_SELL] :
        #     act = GymTrader.ACTIONS[a]
        #     r = 1 if all(action == act) else fakedRewards[a]  # the positive reward for the bingo-ed action, should = reward?
        #     loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, act, r, next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
        #     if loss: self.__recentLoss =loss
        if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]):
            if GymTrader.ACTION_BUY in info['execAction'] :
                loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_BUY], max(reward, 1.0), next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
            else:
                loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_BUY], max(reward, 0.5), next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
                # loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_HOLD], 0.1, next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
        elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]):
            if GymTrader.ACTION_SELL in info['execAction'] :
                loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_SELL], max(reward, 1.0), next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
            else:
                loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_SELL], max(reward, 0.5), next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
                # loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_HOLD], 0.1, next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})
        else:
            if reward <0.01: reward =0.01
            loss = self.wkTrader._agent.gymObserve(self.wkTrader._gymState, GymTrader.ACTIONS[GymTrader.ACTION_HOLD], min(reward, 0.1), next_state, done, bObserveOnly, **{**info, **self._feedbackToAgent})

        if loss: self.__recentLoss =loss

        self.wkTrader._gymState = next_state
        self._total_reward += reward
        self.debug('__idealActionPerMarketEvent(%s) performed gymAct(%s) got reward[%s/%s] done[%s], agent ack-ed loss[%s]'% (ev.desc, action, reward, self._total_reward, done, loss))

    def resetEpisode(self) :
        ret = super(IdealDayTrader, self).resetEpisode()
        self.wkTrader._lstMarketEventProc =[self.__idealActionPerMarketEvent] # replace GymTrader's with __idealActionPerMarketEvent
        self.wkTrader._agent._cbNewReplayFrame = [self.__saveReplayFrame] # the hook agent's OnNewFrame
        # the IdealTrader normally will disable exploration of agent
        self.wkTrader._agent._epsilon = -1.0 # less than epsilonMin
        self.wkTrader._agent._gamma = 0 # idealtrader doesn't concern Q_next
        # self.wkTrader._dailyCapCost = 0.0 # no more daily cost in ideal trader
        # self.wkTrader._maxValuePerOrder = self._startBalance /2

        # no need to do training if to export ReplayFrames
        if 'full' == self._generateReplayFrames or 'direction' in self._generateReplayFrames:
            self.wkTrader._agent._learningRate = -1.0

        return ret

    # to replace BackTest's doAppStep
    def doAppStep(self):

        self._bGameOver = False # always False in IdealTrader
        reachedEnd = False
        if self._wkHistData :
            try :
                ev = next(self._wkHistData)
                if not ev or ev.data.datetime < self._btStartDate: return
                if ev.data.datetime <= self._btEndDate:
                    if self.__dtToday and self.__dtToday == ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0):
                        self.__mdEventsToday.append(ev)
                        return

                    if self.__dtTomrrow and self.__dtTomrrow == ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0):
                        self.__mdEventsTomrrow.append(ev)
                        return

                    # day-close here
                    self.scanEventsAndFakeOrders()
                    for cachedEv in self.__mdEventsToday:
                        self._marketState.updateByEvent(cachedEv)

                        super(BackTestApp, self).doAppStep()
                        self._account.doAppStep()

                        self.OnEvent(cachedEv) # call Trader
                        self._stepNoInEpisode += 1

                    self.__dtToday = self.__dtTomrrow
                    self.__mdEventsToday = self.__mdEventsTomrrow

                    self.__mdEventsTomrrow = []
                    self.__dtTomrrow = ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                    self.__cOpenDays += 1
                    if 0 == (self.__cOpenDays % 100) :
                        self.wkTrader._agent.saveBrain(**self._feedbackToAgent)

                    return # successfully performed a step by pushing an Event

                reachedEnd = True
            except StopIteration:
                reachedEnd = True
                self.info('hist-read: end of playback')
            except Exception as ex:
                self.logexception(ex)

        # this test should be done if reached here
        self.debug('doAppStep() episode[%s] finished: %d steps, KO[%s] end-of-history[%s]' % (self.episodeId, self._stepNoInEpisode, self._bGameOver, reachedEnd))
        
        # for this IdealTrader, collect a single episode of ReplayFrames is enough to export
        # so no more hooking
        self.wkTrader._agent._cbNewReplayFrame = []

        try:
            self.OnEpisodeDone(reachedEnd)
        except Exception as ex:
            self.logexception(ex)

        # print the summary report
        if self._recorder and isinstance(self._episodeSummary, dict):
            self._recorder.pushRow(RECCATE_ESPSUMMARY, self._episodeSummary)

        strReport = self.formatSummary()
        self.info('%s_%s summary:' %(self.ident, self.episodeId))
        for line in strReport.splitlines():
            if len(line) <2: continue
            self.info(line)

        # prepare for the next episode
        self._episodeNo +=1
        if (self._episodeNo > self._episodes) :
            # all tests have been done
            self.stop()
            self.info('all %d episodes have been done, app stopped. obj-in-program: %s' % (self._episodes, self._program.listByType(MetaObj)))

        self._program.stop()
        
        exit(0) # IdealDayTrader is not supposed to run forever, just exit instead of return

    def __scanEventsSequence(self, evseq) :

        price_open, price_high, price_low, price_close = 0.0, 0.0, DUMMY_BIG_VAL, 0.0
        T_high, T_low  = None, None
        if evseq and len(evseq) >0:
            for ev in evseq:
                evd = ev.data
                if EVENT_TICK == ev.type :
                    price_close = evd.price
                    if price_open <= 0.01 :
                        price_open = price_close
                    if price_high < price_close :
                        price_high = price_close
                        T_high = evd.datetime
                    if price_low > price_close :
                        price_low = price_close
                        T_low = evd.datetime
                    continue

                if EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
                    price_close = evd.close
                    if price_high < evd.high :
                        price_high = evd.high
                        T_high = evd.datetime
                    if price_low > evd.low :
                        price_low = evd.low
                        T_low = evd.datetime
                    if price_open <= 0.01 :
                        price_open = evd.open
                    continue

        return price_open, price_high, price_low, price_close, T_high, T_low

    def scanEventsAndFakeOrders(self) :
        '''
        this will generate 3 actions
        '''
        # step 1. scan self.__mdEventsToday and determine TH TL
        price_open, price_high, price_low, price_close, T_high, T_low = self.__scanEventsSequence(self.__mdEventsToday)
        tomorrow_open, tomorrow_high, tomorrow_low, tomorrow_close, tT_high, tT_low = self.__scanEventsSequence(self.__mdEventsTomrrow)

        if not T_high:
            return

        latestDir = OrderData.DIRECTION_NONE
        T_win = timedelta(minutes=2)
        slip = 0.02

        if T_high.month==6 and T_high.day in [25,26]:
             print('here')

        # step 2. determine the stop prices
        sell_stop = price_high -slip
        buy_stop  = min(price_low +slip, price_close*(100.0-self._dayPercentToCatch)/100)

        if (T_high < T_low) and price_close < (sell_stop *0.97): # this is a critical downhill, then enlarger the window to sell
            sell_stop= sell_stop *0.99 -slip

        catchback =0.0 # assume catch-back unnecessaray by default
        cleanup   =price_high*2 # assume no cleanup
        if tomorrow_high :
            tsell_stop = tomorrow_high -slip
            tbuy_stop  = min(tomorrow_low +slip, tomorrow_close*0.99)
            cleanup = max(tsell_stop, price_close -slip)

            if buy_stop > tsell_stop:
                buy_stop =0.0 # no buy today

            if tT_low < tT_high : # tomorrow is an up-hill
                catchback = tbuy_stop
            elif tsell_stop > price_close +slip:
                #catchback = min(tomorrow_high*(100.0- 2*self._dayPercentToCatch)/100, price_close +slip)
                catchback =price_low +slip
        elif (price_close < price_open*(100.0 +self._constraintBuy_closeOverOpen)/100):
            buy_stop =0.0 # forbid to buy
            catchback =0.0

        if cleanup < price_high: # if cleanup is valid, then no more buy/catchback
            catchback =0.0

        if sell_stop <= max(catchback, buy_stop)+slip:
            sell_stop = cleanup # no need to sell

        # step 2. faking the ideal orders
        for ev in self.__mdEventsToday:
            if EVENT_TICK != ev.type and EVENT_KLINE_PREFIX != ev.type[:len(EVENT_KLINE_PREFIX)] :
                continue

            evd = ev.data
            T = evd.datetime

            price = evd.price if EVENT_TICK == ev.type else evd.close
            order = OrderData(self._account)
            order.datetime = T

            if price <= buy_stop :
                order.direction = OrderData.DIRECTION_LONG 
                self.__ordersToPlace.append(copy.copy(order))
                latestDir = order.direction
                continue

            if price >= sell_stop :
                order.direction = OrderData.DIRECTION_SHORT 
                self.__ordersToPlace.append(copy.copy(order))
                latestDir = order.direction
                continue

            if T > max(T_high, T_low) :
                if price < catchback: # whether to catch back after sold
                    order.direction = OrderData.DIRECTION_LONG 
                    self.__ordersToPlace.append(copy.copy(order))
                    latestDir = order.direction

    # def filterFakeOrders(self) :
    #     idx = 0
    #     latestDir = None
    #     cContinuousDir =0
    #     while idx < len(self.__ordersToPlace):
    #         if not latestDir or latestDir == self.__ordersToPlace[idx].direction:
    #             latestDir = self.__ordersToPlace[idx].direction
    #             cContinuousDir +=1
    #             continue

    #             self.__ordersToPlace

    #         self.__ordersToPlace.append(copy.copy(order))

    def scanEventsAndFakeOrders000(self) :
        # step 1. scan self.__mdEventsToday and determine TH TL
        price_open, price_high, price_low, price_close, T_high, T_low = self.__scanEventsSequence(self.__mdEventsToday)
        tomorrow_open, tomorrow_high, tomorrow_low, tomorrow_close, tT_high, tT_low = self.__scanEventsSequence(self.__mdEventsTomrrow)

        if not T_high:
            return

        if T_high.day==27 and T_high.month==2 :
            print('here')

        # step 2. faking the ideal orders
        bMayBuy = price_close >= price_open*(100.0 +self._constraintBuy_closeOverOpen)/100 # may BUY today, >=price_open*1.005
        T_win = timedelta(minutes=2)
        slip = 0.02

        sell_stop = max(price_high -slip, price_close*(100.0 +self._constraintSell_lossBelowHigh)/100)
        buy_stop  = min(price_close*(100.0 -self._constraintBuy_closeOverRecovery)/100, price_low +slip)
        uphill_catchback = price_close + slip

        if tomorrow_high :
            if tomorrow_high > price_close*(100.0 +self._constraintBuy_closeOverRecovery)/100 :
               bMayBuy = True

            if ((tT_low <tT_high and tomorrow_low < price_close) or tomorrow_high < (uphill_catchback * 1.003)) :
                uphill_catchback =0 # so that catch back never happen

        if price_close > price_low*(100.0 +self._constraintBuy_closeOverRecovery)/100 : # if close is at a well recovery edge
            bMayBuy = True

        for ev in self.__mdEventsToday:
            if EVENT_TICK != ev.type and EVENT_KLINE_PREFIX != ev.type[:len(EVENT_KLINE_PREFIX)] :
                continue

            evd = ev.data
            T = evd.datetime

            price = evd.price if EVENT_TICK == ev.type else evd.close
            order = OrderData(self._account)
            order.datetime = T

            if T_low < T_high : # up-hill
                if bMayBuy and (T <= T_low + T_win and price <= buy_stop) :
                    order.direction = OrderData.DIRECTION_LONG 
                    self.__ordersToPlace.append(copy.copy(order))
                if T <= (T_high + T_win) and price >= (price_high -slip):
                    order.direction = OrderData.DIRECTION_SHORT 
                    self.__ordersToPlace.append(copy.copy(order))
                elif T > T_high :
                    # if sell_stop < (uphill_catchback *1.002) and tomorrow_high > price_close:
                    #     continue # too narrow to perform any actions

                    if price > sell_stop:
                        order.direction = OrderData.DIRECTION_SHORT 
                        self.__ordersToPlace.append(copy.copy(order))
                    elif price < uphill_catchback :
                        order.direction = OrderData.DIRECTION_LONG 
                        self.__ordersToPlace.append(copy.copy(order))

            if T_low > T_high : # down-hill
                if price >= (price_high -slip) or (T < T_low and price >= (price_close*(100.0 +self._constraintSell_downHillOverClose)/100)):
                    order.direction = OrderData.DIRECTION_SHORT 
                    self.__ordersToPlace.append(copy.copy(order))
                elif bMayBuy and (T > (T_low - T_win) and T <= (T_low + T_win) and price < round (price_close +price_low*3) /4, 3) :
                    order.direction = OrderData.DIRECTION_LONG 
                    self.__ordersToPlace.append(copy.copy(order))

    def __saveReplayFrame(self, frameId, col_state, col_action, col_reward, col_next_state, col_done) :

        # output the frame into a HDF5 file
        fn_frame = os.path.join(self.wkTrader._outDir, 'RFrm%s_%s.h5' % (NORMALIZE_ID, self.wkTrader._tradeSymbol) )
        dsargs={
            'compression':'lzf'
        }

        with h5py.File(fn_frame, 'a') as h5file:
            g = h5file.create_group('%s%s' % (RFGROUP_PREFIX, frameId))
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs[u'default'] = 'state'
            g.attrs['size'] = col_state.shape[0]
            g.attrs['signature'] = EXPORT_SIGNATURE

            if 'full' == self._generateReplayFrames :
                g.attrs['reward'] = 'reward'
                g.attrs['next_state'] = 'next_state'
                g.attrs['done'] = 'done'

            g.create_dataset(u'title',     data= '%s replay frame[%s] of %s for DQN training' % (self._generateReplayFrames, frameId, self.wkTrader._tradeSymbol))
            st = g.create_dataset('state',      data= col_state, **dsargs)
            st.attrs['dim'] = col_state.shape[1]
            ac = g.create_dataset('action',     data= col_action, **dsargs)
            ac.attrs['dim'] = col_action.shape[1]
            
            if 'full' == self._generateReplayFrames :
                g.create_dataset('reward',     data= col_reward, **dsargs)
                g.create_dataset('next_state', data= col_next_state, **dsargs)
                g.create_dataset('done',       data= col_done, **dsargs)

        self.info('saved frame[%s] len[%s] to file %s with sig[%s]' % (frameId, len(col_state), fn_frame, EXPORT_SIGNATURE))


########################################################################
from Application import Program
from Account import Account_AShare
import HistoryData as hist
import sys, os, platform
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

def main_prog():
    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Gym_AShare.json']

    p = Program()
    p._heartbeatInterval =-1

    SYMBOL = 'SH510050' # '000540' '000001'
    sourceCsvDir = None
    try:
        jsetting = p.jsettings('trainer/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('trainer/objectives')
        if not jsetting is None:
            symbol = jsetting([SYMBOL])[0]
    except Exception as ex:
        symbol = SYMBOL
    SYMBOL = symbol

    if not sourceCsvDir or len(sourceCsvDir) <=0:
        for d in ['e:/AShareSample', '/mnt/e/AShareSample/ETF', '/mnt/m/AShareSample']:
            try :
                if  os.stat(d):
                    sourceCsvDir = d
                    break
            except :
                pass

    if 'Windows' in platform.platform() and '/mnt/' == sourceCsvDir[:5] and '/' == sourceCsvDir[6]:
        drive = '%s:' % sourceCsvDir[5]
        sourceCsvDir = sourceCsvDir.replace(sourceCsvDir[:6], drive)

    # sourceCsvDir='%s/%s' % (sourceCsvDir, SYMBOL)

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)

    p.info('taking input dir %s for symbol[%s]' % (sourceCsvDir, SYMBOL))
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=sourceCsvDir, fields='date,time,open,high,low,close,volume,ammount')

    gymtdr = p.createApp(GymTrader, configNode ='trainer', tradeSymbol=SYMBOL, account=acc)
    
    p.info('all objects registered piror to OfflineSimulator: %s' % p.listByType())
    
    trainer = p.createApp(OfflineSimulator, configNode ='trainer', trader=gymtdr, histdata=csvreader)
    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = '%s/OfflineSimulator.tcsv' % trainer.outdir)
    trainer.setRecorder(rec)

    p.start()
    p.loop()
    p.stop()


if __name__ == '__main__':
#    from vprof import runner
#    runner.run(main_prog, 'cmhp', host='localhost', port=8000)
    main_prog()

'''
Note: The initial version of the distribution of CPU time is:
1) csv.bz2 read took 4%
2）agent.predict to determine action 17%
3) gymStep(mostly marketstate generating) 22%
4) gymObers(mostly brain.fit) 50%
'''

