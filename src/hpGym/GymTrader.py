# encoding: UTF-8
'''
GymTrader impls BaseTrader and represent itself as a Gym Environment
'''
from __future__ import division

# from gym import GymEnv
from Account import Account, OrderData, Account_AShare
from Application import MetaObj
from Trader import MetaTrader, BaseTrader
from BackTest import BackTestApp
from Perspective import PerspectiveState
from MarketData import EVENT_TICK, EVENT_KLINE_PREFIX, EXPORT_FLOATS_DIMS

import hpGym

from abc import abstractmethod
import matplotlib as mpl # pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime

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
        self._epsilonMin = self.getConfig('epsilonMin', 0.02)

        self._wkBrainId = self.getConfig('brainId', None)

        self._trainInterval = self._batchSize /2
        if self._trainInterval < 10:
            self._trainInterval =10

        self._outDir = self.getConfig('outDir', self._gymTrader.dataRoot)
        if '/' != self._outDir[-1]:
            self._outDir +='/'

        self._statusAttrs = {}

        self._brain = self.buildBrain(self._wkBrainId)


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
    def gymAct(self, state):
        '''
        @return one of self.__gymTrader.ACTIONS
        '''
        raise NotImplementedError

    @abstractmethod
    def gymObserve(self, state, action, reward, next_state, done, **feedbacks):
        '''Memory Management and training of the agent
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
    ACTION_BUY  = OrderData.ORDER_BUY
    ACTION_SELL = OrderData.ORDER_SELL
    ACTION_HOLD = 'HOLD'

    ACTIONS = {
        ACTION_HOLD: np.array([1, 0, 0]).astype('float32'),
        ACTION_BUY:  np.array([0, 1, 0]).astype('float32'),
        ACTION_SELL: np.array([0, 0, 1]).astype('float32')
    }

    POS_DIRECTIONS = {
        OrderData.DIRECTION_NONE:  np.array([1, 0, 0]).astype('float32'),
        OrderData.DIRECTION_LONG:  np.array([0, 1, 0]).astype('float32'),
        OrderData.DIRECTION_SHORT: np.array([0, 0, 1]).astype('float32')
    }

    def __init__(self, program, agentClass=None, **kwargs):
        '''Constructor
        '''
        super(GymTrader, self).__init__(program, **kwargs) # redirect to BaseTrader, who will adopt account and so on

        self._agent = None
        self._timeCostYrRate = self.getConfig('timeCostYrRate', 0)
        self._tradeSymbol    = self.getConfig('tradeSymbol', '000001')
        #TODO: the separate the Trader for real account and training account
        
        self.__1stRender = True
        self._AGENTCLASS = None
        agentType = self.getConfig('agent/type', 'DQN')
        if agentType and agentType in hpGym.GYMAGENT_CLASS.keys():
            self._AGENTCLASS = hpGym.GYMAGENT_CLASS[agentType]

        # GymTrader always take PerspectiveState as the market state
        self._marketState = PerspectiveState(None)
        self._gymState = None

        self.__recentLoss = None
        self._total_pnl = 0.0
        self._total_reward = 0.0
        self._feedbackToAgent = {
            'dailyReward': 0.0, 
            'bestRewardTotal': 0.0,
            'bestRewardDays': 0,
        }

        self._dailyCapCost = 0.0 # just to ease calc
        self.__additionalReward = 0.0
        # self.n_actions = 3
        # self._prices_history = []
    @property
    def loss(self) :
        return round(self.__recentLoss.history["loss"][0], 6) if self.__recentLoss else DUMMY_BIG_VAL

    @property
    def withdrawReward(self) : # any read will reset the self.__additionalReward
        ret = self.__additionalReward
        self.__additionalReward = 0.0
        return ret

    def depositReward(self, reward) :
        self.__additionalReward += round(reward, 4)

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

        # self.__stampActStart = datetime.datetime.now()
        # self.__stampActEnd = self.__stampActStart

        self.debug('doAppInit() dummy gymStep to initialize cache')
        while not self._agent.isReady() :
            action = self._agent.gymAct(self._gymState)
            next_state, reward, done, _ = self.gymStep(action)
            self._agent.gymObserve(self._gymState, action, reward, next_state, done) # regardless state-stepping, rewards and loss here

        self.debug('doAppInit() done')
        return True

    def doAppStep(self):
        super(GymTrader, self).doAppStep()
        # perform some dummy steps in order to fill agent._memory[]

    def proc_MarketEvent(self, ev):
        '''processing an incoming MarketEvent'''

        action = self._agent.gymAct(self._gymState)
        strActionAdj =''
        
        # the gymAct only determin the direction, adjust the action to execute per current balance
        if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]) and self._latestCash < 1000:
            action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
            strActionAdj += '<BC'
        elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]) and self._latestPosValue < 1.0:
            action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
            strActionAdj += '<SP'

        if not all(action == GymTrader.ACTIONS[GymTrader.ACTION_HOLD]):
            if not self._account.executable:
                action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
                strActionAdj += '<Frz'

            dtAsOf = self.marketState.getAsOf()
            if (dtAsOf.hour in [14, 23] and dtAsOf.minute in [58,59]) or (dtAsOf.hour in [0, 15] and dtAsOf.minute in [0,1]) :
                action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
                strActionAdj += '<EoD'

        self._action = action

        next_state, reward, done, _ = self.gymStep(self._action)
    
        loss = self._agent.gymObserve(self._gymState, self._action, reward, next_state, done, **self._feedbackToAgent)
        if loss: self.__recentLoss =loss

        self._gymState = next_state
        self._total_reward += reward
        self.debug('proc_MarketEvent(%s) performed gymAct(%s%s) got reward[%s/%s] done[%s], agent ack-ed loss[%s]'% (ev.desc, action, strActionAdj, reward, self._total_reward, done, self.loss))

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

    def gymStep(self, action) :
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
        info = {}

        reward += - round(self._dailyCapCost /240, 4) # this is supposed the capital timecost every minute as there are 4 open hours every day
        # step 1. collected information from the account
        cashAvail, cashTotal, positions = self._account.positionState()
        _, posvalue = self._account.summrizeBalance(positions, cashTotal)
        capitalBeforeStep = cashTotal + posvalue

        # TODO: the first version only support one symbol to play, so simply take the first symbol in the positions        
        symbol = self._tradeSymbol # TODO: should take the __dictOberserves
        latestPrice = self._marketState.latestPrice(symbol)

        maxBuy, maxSell = self._account.maxOrderVolume(symbol, latestPrice)
        # TODO: the first version only support FULL-BUY and FULL-SELL
        if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]) :
            if maxBuy >0 :
                self.debug('gymStep() issuing maxBuy %s x%s' %(latestPrice, maxBuy))
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_BUY, latestPrice, maxBuy, strategy=None)
            else: reward -= 1 # penalty: is the agent blind to buy with no cash? :)
        elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]):
            if  maxSell >0:
                self.debug('gymStep() issuing maxSell %s x%s' %(latestPrice, maxSell))
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_SELL, latestPrice, maxSell, strategy=None)
            else: reward -= 1 # penalty: is the agent blind to sell with no position? :)
        else : reward += self.withdrawReward # only allow withdraw the depositted reward when action=HOLD

        # step 3. calculate the rewards
        prevCap = self._latestCash + self._latestPosValue
        self._latestCash, self._latestPosValue = self._account.summrizeBalance() # most likely the cashAmount changed due to comission
        capitalAfterStep = self._latestCash + self._latestPosValue
        if capitalAfterStep > self._maxBalance :
            self.depositReward(round((capitalAfterStep - self._maxBalance) *5, 4)) # exceeding maxBalance should be encouraged
            self._maxBalance = capitalAfterStep

        reward += round(capitalAfterStep - prevCap, 4)

        instant_pnl = capitalAfterStep - capitalBeforeStep
        self._total_pnl += instant_pnl

        ''' step 4. composing info for tracing

        try :
            self._market_state = self._envMarket.next()
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self.__stepNo >= self._iterationsPerEpisode:
            done = True
            info['status'] = 'Time out.'
        if self.__closed_plot:
            info['status'] = 'Closed plot'

        # try:
        #     self._prices_history.append(self._data_generator.next())
        # except StopIteration:
        #     done = True
        #     info['status'] = 'No more data.'
        # if self.__stepNo >= self._iterationsPerEpisode:
        #     done = True
        #     info['status'] = 'Time out.'
        # if self.__closed_plot:
        #     info['status'] = 'Closed plot'

        '''

        ''' step 5. combine account and market observations as final observations,
            then return
        observation = np.concatenate((self._account_state, self._market_state))
        '''
        observation = self.makeupGymObservation()
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
        market_state = self._marketState.snapshot(self._tradeSymbol)

        # TODO: more observations in the future could be:
        #  - [month, day, hour*60 +minute , weekday ]
        #  - money flow
        #  - market index

        # return the concatenation of account_state and market_state as gymEnv sate
        envState = np.concatenate((account_state, market_state))
        return envState.astype('float32')

    #----------------------------------------------------------------------
    # access to the account observed

    def __OnRenderClosed(self, evt):
        self.__closed_plot = True

########################################################################
class GymTrainer(BackTestApp):
    '''
    GymTrader extends GymTrader by reading history and perform training
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(GymTrainer, self).__init__(program, trader, histdata, **kwargs)
        self._iterationsPerEpisode = self.getConfig('iterationsPerEpisode', 1)
        self.__lastEpisode_loss = DUMMY_BIG_VAL

        self.__savedEpisode_Id = -1
        self.__savedEpisode_opendays = 0
        self.__savedEpisode_loss = DUMMY_BIG_VAL
        self.__savedEpisode_reward = -DUMMY_BIG_VAL
        self.__stampLastSaveBrain = '0000'
        self.__maxKnownOpenDays =0
        self.__prevMaxBalance =0

        # we encourage the train to reach end of history, so give some reward every week for its survive
        self.__rewardDayStepped = float(self._startBalance) * self._initTrader._annualCostRatePcnt /220 / 100

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        # make sure GymTrainer is ONLY wrappering GymTrader
        if not self._initTrader or not isinstance(self._initTrader, GymTrader) :
            self.error('doAppInit() invalid initTrader')
            return False

        if not super(GymTrainer, self).doAppInit() :
            return False

        self.wkTrader.gymReset()
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

    #------------------------------------------------
    # BackTest related entries
    def OnEpisodeDone(self, reachedEnd=True):
        super(GymTrainer, self).OnEpisodeDone(reachedEnd)

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
            rewardMean = self.wkTrader._total_reward /opendays
            self._episodeSummary['dailyReward'] = rewardMean
            self.wkTrader._feedbackToAgent['dailyReward'] = rewardMean
            rewardMeanBest = self.__savedEpisode_reward
            if self.__savedEpisode_opendays>0 :
                rewardMeanBest /= self.__savedEpisode_opendays

            if rewardMean > rewardMeanBest or (rewardMean > rewardMeanBest/2 and opendays > self.__savedEpisode_opendays *1.2):
                lstImproved.append('meanReward')
                self.debug('OnEpisodeDone() meanReward improved from %s/%s to %s/%s' % (self.__savedEpisode_reward, self.__savedEpisode_opendays, self.wkTrader._total_reward, opendays))

            # determin best by loss
            if self.wkTrader.loss < self.__savedEpisode_loss and opendays >= self.__savedEpisode_opendays:
                lstImproved.append('loss')

        # save brain and decrease epsilon if improved
        if len(lstImproved) >0 :
            self.wkTrader._feedbackToAgent['bestRewardTotal'] = self.wkTrader._total_reward
            self.wkTrader._feedbackToAgent['bestRewardDays'] = opendays
            if self.__savedEpisode_loss < DUMMY_BIG_VAL: # do not save for the first episode
                self.wkTrader._feedbackToAgent['improved'] = lstImproved
                self.wkTrader._agent.saveBrain(**self.wkTrader._feedbackToAgent)
                self.__stampLastSaveBrain = datetime.datetime.now()
                self.info('OnEpisodeDone() brain saved per improvements: %s' % lstImproved )

            self.__savedEpisode_opendays = opendays
            self.__savedEpisode_loss = self.wkTrader.loss
            self.__savedEpisode_Id = self.episodeId
            self.__savedEpisode_reward = self.wkTrader._total_reward

        mySummary = {
            'totalReward' : round(self.wkTrader._total_reward, 2),
            'epsilon'     : round(self.wkTrader._agent._epsilon, 6),
            'learningRate': self.wkTrader._agent._learningRate,
            'loss'        : self.wkTrader.loss,
            'lastLoss'    : self.__lastEpisode_loss,
            'savedLoss'   : self.__savedEpisode_loss,
            'savedEId'    : self.__savedEpisode_Id,
            'savedReward' : round(self.__savedEpisode_reward,2),
            'savedODays'  : self.__savedEpisode_opendays,
            'savedTime'   : self.__stampLastSaveBrain.strftime('%Y%m%dT%H%M%S') if isinstance(self.__stampLastSaveBrain, datetime.datetime) else self.__stampLastSaveBrain,
            'frameNum'    : self.wkTrader._agent.frameNum
        }
        self._episodeSummary = {**self._episodeSummary, **mySummary}
        self.__lastEpisode_loss = self.wkTrader.loss

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
        super(GymTrainer, self).resetEpisode()
        return self.wkTrader.gymReset()

    def formatSummary(self, summary=None):
        strReport = super(GymTrainer, self).formatSummary(summary)
        if not isinstance(summary, dict) :
            summary = self._episodeSummary

        strReport += '\n totalReward: %s'  % summary['totalReward']
        strReport += '\n     epsilon: %s'  % summary['epsilon']
        strReport += '\nlearningRate: %s'  % summary['learningRate']
        strReport += '\n        loss: %s from %s' % (summary['loss'], summary['lastLoss'])
        strReport += '\n   savedLoss: %s <-(%s: %sd reward=%s @%s)' % (summary['savedLoss'], summary['savedEId'], summary['savedODays'], summary['savedReward'], summary['savedTime'])
        return strReport

if __name__ == '__main__':
    from Application import Program
    from Account import Account_AShare
    import HistoryData as hist
    import sys, os
    # from keras.backend.tensorflow_backend import set_session
    # import tensorflow as tf

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # set_session(tf.Session(config=config))

    sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Gym_AShare.json']
    p = Program()
    p._heartbeatInterval =-1
    SYMBOL = '000001' # '000540' '000001'

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    csvdir = '/mnt/e/AShareSample' # '/mnt/m/AShareSample'
    for d in ['e:/AShareSample', '/mnt/e/AShareSample', '/mnt/m/AShareSample']:
        try :
            if  os.stat(d):
                csvdir = d
                break
        except :
            pass

    p.info('taking input dir %s for symbol[%s]' % (csvdir, SYMBOL))
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder='%s/%s' % (csvdir, SYMBOL), fields='date,time,open,high,low,close,volume,ammount')
    # marketstate = PerspectiveState('AShare')
    # p.addObj(marketstate)

    gymtdr = p.createApp(GymTrader, configNode ='trainer', tradeSymbol=SYMBOL, account=acc)
    
    p.info('all objects registered piror to GymTrainer: %s' % p.listByType())
    
    trainer = p.createApp(GymTrainer, configNode ='trainer', trader=gymtdr, histdata=csvreader)
    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = '%s/GymTrainer.tcsv' % trainer.outdir)
    trainer.setRecorder(rec)

    p.start()
    p.loop()
    p.stop()

