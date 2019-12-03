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
from Perspective import PerspectiveDict
from MarketData import EVENT_TICK, EVENT_KLINE_PREFIX

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

        self._memorySize = self.getConfig('memorySize', 2000)
        self._memory = [None] * self._memorySize
        self._idxMem = 0

        self._trainInterval = self.getConfig('trainInterval', 10)
        self._learningRate = self.getConfig('learningRate', 0.001)
        self._batchSize = self.getConfig('batchSize', 64)

        self._gamma = self.getConfig('gamma', 0.95)
        self._epsilon = self.getConfig('epsilon', 1.0) # rand() <= self._epsilon will trigger a random explore
        self._epsilonMin = self.getConfig('epsilonMin', 0.01)
        #TODO ?? self.__epsilonDecrement = (self._epsilon - self._epsilonMin) * self._trainInterval / (self._epsilon * episode_length)  # linear decrease rate

        self.__wkBrainId = None
        self._brain = self.buildBrain()

    @abstractmethod
    def isReady(self) :
        return not None in self._memory

    def getConfig(self, configName, defaultVal) :
        try :
            if configName in self._kwargs.keys() :
                return self._kwargs[configName]

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
    def buildBrain(self):
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
    def gymObserve(self, state, action, reward, next_state, done, warming_up=False):
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
        #TODO: the separate the Trader for real account and training account
        
        self.__1stRender = True
        self._AGENTCLASS = None
        agentType = self.getConfig('agent/type', 'DQN')
        if agentType and agentType in hpGym.GYMAGENT_CLASS.keys():
            self._AGENTCLASS = hpGym.GYMAGENT_CLASS[agentType]

        # step 1. GymTrader always take PerspectiveDict as the market state
        self._marketState = PerspectiveDict(None)
        self._gymState = None
        self._total_pnl = 0.0
        self._total_reward = 0.0

        # self.n_actions = 3
        # self._prices_history = []

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(GymTrader, self).doAppInit() :
            return False

        if not self._AGENTCLASS :
            return False

        # the self._account should be good here

        # step 1. GymTrader always take PerspectiveDict as the market state
        self._marketState._exchange = self._account.exchange
        self._account._marketState = self._marketState

        agentKwArgs = self.getConfig('agent', {})
        self._agent = self._AGENTCLASS(self, jsettings=self.subConfig('agent'), **agentKwArgs)
        # gymReset() will be called by agent above, self._gymState = self.gymReset() # will perform self._action = ACTIONS[ACTION_HOLD]

        while not self._agent.isReady() :
            action = self._agent.gymAct(self._gymState)
            next_state, reward, done, _ = self.gymStep(action)
            self._agent.gymObserve(self._gymState, action, reward, next_state, done, warming_up=True) # regardless state-stepping, rewards and loss here

        return True

    def doAppStep(self):
        super(GymTrader, self).doAppStep()
        # perform some dummy steps in order to fill agent._memory[]


    def OnEvent(self, ev): 
        # step 2. 收到行情后，在启动策略前的处理
        self._marketState.updateByEvent(ev)
        self._action = self._agent.gymAct(self._gymState)
        next_state, reward, done, _ = self.gymStep(self._action)
        loss = self._agent.gymObserve(self._gymState, self._action, reward, next_state, done)
        self._gymState = next_state
        # self._total_reward += reward has already been performed in above self.gymStep()

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

        observation = self.makeupGymObservation()
        self._shapeOfState = observation.shape
        self._action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]
        self._gymState = observation
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
        instant_pnl = 0
        info = {}

        # step 1. collected information from the account
        cashAvail, cashTotal, positions = self.getAccountState()
        capitalBeforeStep = self.__summrizeAccount(positions, cashTotal)

        # TODO: the first version only support one symbol to play, so simply take the first symbol in the positions        
        symbol = None # TODO: should take the __dictOberserves
        latestPrice = 0
        posAvail =0
        for s, pos in positions.values() :
            symbol =  s
            latestPrice = pos.price
            posAvailVol = pos.posAvail
            capitalBeforeStep += pos.price * pos.position * self._account.contractSize
            break

        reward = - self._timeCostYrRate # initialize with a time cost
        # reward = - capitalBeforeStep *self._timeCostYrRate/100/365

        if not symbol or latestPrice <=0:
            action = GymTrader.ACTIONS[GymTrader.ACTION_HOLD]

        #TODO?? if capitalBeforeStep <=0:
        #    done = True

        # step 2. perform the action buy/sell/hold by driving self._account
        if all(action == GymTrader.ACTIONS[GymTrader.ACTION_BUY]):
            # TODO: the first version only support FULL-BUY and FULL-SELL
            price  = latestPrice + self._account.priceTick
            volume = round(cashAvail / latestPrice / self._account.contractSize, 0)
            vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_BUY, price, volume, strategy=None)
            # cash will be updated in callback onOrderPlaced()
            # turnover, commission, slippage = self._account.calcAmountOfTrade(symbol, price, volume)
            # reward -= commission + slippage # TODO maybe after the order is comfirmed
        elif all(action == GymTrader.ACTIONS[GymTrader.ACTION_SELL]) and posAvail >0:
            price  = latestPrice - self._account.priceTick
            if price <= self._account.priceTick :
                price = self._account.priceTick 

            volume = - posAvail
            vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_SELL, price, volume, strategy=None)
            # cash will be updated in callback onOrderPlaced()
            # turnover, commission, slippage = self._account.calcAmountOfTrade(symbol, price, volume)
            # reward -= commission + slippage # TODO maybe after the order is comfirmed
            # if positions[self._symbol] ==0:
            #     self._exit_price = calculate based on price and commision # TODO maybe after the order is comfirmed
            #     instant_pnl = self._entry_price - self._exit_price
            #     self._entry_price = 0

        # step 3. calculate the rewards
        capitalAfterStep = self.__summrizeAccount() # most likely the cashAmount changed due to comission
        instant_pnl = capitalAfterStep - capitalBeforeStep
        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

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
            numpy.array: observation array.
        '''
        # part 1. build up the account_state
        cashAvail, cashTotal, positions = self.getAccountState()
        capitalBeforeStep = self.__summrizeAccount(positions, cashTotal)
        stateCapital = [cashAvail, cashTotal, capitalBeforeStep]
        # POS_COLS = PositionData.COLUMNS.split(',')
        # del(POS_COLS['exchange', 'stampByTrader', 'stampByBroker'])
        # del(POS_COLS['symbol']) # TODO: this version only support single Symbol, so regardless field symbol
        POS_COLS = 'position,posAvail,price,avgPrice'
        statePOS = [0.0,0.0,0.0,0.0] # supposed to be [[0.0,0.0,0.0,0.0],...] when mutliple-symbols
        for s, pos in positions.items() :
            # row = []
            # for c in POS_COLS:
            #     row.append(pos.__dict__[c])
            # statePOS.append(row)
            statePOS = [pos.position, pos.posAvailVol, pos.price, pos.avgPrice]
            break

        account_state = np.concatenate([stateCapital + statePOS], axis=0)

        # part 2. build up the market_state
        market_state = self._marketState.snapshot('000001')

        # return the concatenation of account_state and market_state as gymEnv sate
        ret = np.concatenate((account_state, market_state))
        return ret.astype('float32')

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])

    #----------------------------------------------------------------------
    # access to the account observed

    def __OnRenderClosed(self, evt):
        self.__closed_plot = True

    def getAccountState(self) :
        ''' get the account capitial including cash and positions
        '''
        if not self._account:
            return 0.0, 0.0, {}

        positions = self._account.getAllPositions()
        cashAvail, cashTotal = self._account.cashAmount()
        return cashAvail, cashTotal, positions

    def __summrizeAccount(self, positions=None, cashTotal=0) :
        ''' sum up the account capitial including cash and positions
        '''
        if positions is None:
            _, cashTotal, positions = self.getAccountState()

        posValueSubtotal =0
        for s, pos in positions.items():
            posValueSubtotal += pos.position * pos.price * self._account.contractSize

        return cashTotal + posValueSubtotal


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

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        # make sure GymTrainer is ONLY wrappering GymTrader
        if not self._initTrader or not isinstance(self._initTrader, GymTrader) :
            return False

        return super(GymTrainer, self).doAppInit()

    def OnEvent(self, ev): 
        symbol  = None
        try :
            symbol = ev.data.symbol
        except:
            pass

        if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            self._account.matchTrades(ev)

        self.wkTrader.OnEvent(ev) # to perform the gym step

        if not self._dataBegin_date:
            self._dataBegin_date = self.wkTrader.marketState.getAsOf(symbol)

        
    # end of BaseApplication routine
    #----------------------------------------------------------------------

    #------------------------------------------------
    # BackTest related entries
    def OnEpisodeDone(self):
        super(GymTrainer, self).OnEpisodeDone()
        self.info('OnEpisodeDone() trained episode[%d/%d], total-reward[%s] epsilon[%s] loss[%d]' % (self.__episodeNo, self._episodes, 
            round(self.wkTrader._total_reward, 2), round(self.wkTrader._agent.epsilon, 2), round(self.wkTrader.loss.history["loss"][0], 4) ))

        # maybe self.wkTrader.gymRender()

    def resetEpisode(self) :
        '''
        reset the gym environment, will be called when each episode starts
        reset the trading environment / rewards / data generator...
        @return:
            observation (numpy.array): observation of the state
        '''
        super(GymTrainer, self).resetEpisode()
        return self.wkTrader.gymReset()

if __name__ == '__main__':
    from Application import Program
    from Account import Account_AShare
    import HistoryData as hist
    import sys, os

    sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Gym_AShare.json']
    p = Program()
    p._heartbeatInterval =-1
    SYMBOL = '000001' # '000540' '000001'

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    csvdir = '/mnt/e/AShareSample' # '/mnt/m/AShareSample'
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder='%s/%s' % (csvdir, SYMBOL), fields='date,time,open,high,low,close,volume,ammount')
    # marketstate = PerspectiveDict('AShare')
    # p.addObj(marketstate)

    gymtdr = p.createApp(GymTrader, configNode ='trainer', account=acc)
    
    p.info('all objects registered piror to GymTrainer: %s' % p.listByType())
    
    p.createApp(GymTrainer, configNode ='trainer', trader=gymtdr, histdata=csvreader)

    p.start()
    p.loop()
    p.stop()

