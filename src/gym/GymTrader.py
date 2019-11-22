# encoding: UTF-8
'''
GymTrader impls BaseTrader and represent itself as a Gym Environment
'''

# from gym import GymEnv
from Trader import BaseTrader

from abc import ABC, abstractmethod
import matplotlib as mpl # pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
class GymTrader(BaseTrader):
    '''
    GymTrader impls BaseTrader and represent itself as a Gym Environment
    '''
    ACTION_BUY  = OrderData.ORDER_BUY
    ACTION_SELL = OrderData.ORDER_SELL
    ACTION_HOLD = 'HOLD'

    ACTIONS = {
        ACTION_HOLD: np.array([1, 0, 0]),
        ACTION_BUY:  np.array([0, 1, 0]),
        ACTION_SELL: np.array([0, 0, 1])
    }

    POS_DIRECTIONS = {
        OrderData.DIRECTION_NONE:  np.array([1, 0, 0]),
        OrderData.DIRECTION_LONG:  np.array([0, 1, 0]),
        OrderData.DIRECTION_SHORT: np.array([0, 0, 1])
    }

    def __init__(self, program, **kwargs):
        '''Constructor
        @param envAccount (AccountGEnv): the account env to observe and drive
        @param envMarket (MarketGEnv): the market env to observe
        @param timeCostYrRate (float%): the time interest cost of the capital
        '''
        super(GymTrader, self).__init__(program, **kwargs) # redirect to BaseTrader, who will adopt account and so on

        self._timeCostYrRate = self.getConfig('timeCostYrRate', 0)
        self._iterationsPerEpisode = self.getConfig('iterationsPerEpisode', 1)
        #TODO: the separate the Trader for real account and training account
        
        self.__1st_render = True
        # self.n_actions = 3
        # self._prices_history = []
        self.gymReset()

    '''
    def DEAD__init__(self, envAccount, envMarket, spread_coefficients, episode_length=10000, timeCostYrRate=0.0) :
        """Initialisation function

        @param envAccount (AccountGEnv): the account env to observe and drive
        @param envMarket (MarketGEnv): the market env to observe
        @param timeCostYrRate (float%): the time interest cost of the capital
        """
        self._account = envAccount
        self._envMarket = envMarket

        assert data_generator.n_products == len(spread_coefficients)
        assert history_length > 0
        # self._data_generator = data_generator
        self._spread_coefficients = spread_coefficients
        self._first_render = True
        # self._trading_fee = trading_fee
        self._timeCostYrRate = timeCostYrRate
        self._episode_length = episode_length
        self.n_actions = 3
        self._prices_history = []
        # self._history_length = history_length
        self.gymReset()
    '''

    #------------------------------------------------
    # GymEnv related methods
    def gymReset(self) :
        '''
        reset the gym environment, will be called when each episode starts
        reset the trading environment / rewards / data generator...
        @return:
            observation (numpy.array): observation of the state
        '''
        self.__execStamp_episodeStart = datetime.now()
        self.__closed_plot = False
        self.__stepId = 0
        
        self.info('gymReset() iteration[%d/%d], elapsed %s' % (self.__testRoundId, self._testRounds, str(self.__execStamp_episodeStart - self.__execStamp_appStart)))

        # step 1. start over the market state
        if self.__wkMarketState:
            self._program.removeObj(self.__wkMarketState)
        
        if self._initMarketState:
            self.__wkMarketState = copy.deepcopy(self._initMarketState)
            self._program.addObj(self.__wkMarketState)

        # step 2. create clean trader and account from self._initAcc and  
        if self.__wkTrader:
            self._program.removeObj(self.__wkTrader)
        self.__wkTrader = copy.deepcopy(self._initTrader)
        self._program.addApp(self.__wkTrader)
        self.__wkTrader._marketstate = self.__wkMarketState

        if self._account :
            self._program.removeApp(self._account)
            self._account =None
        
        self._account = copy.deepcopy(self._initAcc)
        self._program.addApp(self._account)
        self._account.setCapital(self._startBalance, True) # 回测时的起始本金（默认10万）
        self._account._marketstate = self.__wkMarketState
        self.__wkTrader._account = self._account
        self.__wkHistData.resetRead()
           
        self._dataBegin_date = None
        self._dataBegin_closeprice = 0.0
        
        self._dataEnd_date = None
        self._dataEnd_closeprice = 0.0

        # 当前最新数据，用于模拟成交用
        self.tick = None
        self.bar  = None
        self.__dtData  = None      # 最新数据的时间

        if self.__wkMarketState :
            for i in range(30) : # initially feed 20 data from histread to the marketstate
                ev = next(self.__wkHistData)
                if not ev : continue
                self.__wkMarketState.updateByEvent(ev)

            if len(self.__wkTrader._dictObjectives) <=0:
                sl = self.__wkMarketState.listOberserves()
                for symbol in sl:
                    self.__wkTrader.openObjective(symbol)

        # step 4. subscribe account events
        self.subscribeEvent(Account.EVENT_ORDER)
        self.subscribeEvent(Account.EVENT_TRADE)

        observation = self.__build_gym_observation()
        self.state_shape = observation.shape
        self._action = self.ACTIONS[ACTION_HOLD]
        return observation

    '''
    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...

        Returns:
            observation (numpy.array): observation of the state
        """
        self.__stepId = 0
        # self._data_generator.rewind()
        # self._total_reward = 0
        # self._total_pnl = 0
        # self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]
        # self._entry_price = 0
        # self._exit_price = 0

        self._account.reset()
        self._envMarket.reset()
        self.__closed_plot = False

        # for i in range(self._history_length):
        #     self._prices_history.append(self._data_generator.next())

        observation = self.__build_gym_observation()
        self.state_shape = observation.shape
        self._action = self.ACTIONS[ACTION_HOLD]
        return observation
    '''

    def gymStep(self, action) :
        '''Take an action (buy/sell/hold) and computes the immediate reward.

        @param action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        assert any([(action == x).all() for x in self.ACTIONS.values()])
        self._action = action
        self.__stepId += 1
        done = False
        instant_pnl = 0
        info = {}

        # step 1. collected information from the account
        positions = self._account.getAllPositions()
        cashAvail, cashTotal = self._account.cashAmount()
        totalCapital = cashTotal
        for s, pos in positions.values() :
            totalCapital += pos.price * pos.position * self._account.contractSize

        reward = - self._timeCostYrRate # initialize with a time cost
        # reward = - totalCapital *self._timeCostYrRate/100/365

        # step 2. perform the action buy/sell/hold by driving self._account
        if all(action == self.ACTIONS[ACTION_BUY]):
            volume = determine the volume to buy
            price = determine the price
            vtOrderIDList = self._account.sendOrder(self._symbol, OrderData.ORDER_BUY, price, volume, strategy=None)
            commission = calcluate the commission fee
            reward -= commission
            if positions[self._symbol] ==0:
                self._entry_price = calculate based on price and commision # TODO maybe after the order is comfirmed
        elif all(action == self.ACTIONS[ACTION_SELL]):
            volume = determine the volume to buy
            price = determine the price
            vtOrderIDList = self._account.sendOrder(self._symbol, OrderData.ORDER_SELL, price, volume, strategy=None)
            commission = calcluate the commission fee
            reward -= commission
            if positions[self._symbol] ==0:
                self._exit_price = calculate based on price and commision # TODO maybe after the order is comfirmed
                instant_pnl = self._entry_price - self._exit_price
                self._entry_price = 0
        '''

    def step(self, action):
        '''Take an action (buy/sell/hold) and computes the immediate reward.

        @param action (numpy.array): Action to be taken, one-hot encoded.

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        assert any([(action == x).all() for x in self.ACTIONS.values()])
        self._action = action
        self.__stepId += 1
        done = False
        instant_pnl = 0
        info = {}

        # step 1. collected information from the account
        positions = self._account.getAllPositions()
        cashAvail, cashTotal = self._account.cashAmount()
        totalCapital = cashTotal
        for s, pos in positions.values() :
            totalCapital += pos.price * pos.position * self._account.contractSize

        reward = - self._timeCostYrRate # initialize with a time cost
        # reward = - totalCapital *self._timeCostYrRate/100/365

        ''' step 2. perform the action buy/sell/hold
        TODO: drive self._account refer to vnAccount.step
        if all(action == self.ACTIONS[ACTION_BUY]):
            volume = determine the volume to buy
            price = determine the price
            vtOrderIDList = self._account.sendOrder(self._symbol, OrderData.ORDER_BUY, price, volume, strategy=None)
            commission = calcluate the commission fee
            reward -= commission
            if positions[self._symbol] ==0:
                self._entry_price = calculate based on price and commision # TODO maybe after the order is comfirmed
        elif all(action == self.ACTIONS[ACTION_SELL]):
            volume = determine the volume to buy
            price = determine the price
            vtOrderIDList = self._account.sendOrder(self._symbol, OrderData.ORDER_SELL, price, volume, strategy=None)
            commission = calcluate the commission fee
            reward -= commission
            if positions[self._symbol] ==0:
                self._exit_price = calculate based on price and commision # TODO maybe after the order is comfirmed
                instant_pnl = self._entry_price - self._exit_price
                self._entry_price = 0
        '''

        if all(action == self.ACTIONS[ACTION_BUY]):
            reward -= self._trading_fee
            if all(self._position == self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]):
                self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_LONG]
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
            elif all(self._position == self.POS_DIRECTIONS[OrderData.DIRECTION_SHORT]):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                instant_pnl = self._entry_price - self._exit_price
                self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]
                self._entry_price = 0
        elif all(action == self.ACTIONS[ACTION_SELL]):
            reward -= self._trading_fee
            if all(self._position == self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]):
                self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_SHORT]
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
            elif all(self._position == self.POS_DIRECTIONS[OrderData.DIRECTION_LONG]):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                instant_pnl = self._exit_price - self._entry_price
                self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]
                self._entry_price = 0

        ''' step 2.1. build up the account_state (numpy.array) based on
            postions, cache amount, maybe a sum-up
       
        self._account_state = ...
        '''

        ''' step 3. calculate the rewards
        TODO: drive self._account refer to vnAccount.step
        '''
        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

        ''' step 4. market observation and determine game over upon:
            a) market observation rearched end
            b) the account is well lost

        try :
            self._market_state = self._envMarket.next()
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self.__stepId >= self._iterationsPerEpisode:
            done = True
            info['status'] = 'Time out.'
        if self.__closed_plot:
            info['status'] = 'Closed plot'

        '''
        try:
            self._prices_history.append(self._data_generator.next())
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self.__stepId >= self._iterationsPerEpisode:
            done = True
            info['status'] = 'Time out.'
        if self.__closed_plot:
            info['status'] = 'Closed plot'

        ''' step 5. combine account and market observations as final observations,
            then return
        observation = np.concatenate((self._account_state, self._market_state))
        '''
        observation = self.__build_gym_observation()
        return observation, reward, done, info
    
    def _handle_close(self, evt):
        self.__closed_plot = True

    def render(self, savefig=False, filename='myfig'):
        """Matlplotlib rendering of each step.

        @param savefig (bool): Whether to save the figure as an image or not.
        @param filename (str): Name of the image file.
        """
        if self.__1st_render:
            self._f, self._ax = plt.subplots(
                len(self._spread_coefficients) + int(len(self._spread_coefficients) > 1),
                sharex=True
            )

            if len(self._spread_coefficients) == 1:
                self._ax = [self._ax]

            self._f.set_size_inches(12, 6)
            self.__1st_render = False
            self._f.canvas.mpl_connect('close_event', self._handle_close)

        if len(self._spread_coefficients) > 1:
            # TODO: To be checked
            for prod_i in range(len(self._spread_coefficients)):
                bid = self._prices_history[-1][2 * prod_i]
                ask = self._prices_history[-1][2 * prod_i + 1]
                self._ax[prod_i].plot([self.__stepId, self.__stepId + 1],
                                      [bid, bid], color='white')
                self._ax[prod_i].plot([self.__stepId, self.__stepId + 1],
                                      [ask, ask], color='white')
                self._ax[prod_i].set_title('Product {} (spread coef {})'.format(
                    prod_i, str(self._spread_coefficients[prod_i])))

        # Spread price
        prices = self._prices_history[-1]
        bid, ask = calc_spread(prices, self._spread_coefficients)
        self._ax[-1].plot([self.__stepId, self.__stepId + 1],
                          [bid, bid], color='white')
        self._ax[-1].plot([self.__stepId, self.__stepId + 1],
                          [ask, ask], color='white')
        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        if (self._action == self.ACTIONS[ACTION_SELL]).all():
            self._ax[-1].scatter(self.__stepId + 0.5, bid + 0.03 *
                                 yrange, color='orangered', marker='v')
        elif (self._action == self.ACTIONS[ACTION_BUY]).all():
            self._ax[-1].scatter(self.__stepId + 0.5, ask - 0.03 *
                                 yrange, color='lawngreen', marker='^')
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + [OrderData.DIRECTION_NONE, OrderData.DIRECTION_LONG, OrderData.DIRECTION_SHORT][list(self._position).index(1)] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price)
        self._f.tight_layout()
        plt.xticks(range(self.__stepId)[::5])
        plt.xlim([max(0, self.__stepId - 80.5), self.__stepId + 0.5])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename)

    def __build_gym_observation(self):
        """Concatenate all necessary elements to create the observation.

        Returns:
            numpy.array: observation array.
        """
        account_state = self._account.__build_gym_observation()
        market_state = self._envMarket.__build_gym_observation()
        return np.concatenate((account_state, market_state))
        # return np.concatenate(
        #     [prices for prices in self._prices_history[-self._history_length:]] +
        #     [
        #         np.array([self._entry_price]),
        #         np.array(self._position)
        #     ]
        # )

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])

"""
########################################################################
class AccountGEnv(GymEnv):
    '''Class for a sub-env based on a trading account
    '''
    def __init__(self, envTrading, account):
        '''Constructor
            @param envTrading (TradingEnv) the master TradingEnv
            @param account (nvApp.Account) the account to observe and drive
        '''
        self._envTrading = envTrading
        self._account = account
        self.gymReset()

    def reset(self):
        '''Reset the account
            1) for a real trading account, there is likely nothing to do at this step, maybe perform a reconnecting
            2) for a training or backtest account, reset the context data

        Returns:
            observation (numpy.array) collected from self._account consists of 
               a) positions: total and available to act
               b) cash amount: total and available to act
               c) ?? maybe the outgoing and/or failed orders
        '''
        self._total_reward = 0
        self._total_pnl = 0
        self._entry_price = 0
        self._exit_price = 0
        self._current_value = 0
        self.__closed_plot = False

        # load all the history data in the memory
        for i in range(self._history_length):
            self._prices_history.append(self._data_generator.next())

        observation = self.__build_gym_observation()
        self.state_shape = observation.shape
        self._action = self.ACTIONS[ACTION_HOLD]
        return observation

    @abstractmethod
    def __build_gym_observation(self):
        '''collect observation
            mostly call self._account to collect
        Returns:
            numpy.array: observation array.
        '''
        pass
        ''' TODO:
        call the account to collection postions, cache amount,
        sum-up to update self._current_value and so on
        '''

        return np.concatenate(
            [prices for prices in self._prices_history[-self._history_length:]] +
            [
                np.array([self._entry_price]),
                np.array(self._position)
            ]
        )


########################################################################
class MarketGEnv(GymEnv):
    '''class for a sub-env based on a trading markets
    '''

    def __init__(self, envTrading, perspectiveId):
        '''Constructor
            @param envTrading (TradingEnv) the master TradingEnv
            @param account (nvApp.Account) the account to observe and drive
        '''
        self._envTrading = envTrading
        self._perspectiveId = perspectiveId
        self.gymReset()

    def reset(self, perspectiveId):
        '''Reset the market data
            1) for a real trading market, it is the time to rebuild the CURRENT market perspective
            2) for a training or backtest, reset the market history data

        Returns:
            observation (numpy.array): converted from one of the market perspective classes, 
                which should be identified by unique class ids
        '''

        ''' TODO:
        perspectiveClazz = perspective.find(self._perspectiveId)
        self._histdata = perspectiveClazz(startTime=..., endTime=...) 
        '''

        _iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self.POS_DIRECTIONS[OrderData.DIRECTION_NONE]
        self._entry_price = 0
        self._exit_price = 0
        self.__closed_plot = False

        # load all the history data in the memory
        for i in range(self._history_length):
            self._prices_history.append(self._data_generator.next())

        observation = self.__build_gym_observation()
        self.state_shape = observation.shape
        self._action = self.ACTIONS[ACTION_HOLD]
        return observation
"""