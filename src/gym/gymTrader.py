# encoding: UTF-8
'''
gymTrader impls BaseTrader and represent itself as a Gym Environment
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
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }

    def __init__(self, program, **kwargs):
        """Constructor
        @param envAccount (AccountGEnv): the account env to observe and drive
        @param envMarket (MarketGEnv): the market env to observe
        @param timeCostYrRate (float%): the time interest cost of the capital
        """
        super(GymTrader, self).__init__(program, **kwargs)

    def __init__(self, envAccount, envMarket, spread_coefficients, episode_length=10000, timeCostYrRate=0.0) :
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
        self.reset()

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...

        Returns:
            observation (numpy.array): observation of the state
        """
        self._iteration = 0
        # self._data_generator.rewind()
        # self._total_reward = 0
        # self._total_pnl = 0
        # self._position = self._positions['flat']
        # self._entry_price = 0
        # self._exit_price = 0

        self._account.reset()
        self._envMarket.reset()
        self._closed_plot = False

        # for i in range(self._history_length):
        #     self._prices_history.append(self._data_generator.next())

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

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
        assert any([(action == x).all() for x in self._actions.values()])
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0
        info = {}
        reward = - self._timeCostYrRate

        ''' step 1. collected information from the account
        TODO: make the self._account as vnApp.Account
        positions = self._account.getAllPositions()
        cashAvail, cashTotal = self._account.cashAmount()
        totalCapital = ...
        reward = - totalCapital *self._timeCostYrRate/100/365
        '''

        ''' step 2. perform the action buy/sell/hold
        TODO: drive self._account refer to vnAccount.step
        if all(action == self._actions['buy']):
            volume = determine the volume to buy
            price = determine the price
            vtOrderIDList = self._account.sendOrder(self._symbol, OrderData.ORDER_BUY, price, volume, strategy=None)
            commission = calcluate the commission fee
            reward -= commission
            if positions[self._symbol] ==0:
                self._entry_price = calculate based on price and commision # TODO maybe after the order is comfirmed
        elif all(action == self._actions['sell']):
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

        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
            elif all(self._position == self._positions['short']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions['flat']
                self._entry_price = 0
        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
            elif all(self._position == self._positions['long']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions['flat']
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
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        '''
        try:
            self._prices_history.append(self._data_generator.next())
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        ''' step 5. combine account and market observations as final observations,
            then return
        observation = np.concatenate((self._account_state, self._market_state))
        '''
        observation = self._get_observation()
        return observation, reward, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True

    def render(self, savefig=False, filename='myfig'):
        """Matlplotlib rendering of each step.

        @param savefig (bool): Whether to save the figure as an image or not.
        @param filename (str): Name of the image file.
        """
        if self._first_render:
            self._f, self._ax = plt.subplots(
                len(self._spread_coefficients) + int(len(self._spread_coefficients) > 1),
                sharex=True
            )

            if len(self._spread_coefficients) == 1:
                self._ax = [self._ax]

            self._f.set_size_inches(12, 6)
            self._first_render = False
            self._f.canvas.mpl_connect('close_event', self._handle_close)

        if len(self._spread_coefficients) > 1:
            # TODO: To be checked
            for prod_i in range(len(self._spread_coefficients)):
                bid = self._prices_history[-1][2 * prod_i]
                ask = self._prices_history[-1][2 * prod_i + 1]
                self._ax[prod_i].plot([self._iteration, self._iteration + 1],
                                      [bid, bid], color='white')
                self._ax[prod_i].plot([self._iteration, self._iteration + 1],
                                      [ask, ask], color='white')
                self._ax[prod_i].set_title('Product {} (spread coef {})'.format(
                    prod_i, str(self._spread_coefficients[prod_i])))

        # Spread price
        prices = self._prices_history[-1]
        bid, ask = calc_spread(prices, self._spread_coefficients)
        self._ax[-1].plot([self._iteration, self._iteration + 1],
                          [bid, bid], color='white')
        self._ax[-1].plot([self._iteration, self._iteration + 1],
                          [ask, ask], color='white')
        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        if (self._action == self._actions['sell']).all():
            self._ax[-1].scatter(self._iteration + 0.5, bid + 0.03 *
                                 yrange, color='orangered', marker='v')
        elif (self._action == self._actions['buy']).all():
            self._ax[-1].scatter(self._iteration + 0.5, ask - 0.03 *
                                 yrange, color='lawngreen', marker='^')
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + ['flat', 'long', 'short'][list(self._position).index(1)] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price)
        self._f.tight_layout()
        plt.xticks(range(self._iteration)[::5])
        plt.xlim([max(0, self._iteration - 80.5), self._iteration + 0.5])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename)

    # def _get_observation(self):
    #     """Concatenate all necessary elements to create the observation.

    #     Returns:
    #         numpy.array: observation array.
    #     """
    #     account_state = self._account._get_observation()
    #     market_state = self._envMarket._get_observation()
    #     return np.concatenate((account_state, market_state))
    #     # return np.concatenate(
    #     #     [prices for prices in self._prices_history[-self._history_length:]] +
    #     #     [
    #     #         np.array([self._entry_price]),
    #     #         np.array(self._position)
    #     #     ]
    #     # )

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])


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
        self.reset()

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
        self._closed_plot = False

        # load all the history data in the memory
        for i in range(self._history_length):
            self._prices_history.append(self._data_generator.next())

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

    @abstractmethod
    def _get_observation(self):
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
        self.reset()

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
        self._position = self._positions['flat']
        self._entry_price = 0
        self._exit_price = 0
        self._closed_plot = False

        # load all the history data in the memory
        for i in range(self._history_length):
            self._prices_history.append(self._data_generator.next())

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation
