# encoding: UTF-8
'''
Deep-NeuralNetworks based impls of Advisor and Trader
'''
from __future__ import division

from EventData    import EventData, datetime2float, EVENT_NAME_PREFIX
from MarketData   import *
from Perspective  import PerspectiveState
from Application  import BaseApplication, BOOL_STRVAL_TRUE
from TradeAdvisor import *
from Trader       import MetaTrader, BaseTrader
from Account      import OrderData

from tensorflow.keras.models import model_from_json

NN_FLOAT = 'float32'

def _loadBrain(app, brainDir) :
    ''' load the previous saved brain
    @param brainDir must be given, in which there are model.json definition and weights.h5 parameters
    '''
    try : 
        # step 1. read the model file in json
        app.debug('loading saved brain from %s' % brainDir)
        with open('%smodel.json' % brainDir, 'r') as mjson:
            model_json = mjson.read()
        brain = model_from_json(model_json)

        # step 2. read the weights of the model
        app.debug('loading saved brain weights from %s' %brainDir)
        brain.load_weights('%sweights.h5' % brainDir)

        app.info('loaded brain from %s' % (brainDir))
        return brain

    except Exception as ex:
        app.logexception(ex)

    return None

########################################################################
class DnnAdvisor(TradeAdvisor):
    '''
    DnnAdvisor impls TradeAdvisor by employing a pre-trained DNN model
    '''
    def __init__(self, program, **kwargs) :
        self._brainId = None
        super(DnnAdvisor, self).__init__(program, **kwargs)

        self._brainId = self.getConfig('brainId', "default")
        self.__stateSize, self.__actionSize = 1548, 3 #TODO

    @property
    def ident(self) :
        return 'NNAdv.%s.%s' % (self._brainId, self._id) if self._brainId else super(DnnAdvisor,self).ident

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

        brainDir = '%s%s.S%dI%dA%d/' % (self.dataRoot, self._brainId, self.__stateSize, EXPORT_FLOATS_DIMS, self.__actionSize)
        self._brain = _loadBrain(self, brainDir)
        if not self._brain:
            self.error('doAppInit() failed to load brain[%s]' %self._brainId)
            return False

        return super(DnnAdvisor, self).doAppInit()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

########################################################################
class DnnTrader(BaseTrader):
    '''
    DnnTrader impls BaseTrader by employing a pre-trained DNN model,
    this basic DnnTrader only trade a single objective specified via self._tradeSymbol
    '''
    def __init__(self, program, **kwargs):
        '''Constructor
        '''
        super(DnnTrader, self).__init__(program, **kwargs) # redirect to BaseTrader, who will adopt account and so on

        self._timeCostYrRate = self.getConfig('timeCostYrRate', 0)
        self._tradeSymbol    = self.getConfig('tradeSymbol', '000001')

        self._advDir = OrderData.DIRECTION_NONE

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(DnnTrader, self).doAppInit() :
            self.error('doAppInit() super-of[DnnTrader] failed')
            return False

        # step 1. redirect the marketState and account
        self._marketState._exchange = self._account.exchange
        self._account._marketState = self._marketState

        self.debug('doAppInit() done')
        return True

    def OnAdvice(self, evAdvice):
        '''
        processing an TradeAdvice, this basic DnnTrader takes whatever the advice told

        Take an action (buy/sell/hold) and computes the immediate reward.
        @param action (numpy.array): Action to be taken, one-hot encoded.
        @returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        '''
        if not evAdvice or EVENT_ADVICE != evAdvice.type :
            return

        adv = evAdvice.data

        # if bObserveOnly:
        #     return GymTrader.ACTIONS[GymTrader.ACTION_HOLD]

        dirToExec = ADVICE_DIRECTIONS[np.argmax([adv.dirNONE, adv.dirLONG, adv.dirSHORT])]
        strExec =''

        self.debug('OnAdvice() processing advDir[%s] %s' % (self._advDir, adv.desc))
        prevCap = self._latestCash + self._latestPosValue

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
        if OrderData.DIRECTION_LONG == dirToExec :
            if maxBuy <=0 :
                dirExeced = OrderData.DIRECTION_NONE
            else:
                strExec = '%s:%sx%s' %(dirToExec, latestPrice, maxBuy)
                self.debug('OnAdvice() issuing max%s' % strExec)
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_BUY, latestPrice, maxBuy, strategy=None)
                dirExeced = OrderData.DIRECTION_LONG

        elif OrderData.DIRECTION_SHORT == dirToExec :
            if maxSell <=0:
                dirExeced = OrderData.DIRECTION_NONE
            else:
                strExec = '%s:%sx%s' %(dirToExec, latestPrice, maxBuy)
                self.debug('OnAdvice() issuing max%s' % strExec)
                self._account.cancelAllOrders()
                vtOrderIDList = self._account.sendOrder(symbol, OrderData.ORDER_SELL, latestPrice, maxSell, strategy=None)

        # step 3. calculate the rewards
        self._latestCash, self._latestPosValue = self._account.summrizeBalance() # most likely the cashAmount changed due to comission
        capitalAfterStep = self._latestCash + self._latestPosValue

        # instant_pnl = capitalAfterStep - capitalBeforeStep
        # self._total_pnl += instant_pnl
    
    # end of impl/overwrite of BaseApplication
    #----------------------------------------------------------------------

