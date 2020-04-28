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
class DnnAdvisor_S1548I4A3(TradeAdvisor):
    '''
    DnnAdvisor_S1548I4A3 impls TradeAdvisor by employing a pre-trained DNN model for state S1548I4A3
    '''
    STATE_DIMS  = 1548
    ITEM_FLOATS = EXPORT_FLOATS_DIMS
    ACTION_DIMS = len(ADVICE_DIRECTIONS) # =3

    def __init__(self, program, **kwargs) :
        self._brainId = None
        super(DnnAdvisor_S1548I4A3, self).__init__(program, **kwargs)

        self._brainId = self.getConfig('brainId', "default")

    @property
    def ident(self) :
        return 'S1548I4A3.%s.%s' % (self._brainId, self._id) if self._brainId else super(DnnAdvisor_S1548I4A3,self).ident

    def generateAdviceOnMarketEvent(self, ev):
        '''processing an incoming MarketEvent and generate an advice'''

        if MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            return None

        d = ev.data
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]

        market_state = self._marketState.exportKLFloats(symbol)
        market_state = np.array(market_state).astype(NN_FLOAT).reshape(1, DnnAdvisor_S1548I4A3.STATE_DIMS)

        act_values = self._brain.predict(market_state)
        # action = [0.0] * DnnAdvisor_S1548I4A3.ACTION_DIMS
        # idxAct = np.argmax(act_values[0])
        # action[idxAct] = 1.0
        advice = AdviceData(self.ident, symbol, d.exchange)
        advice.dirNONE, advice.dirLONG, advice.dirSHORT = act_values[0][0], act_values[0][1], act_values[0][2]
        advice.price = d.price if EVENT_TICK == ev.type else d.close

        return advice

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        brainDir = '%s%s.S1548I4A3/' % (self.dataRoot, self._brainId)
        self._brain = _loadBrain(self, brainDir)
        if not self._brain:
            self.error('doAppInit() failed to load brain[%s]' %self._brainId)
            return False

        return super(DnnAdvisor_S1548I4A3, self).doAppInit()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

