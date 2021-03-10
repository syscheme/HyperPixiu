# encoding: UTF-8
'''
Deep-NeuralNetworks based impls of Advisor and Trader
'''
from __future__ import division

from EventData    import EventData, datetime2float, EVENT_NAME_PREFIX
from MarketData   import *
from Perspective  import PerspectiveState, Formatter_F1548, Formatter_2dImg32x18, Formatter_Snail32x32, Formatter_1d518
from Application  import BaseApplication, BOOL_STRVAL_TRUE
from TradeAdvisor import *
from Trader       import MetaTrader, BaseTrader
from Account      import OrderData

import HistoryData as hist
import ReplaySample as rs

from dnn.Makeups  import Model88, Model88_sliced

import tensorflow as tf
import numpy as np
import os, h5py

########################################################################
class DnnAdvisor(TradeAdvisor):
    '''
    DnnAdvisor impls TradeAdvisor by employing a pre-trained DNN model for state S1548I4A3
    '''
    STATE_DIMS  = 1548
    ITEM_FLOATS = EXPORT_FLOATS_DIMS
    ACTION_DIMS = len(ADVICE_DIRECTIONS) # =3

    FORMATTERS ={
        '1d518': Formatter_1d518,
        '2dImg32x18': Formatter_2dImg32x18,
        'F1548': Formatter_F1548,
        '2dImgSnail16': Formatter_Snail32x32,
    }

    def __init__(self, program, **kwargs) :
        self._brain, self.__id, self._fnModel = None, None, None
        super(DnnAdvisor, self).__init__(program, **kwargs)

        self._processor = self.getConfig('processor', None)
        self._type      = self.getConfig('type', "dnn_sliced2d")
        self._fnModel  = self.getConfig('modelPath', None)

        defaultFormatId  = list(DnnAdvisor.FORMATTERS.keys())[0]
        stateFormatId = self.getConfig('stateFormat', defaultFormatId)
        fmtClass = DnnAdvisor.FORMATTERS[stateFormatId] if stateFormatId in DnnAdvisor.FORMATTERS.keys() else DnnAdvisor.FORMATTERS[DnnAdvisor.FORMATTERS.keys()]
        self.__fmtr = fmtClass()
        
    # @property
    # def ident(self) :
    #     if self.__id: return self.__id

    #     id = None
    #     if self._brain : id = self._brain.modelId
    #     elif self._fnModel and len(self._fnModel)>0:
    #         id = os.path.basename(self._fnModel)
    #         if '.h5' == id[-3:] : id = id[:-3]
    #         if '.json' == id[-5:] : id = id[:-5]

    #         id = 'dnn.%s' % id

    #     return id if id else 'DnnAdvisor'

    def generateAdviceOnMarketEvent(self, ev, lastAdv=None):
        '''processing an incoming MarketEvent and generate an advice
            @param lastAdv is a reference in the case the advisor wish to refer to 
        '''

        if MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] :
            self.debug('generateAdviceOnMarketEvent() ignored event %s' % ev.type)
            return None

        if not EVENT_KLINE_PREFIX in ev.type and not EVENT_TICK in ev.type :
            self.debug('generateAdviceOnMarketEvent() not yet support event %s' % ev.type)
            return None

        d = ev.data
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]

        # because the gain-size of S1548I4A3 is KL1m, it is unnecessary to perform perdiction more freq-ly than once-per-minute
        if lastAdv and d.asof.minute == lastAdv.asof.minute:
            secAgo = datetime2float(d.asof) - datetime2float(lastAdv.asof)
            if (EVENT_TICK == ev.type and d.asof.second <5.0) or secAgo < 30.0 : # maximal twice during a same minute
                self.debug('generateAdviceOnMarketEvent() recently adviced %ss ago, skip predicting on event: %s' % (secAgo, ev.desc))
                return None

        # floatstate = self._marketState.exportF1548(symbol)
        floatstate = self._marketState.format(self.__fmtr, symbol)
        if not floatstate:
            self.debug('generateAdviceOnMarketEvent() rack of marketState on %s' % ev.desc)
            return None # skip advising pirior to plenty state data

        floatstate = np.array([floatstate]).astype(rs.SAMPLE_FLOAT)
        act_values = self._brain.predict(floatstate)
        # action = [0.0] * DnnAdvisor_S1548I4A3.ACTION_DIMS
        # idxAct = np.argmax(act_values[0])
        # action[idxAct] = 1.0
        advice = AdviceData(self.ident, symbol, d.exchange)
        advice.dirNONE, advice.dirLONG, advice.dirSHORT = act_values[0][0], act_values[0][1], act_values[0][2]
        advice.price = d.close if EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] else d.price

        return advice

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        if self._processor and len(self._processor) >1: # if specified to run on a given processor
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            devname = None
            if '/device:' in self._processor:
                devname = self._processor
            else:
                self._processor = self._processor.upper()
                for x in local_device_protos:
                    if x.device_type == self._processor or self._processor == x.name[len('/device:'):]:
                        devname = x.name
                        break
            if devname:
                devname = devname[len('/device:'):]
                with tf.device('/%s' % devname) :
                    self._brain = Model88_sliced.load(self._fnModel) if 'dnn_sliced2d' == conf_type \
                        else Model88.load(self._fnModel)

        if not self._brain:
            self._brain = Model88_sliced.load(self._fnModel) if 'dnn_sliced2d' == self._type \
                else Model88.load(self._fnModel)

        if not self._brain:
            self.error('doAppInit() failed to load brain[%s]' %self._brainId)
            return False

        self.__fmtr._channels = self._brain._input_shape[-1]

        return super(DnnAdvisor, self).doAppInit()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

