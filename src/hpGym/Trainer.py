from BackTest import BackTestApp, AccountWrapper

########################################################################
class GymTrainer(BackTestApp):
    '''
    GymTrader extends GymTrader by reading history and perform training
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(GymTrainer, self).__init__(program, **kwargs)
        self._iterationsPerEpisode = self.getConfig('iterationsPerEpisode', 1)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ

        # make sure GymTrainer is ONLY wrappering GymTrader
        if not self._initTrader or not isinstance(self._initTrader, GymTrader) :
            return False

        return super(GymTrainer, self).doAppInit() :

    def OnEvent(self, ev): 
        symbol  = None
        try :
            symbol = ev.data.symbol
        except:
            pass

        if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            self._account.matchTrades(ev)

        self.__wkTrader.OnEvent(ev) # to perform the gym step

        if not self._dataBegin_date:
            self._dataBegin_date = self.__wkTrader.marketState.getAsOf(symbol)

        
    # end of BaseApplication routine
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    # Overrides of events handling of Trader
    def eventHdl_Order(self, ev):
        return self.__wkTrader.eventHdl_Order(ev)
            
    def eventHdl_Trade(self, ev):
        return self.__wkTrader.eventHdl_Trade(ev)

    def onDayOpen(self, symbol, date):
        return self.__wkTrader.onDayOpen(symbol, date)

    def proc_MarketEvent(self, ev):
        self.error('proc_MarketEvent() should not be here')

    # end of Trader routine
    #----------------------------------------------------------------------

    #------------------------------------------------
    def OnEpisodeDone(self):
        super(GymTrainer, self).OnEpisodeDone()
        self.info('OnEpisodeDone() trained episode[%d/%d], total-reward[%s] epsilon[%s] loss[%d]' % (self.__episodeNo, self._episodes, 
            round(self.__wkTrader._total_reward, 2), round(self.__wkTrader._agent.epsilon, 2), round(self.__wkTrader.loss..history["loss"][0],4) )))

        # maybe self.__wkTrader.gymRender()

    # GymEnv related methods
    def resetEpisode(self) :
        '''
        reset the gym environment, will be called when each episode starts
        reset the trading environment / rewards / data generator...
        @return:
            observation (numpy.array): observation of the state
        '''
        super(GymTrainer, self).resetEpisode()
        return self.__wkTrader.gymReset()
