########################################################################
class MDPerspective(object):
    '''
    Data structure of Perspective：
    1. Ticks
    2. 1min KLines
    3. 5min KLines
    4. 1day KLines
    '''

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol =None):
        '''Constructor'''
        super(MDPerspective, self).__init__()
        self._data = {
            ticks:   [],
            KL_1min: [],
            KL_5min: [],
            KL_1day: [],
        }
        
        self._stampAsOf = None
        self._symbol    = EventData.EMPTY_STRING
        self._vtSymbol  = EventData.EMPTY_STRING
        self._exchange  = exchange
        if symbol and len(symbol)>0:
            self.symbol = self.vtSymbol = symbol
            if  len(exchange)>0 :
                self.vtSymbol = '.'.join([self.symbol, self.exchange])


########################################################################
class PerspectiveStack(object):
    '''
    Perspective合成器，支持：
    1. 基于x分钟K线输入（X可以是1、5、day	）
    2. 基于Tick输入
    '''

    #----------------------------------------------------------------------
    def __init__(self, exchange, symbol, KLDepth_1min=60, KLDepth_5min=240, KLDepth_1day=220, tickDepth=120) :
        '''Constructor'''
        self._depth ={
            ticks:   tickDepth,
            KL_1min: KLDepth_1min,
            KL_5min: KLDepth_5min,
            KL_1day: KLDepth_1day,
        }

        self._symbol   =symbol
        self._exchange =exchange
        self._mergerTick21Min    = TickToKLineMerger(self.cbMergedKLine1min)
        self._mergerKline1To5m   = KlineToXminMerger(self.cbMergedKLine5min, xmin=5)
        self._mergerKline5mToDay = KlineToXminMerger(self.cbMergedKLineDay,  xmin=240)
        self._currentPersective =  MDPerspective(self._symbol, self._exchange)
        for i in range(self._depth.ticks) ：
            self._currentPersective._data.ticks = [TickData(self._symbol, self._exchange)] + self._currentPersective._data.ticks
        for i in range(self._depth.KL_1min) ：
            self._currentPersective._data.KL_1min = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KL_1min
        for i in range(self._depth.KL_5min) ：
            self._currentPersective._data.KL_5min = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KL_5min
        for i in range(self._depth.KLDepth_1day) ：
            self._currentPersective._data.KLDepth_1day = [KLineData(self._symbol, self._exchange)] + self._currentPersective._data.KLDepth_1day

        self._stampStepped = self._stampNoticed = datetime.now()
        
    #----------------------------------------------------------------------
    def pushKLineXmin(self, kline, scaleXmin=1) :
        self._pushKLine(self, kline, scaleXmin) :
        if self._stampNoticed < self._stampStepped :
            self._currentPersective._stampAsOf = self._stampStepped
            self.OnNewPerspective(copy(self._currentPersective))
            self._stampNoticed = datetime.now()

    def OnNewPerspective(self, persective) :
        pass
           
    #----------------------------------------------------------------------
    # private callbacks for the mergers
    def cbMergedKLine1min(self, kline):
        self._pushKLine(kline, scaleXmin=1)

    def cbMergedKLine5min(self, kline):
        self._pushKLine(kline, scaleXmin=5)

    def cbMergedKLineDay(self, kline):
        self._pushKLine(kline, scaleXmin=240)

    #----------------------------------------------------------------------
    def _pushKLine(self, kline, scaleXmin=1):
        '''X分钟K线更新'''

        merger = None
        if scaleXmin ==1:
            klines = self._currentPersective.KL_1min
            merger = self._mergerKline1To5m
        elif scaleXmin ==5:
            klines = self._currentPersective.KL_5min
            merger = self._mergerKline5mToDay
        elif scaleXmin ==240:
            klines = self._currentPersective.KL_1day

        if (kline.exchange.find('_k2x') or kline.exchange.find('_t2k')) and klines[0].datetime > kline.datetime:
            return // ignore the late merging

        if klines[0].datetime.minute == kline.datetime.minute && klines[0].datetime.date == kline.datetime.date:
            klines[0] = kline
        else:
            klines = [kline] + klines[1:] # shift the existing list and push new kline at the front
            self._stampStepped = datetime.now()

        if merger:
            merger.pushKLine(kline)
        
    def pushTick(self, tick):
        '''Tick更新'''

        merger = self._mergerTick21Min
        ticks = self._currentPersective._data.ticks
        if ticks[0].datetime and tick.datetime < ticks[0].datetime
            return

        ticks = [tick] + ticks[1:] # shift the existing list and push new tick at the front
        self._stampStepped = datetime.now()
        if merger:
            merger.pushTick(tick)

