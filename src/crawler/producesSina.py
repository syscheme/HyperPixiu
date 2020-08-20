# encoding: UTF-8

from __future__ import division

from Simulator import IdealTrader_Tplus1, ShortSwingScanner
from EventData import Event, EventData
from MarketData import *
import HistoryData as hist
from crawler.crawlSina import SinaCrawler

from datetime import datetime, timedelta
import os
import fnmatch

def _makeupMux(simulator, dirOffline):

    dtStart, _ = simulator._wkHistData.datetimeRange
    symbol = simulator._tradeSymbol

    # part.1 the weekly tcsv collection by advisors that covers KL5m, MF1m, and Ticks
    # in the filename format such as SZ000001_sinaWk20200629.tcsv
    fnAll = hist.listAllFiles(dirOffline)
    fnFilter = '%s_sinaWk[0-9]*.tcsv' % symbol 
    for fn in fnAll:
        if not fnmatch.fnmatch(os.path.basename(fn), fnFilter):
            continue

        simulator.debug("__makeupMux() loading offline file: %s" %(fn))
        f = open(fn, "rb")
        pb = hist.TaggedCsvStream(f, program=simulator.program)
        pb.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        pb.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        pb.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        pb.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        simulator._wkHistData.addStream(pb)

    # part.2 the online daily-data that supply a long term data that might be not covered by the above advisors
    crawl = SinaCrawler(simulator.program, None)
    days = (datetime.now() - dtStart).days +2
    if days > 300: days =300

    # part.2.1 EVENT_KLINE_1DAY
    evtype = EVENT_KLINE_1DAY
    simulator.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
    httperr, dataseq = crawl.GET_RecentKLines(symbol, 240, days)
    if 200 != httperr or len(dataseq) <=0:
        simulator.error("__makeupMux() GET_RecentKLines(%s:%s) failed, err(%s) len(%d)" %(symbol, evtype, httperr, len(dataseq)))
    else:
        # succ at query
        pb, c = hist.Playback(symbol, program=simulator.program), 0
        for i in dataseq:
            ev = Event(evtype)
            ev.setData(i)
            pb.enquePending(ev)
            c+=1

        simulator._wkHistData.addStream(pb)
        simulator.info('__makeupMux() added online query as source of event[%s] len[%d]' % (evtype, c))

    evtype = EVENT_MONEYFLOW_1DAY
    simulator.debug('taking online query as source of event[%s] of %ddays' % (evtype, days))
    httperr, dataseq = crawl.GET_MoneyFlow(symbol, days, False)
    if 200 != httperr or len(dataseq) <=0:
        simulator.error("__makeupMux() GET_MoneyFlow(%s:%s) failed, err(%s) len(%d)" %(symbol, evtype, httperr, len(dataseq)))
    else:
        # succ at query
        pb, c = hist.Playback(symbol, program=simulator.program), 0
        for i in dataseq:
            ev = Event(evtype)
            ev.setData(i)
            pb.enquePending(ev)
            c+=1

        simulator._wkHistData.addStream(pb)
        simulator.info('__makeupMux() added online query as source of event[%s] len[%d]' % (evtype, c))

    return simulator._wkHistData.size >0

########################################################################
class Sina_Tplus1(IdealTrader_Tplus1):
    '''
    Sina_Tplus1 extends IdealTrader_Tplus1 based on online and offline data collected from Sina
    '''
    def __init__(self, program, trader, symbol, dirOfflineData, **kwargs):
        '''Constructor
        '''
        mux = hist.PlaybackMux(program=program) # not start/end data specified, startDate =startDate, endDate=endDate)

        super(Sina_Tplus1, self).__init__(program, trader, histdata=mux, **kwargs) # mux will be kept as self._wkHistData

        self._dirOfflineData = dirOfflineData
        self._tradeSymbol = symbol

    def doAppInit(self): # return True if succ
        # load the offline tcsv streams and the online daily streams int self._wkHistData as a PlaybackMux
        if not _makeupMux(self, self._dirOfflineData):
            return False

        if not super(Sina_Tplus1, self).doAppInit() :
            return False
        
        return True

    # Directly takes that of IdealTrader_Tplus1
    #  - def OnEvent(self, ev)
    #  - def resetEpisode(self)
    #  - def OnEpisodeDone(self, reachedEnd=True)
    #  - def doAppStep(self)

########################################################################
class SinaSwingScanner(ShortSwingScanner):
    '''
    ShortSwingScanner extends OfflineSimulator by scanning the MarketEvents occurs up to several days, determining
    the short trend
    '''
    def __init__(self, program, trader, symbol, dirOfflineData, f4schema=None, **kwargs):
        '''Constructor
        '''
        mux = hist.PlaybackMux(program=program) # not start/end data specified, startDate =startDate, endDate=endDate)

        super(SinaSwingScanner, self).__init__(program, trader, mux, **kwargs)

        self._dirOfflineData = dirOfflineData
        self._tradeSymbol = symbol

    def doAppInit(self): # return True if succ
        # load the offline tcsv streams and the online daily streams int self._wkHistData as a PlaybackMux
        if not _makeupMux(self, self._dirOfflineData):
            return False

        if not super(SinaSwingScanner, self).doAppInit() :
            return False

        return True
