# encoding: UTF-8

'''
BackTest inherits from Trader
'''
from __future__ import division

from Application import *
from Account import *
from Trader import *
import HistoryData as hist
from MarketData import TickData, KLineData, NORMALIZE_ID, EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY, MARKETDATE_EVENT_PREFIX

from Perspective import *
import vn.VnTrader as vn

from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import multiprocessing
import copy

import os
import h5py, tarfile
# import pymongo
import pandas as pd
import numpy as np
import shutil
import codecs
import math

# TODO: move the plotting stuf into a separate python program
# import matplotlib as mpl # pip install matplotlib
# import matplotlib.pyplot as plt
# mpl.use('Agg')

# # 如果安装了seaborn则设置为白色风格
# try:
#     import seaborn as sns       
#     sns.set_style('whitegrid')  
# except ImportError:
#     pass

RFGROUP_PREFIX = 'ReplayFrame:'
RECCATE_ESPSUMMARY = 'EspSum'
COLUMNS_ESPSUMMARY ='episodeNo,endBalance,openDays,startDate,endDate,totalDays,tradeDay_1st,tradeDay_last,profitDays,lossDays,maxDrawdown,maxDdPercent,' \
    + 'totalNetPnl,dailyNetPnl,totalCommission,dailyCommission,totalSlippage,dailySlippage,totalTurnover,dailyTurnover,totalTradeCount,dailyTradeCount,totalReturn,annualizedReturn,dailyReturn,' \
    + 'returnStd,sharpeRatio,episodeDuration,stepsInEpisode,totalReward,dailyReward,epsilon,loss,savedEId,savedLoss,savedReward,savedODays,savedTime,frameNum,reason'

########################################################################
class BackTestApp(MetaTrader):
    '''
    BackTest is a wrapprer of Trader, with same interface of Trader
    '''
    #----------------------------------------------------------------------
    def __init__(self, program, trader, histdata, **kwargs):
        """Constructor"""

        super(BackTestApp, self).__init__(program, **kwargs)
        self._initTrader = trader
        self._initMarketState = None # to populate from _initTrader
        self._originAcc = None # to populate from _initTrader then wrapper

        self._account = None # the working account inherit from MetaTrader
        self._marketState = None
        self.__wkTrader = None
        self._wkHistData = histdata
        
        self.setRecorder(self._initTrader.recorder)

        # 回测相关属性
        # -----------------------------------------
        self._startBalance = 100000      # 10w

        self._btStartDate    = datetime.strptime(self.getConfig('backTest/startDate', '2000-01-01'), '%Y-%m-%d').replace(hour=0, minute=0, second=0) # 回测数据开始日期，datetime对象
        self._btEndDate      = datetime.strptime(self.getConfig('backTest/endDate', '2999-12-31'), '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        self._startBalance   = self.getConfig('backTest/startBalance', 100000)
        self._episodes       = self.getConfig('backTest/episodes', 1)
        self._pctMaxDrawDown = self.getConfig('backTest/pctMaxDrawDown', 21) # we allow 30% lost during a episode
        self._warmupDays     = self.getConfig('backTest/warmupDays', 5) # observe for a week by default to make the market state not so empty
        self._plotReport     = self.getConfig('backTest/plotReport', 'False').lower() in BOOL_STRVAL_TRUE

        self._episodeNo = 1 # count start from 1 to ease reading
        self._stepNoInEpisode =0
        self.__execStamp_appStart = datetime.now()
        self.__execStamp_episodeStart = self.__execStamp_appStart

        self._dailyCapCost = float(self._startBalance) * self._initTrader._annualCostRatePcnt /220 / 100  # assuming 220 opendays every year
        self._maxBalance = self._startBalance

        # backtest will always clear the datapath
        try :
            shutil.rmtree(self.outdir)
        except:
            pass
        try :
            os.makedirs(self.outdir)
        except:
            pass

        self._bGameOver = False
        self._openDays = 0

    @property
    def episodeId(self) :
        return 'E' + str(self._episodeNo).zfill(6)

    @property
    def wkTrader(self) :
        return self.__wkTrader

    def setTimeRange(self, dtStart, dtEnd = None) :
        if dtStart: self._btStartDate = dtStart
        if dtEnd:   self._btEndDate = dtEnd

        return self._btStartDate, self._btEndDate

    def setRecorder(self, recorder) :
        self._recorder = recorder
        if self._recorder :
            self._recorder.registerCategory(Account.RECCATE_ORDER,       params= {'columns' : OrderData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_TRADE,       params= {'columns' : TradeData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYPOS,    params= {'columns' : DailyPosition.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYRESULT, params= {'columns' : DailyResult.COLUMNS})
            self._recorder.registerCategory(RECCATE_ESPSUMMARY,          params= {'columns' : COLUMNS_ESPSUMMARY })

        if self.__wkTrader :
            self.__wkTrader._recorder = self._recorder
            
        return self._recorder

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    @property
    def ident(self) :
        tdrCore = self.__wkTrader if self.__wkTrader else self._initTrader
        return '%s/%s' % (self.__class__.__name__, (tdrCore.ident if tdrCore else self._id))

    def stop(self):
        """退出程序前调用，保证正常退出"""
        if self._marketState:
            self._program.removeObj(self._marketState)
        self._marketState = None

        # step 2. create clean trader and account from self._account and  
        if self.__wkTrader:
            self._program.removeApp(self.__wkTrader)
        self.__wkTrader = None

        if self._account :
            self._program.removeApp(self._account)

        return super(BackTestApp, self).stop()

    def __taglog(self, message):
        if not self.marketState: return message
        return '%s@%s ' % (self.episodeId, self.marketState.getAsOf().strftime('%Y%m%dT%H%M%S')) + message

    def debug(self, message):
        super(BackTestApp, self).debug(self.__taglog(message))
    
    def info(self, message):
        super(BackTestApp, self).info(self.__taglog(message))

    def warn(self, message):
        super(BackTestApp, self).warn(self.__taglog(message))

    def error(self, message):
        super(BackTestApp, self).error(self.__taglog(message))

    def log(self, level, message):
        super(BackTestApp, self).log(level, self.__taglog(message))

    def doAppInit(self): # return True if succ
        if not super(BackTestApp, self).doAppInit() :
            return False

        self.__dtLastData = self._btStartDate

        # step 1. wrapper the Trader
        if not self._initTrader :
            return False
        
        self.debug('doAppInit() taking trader-template[%s]' % (self._initTrader.ident))
        self._program.removeApp(self._initTrader)
        self._program.addApp(self)
        if not self._initTrader.doAppInit() :
            self.info('doAppInit() failed to initialize trader-template[%s]' % (self._initTrader.ident))
            return False
            
        self._program.removeApp(self._initTrader) # in the case self._initTrader register itself again
        self.info('doAppInit() wrapped[%s]' % (self._initTrader.ident))

        self._initMarketState = self._initTrader._marketState
        self._originAcc = self._initTrader.account
        if self._originAcc and not isinstance(self._originAcc, AccountWrapper):
            self._program.removeApp(self._originAcc.ident)

        # # ADJ_1. adjust the Trader._dictObjectives to append suffix MarketData.TAG_BACKTEST
        # for obj in self._dictObjectives.values() :
        #     if len(obj["dsTick"]) >0 :
        #         obj["dsTick"] += MarketData.TAG_BACKTEST
        #     if len(obj["ds1min"]) >0 :
        #         obj["ds1min"] += MarketData.TAG_BACKTEST

        # step 3.1 subscribe the TradeAdvices
        self.subscribeEvents([EVENT_ADVICE, EVENT_TICK_OF_ADVICE])
        
        # step 3.2 subscribe the market events
        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
        self.subscribeEvents([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY])

        self.resetEpisode()
        _quitEpisode = False
        return True

    def doAppStep(self):
        super(BackTestApp, self).doAppStep()
        self._account.doAppStep()

        reachedEnd = False

        if self._wkHistData and not self._bGameOver:
            try :
                ev = next(self._wkHistData)
                if not ev or ev.data.datetime < self._btStartDate: return
                if ev.data.datetime <= self._btEndDate:
                    self._marketState.updateByEvent(ev)
                    s = ev.data.symbol
                    price, asofP = self._marketState.latestPrice(s)
                    self.debug('hist-read: symbol[%s]%s asof[%s] lastPrice[%s] OHLC%s' % (s, ev.type[len(MARKETDATE_EVENT_PREFIX):], asofP.strftime('%Y%m%dT%H%M'), price, self._marketState.dailyOHLC_sofar(s)))
                    self.postEvent(ev) # self.OnEvent(ev) # call Trader
                    self._stepNoInEpisode += 1
                    return # successfully performed a step by pushing an Event

                reachedEnd = True

            except StopIteration:
                reachedEnd = True
                self.info('hist-read: end of playback')
            except Exception as ex:
                self.logexception(ex)

        # this test should be done if reached here
        self.info('doAppStep() episode[%s/%s] finished: %d steps, KO[%s] end-of-history[%s]' % (self.episodeId, self._episodes, self._stepNoInEpisode, self._bGameOver, reachedEnd))
        try:
            self.OnEpisodeDone(reachedEnd)
        except Exception as ex:
            self.logexception(ex)

        # print the summary report
        if self._recorder and isinstance(self._episodeSummary, dict):
            self._recorder.pushRow(RECCATE_ESPSUMMARY, self._episodeSummary)

        strReport = self.formatSummary()
        self.info('%s_%s summary:' %(self.ident, self.episodeId))
        for line in strReport.splitlines():
            if len(line) <2: continue
            self.info(line)

        strReport += '\n'
        with codecs.open('%s/%s_summary.txt' %(self._initTrader.outdir, self.episodeId), "w","utf-8") as rptfile:
            rptfile.write(strReport)
            self.info('doAppStep() episode[%s/%s] summary report generated' %(self.episodeId, self._episodes))

        # prepare for the next episode
        self._episodeNo +=1
        if (self._episodeNo > self._episodes) :
            # all tests have been done
            self.stop()
            self.info('all %d episodes have been done, took %s, app stopped. obj-in-program: %s' % (self._episodes, str(datetime.now() - self.__execStamp_appStart), self._program.listByType(MetaObj)))
            exit(0)
            return

        self.info('-' *30)
        self.debug('doAppStep() starting over new episode[%s]' %(self.episodeId))
        self.resetEpisode()
        self._bGameOver =False

    def OnEvent(self, ev): 
        # step 2. 收到行情后，在启动策略前的处理
        evd = ev.data
        matchNeeded = False
        if not self.__dtLastData or self.__dtLastData < evd.asof :
            self.__dtLastData = evd.asof
            self._dataEnd_date = evd.date
            if EVENT_TICK == ev.type:
                self._dataEnd_closeprice = evd.price
                matchNeeded = True
            elif EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
                self._dataEnd_closeprice = evd.close
                matchNeeded = True

            if not self._dataBegin_date:
                # NO self._dataBegin_date = evd.date here because events other than tick and KL could be in
                if EVENT_TICK == ev.type:
                    self._dataBegin_date = evd.date
                    self._dataBegin_openprice = evd.price
                elif EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
                    self._dataBegin_date = evd.date
                    self._dataBegin_openprice = evd.close

        if matchNeeded :
            self._account.matchTrades(ev)

        return self.__wkTrader.OnEvent(ev)

    # end of BaseApplication routine
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    # Overrides of Events handling
    def eventHdl_Order(self, ev):
        return self.__wkTrader.eventHdl_Order(ev)
            
    def eventHdl_Trade(self, ev):
        return self.__wkTrader.eventHdl_Trade(ev)

    def eventHdl_DayOpen(self, symbol, date):
        return self.__wkTrader.eventHdl_DayOpen(symbol, date)

    #------------------------------------------------
    # 数据回放结果计算相关

    def OnEpisodeDone(self, reachedEnd=True):

        additionAttrs = {
            'openDays' : len(self._account.dailyResultDict),
            'episodeDuration' : round(datetime2float(datetime.now()) - datetime2float(self.__execStamp_episodeStart), 3),
            'episodeNo' : self._episodeNo,
            'episodes' : self._episodes,
            'stepsInEpisode' : self._stepNoInEpisode,
        }

        self._episodeSummary = {**self._episodeSummary, **additionAttrs}

        self.info('OnEpisodeDone() episode[%d/%d], processed %d events in %d opendays took %ssec, composing summary' % 
            (additionAttrs['episodeNo'], additionAttrs['episodes'], additionAttrs['stepsInEpisode'], additionAttrs['openDays'], additionAttrs['episodeDuration']) )

        tradeDays, summary = calculateSummary(self._startBalance, self._account.dailyResultDict)

        self._account.OnPlaybackEnd()

        if not summary or not isinstance(summary, dict) :
            self.error('no summary given: %s' % summary)
        else: 
            self._episodeSummary = {**self._episodeSummary, **summary}

        if not tradeDays is None:
            # has been covered by tcsv recorder
            # csvfile = '%s/%s_DR.csv' %(self._initTrader.outdir, self.episodeId)
            # self.debug('OnEpisodeDone() episode[%s], saving trade-days into %s' % (self.episodeId, csvfile))
            # try :
            #     os.makedirs(os.path.dirname(csvfile))
            # except:
            #     pass

            # tradeDays.to_csv(csvfile)

            if self._plotReport :
                self.plotResult(tradeDays)

    def resetEpisode(self) :

        self._episodeSummary = {}
        self.__execStamp_episodeStart = datetime.now()
        self._stepNoInEpisode =0
        self.debug('resetEpisode() initializing episode[%d/%d], elapsed %s obj-in-program: %s' % (self._episodeNo, self._episodes, str(self.__execStamp_episodeStart - self.__execStamp_appStart), self._program.listByType(MetaObj)))

        # NOTE: Any applications must be created prior to program.start()
        # if self._recorder:
        #     self._program.removeApp(self._recorder)
        # self._recorder = self._program.createApp(hist.TaggedCsvRecorder, filepath ='%s/BT_%s.tcsv' % (self._initTrader.outdir, self.episodeId))

        # step 1. start over the market state
        if self._marketState:
            self._program.removeObj(self._marketState)
        
        if self._initMarketState:
            self._marketState = copy.deepcopy(self._initMarketState)
            self._program.addObj(self._marketState)

        # step 2. create clean trader and account from self._account and  
        if self.__wkTrader:
            self._program.removeObj(self.__wkTrader)
        self.__wkTrader = copy.deepcopy(self._initTrader)
        # self._program.addApp(self.__wkTrader)
        self.__wkTrader._marketState = self._marketState
        # self.__wkTrader._recorder = self._recorder

        if self._account :
            self._program.removeApp(self._account)
            self._account =None
        
        # step 3. wrapper the broker drivers of the accounts
        if self._originAcc and not isinstance(self._originAcc, AccountWrapper):
            self._program.removeApp(self._originAcc.ident)
            self._account = AccountWrapper(self._program, btTrader =self, account=copy.copy(self._originAcc)) # duplicate the original account for test espoches
            self._account.hostTrader(self) # adopt the account by pointing its._trader to self
            self._account.setCapital(self._startBalance, True)
            self.__wkTrader._dailyCapCost = self._dailyCapCost
            self._program.addApp(self._account)
            self._account._marketState = self._marketState
            self._account._warmupDays = self._warmupDays
            self.__wkTrader._account = self._account
            self.info('doAppInit() wrappered account[%s] to [%s] with startBalance[%d]' % (self._originAcc.ident, self._account.ident, self._startBalance))

        self._maxBalance = self._startBalance
        self._wkHistData.resetRead()
           
        self._dataBegin_date, self._dataEnd_date = None, None
        self._dataBegin_openprice, self._dataEnd_closeprice = 0.0, 0.0

        # 当前最新数据，用于模拟成交用
        self.tick = None
        self.bar  = None
        self.__dtLastData  = None      # 最新数据的时间

        if self._marketState :
            for i in range(30) : # initially feed 20 data from histread to the marketstate
                ev = next(self._wkHistData)
                if not ev : continue
                self._marketState.updateByEvent(ev)

            if len(self.__wkTrader._dictObjectives) <=0:
                sl = self._marketState.listOberserves()
                for symbol in sl:
                    self.__wkTrader.openObjective(symbol)

        # step 4. subscribe account events
        self.subscribeEvents([Account.EVENT_ORDER, Account.EVENT_TRADE])

        self.info('resetEpisode() done for episode[%d/%d], obj-in-program: %s' % (self._episodeNo, self._episodes, self._program.listByType(MetaObj)))
        return True

    #----------------------------------------------------------------------
    def __fieldName(self, summaryKey, zh=False):
        __CHINESE_FIELDNAMES= {
        'playbackRange'   : u'    回放始末',
        'executeRange'    : u'  交易日始末',
        'executeDays'     : u'    交易日数',
        'startBalance'    : u'    起始资金',
        'endBalance'      : u'    结束资金',
        'totalTurnover'   : u'  总成交金额',
        'totalTradeCount' : u'  总成交笔数',
        'totalCommission' : u'    总手续费',
        'totalSlippage'   : u'      总滑点',
        'totalNetPnl'     : u'      总盈亏',
        'totalReturn'     : u'    总收益率',
        'annualizedReturn': u'    年化收益',
        'maxDrawdown'     : u'    最大回撤',
        'maxDdPercent'    : u'  最大回撤率',
        'sharpeRatio'     : u'      夏普率',
        'dailyNetPnl'     : u'    日均盈亏',
        'dailyCommission' : u'  日均手续费',
        'dailySlippage'   : u'    日均滑点',
        'dailyTurnover'   : u'日均成交金额',
        'dailyTradeCount' : u'日均成交笔数',
        'dailyReturn'     : u'  日均收益率',
        'returnStd'       : u'  收益标准差',
        }

        return __CHINESE_FIELDNAMES[summaryKey] if zh else summaryKey

    def formatSummary(self, summary=None):
        """显示按日统计的交易结果"""
        if not summary :
            summary = self._episodeSummary
        if isinstance(summary, str) : return summary
        if not isinstance(summary, dict) :
            self.error('no summary given: %s' % summary)
            return ''

        originGain = 0.0
        if self._dataBegin_openprice >0 :
            originGain = (self._dataEnd_closeprice - self._dataBegin_openprice)*100 / self._dataBegin_openprice

        # 输出统计结果
        strReport  = '\n%s_R%d/%d took %s' %(self.ident, self._episodeNo, self._episodes, str(datetime.now() - self.__execStamp_episodeStart))
        strReport += u'\n%s: %-10s ~ %-10s'  % (self.__fieldName('playbackRange'), self._btStartDate.strftime('%Y-%m-%d'), self._btEndDate.strftime('%Y-%m-%d'))
        strReport += u'\n%s: %-10s(open:%.2f) ~ %-10s(close:%.2f): %sdays %s%%' % \
                    (self.__fieldName('executeRange'), summary['startDate'], self._dataBegin_openprice, summary['endDate'], self._dataEnd_closeprice, summary['totalDays'], formatNumber(originGain))
        strReport += u'\n%s: %s (%s-profit, %s-loss) %s ~ %s +%s' % (self.__fieldName('executeDays'), summary['daysHaveTrade'], summary['profitDays'], summary['lossDays'], 
                    summary['tradeDay_1st'], summary['tradeDay_last'], summary['endLazyDays'])
        
        strReport += u'\n%s: %-12s' % (self.__fieldName('startBalance'), formatNumber(self._startBalance,2))
        strReport += u'\n%s: %-12s' % (self.__fieldName('endBalance'), formatNumber(summary['endBalance']))
    
        strReport += u'\n%s: %-12s' % (self.__fieldName('totalTurnover'), formatNumber(summary['totalTurnover']))
        strReport += u'\n%s: %s'    % (self.__fieldName('totalTradeCount'), formatNumber(summary['totalTradeCount'],0))
        strReport += u'\n%s: %s'    % (self.__fieldName('totalCommission'), formatNumber(summary['totalCommission']))
        strReport += u'\n%s: %s'    % (self.__fieldName('totalSlippage'), formatNumber(summary['totalSlippage']))

        strReport += u'\n%s: %s'    % (self.__fieldName('totalNetPnl'), formatNumber(summary['totalNetPnl']))
        strReport += u'\n%s: %s%%'  % (self.__fieldName('totalReturn'), formatNumber(summary['totalReturn']))
        strReport += u'\n%s: %s%%'  % (self.__fieldName('annualizedReturn'), formatNumber(summary['annualizedReturn']))
        strReport += u'\n%s: %s'    % (self.__fieldName('maxDrawdown'), formatNumber(summary['maxDrawdown']))
        strReport += u'\n%s: %s%%'  % (self.__fieldName('maxDdPercent'), formatNumber(summary['maxDdPercent']))
        strReport += u'\n%s: %s'    % (self.__fieldName('sharpeRatio'), formatNumber(summary['sharpeRatio'], 3))
        
        strReport += u'\n%s: %s'    % (self.__fieldName('dailyNetPnl'), formatNumber(summary['dailyNetPnl']))
        strReport += u'\n%s: %s'    % (self.__fieldName('dailyCommission'), formatNumber(summary['dailyCommission']))
        strReport += u'\n%s: %s'    % (self.__fieldName('dailySlippage'), formatNumber(summary['dailySlippage']))
        strReport += u'\n%s: %s'    % (self.__fieldName('dailyTurnover'), formatNumber(summary['dailyTurnover']))
        strReport += u'\n%s: %s'    % (self.__fieldName('dailyTradeCount'), formatNumber(summary['dailyTradeCount']))
        strReport += u'\n%s: %s%%'  % (self.__fieldName('dailyReturn'), formatNumber(summary['dailyReturn']))
        strReport += u'\n%s: %s%%'  % (self.__fieldName('returnStd'), formatNumber(summary['returnStd']))
        if 'reason' in summary.keys() : strReport += u'\n  reason: %s' % summary['reason']
        
        return strReport

''' 
    TODO: move the plotting stuf into a separate python program
    #----------------------------------------------------------------------
    def plotResult(self, tradeDays):
        # 绘图
        
        filename = '%s%s_DR.png' %(self._initTrader.outdir, self.episodeId)
        self.debug('plotResult() episode[%s] plotting result to %s' % (self.episodeId, filename))

        plt.rcParams['agg.path.chunksize'] =10000

        fig = plt.figure(figsize=(10, 16))
        
        pBalance = plt.subplot(4, 1, 1)
        pBalance.set_title(self._id + ' Balance')
        tradeDays['endBalance'].plot(legend=True)
        
        pDrawdown = plt.subplot(4, 1, 2)
        pDrawdown.set_title('Drawdown')
        pDrawdown.fill_between(range(len(tradeDays)), tradeDays['drawdown'].values)
        
        pPnl = plt.subplot(4, 1, 3)
        pPnl.set_title('Daily Pnl') 
        tradeDays['netPnl'].plot(kind='bar', legend=False, grid=False, xticks=[])

        pKDE = plt.subplot(4, 1, 4)
        pKDE.set_title('Daily Pnl Distribution')
        tradeDays['netPnl'].hist(bins=50)
        
        plt.savefig(filename, dpi=400, bbox_inches='tight')
        plt.show()
        plt.close()
       
    #----------------------------------------------------------------------
    def runOptimization(self, strategyClass, optimizationSetting):
        """优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.debug(u'优化设置有问题，请检查')
        
        # 遍历优化
        self.resultList =[]
        for setting in settingList:
            self.clearBackTesting()
            self.debug('-' * 30)
            self.debug('setting: %s' %str(setting))
            self.initStrategy(strategyClass, setting)
            self.runBacktesting()

            df, d = calculateSummary(self._startBalance, self._account.dailyResultDict)
            try:
                targetValue = d[targetName]
            except KeyError:
                targetValue = 0
            self.resultList.append(([str(setting)], targetValue, d))
        
        # 显示结果
        self.resultList.sort(reverse=True, key=lambda result:result[1])
        self.debug('-' * 30)
        self.debug(u'优化结果：')
        for result in self.resultList:
            self.debug(u'参数：%s，目标：%s' %(result[0], result[1]))    
        return self.resultList
            
    #----------------------------------------------------------------------
    def runParallelOptimization(self, strategyClass, optimizationSetting):
        """并行优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.debug(u'优化设置有问题，请检查')
        
        # 多进程优化，启动一个对应CPU核心数量的进程池
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        l = []

        for setting in settingList:
            l.append(pool.apply_async(optimize, (strategyClass, setting,
                                                 targetName, self.mode, 
                                                 self.startDate, self.initDays, self.endDate,
                                                 self.dbName, self.symbol)))
        pool.close()
        pool.join()
        
        # 显示结果
        resultList = [res.get() for res in l]
        resultList.sort(reverse=True, key=lambda result:result[1])
        self.debug('-' * 30)
        self.debug(u'优化结果：')
        for result in resultList:
            self.debug(u'参数：%s，目标：%s' %(result[0], result[1]))    
            
        return resultList

    #----------------------------------------------------------------------
    def calculateTransactions(self):
        """
        计算回测结果
        """
        self.debug(u'计算回测结果')
        
        # 首先基于回测后的成交记录，计算每笔交易的盈亏
        self.clearResult()

        buyTrades = []              # 未平仓的多头交易
        sellTrades = []             # 未平仓的空头交易

        # ---------------------------
        # scan all 交易
        # ---------------------------
        # convert the trade records into result records then put them into resultList
        for trade in self._dictTrades.values():
            # 复制成交对象，因为下面的开平仓交易配对涉及到对成交数量的修改
            # 若不进行复制直接操作，则计算完后所有成交的数量会变成0
            trade = copy.copy(trade)
            
            # buy交易
            # ---------------------------
            if trade.direction == OrderData.DIRECTION_LONG:

                if not sellTrades:
                    # 如果尚无空头交易
                    buyTrades.append(trade)
                    continue

                # 当前多头交易为平空
                while True:
                    entryTrade = sellTrades[0]
                    exitTrade = trade
                    
                    # 清算开平仓交易
                    closedVolume = min(exitTrade.volume, entryTrade.volume)
                    result = TradingResult(entryTrade.price, entryTrade.dt, 
                                           exitTrade.price, exitTrade.dt,
                                           -closedVolume, self._ratePer10K, self._slippage, self._account.size)

                    self.resultList.append(result)
                    
                    self.posList.extend([-1,0])
                    self.tradeTimeList.extend([result.entryDt, result.exitDt])
                    
                    # 计算未清算部分
                    entryTrade.volume -= closedVolume
                    exitTrade.volume -= closedVolume
                    
                    # 如果开仓交易已经全部清算，则从列表中移除
                    if not entryTrade.volume:
                        sellTrades.pop(0)
                    
                    # 如果平仓交易已经全部清算，则退出循环
                    if not exitTrade.volume:
                        break
                    
                    # 如果平仓交易未全部清算，
                    if exitTrade.volume:
                        # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                        # 等于新的反向开仓交易，添加到队列中
                        if not sellTrades:
                            buyTrades.append(exitTrade)
                            break
                        # 如果开仓交易还有剩余，则进入下一轮循环
                        else:
                            pass

                continue 
                # end of # 多头交易

            # 空头交易        
            # ---------------------------
            if not buyTrades:
                # 如果尚无多头交易
                sellTrades.append(trade)
                continue

            # 当前空头交易为平多
            while True:
                entryTrade = buyTrades[0]
                exitTrade = trade
                
                # 清算开平仓交易
                closedVolume = min(exitTrade.volume, entryTrade.volume)
                result = TradingResult(entryTrade.price, entryTrade.dt, 
                                       exitTrade.price, exitTrade.dt,
                                       closedVolume, self._ratePer10K, self._slippage, self._account.size)

                self.resultList.append(result)
                self.posList.extend([1,0])
                self.tradeTimeList.extend([result.entryDt, result.exitDt])

                # 计算未清算部分
                entryTrade.volume -= closedVolume
                exitTrade.volume -= closedVolume
                
                # 如果开仓交易已经全部清算，则从列表中移除
                if not entryTrade.volume:
                    buyTrades.pop(0)
                
                # 如果平仓交易已经全部清算，则退出循环
                if not exitTrade.volume:
                    break
                
                # 如果平仓交易未全部清算，
                if exitTrade.volume:
                    # 且开仓交易已经全部清算完，则平仓交易剩余的部分
                    # 等于新的反向开仓交易，添加到队列中
                    if not buyTrades:
                        sellTrades.append(exitTrade)
                        txnstr += '%-dx%.2f' % (trade.volume, trade.price)
                        break
                    # 如果开仓交易还有剩余，则进入下一轮循环
                    else:
                        pass                    

                continue 
                # end of 空头交易

        # end of scanning tradeDict
        
        # ---------------------------
        # 结算日
        # ---------------------------
        # 到最后交易日尚未平仓的交易，则以最后价格平仓
        for trade in buyTrades:
            result = TradingResult(trade.price, trade.dt, self._dataEnd_closeprice, self.__dtLastData, 
                                   trade.volume, self._ratePer10K, self._slippage, self._account.size)
            self.resultList.append(result)
            txnstr += '%+dx%.2f' % (trade.volume, trade.price)
            
        for trade in sellTrades:
            result = TradingResult(trade.price, trade.dt, self._dataEnd_closeprice, self.__dtLastData, 
                                   -trade.volume, self._ratePer10K, self._slippage, self._account.size)
            self.resultList.append(result)
            txnstr += '%-dx%.2f' % (trade.volume, trade.price)

        # return resultList;
        return self.settleResult()
        
'''

########################################################################
class OnlineSimulator(MetaTrader):
    '''
    OnlineSimulator is a wrapprer of Trader, with wrapper dummy account
    '''
    #----------------------------------------------------------------------
    def __init__(self, program, trader, **kwargs):
        """Constructor"""

        super(OnlineSimulator, self).__init__(program, **kwargs)
        self.__wkTrader = trader

        self._originAcc = None # to populate from _initTrader then wrapper
        self._account = None # the working account inherit from MetaTrader
        self._marketState = None

        self.__stampLastSaveState = None

        self.setRecorder(self.__wkTrader.recorder)

        # attributes of virtual account
        # -----------------------------------------
        self._startBalance = 100000      # 10w
        self._startBalance = self.getConfig('backTest/startBalance', 100000)

        self.__execStamp_appStart = datetime.now()
        self.__dtLastData = None
        self._maxBalance = self._startBalance
        self._openDays = 0

        # backtest will always clear the datapath
        # try :
        #     shutil.rmtree(self.__wkTrader.outdir)
        # except:
        #     pass

        # try :
        #     os.makedirs(self.__wkTrader.outdir)
        # except:
        #     pass

    @property
    def wkTrader(self) :
        return self.__wkTrader

    def setRecorder(self, recorder) :
        self._recorder = recorder
        if self._recorder :
            self._recorder.registerCategory(Account.RECCATE_ORDER,       params= {'columns' : OrderData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_TRADE,       params= {'columns' : TradeData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYPOS,    params= {'columns' : DailyPosition.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYRESULT, params= {'columns' : DailyResult.COLUMNS})

            self._recorder.registerCategory(EVENT_TICK,           params={'columns': TickData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_1MIN,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_5MIN,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_KLINE_1DAY,     params={'columns': KLineData.COLUMNS})
            self._recorder.registerCategory(EVENT_MONEYFLOW_1MIN, params={'columns': MoneyflowData.COLUMNS})
            self._recorder.registerCategory(EVENT_MONEYFLOW_1DAY, params={'columns': MoneyflowData.COLUMNS})

        if self.__wkTrader :
            self.__wkTrader._recorder = self._recorder
            
        return self._recorder

    def __saveMarketState(self) :
        try :
            self.program.saveObject(self.marketState, 'marketState')
        except Exception as ex:
            self.logexception(ex)

    def __restoreMarketState(self) :
        try :
            return self.program.loadObject('marketState') # '%s/marketState' % self.__class__)
        except Exception as ex:
            self.logexception(ex)
        return None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    @property
    def ident(self) :
        ret = self.__class__.__name__
        if self._id and len(self._id) >0:
            ret += '.%s' % self._id
        if not self.__wkTrader:
            return ret
        
        ret += '/%s' % self.__wkTrader.__class__.__name__
        try :
            if self.__wkTrader._tradeSymbol:
                ret += '.%s' % self.__wkTrader._tradeSymbol
        except:
            pass
        return ret

    def stop(self):
        """退出程序前调用，保证正常退出"""
        if self._marketState:
            self._program.removeObj(self._marketState)
        self._marketState = None

        # step 2. create clean trader and account from self._account and  
        if self.__wkTrader:
            self._program.removeApp(self.__wkTrader)
        self.__wkTrader = None

        if self._account :
            self._program.removeApp(self._account)

        return super(OnlineSimulator, self).stop()

    def doAppInit(self): # return True if succ
        if not super(OnlineSimulator, self).doAppInit() :
            return False

        # step 1. wrapper the Trader
        if not self.__wkTrader :
            return False
        
        self.info('doAppInit() taking trader[%s]' % (self.__wkTrader.ident))
        self._program.removeApp(self.__wkTrader)
        self._program.addApp(self)
        if not self.__wkTrader.doAppInit() :
            self.info('doAppInit() failed to initialize trader[%s]' % (self.__wkTrader.ident))
            return False

        symbolBy = None
        try :
            if self.__wkTrader._tradeSymbol:
                symbolBy = self.__wkTrader._tradeSymbol
        except:
            pass

        if not symbolBy and len(self.__wkTrader._dictObjectives) >0:
            symbolBy = list(self.__wkTrader._dictObjectives.keys())[0]

        if symbolBy:
            svfn = '%s%s/%s.ss' % (self.dataRoot, self.program.baseName, symbolBy)
            if self.program.shelveFilename != svfn:
                self.debug('doAppInit() taking safestore[%s] instead of [%s]' % (svfn, self.program.shelveFilename))
                self.program.setShelveFilename(svfn)

        # step 2. init the marketState
        prevState = self.__restoreMarketState()
        if prevState:
            self.__wkTrader._marketState = prevState
            self.info('doAppInit() previous market state restored: %s' % self.__wkTrader._marketState.descOf(None))
        self._marketState = self.__wkTrader._marketState

        # step 3. connects the trader and account 
        # prevAccount = self.program.loadObject('%s/account' % 'OnlineSimulator') # '%s/marketState' % self.__class__)
        originAcc = self.__wkTrader.account
        bAccRestored = originAcc.restore()

        if self._account :
            self._program.removeApp(self._account)
            self._account =None
        
        # step 4. wrapper the broker drivers of the accounts
        self._maxBalance = self._startBalance
        if originAcc and not isinstance(originAcc, AccountWrapper):
            self._program.removeApp(originAcc)
            self._account = AccountWrapper(self._program, btTrader =self, account=originAcc)
            self._account.hostTrader(self) # adopt the account by pointing its._trader to self
            if not bAccRestored:
                self._account.setCapital(self._startBalance, True)
                self.__wkTrader._dailyCapCost = 0

            self._account._warmupDays =0 # OnlineSimulator will not be constrainted by warmupDays

            self._program.addApp(self._account)
            self._account._marketState = self._marketState
            self.__wkTrader._account = self._account
            
            cashAvail, cashTotal, positions = self._account.positionState()
            _, posvalue = self._account.summrizeBalance(positions, cashTotal)
            capitalTotal = cashTotal + posvalue

            if capitalTotal > self._maxBalance:
                self._maxBalance = capitalTotal
            self.info('doAppInit() wrappered account[%s] to [%s] with restored[%s] capitalTotal[%s=%scash+%spos] max[%s]' % (self._account.account.ident, self._account.ident, 'T' if prevState else 'F', capitalTotal, cashTotal, posvalue, self._maxBalance))
            
        self._account.account.save()

        if len(self.__wkTrader._dictObjectives) <=0:
            sl = self._marketState.listOberserves()
            for symbol in sl:
                self.__wkTrader.openObjective(symbol)

        # step 5.1 subscribe the TradeAdvices
        self.subscribeEvents([EVENT_ADVICE, EVENT_TICK_OF_ADVICE])

        # step 5.2 subscribe the account and market events
        self.subscribeEvents([Account.EVENT_ORDER, Account.EVENT_TRADE])

        self.subscribeEvents([EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY])
        self.subscribeEvents([EVENT_MONEYFLOW_1MIN, EVENT_MONEYFLOW_1DAY])

        self.info('doAppInit() done, obj-in-program: %s' % (self._program.listByType(MetaObj)))
        return True

    def doAppStep(self):
        super(OnlineSimulator, self).doAppStep()
        self._account.doAppStep()

        stampNow = datetime.now()
        saveInterval = timedelta(hours=1)
        if Account.STATE_OPEN == self._account.account._state:
            saveInterval = timedelta(minutes=3)
            today = stampNow.strftime('%Y-%m-%d')
            if self._account.account._dateToday == today and stampNow > (self._account.account.__class__.tradeEndOfDay(stampNow) + DAYCLOSE_TIME_ERR) and self._marketState.getAsOf().strftime('%Y-%m-%d') == today:
                self._account.onDayClose()
                self._account.account.save()
                self.__saveMarketState()
            
        if not self.__stampLastSaveState:
            self.__stampLastSaveState = stampNow
            
        if stampNow - self.__stampLastSaveState > saveInterval:
            self.__stampLastSaveState = stampNow
            self.__saveMarketState()

    def OnEvent(self, ev): 

        if EVENT_TICK_OF_ADVICE == ev.type :
            d = copy.copy(ev.data)
            ev = Event(EVENT_TICK)
            ev.setData(d)
            self.debug('OnEvent(%s) treating as: %s' % (EVENT_TICK_OF_ADVICE, ev.desc))

        # step 2. 收到行情后，在启动策略前的处理
        evd = ev.data
        matchNeeded = False
        if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            if not self.__dtLastData or self.__dtLastData < evd.asof :
                self.__dtLastData = evd.asof
                if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
                    matchNeeded = True

        if matchNeeded :
            self._account.matchTrades(ev)

        return self.__wkTrader.OnEvent(ev)

    # end of BaseApplication routine
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    # Overrides of Events handling
    def eventHdl_Order(self, ev):
        self._account.account.save()
        return self.__wkTrader.eventHdl_Order(ev)
            
    def eventHdl_Trade(self, ev):
        self._account.account.save()
        return self.__wkTrader.eventHdl_Trade(ev)

    def eventHdl_DayOpen(self, symbol, date):
        self._account.account.save()
        return self.__wkTrader.eventHdl_DayOpen(symbol, date)

########################################################################
class OptimizationSetting(object):
    """优化设置"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.paramDict = OrderedDict()
        
        self.optimizeTarget = ''        # 优化目标字段
        
    #----------------------------------------------------------------------
    def addParameter(self, name, start, end=None, step=None):
        """增加优化参数"""
        if end is None and step is None:
            self.paramDict[name] = [start]
            return 
        
        if end < start:
            print(u'参数起始点必须不大于终止点')
            return
        
        if step <= 0:
            print(u'参数布进必须大于0')
            return
        
        l = []
        param = start
        
        while param <= end:
            l.append(param)
            param += step
        
        self.paramDict[name] = l
        
    #----------------------------------------------------------------------
    def generateSetting(self):
        """生成优化参数组合"""
        # 参数名的列表
        nameList = self.paramDict.keys()
        paramList = self.paramDict.values()
        
        # 使用迭代工具生产参数对组合
        productList = list(product(*paramList))
        
        # 把参数对组合打包到一个个字典组成的列表中
        settingList = []
        for p in productList:
            d = dict(zip(nameList, p))
            settingList.append(d)
    
        return settingList
    
    #----------------------------------------------------------------------
    def setOptimizeTarget(self, target):
        """设置优化目标字段"""
        self.optimizeTarget = target

#----------------------------------------------------------------------
def calculateSummary(startBalance, dayResultDict):
    '''
    @param dayResultDict - OrderedDict of account DailyResult during the date-window
    @return panda dataframe of formated-'DailyResult', summary-stat
    '''

    # step 1 convert OrderedDict of DailyResult to DataFrame
    if not dayResultDict or len(dayResultDict) <=0:
        return None, 'NULL dayResultDict'

    columns ={}
    cDaysHaveTrade =0
    for k, dr in dayResultDict.items():
        if not dr.date: dr.date = k
        if len(columns) <=0 :
            for k in dr.__dict__.keys() :
                if 'tradeList' == k : # to exclude some columns
                    continue
                columns[k] =[]

        for k, v in dr.__dict__.items() :
            if k in columns :
                columns[k].append(v)
            
    df = pd.DataFrame.from_dict(columns).set_index('date')

    # step 2. append the DataFrame with new columns [balance, return, highlevel, drawdown, ddPercent]
    # df['balance'] = df['netPnl'].cumsum() + startBalance
    df['return'] = (np.log(df['endBalance']) - np.log(df['endBalance'].shift(1))).fillna(0)
    df['highlevel'] = df['endBalance'].rolling(min_periods=1,window=len(df),center=False).max()
    df['drawdown'] = df['endBalance'] - df['highlevel']
    df['ddPercent'] = df['drawdown'] / df['highlevel'] * 100
    
    # step 3. calculate the overall performance summary
    startDate = df.index[0]
    endDate = df.index[-1]

    totalDays  = len(df)
    profitDays = len(df[df['netPnl']>0.01])
    lossDays   = len(df[df['netPnl']<-0.01])
    daysHaveTrades = df[(df['tcBuy'] + df['tcSell']) >0].index
    tcBuys = df['tcBuy'].sum()
    tcSells = df['tcSell'].sum()
    tradeDay_1st = daysHaveTrades[0] if len(daysHaveTrades) >0 else None
    tradeDay_last  = daysHaveTrades[-1] if len(daysHaveTrades) >0 else None
    endLazyDays = totalDays - df.index.get_loc(tradeDay_last) -1 if tradeDay_last else totalDays
    
    endBalance   = round(df['endBalance'].iloc[-1], 2)
    maxDrawdown  = round(df['drawdown'].min(),2)
    maxDdPercent = round(df['ddPercent'].min(),2)
    
    totalNetPnl = df['netPnl'].sum()
    dailyNetPnl = totalNetPnl / totalDays
    
    totalCommission = df['commission'].sum()
    dailyCommission = totalCommission / totalDays
    
    totalSlippage = df['slippage'].sum()
    dailySlippage = totalSlippage / totalDays
    
    totalTurnover = df['turnover'].sum()
    dailyTurnover = totalTurnover / totalDays
    
    totalTradeCount = tcBuys + tcSells
    dailyTradeCount = totalTradeCount / totalDays
    
    totalReturn = (endBalance/startBalance - 1) * 100
    annualizedReturn =  (math.exp(math.log(endBalance/startBalance) / (totalDays /Account_AShare.ANNUAL_TRADE_DAYS)) -1) *100 # = totalReturn / totalDays * 240
    dailyReturn = df['return'].mean() * 100
    returnStd = df['return'].std() * 100
    
    if returnStd:
        sharpeRatio = dailyReturn / returnStd * np.sqrt(240)
    else:
        sharpeRatio = 0
        
    # 返回结果
    summary = {
        'startDate': startDate,
        'endDate': endDate,
        'totalDays': totalDays,
        'profitDays': profitDays,
        'lossDays': lossDays,
        'endBalance': round(endBalance, 3),
        'maxDrawdown': round(maxDrawdown, 3),
        'maxDdPercent': round(maxDdPercent, 3),
        'totalNetPnl': round(totalNetPnl, 3),
        'dailyNetPnl': round(dailyNetPnl, 3),
        'totalCommission': round(totalCommission, 3),
        'dailyCommission': round(dailyCommission, 3),
        'totalSlippage': round(totalSlippage, 3),
        'dailySlippage': round(dailySlippage, 3),
        'totalTurnover': round(totalTurnover, 3),
        'dailyTurnover': round(dailyTurnover, 3),
        # 'tcSells':tcSell,
        # 'tcBuys':tcBuys,
        'totalTradeCount': totalTradeCount,
        'dailyTradeCount': round(dailyTradeCount, 3),
        'totalReturn':    round(totalReturn, 3),
        'annualizedReturn': round(annualizedReturn, 3),
        'dailyReturn': round(dailyReturn, 3),
        'returnStd': returnStd,
        'sharpeRatio': sharpeRatio,
        'daysHaveTrade': len(daysHaveTrades),
        'tradeDay_1st': tradeDay_1st,
        'tradeDay_last': tradeDay_last,
        'endLazyDays' : endLazyDays
    }
    
    return df, summary

#----------------------------------------------------------------------
def optimize(strategyClass, setting, targetName,
             mode, startDate, initDays, endDate,
             dbName, symbol):

    """多进程优化时跑在每个进程中运行的函数"""
    account = BTAccount_AShare() # should be BTAccountWrapper
    account.setBacktestingMode(mode)
    account.setStartDate(startDate, initDays)
    account.setEndDate(endDate)
    account.setSlippage(slippage)
    account.setRate(rate)
    account.setSize(size)
    account.setPriceTick(priceTick)
    account.setDatabase(dbName, symbol)
    
    account.initStrategy(strategyClass, setting)
    account.runBacktesting()
    
    df, d = calculateSummary(startBalance, account.dailyResultDict)
    try:
        targetValue = d[targetName]
    except KeyError:
        targetValue = 0
                    
    return (str(setting), targetValue, d)    
    
########################################################################
class AccountWrapper(MetaAccount):
    """
    回测BrokerDriver
    函数接口和BrokerDriver保持一样，
    从而实现同一套代码从回测到实盘。
    """

    #----------------------------------------------------------------------
    def __init__(self, program, btTrader, account, **kwargs) :
        """Constructor"""

        super(AccountWrapper, self).__init__(program, **kwargs)

        self._btTrader = btTrader   # refer to the BackTest App
        self._nest  = account
        # self._nest._mode = Account.BROKER_API_SYNC
        self._tradeCount = 0

        # 日线回测结果计算用
        self.__dailyResultDict = OrderedDict()
        self._warmupDays =0

    @property
    def dailyResultDict(self):
        return self.__dailyResultDict

    @property
    def account(self):
        return self._nest.account

    @property
    def executable(self):
        if len(self.__dailyResultDict) < self._warmupDays :
            return False

        return self._nest.executable

    #----------------------------------------------------------------------
    # impl of BaseApplication
    def doAppInit(self): 
        return self._nest.doAppInit()

    def doAppStep(self):
        ''' 
        this is a 'duplicated' impl of Account in order to call BackTestAcc._broker_xxxx() 
        instead of those of Account
        '''
        outgoingOrders = []
        ordersToCancel = []
        cStep =0
        fstampNow = datetime2float(self.datetimeAsOfMarket())
        strExpired = ''
        cExpired=0

        with self._nest._lock:
            outgoingOrders = copy.deepcopy(list(self._nest._dictOutgoingOrders.values()))

            # find out those expired orders, append them into _lstOrdersToCancel
            if fstampNow >0.0 :
                for odid,odata in self._nest._dictLimitOrders.items():
                    if odata.msecTTL <=0 or odata.fstampSubmitted + odata.msecTTL > fstampNow: continue
                    if odid in self._lstOrdersToCancel : continue
                    self._lstOrdersToCancel.append(odid)
                    strExpired += 'O[%s],' % odata.desc
                    cExpired +=1

                for odid,odata in self._nest._dictStopOrders.items():
                    if odata.msecTTL <=0 or odata.fstampSubmitted + odata.msecTTL > fstampNow: continue
                    if odid in self._lstOrdersToCancel : continue
                    self._lstOrdersToCancel.append(odid)
                    strExpired += 'O[%s],' % odata.desc
                    cExpired +=1

            # find out he orderData by brokerOrderId
            for odid in self._nest._lstOrdersToCancel :
                orderData = None
                try :
                    if OrderData.STOPORDERPREFIX in odid :
                        orderData = self._nest._dictStopOrders[odid]
                    else :
                        orderData = self._nest._dictLimitOrders[odid]
                except KeyError:
                    pass

                if orderData :
                    ordersToCancel.append(copy.copy(orderData))
                
                cStep +=1

            self._nest._lstOrdersToCancel = []

        for co in ordersToCancel:
            self._broker_cancelOrder(co)
            cStep +=1

        for no in outgoingOrders:
            self._broker_placeOrder(no)
            cStep +=1

        if (cExpired>0):
            self.warn('step() placed %d orders, cancelled %d orders including %d expired: %s'% (len(outgoingOrders), len(ordersToCancel), cExpired, strExpired))
        elif (len(ordersToCancel) + len(outgoingOrders)) >0:
            self.info('step() placed %d orders, cancelled %d orders'% (len(outgoingOrders), len(ordersToCancel)))

        return cStep

    def OnEvent(self, ev):
        return self._nest.OnEvent(ev)

    def debug(self, msg):
        fwdTo = self._btTrader if self._btTrader else self._nest
        fwdTo.debug(msg)
        
    def info(self, msg):
        fwdTo = self._btTrader if self._btTrader else self._nest
        fwdTo.info(msg)

    def warn(self, msg):
        fwdTo = self._btTrader if self._btTrader else self._nest
        fwdTo.warn(msg)

    def error(self, msg):
        fwdTo = self._btTrader if self._btTrader else self._nest
        fwdTo.error(msg)

    #----------------------------------------------------------------------
    # most of the methods are just forward to the self._nest
    @property
    def trader(self): return self._nest.trader
    def hostTrader(self, trader): self._nest.hostTrader(trader)
    @property
    def priceTick(self): return self._nest.priceTick
    @property
    def contractSize(self): return self._nest.contractSize
    @property
    def cashSymbol(self): return self._nest.cashSymbol
    @property
    def dbName(self): return self._nest.dbName
    @property
    def ident(self) : return self._nest.ident
    @property
    def nextOrderReqId(self): return self._nest.nextOrderReqId
    @property
    def collectionName_dpos(self): return self._nest.collectionName_dpos
    @property
    def collectionName_trade(self): return self._nest.collectionName_dpos
    def getPosition(self, symbol): return self._nest.getPosition(symbol) # returns PositionData
    def cashAmount(self): return self._nest.cashAmount() # returns (avail, total)
    def positionState(self) : return self._nest.positionState()
    def summrizeBalance(self, positions=None, cashTotal=0) : return self._nest.summrizeBalance(positions=positions, cashTotal=cashTotal)
    def cashChange(self, dAvail=0, dTotal=0): return self._nest.cashChange(dAvail, dTotal)
    def record(self, category, data): return self._nest.record(category, data)
    def postEvent_Order(self, orderData): return self._nest.postEvent_Order(orderData)
    # def sendOrder(self, vtSymbol, orderType, price, volume, reason): return self._nest.sendOrder(vtSymbol, orderType, price, volume, reason)
    def cancelOrder(self, brokerOrderId): return self._nest.cancelOrder(brokerOrderId)
    def batchCancel(self, brokerOrderIds): return self._nest.batchCancel(brokerOrderIds)
    def cancelAllOrders(self): return self._nest.cancelAllOrders()
    def sendStopOrder(self, vtSymbol, orderType, price, volume, reason): return self._nest.sendStopOrder(vtSymbol, orderType, price, volume, reason)
    def findOrdersOfStrategy(self, strategyId, symbol=None): return self._nest.findOrdersOfStrategy(strategyId, symbol)
    
    def datetimeAsOfMarket(self): return self._btTrader.wkTrader._dtData
    def _broker_onOrderPlaced(self, orderData): return self._nest._broker_onOrderPlaced(orderData)
    def _broker_onCancelled(self, orderData): return self._nest._broker_onCancelled(orderData)
    def _broker_onOrderDone(self, orderData): return self._nest._broker_onOrderDone(orderData)
    def _broker_onTrade(self, trade): return self._nest._broker_onTrade(trade)
    def _broker_onGetAccountBalance(self, data, reqid): return self._nest._broker_onGetAccountBalance(data, reqid)
    def _broker_onGetOrder(self, data, reqid): return self._nest._broker_onGetOrder(data, reqid)
    def _broker_onGetOrders(self, data, reqid): return self._nest._broker_onGetOrders(data, reqid)
    def _broker_onGetMatchResults(self, data, reqid): return self._nest._broker_onGetMatchResults(data, reqid)
    def _broker_onGetMatchResult(self, data, reqid): return self._nest._broker_onGetMatchResult(data, reqid)
    def _broker_onGetTimestamp(self, data, reqid): return self._nest._broker_onGetTimestamp(data, reqid)

    def calcAmountOfTrade(self, symbol, price, volume): return self._nest.calcAmountOfTrade(symbol, price, volume)
    def maxOrderVolume(self, symbol, price): return self._nest.maxOrderVolume(symbol, price)
    def roundByPriceTick(self, price, dir=OrderData.DIRECTION_NONE): return self._nest.roundByPriceTick(price, dir)
    def onStart(self): return self._nest.onStart()
    # must be duplicated other than forwarding to _nest def doAppStep(self) : return self._nest.doAppStep()
    def onDayClose(self):
        self._nest.onDayClose()

        # save the calculated daily result into the this wrapper for late calculating
        if self._nest._datePrevClose :
            self.__dailyResultDict[self._nest._datePrevClose] = self._nest._todayResult

    def onTimer(self, dt): return self._nest.onTimer(dt)
    # def saveDB(self): return self._nest.saveDB()
    def loadDB(self, since =None): return self._nest.loadDB(since =None)
    def calcDailyPositions(self): return self._nest.calcDailyPositions()
    def log(self, message): return self._nest.log(message)
    def stdout(self, message): return self._nest.stdout(message)
    def saveSetting(self): return self._nest.saveSetting()
    def updateDailyStat(self, dt, price): return self._nest.updateDailyStat(dt, price)
    def evaluateDailyStat(self, startdate, enddate): return self._nest.evaluateDailyStat(startdate, enddate)

    #------------------------------------------------
    # overwrite of Account
    #------------------------------------------------    
    def sendOrder(self, symbol, orderType, price, volume, reason):

        orderData = OrderData(self)
        # 代码编号相关
        orderData.symbol      = symbol
        orderData.exchange    = self._exchange
        orderData.price       = self.roundByPriceTick(price) # 报单价格
        orderData.totalVolume = volume    # 报单总数量
        orderData.datetime    = self.datetimeAsOfMarket()
        orderData.reason      = reason if reason else ''
        orderData.msecTTL     = self._nest._msecOrderTTL if self._nest._msecOrderTTL >0 else 0.0

        # 报单方向
        if orderType == OrderData.ORDER_BUY:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_SELL:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_CLOSE
        elif orderType == OrderData.ORDER_SHORT:
            orderData.direction = OrderData.DIRECTION_SHORT
            orderData.offset = OrderData.OFFSET_OPEN
        elif orderType == OrderData.ORDER_COVER:
            orderData.direction = OrderData.DIRECTION_LONG
            orderData.offset = OrderData.OFFSET_CLOSE     

        if isinstance(self._nest, Account_AShare) :
            if orderData.datetime.hour >=15 or (14 == orderData.datetime.hour and orderData.datetime.minute >=58) :
                return ''
        else:
            if (23 == orderData.datetime.hour and orderData.datetime.minute >=58) :
                return ''

        self._broker_placeOrder(orderData)
        return orderData.reqId

    def _broker_placeOrder(self, orderData):
        """发单"""
        orderData.brokerOrderId = "$" + orderData.reqId
        orderData.status = OrderData.STATUS_SUBMITTED
        self.debug('faking order placed: %s' % orderData.desc)

        # redirectly simulate a place ok
        self._broker_onOrderPlaced(orderData)

    def _broker_cancelOrder(self, orderData) :
        # simuate a cancel by orderData
        orderData.status = OrderData.STATUS_CANCELLED
        orderData.stampCanceled = self.datetimeAsOfMarket().strftime('%H:%M:%S.%f')[:3]
        self.debug('faking order cancelled: %s' % orderData.desc)
        self._broker_onCancelled(orderData)

    def onDayOpen(self, newDate):
        # instead that the true Account is able to sync with broker,
        # Backtest should perform cancel to restore available/frozen positions
        if self._nest._dateToday and newDate != self._nest._dateToday :
            self.onDayClose()

        clist = []
        with self._nest._lock :
            # A share will not keep yesterday's order alive
            self._nest._dictOutgoingOrders.clear()
            for o in self._nest._dictLimitOrders.values():
                clist.append(o.brokerOrderId)
            for o in self._nest._dictStopOrders.values():
                clist.append(o.brokerOrderId)

        if len(clist) >0:
            self.batchCancel(clist)
            self.debug('BT.onDayOpen() batchCancelled: %s' % clist)

        self._nest.onDayOpen(newDate)

    #----------------------------------------------------------------------
    def setCapital(self, capital, resetAvail=False):
        """设置资本金"""
        cachAvail, cashTotal = self.cashAmount()
        dCap = capital-cashTotal
        dAvail = dCap
        if resetAvail :
            dAvail = capital-cachAvail

        self.cashChange(dAvail, dCap)
     
    def matchTrades(self, ev):
        ''' 模拟撮合成交 '''

        symbol = None
        maxCrossVolume =-1
        dtEvent = None

        # 先确定会撮合成交的价格
        if EVENT_TICK == ev.type:
            tkdata = ev.data
            symbol = tkdata.symbol
            dtEvent = tkdata.datetime
            buyCrossPrice      = tkdata.a1P
            sellCrossPrice     = tkdata.b1P
            buyBestCrossPrice  = tkdata.a1P
            sellBestCrossPrice = tkdata.b1P
        elif EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            kldata = ev.data
            symbol = kldata.symbol
            dtEvent = kldata.datetime
            # bestPrice          = round(((kldata.open + kldata.close) *4 + kldata.high + kldata.low) /10, 3)

            buyCrossPrice      = kldata.low        # 若买入方向限价单价格高于该价格，则会成交
            sellCrossPrice     = kldata.high      # 若卖出方向限价单价格低于该价格，则会成交
            maxCrossVolume     = kldata.volume
            buyBestCrossPrice  = ((kldata.open + kldata.close + kldata.high) *3 + kldata.low)  /10  # 在当前时间点前发出的买入委托可能的最优成交价
            sellBestCrossPrice = ((kldata.open + kldata.close + kldata.low)  *3 + kldata.high) /10  # 在当前时间点前发出的卖出委托可能的最优成交价
            
            # 张跌停封板
            if buyCrossPrice <= kldata.open*0.9 :
                buyCrossPrice =0
            if sellCrossPrice >= kldata.open*1.1 :
                sellCrossPrice =0

        if not symbol :
            return # ignore those non-tick/kline events

        buyBestCrossPrice  = self.roundByPriceTick(buyBestCrossPrice)
        sellBestCrossPrice = self.roundByPriceTick(sellBestCrossPrice)

        # 先撮合限价单
        self.__crossLimitOrder(symbol, dtEvent, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume)
        # 再撮合停止单
        self.__crossStopOrder(symbol, dtEvent, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume)

    def __crossLimitOrder(self, symbol, dtAsOf, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume=-1):
        """基于最新数据撮合限价单
        A limit order is an order placed with a brokerage to execute a buy or 
        sell transaction at a set number of shares and at a specified limit
        price or better. It is a take-profit order placed with a bank or
        brokerage to buy or sell a set amount of a financial instrument at a
        specified price or better; because a limit order is not a market order,
        it may not be executed if the price set by the investor cannot be met
        during the period of time in which the order is left open.
        """
        # 遍历限价单字典中的所有限价单vtTradeID

        trades = []
        finishedOrders = []
        pendingOrders =[]

        strCrossed =''
        if not dtAsOf:
            dtAsOf  = self.datetimeAsOfMarket()

        with self._nest._lock:
            for orderID, order in self._nest._dictLimitOrders.items():
                if order.symbol != symbol:
                    continue

                # 推送委托进入队列（未成交）的状态更新
                if not order.status:
                    order.status = OrderData.STATUS_SUBMITTED

                # 判断是否会成交
                buyCross = (order.direction == OrderData.DIRECTION_LONG and 
                            order.price>=buyCrossPrice and
                            buyCrossPrice > 0)      # 国内的tick行情在涨停时a1P为0，此时买无法成交
                
                sellCross = (order.direction == OrderData.DIRECTION_SHORT and 
                            order.price<=sellCrossPrice and
                            sellCrossPrice > 0)    # 国内的tick行情在跌停时bidP1为0，此时卖无法成交
                
                if not buyCross and not sellCross:
                    pendingOrders.append(order)
                    continue

                # 如果发生了成交， 推送成交数据
                self._tradeCount += 1            # 成交编号自增1
                tradeID = str(self._tradeCount)
                trade = TradeData(self._nest)
                trade.brokerTradeId = tradeID
                # tradeID will be generated in Account: trade.tradeID = tradeID
                trade.symbol = order.symbol
                trade.exchange = order.exchange
                trade.orderReq = order.reqId
                trade.orderID  = order.brokerOrderId
                trade.direction = order.direction
                trade.offset    = order.offset
                trade.volume    = order.totalVolume
                trade.datetime  = dtAsOf

                if buyCross:
                    trade.price = min(order.price, buyBestCrossPrice)
                else:
                    trade.price  = max(order.price, sellBestCrossPrice)

                trades.append(trade)

                order.tradedVolume = trade.volume
                order.status = OrderData.STATUS_ALLTRADED
                order.stampFinished = dtAsOf.strftime('%H:%M:%S.%f')[:3]

                if order.tradedVolume < order.totalVolume :
                    order.status = OrderData.STATUS_PARTTRADED
                finishedOrders.append(order)
                strCrossed += 'O[%s]->T[%s],' % (order.desc, trade.desc)

        if len(finishedOrders) + len(pendingOrders) >0:
            strPendings = ''
            for o in pendingOrders:
                strPendings += 'O[%s],' % o.desc
            if len(finishedOrders) >0:
                self.info('crossLimitOrder() crossed %d orders:%s; %d pendings: %s'% (len(finishedOrders), strCrossed, len(pendingOrders), strPendings))
            else:
                self.debug('crossLimitOrder() %d pending orders: %s'% (len(pendingOrders), strPendings))

        for o in finishedOrders:
            self._broker_onOrderDone(o)
            
        for t in trades:
            self._broker_onTrade(t)

    #----------------------------------------------------------------------
    def __crossStopOrder(self, symbol, dtAsOf, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume=-1): 
        """基于最新数据撮合停止单
            A stop order is an order to buy or sell a security when its price moves past
            a particular point, ensuring a higher probability of achieving a predetermined 
            entry or exit price, limiting the investor's loss or locking in a profit. Once 
            the price crosses the predefined entry/exit point, the stop order becomes a
            market order.
        """
        if not dtAsOf:
            dtAsOf  = self.datetimeAsOfMarket()

        # TODO implement StopOrders
        #########################################################
        # # 遍历停止单字典中的所有停止单
        # with self._nest. _lock:
        #     for stopOrderID, so in self._nest._dictStopOrders.items():
        #         if order.symbol != symbol:
        #             continue

        #         # 判断是否会成交
        #         buyCross  = (so.direction==OrderData.DIRECTION_LONG)  and so.price<=buyCrossPrice
        #         sellCross = (so.direction==OrderData.DIRECTION_SHORT) and so.price>=sellCrossPrice
                
        #         # 忽略未发生成交
        #         if not buyCross and not sellCross : # and (so.volume < maxCrossVolume):
        #             continue;

        #         # 更新停止单状态，并从字典中删除该停止单
        #         so.status = OrderData.STOPORDER_TRIGGERED
        #         if stopOrderID in self._dictStopOrders:
        #             del self._dictStopOrders[stopOrderID]                        

        #         # 推送成交数据
        #         self.tdDriver.tradeCount += 1            # 成交编号自增1
        #         tradeID = str(self.tdDriver.tradeCount)
        #         trade = VtTradeData()
        #         trade.vtSymbol = so.vtSymbol
        #         trade.tradeID = tradeID
        #         trade.vtTradeID = tradeID
                    
        #         if buyCross:
        #             self.strategy.pos += so.volume
        #             trade.price = max(bestCrossPrice, so.price)
        #         else:
        #             self.strategy.pos -= so.volume
        #             trade.price = min(bestCrossPrice, so.price)                
                    
        #         self.tdDriver.limitOrderCount += 1
        #         orderID = str(self.tdDriver.limitOrderCount)
        #         trade.orderID = orderID
        #         trade.vtOrderID = orderID
        #         trade.direction = so.direction
        #         trade.offset = so.offset
        #         trade.volume = so.volume
        #         trade.tradeTime = self.__dtLastData.strftime('%H:%M:%S')
        #         trade.dt = self.__dtLastData
                    
        #         self._dictTrades[tradeID] = trade
                    
        #         # 推送委托数据
        #         order = VtOrderData()
        #         order.vtSymbol = so.vtSymbol
        #         order.symbol = so.vtSymbol
        #         order.orderID = orderID
        #         order.vtOrderID = orderID
        #         order.direction = so.direction
        #         order.offset = so.offset
        #         order.price = so.price
        #         order.totalVolume = so.volume
        #         order.tradedVolume = so.volume
        #         order.status = OrderData.STATUS_ALLTRADED
        #         order.orderTime = trade.tradeTime
                    
        #         self.tdDriver.limitOrderDict[orderID] = order
                    
        #         self._account.onTrade(trade)

        #         # 按照顺序推送数据
        #         self.strategy.onStopOrder(so)
        #         self.strategy.onOrder(order)
        #         self.strategy.onTrade(trade)

    def OnPlaybackEnd(self) :
        # ---------------------------
        # 结算日
        # ---------------------------
        # step 1 fake a if newDate == self._dateToday
        dtAsOf  = self.datetimeAsOfMarket()
        dtFakedTomorrow = (dtAsOf + timedelta(days=1)).replace(hour=0, minute=0, second=1)
        self.info('OnPlaybackEnd() faking a day-open(%s)' % dtFakedTomorrow)
        self.onDayOpen(dtFakedTomorrow.strftime('%Y-%m-%d'))

        # step 1 reached playback-end, sell all position and turn them into cash via the latest price
        self.info('OnPlaybackEnd() faking trades to clean all positions into cash')
        cashAvail, cashTotal, currentPositions = self.positionState()
        for symbol, pos in currentPositions.items() :
            if symbol == self.cashSymbol:
                continue

            # fake a sold-succ trade into self._dictTrades
            self._tradeCount += 1            # 成交编号自增1
            trade = TradeData(self._nest)
            trade.brokerTradeId = str(self._tradeCount)
            trade.symbol = symbol
            # trade.exchange = self._nest.exchange
            trade.orderReq = self.nextOrderReqId +"$BTEND"
            trade.orderID = 'O' + trade.orderReq
            trade.direction = OrderData.DIRECTION_SHORT
            trade.offset = OrderData.OFFSET_CLOSE
            trade.price  = pos.price
            trade.volume = pos.posAvail
            trade.datetime = dtAsOf

            self._broker_onTrade(trade)
            self.info('OnPlaybackEnd() faked a trade: %s' % trade.desc)

########################################################################
class OfflineSimulator(BackTestApp):
    '''
    OfflineSimulator extends BackTestApp by reading history and perform training
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(OfflineSimulator, self).__init__(program, trader, histdata, **kwargs)

        self._masterExportHomeDir = self.getConfig('master/homeDir', None) # this agent work as the master when configured, usually point to a dir under webroot
        if self._masterExportHomeDir and '/' != self._masterExportHomeDir[-1]: self._masterExportHomeDir +='/'
        
        # the base URL of local web for the slaves to GET/POST the tasks
        # current OfflineSimulator works as slave if this masterExportURL presents but masterExportHomeDir abendons
        self._masterExportURL = self.getConfig('master/exportURL', self._masterExportHomeDir)

        self.__savedEpisode_Id = -1
        self.__savedEpisode_opendays = 0
        self.__maxKnownOpenDays =0
        self.__prevMaxBalance =0

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        # make sure OfflineSimulator is ONLY wrappering GymTrader
        if not self._initTrader or not isinstance(self._initTrader, BaseTrader) :
            self.error('doAppInit() invalid initTrader')
            return False

        if not super(OfflineSimulator, self).doAppInit() :
            return False

        self._account.account._skipSavingByEvent = True
        return True

    def OnEvent(self, ev): # this overwrite BackTest's because there are some different needs
        symbol  = None

        # OfflineSimulator MUST get rid of SysClock
        if EVENT_SYS_CLOCK == ev.type :
            return

        if EVENT_TICK_OF_ADVICE == ev.type :
            d = copy.copy(ev.data)
            ev = Event(EVENT_TICK)
            ev.setData(d)
            self.debug('OnEvent(%s) treating as: %s' % (EVENT_TICK_OF_ADVICE, ev.desc))

        if EVENT_TICK == ev.type or EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
            try :
                symbol = ev.data.symbol
            except:
                pass
            self._account.matchTrades(ev)

        self.wkTrader.OnEvent(ev) # to perform the real handling

        asOf = self.wkTrader.marketState.getAsOf(symbol)

        # get some additional reward when survived for one more day
        self._dataEnd_date = asOf
        self._dataEnd_closeprice, _ = self.wkTrader.marketState.latestPrice(symbol)

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
        super(OfflineSimulator, self).OnEpisodeDone(reachedEnd)

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
            # determin if best episode
            pass

        # save brain and decrease epsilon if improved
        if len(lstImproved) >0 :
            self.__savedEpisode_opendays = opendays
            self.__savedEpisode_Id = self.episodeId

        mySummary = {
            'savedEId'    : self.__savedEpisode_Id,
            'savedODays'  : self.__savedEpisode_opendays,
        }

        self._episodeSummary = {**self._episodeSummary, **mySummary}

########################################################################
class IdealTrader_Tplus1(OfflineSimulator):
    '''
    IdealTrader extends OfflineSimulator by scanning the MarketEvents occurs in a day, determining
    the ideal actions then pass the events down to the models
    '''
    def __init__(self, program, trader, histdata, **kwargs):
        '''Constructor
        '''
        super(IdealTrader_Tplus1, self).__init__(program, trader, histdata, **kwargs)

        self._dayPercentToCatch                = self.getConfig('constraints/dayPercentToCatch',          1.0) # pecentage of daychange to catch, otherwise will keep position empty
        self._constraintBuy_closeOverOpen      = self.getConfig('constraints/buy_closeOverOpen',          0.5) #pecentage price-close more than price-open - indicate buy
        self._constraintBuy_closeOverRecovery  = self.getConfig('constraint/buy_closeOverRecovery',   2.0) #pecentage price-close more than price-low at the recovery edge - indicate buy
        self._constraintSell_lossBelowHigh     = self.getConfig('constraint/sell_lossBelowHigh',         2.0) #pecentage price-close less than price-high at the loss edge - indicate sell
        self._constraintSell_downHillOverClose = self.getConfig('constraint/sell_downHillOverClose', 0.5) #pecentage price more than price-close triggers sell during a downhill-day to reduce loss
        self._generateReplayFrames             = self.getConfig('generateReplayFrames', 'directionOnly').lower()
        self._h5compression                    = self.getConfig('h5compression', 'lzf').lower()

        self._samplingRate                     = self.getConfig('samplingRate', 0.2) # 20%
        if self._samplingRate < 0.001 : self._samplingRate = 0.001
        if self._samplingRate > 1 :     self._samplingRate = 1

        self._pctMaxDrawDown =99.0 # IdealTrader will not be constrainted by max drawndown, so overwrite it with 99%
        self._warmupDays =0 # IdealTrader will not be constrainted by warmupDays

        self.__cOpenDays =0

        self.__adviceSeq = [] # list of faked OrderData, the OrderData only tells the direction withno amount

        self.__dtToday = None
        self.__mdEventsToday = [] # list of the datetime of open, high, low, close price occured today

        self.__dtTomrrow = None
        self.__mdEventsTomrrow = [] # list of the datetime of open, high, low, close price occured 'tomorrow'

        self.__sampleFrmSize  = 1024*8
        self.__sampleFrm = [None]  * self.__sampleFrmSize
        self.__sampleIdx, self.__frameNo = 0, 0
        self.__lastestDir, self.__lastmstate, self.__samplingYield  = None, None, 0

    def doAppInit(self): # return True if succ
        if not super(IdealTrader_Tplus1, self).doAppInit() :
            return False

        if self._recorder:
            self._recorder.registerCategory(EVENT_ADVICE, params= {'columns' : AdviceData.COLUMNS})

        self._tradeSymbol = self.wkTrader.objectives[0] # idealTrader only cover a single symbol from the objectives
        self._episodes =1 # idealTrader only run one loop
        return True
    
    # to replace OfflineSimulator's OnEvent with some TradeAdvisor logic and execute the advice as order directly
    def OnEvent(self, ev):
        '''processing an incoming MarketEvent'''

        super(IdealTrader_Tplus1, self).OnEvent(ev) # self.wkTrader._dtData = d.asof # self.wkTrader.OnEvent(ev)

        if not MARKETDATE_EVENT_PREFIX in ev.type: return
        
        d = ev.data
        tokens = (d.vtSymbol.split('.'))
        symbol = tokens[0]
        self.wkTrader._dtData = d.asof
        
        # see if need to perform the next order pre-determined
        dirToExec = OrderData.DIRECTION_NONE
        action = [0] * len(ADVICE_DIRECTIONS)

        if len(self.__adviceSeq) >0 :
            nextAdvice = self.__adviceSeq[0]
            if nextAdvice.datetime <= ev.data.datetime :
                dirToExec = nextAdvice.dirString()
                del self.__adviceSeq[0]

                # fake a TradeAdvice here to forward to wkTrader
                self.debug('OnEvent(%s) excuted a faked advice[%s] upon mstate: %s' % (d.desc, dirToExec, self._marketState.descOf(nextAdvice.symbol)))
                # action[ADVICE_DIRECTIONS.index(dirToExec)] =1.0
                # advice = AdviceData(self.ident, symbol, self._marketState.exchange)
                # advice.dirNONE, advice.dirLONG, advice.dirSHORT = action[0], action[1], action[2]
                # advice.price = nextOrder.price if EVENT_TICK == ev.type else d.close
                # advice.datetime  = nextOrder.asof
                # advice.advisorId = '%s' % self.ident
            
                # advice.Rdaily = 0.0
                # advice.Rdstd  = 0.0
                # advice.pdirNONE, advice.pdirLONG, advice.pdirSHORT = 0.0, 0.0, 0.0
                # advice.pdirPrice = 0.0
                # advice.pdirAsOf  = nextOrder.datetime
                # advice.dirString() # generate the dirString to ease reading

                if self._recorder:
                    self._recorder.pushRow(EVENT_ADVICE, nextAdvice)

                evAdv = Event(EVENT_ADVICE)
                evAdv.setData(nextAdvice)
                super(IdealTrader_Tplus1, self).OnEvent(evAdv) # to perform the real handling

        action[ADVICE_DIRECTIONS.index(dirToExec)] =1
        self._mstate = self._marketState.exportF1548(self._tradeSymbol)
        # self._mstate = self._marketState.exportImg6C_3Liner16xx(self._tradeSymbol) # self._mstate = self._marketState.exportImg6C_3Liner16x32R(self._tradeSymbol) # self._marketState.exportImg6C_3Snail16x16(self._tradeSymbol)

        if not self._mstate: return

        # if bFullState:
        orderDir = self.__lastestDir
        if self.__samplingYield <=0 or dirToExec != self.__lastestDir :
            self.__samplingYield = int (1.0/ self._samplingRate -1)

            if dirToExec != self.__lastestDir and self.__lastmstate: # the (state, dir) piror to dir-change sounds important to save
                self.__pushStateAction(self.__lastmstate, self.__lastestDir)

            self.__pushStateAction(self._mstate, action)
            self.__lastestDir, self.__lastmstate = dirToExec, None
        else :
            self.__lastmstate, self.__samplingYield = self._mstate, self.__samplingYield -1

        if orderDir != dirToExec:
            self.info('OnEvent(%s) changedir %s->%s upon mstate: %s' % (ev.desc, orderDir, dirToExec, self._marketState.descOf(self._tradeSymbol)))
        else:
            self.debug('OnEvent(%s) continue %s upon mstate: %s' % (ev.desc, dirToExec, self._marketState.descOf(self._tradeSymbol)))

    def resetEpisode(self) :
        ret = super(IdealTrader_Tplus1, self).resetEpisode()
        self.__adviceSeq = []

        return ret

    def OnEpisodeDone(self, reachedEnd=True):
        super(IdealTrader_Tplus1, self).OnEpisodeDone(reachedEnd)
        if self.__sampleIdx >0 and not None in self.__sampleFrm :
            self.__saveFrame(self.__sampleFrm[:self.__sampleIdx])

    # to replace BackTest's doAppStep
    def doAppStep(self):

        self._bGameOver = False # always False in IdealTrader
        reachedEnd = False
        if self._wkHistData :
            try :
                ev = next(self._wkHistData)
                if not ev or ev.data.datetime < self._btStartDate: return
                if ev.data.datetime <= self._btEndDate:
                    if self.__dtToday and self.__dtToday == ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0):
                        self.__mdEventsToday.append(ev)
                        return

                    if self.__dtTomrrow and self.__dtTomrrow == ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0):
                        self.__mdEventsTomrrow.append(ev)
                        return

                    # day-close here
                    self.scanEventsForAdvices()
                    for cachedEv in self.__mdEventsToday:
                        # !!NEVER!!! self._marketState.updateByEvent(cachedEv), let Trade.OnEvent() do so, otherwise will mess up the output RFrames
                        super(BackTestApp, self).doAppStep() # yes, this is to the super of BackTestApp
                        self._account.doAppStep()

                        self.postEvent(cachedEv) # call Trader
                        self._stepNoInEpisode += 1

                    self.__dtToday = self.__dtTomrrow
                    self.__mdEventsToday = self.__mdEventsTomrrow

                    self.__mdEventsTomrrow = []
                    self.__dtTomrrow = ev.data.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                    self.__cOpenDays += 1

                    return # successfully performed a step by pushing an Event

                reachedEnd = True
            except StopIteration:
                reachedEnd = True
                self.info('hist-read: end of playback')
            except Exception as ex:
                self.logexception(ex)

        # this test should be done if reached here
        self.debug('doAppStep() episode[%s] finished: %d steps, KO[%s] end-of-history[%s]' % (self.episodeId, self._stepNoInEpisode, self._bGameOver, reachedEnd))
        
        try:
            self.OnEpisodeDone(reachedEnd)
        except Exception as ex:
            self.logexception(ex)

        # print the summary report
        if self._recorder and isinstance(self._episodeSummary, dict):
            self._recorder.pushRow(RECCATE_ESPSUMMARY, self._episodeSummary)

        strReport = self.formatSummary()
        self.info('%s_%s summary:' %(self.ident, self.episodeId))
        for line in strReport.splitlines():
            if len(line) <2: continue
            self.info(line)

        strReport += '\n'
        with codecs.open(os.path.join(self.wkTrader.outdir, 'summary_%s.txt' % self._tradeSymbol), "w","utf-8") as rptfile:
            rptfile.write(strReport)
            self.debug('doAppStep() episode[%s] summary report generated' %(self.episodeId))

        # prepare for the next episode
        self._episodeNo +=1
        if (self._episodeNo > self._episodes) :
            # all tests have been done
            self.stop()
            self.info('all %d episodes have been done, app stopped. obj-in-program: %s' % (self._episodes, self._program.listByType(MetaObj)))

        self._program.stop()
        
        exit(0) # IdealTrader_Tplus1 is not supposed to run forever, just exit instead of return

    def __pushStateAction(self, mstate, action):

        if (self.__sampleIdx + self.__frameNo) <=0 and all(v == 0.0 for v in mstate): return # skip the leading all[0.0]

        self.__sampleIdx = self.__sampleIdx % self.__sampleFrmSize
        if 0 == self.__sampleIdx and not None in self.__sampleFrm :
            # frame full, output it into a HDF5 file
            self.__saveFrame(self.__sampleFrm)

        self.__sampleFrm[self.__sampleIdx] = (mstate, action)
        self.__sampleIdx +=1

    def __saveFrame(self, rangedFrame):
        metrix     = np.array(rangedFrame)
        col_state  = np.concatenate(metrix[:, 0]).reshape(len(rangedFrame), len(rangedFrame[0][0]))
        col_action = np.concatenate(metrix[:, 1]).reshape(len(rangedFrame), len(rangedFrame[0][1]))

        fn_frame = os.path.join(self.wkTrader.outdir, 'RFrm%s_%s.h5' % (NORMALIZE_ID, self._tradeSymbol) )
        
        h5args =copy.copy(hist.H5DSET_DEFAULT_ARGS)
        if self._h5compression and len(self._h5compression)>0:
            h5args['compression'] = self._h5compression
        
        with h5py.File(fn_frame, 'a') as h5file:
            frameId = 'RF%s' % str(self.__frameNo).zfill(3)
            self.__frameNo += 1

            g = h5file.create_group(frameId)
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs[u'default'] = 'state'
            g.attrs['size'] = col_state.shape[0]
            g.attrs['signature'] = EXPORT_SIGNATURE

            g.create_dataset(u'title',      data= '%s replay %s of %s by %s' % (self._generateReplayFrames, frameId, self._tradeSymbol, self.ident))
            st = g.create_dataset('state',  data= col_state, **h5args)
            st.attrs['dim'] = col_state.shape[1]
            ac = g.create_dataset('action', data= col_action, **h5args)
            ac.attrs['dim'] = col_action.shape[1]
            
        self.info('saved %s len[%s] into file %s with sig[%s]' % (frameId, len(col_state), fn_frame, EXPORT_SIGNATURE))

    def __scanEventsSequence(self, evseq) :

        price_open, price_high, price_low, price_close = 0.0, 0.0, DUMMY_BIG_VAL, 0.0
        T_high, T_low  = None, None
        if evseq and len(evseq) >0:
            for ev in evseq:
                evd = ev.data
                if EVENT_TICK == ev.type :
                    price_close = evd.price
                    if price_open <= 0.01 :
                        price_open = price_close
                    if price_high < price_close :
                        price_high = price_close
                        T_high = evd.datetime
                    if price_low > price_close :
                        price_low = price_close
                        T_low = evd.datetime
                    continue

                if EVENT_KLINE_PREFIX == ev.type[:len(EVENT_KLINE_PREFIX)] :
                    price_close = evd.close
                    if price_high < evd.high :
                        price_high = evd.high
                        T_high = evd.datetime
                    if price_low > evd.low :
                        price_low = evd.low
                        T_low = evd.datetime
                    if price_open <= 0.01 :
                        price_open = evd.open
                    continue

        return price_open, price_high, price_low, price_close, T_high, T_low

    def scanEventsForAdvices(self) :
        '''
        this will generate 3 actions
        '''
        # step 1. scan self.__mdEventsToday and determine TH TL
        price_open, price_high, price_low, price_close, T_high, T_low = self.__scanEventsSequence(self.__mdEventsToday)
        tomorrow_open, tomorrow_high, tomorrow_low, tomorrow_close, tT_high, tT_low = self.__scanEventsSequence(self.__mdEventsTomrrow)

        if not T_high:
            return

        latestDir = OrderData.DIRECTION_NONE
        T_win = timedelta(minutes=2)
        slip = 0.02

        # if T_high.month==6 and T_high.day in [25,26]:
        #      print('here')

        # step 2. determine the stop dailizedGainRates
        sell_stop = price_high -slip
        buy_stop  = min(price_low +slip, price_close*(100.0-self._dayPercentToCatch)/100)

        if (T_high < T_low) and price_close < (sell_stop *0.97): # this is a critical downhill, then enlarger the window to sell
            sell_stop= sell_stop *0.99 -slip

        catchback =0.0 # assume catch-back unnecessaray by default
        cleanup   =price_high*2 # assume no cleanup
        if tomorrow_high :
            tsell_stop = tomorrow_high -slip
            tbuy_stop  = min(tomorrow_low +slip, tomorrow_close*0.99)
            cleanup = max(tsell_stop, price_close -slip)

            if buy_stop > tsell_stop:
                buy_stop =0.0 # no buy today

            if tT_low < tT_high : # tomorrow is an up-hill
                catchback = tbuy_stop
            elif tsell_stop > price_close +slip:
                #catchback = min(tomorrow_high*(100.0- 2*self._dayPercentToCatch)/100, price_close +slip)
                catchback =price_low +slip
        elif (price_close < price_open*(100.0 +self._constraintBuy_closeOverOpen)/100):
            buy_stop =0.0 # forbid to buy
            catchback =0.0

        if cleanup < price_high: # if cleanup is valid, then no more buy/catchback
            catchback =0.0

        if sell_stop <= max(catchback, buy_stop)+slip:
            sell_stop = cleanup # no need to sell

        # step 2. faking the ideal orders
        for ev in self.__mdEventsToday:
            if EVENT_TICK != ev.type and EVENT_KLINE_PREFIX != ev.type[:len(EVENT_KLINE_PREFIX)] :
                continue

            evd = ev.data
            T = evd.datetime

            price = evd.price if EVENT_TICK == ev.type else evd.close
            # order = OrderData(self._account)
            # order.datetime = T

            tokens = (evd.vtSymbol.split('.'))
            symbol = tokens[0]
            advice = AdviceData(self.ident, symbol, self._marketState.exchange)
            advice.price = price
            advice.datetime = T
            advice.advisorId = '%s' % self.ident
            advice.Rdaily = 0.0
            advice.Rdstd  = 0.0
            advice.pdirNONE, advice.pdirLONG, advice.pdirSHORT = 0.0, 0.0, 0.0
            advice.pdirPrice = 0.0
            advice.pdirAsOf  = T
            advice.dirNONE, advice.dirLONG, advice.dirSHORT = 0.0, 0.0, 0.0

            if price <= buy_stop :
                advice.dirLONG = 1
                latestDir = advice.dirString()
                self.__adviceSeq.append(copy.copy(advice))
                continue

            if price >= sell_stop :
                advice.dirSHORT = 1
                latestDir = advice.dirString()
                self.__adviceSeq.append(copy.copy(advice))
                continue

            if T > max(T_high, T_low) :
                if price < catchback: # whether to catch back after sold
                    advice.dirLONG = 1
                    latestDir = advice.dirString()
                    self.__adviceSeq.append(copy.copy(advice))

    def scanEventsForAdvices000(self) :
        # step 1. scan self.__mdEventsToday and determine TH TL
        price_open, price_high, price_low, price_close, T_high, T_low = self.__scanEventsSequence(self.__mdEventsToday)
        tomorrow_open, tomorrow_high, tomorrow_low, tomorrow_close, tT_high, tT_low = self.__scanEventsSequence(self.__mdEventsTomrrow)

        if not T_high:
            return

        # if T_high.day==27 and T_high.month==2 :
        #     print('here')

        # step 2. faking the ideal orders
        bMayBuy = price_close >= price_open*(100.0 +self._constraintBuy_closeOverOpen)/100 # may BUY today, >=price_open*1.005
        T_win = timedelta(minutes=2)
        slip = 0.02

        sell_stop = max(price_high -slip, price_close*(100.0 +self._constraintSell_lossBelowHigh)/100)
        buy_stop  = min(price_close*(100.0 -self._constraintBuy_closeOverRecovery)/100, price_low +slip)
        uphill_catchback = price_close + slip

        if tomorrow_high :
            if tomorrow_high > price_close*(100.0 +self._constraintBuy_closeOverRecovery)/100 :
               bMayBuy = True

            if ((tT_low <tT_high and tomorrow_low < price_close) or tomorrow_high < (uphill_catchback * 1.003)) :
                uphill_catchback =0 # so that catch back never happen

        if price_close > price_low*(100.0 +self._constraintBuy_closeOverRecovery)/100 : # if close is at a well recovery edge
            bMayBuy = True

        for ev in self.__mdEventsToday:
            if EVENT_TICK != ev.type and EVENT_KLINE_PREFIX != ev.type[:len(EVENT_KLINE_PREFIX)] :
                continue

            evd = ev.data
            T = evd.datetime

            price = evd.price if EVENT_TICK == ev.type else evd.close
            tokens = (evd.vtSymbol.split('.'))
            symbol = tokens[0]
            advice = AdviceData(self.ident, symbol, self._marketState.exchange)
            advice.price = price
            advice.datetime = T
            advice.advisorId = '%s' % self.ident
            advice.Rdaily = 0.0
            advice.Rdstd  = 0.0
            advice.pdirNONE, advice.pdirLONG, advice.pdirSHORT = 0.0, 0.0, 0.0
            advice.pdirPrice = 0.0
            advice.pdirAsOf  = T
            advice.dirNONE, advice.dirLONG, advice.dirSHORT = 0.0, 0.0, 0.0

            if T_low < T_high : # up-hill
                if bMayBuy and (T <= T_low + T_win and price <= buy_stop) :
                    advice.dirLONG = 1
                    latestDir = advice.dirString()
                    self.__adviceSeq.append(copy.copy(advice))
                if T <= (T_high + T_win) and price >= (price_high -slip):
                    advice.dirSHORT = 1
                    latestDir = advice.dirString()
                    self.__adviceSeq.append(copy.copy(advice))
                elif T > T_high :
                    # if sell_stop < (uphill_catchback *1.002) and tomorrow_high > pSHORTrice_close:
                    #     continue # too narrow to perform any actions

                    if price > sell_stop:
                        advice.dirSHORT = 1
                        latestDir = advice.dirString()
                        self.__adviceSeq.append(copy.copy(advice))
                    elif price < uphill_catchback :
                        advice.dirLONG = 1
                        latestDir = advice.dirString()
                        self.__adviceSeq.append(copy.copy(advice))

            if T_low > T_high : # down-hill
                if price >= (price_high -slip) or (T < T_low and price >= (price_close*(100.0 +self._constraintSell_downHillOverClose)/100)):
                    advice.dirSHORT = 1
                    latestDir = advice.dirString()
                    self.__adviceSeq.append(copy.copy(advice))
                elif bMayBuy and (T > (T_low - T_win) and T <= (T_low + T_win) and price < round (price_close +price_low*3) /4, 3) :
                    advice.dirLONG = 1
                    latestDir = advice.dirString()
                    self.__adviceSeq.append(copy.copy(advice))


########################################################################
class ShortSwingScanner(OfflineSimulator):
    '''
    ShortSwingScanner extends OfflineSimulator by scanning the MarketEvents occurs up to several days, determining
    the short trend
    '''
    DAILIZED_GAIN_PCTS      = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0] # should be up to the stat data

    def __init__(self, program, trader, histdata, f4schema=None, **kwargs):
        '''Constructor
        '''
        super(ShortSwingScanner, self).__init__(program, trader, histdata, **kwargs)

        self._daysLong      = self.getConfig('constraints/futureDays',  3) # long-term prospect, default 1week(5days)
        self._daysShort     = self.getConfig('constraints/shortFuture', 2) # short-term prospect, default 2days
        self._byEvent       = self.getConfig('constraints/byEvent',   EVENT_KLINE_5MIN)
        self._h5compression = self.getConfig('h5compression', 'lzf').lower()
        self._h5compression = self.getConfig('h5compression', 'lzf').lower()
        self._h5filename    = self.getConfig('h5filename', None)

        self._f4schema = f4schema if isinstance(f4schema,dict) else { # the default schema is based on KL only
            'asof':1, 
            EVENT_KLINE_5MIN     : 50,
            EVENT_KLINE_1DAY     : 150,
        }

        if not self._byEvent or MARKETDATE_EVENT_PREFIX != self._byEvent[:len(MARKETDATE_EVENT_PREFIX)] :
            self._byEvent = EVENT_KLINE_1DAY

        self._warmupDays =0 # IdealTrader will not be constrainted by warmupDays
        self.__cOpenDays =0
        self.__stampByEvent = None

        self.__dtToday = None
        self.__momentsToSample = ['10:00:00', '11:00:00', '13:30:00', '14:30:00', '15:00:00']
        self.__stateOfMoments = {}

        self.__eventsOfDays = EvictableStack(evictSize=self._daysLong+1, nildata=[]) # list of days, each item contains events of days that up to self._daysLong
        self.__psptReadAhead = PerspectiveState('Dummy') # Perspective('Dummy')
        
        self.__sampleFrmSize  = 1024*8
        self.__sampleFrm = [None]  * self.__sampleFrmSize
        self.__sampleIdx, self.__frameNo = 0, 0

    @property
    def stateOfMoments(self) : return self.__stateOfMoments

    @property
    def measureDays(self) : return self._daysLong

    def setSampling(self, h5filename, momentsToSample=['10:00:00', '11:00:00', '13:30:00', '14:30:00', '15:00:00']):
        if h5filename : self._h5filename = h5filename
        self.__momentsToSample = momentsToSample

    # overwrite parent's by adjust startDate to measureDays ealier
    def setTimeRange(self, dtStart, dtEnd = None) :
        super(ShortSwingScanner, self).setTimeRange(dtStart, dtEnd)
        dtStart - timedelta(days = 2 + self._daysLong)
        return self._btStartDate, self._btEndDate

    def doAppInit(self): # return True if succ
        if not super(ShortSwingScanner, self).doAppInit() :
            return False

        if self._recorder:
            self._recorder.registerCategory(EVENT_ADVICE, params= {'columns' : AdviceData.COLUMNS})

        self._tradeSymbol = self.wkTrader.objectives[0] # idealTrader only cover a single symbol from the objectives
        self._episodes =1 # idealTrader only run one loop
        return True
    
    # to replace OfflineSimulator's OnEvent with some TradeAdvisor logic and execute the advice as order directly
    def OnEvent(self, ev):
        '''processing an incoming MarketEvent'''
        pass # do nothing here

    def OnEpisodeDone(self, reachedEnd=True):
        # NO neccessary to call super(ShortSwingScanner, self).OnEpisodeDone(reachedEnd) as we donot generate reports
        if self.__sampleIdx >0 : # if self.__sampleIdx >0 and not None in self.__sampleFrm :
            self.__saveFrame(self.__sampleFrm[:self.__sampleIdx])

        self.program.stop()

    # to replace BackTest's doAppStep
    def doAppStep(self):

        self._bGameOver = False # always False in IdealTrader
        reachedEnd = False
        if self._wkHistData :
            try :
                ev = next(self._wkHistData)
                if not ev or MARKETDATE_EVENT_PREFIX != ev.type[:len(MARKETDATE_EVENT_PREFIX)] : return
                if ev.data.datetime < self._btStartDate: return

                symbol = ev.data.symbol
                if ev.data.datetime <= self._btEndDate:
                    if not self.__psptReadAhead.updateByEvent(ev) : # .push(ev)
                        return 0

                    if symbol != self._tradeSymbol :# if not symbol in self.wkTrader.objectives:
                        return 0

                    stamp    = self.__psptReadAhead.getAsOf(symbol, self._byEvent)
                    price, _ = self.__psptReadAhead.latestPrice(symbol)
                    ohlc     = self.__psptReadAhead.dailyOHLC_sofar(symbol)

                    if not ohlc or self.__stampByEvent and self.__stampByEvent == stamp:
                        return 0
                    
                    if 0 != stamp.minute %3 : return # every 15min with self._byEvent=KL5m to reduce samples

                    self.__stampByEvent = stamp

                    dayOfEvent = ev.data.datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
                    if not self.__dtToday or self.__dtToday < dayOfEvent: # a new day comes

                        for i in range(0, self.__eventsOfDays.size): # 0 maps today's close price
                            label = 'priceIn%02dd' % i
                            eventsDaysAgo = self.__eventsOfDays[i]
                            for e in range(len(eventsDaysAgo)):
                                eventsDaysAgo[e][label] = ohlc.close
                        
                        if self.__eventsOfDays.size >= self.__eventsOfDays.evictSize:
                            # evicting the oldest date into the __sampleFrm, which were supposed filled all days result
                            eventsDaysAgo = self.__eventsOfDays[self.__eventsOfDays.size -1]
                            self.__saveEventsOfDay(eventsDaysAgo)
                            self.debug('doAppStep() committed %s into frames' % ','.join([i['ident'] for i in eventsDaysAgo]))

                        self.__eventsOfDays.push([])

                        if self.__dtToday:
                            self.debug('doAppStep() %s day-%03d[%s] applied onto [%d,%d]days-ago by %s' % (symbol, self.__cOpenDays, self.__dtToday.strftime('%Y-%m-%d'), self._daysShort, self._daysLong, self._byEvent))

                        self.__cOpenDays += 1
                        self.__dtToday = dayOfEvent
                        self.__stateOfMoments = {'date': self.__dtToday }

                    # in a same day
                    moment = ev.data.datetime.strftime('%H:%M:00')
                    if len(self.__momentsToSample) >0 and not moment in self.__momentsToSample:
                        return 1

                    stateOfEvent = {
                        'ident'   : '%s@%s' % (symbol, ev.data.datetime.strftime('%Y-%m-%dT%H:%M:%S')),
                        'ohlc'   : [ohlc.open, ohlc.high, ohlc.low, price],
                        'stateD4f': self.__psptReadAhead.exportImg6C_3Liner16x32R(symbol, self.outdir), # floatsD4(d4wished = self._f4schema), 
                        }

                    self.__eventsOfDays[0].append(stateOfEvent)
                    self.__stateOfMoments[moment] = stateOfEvent
                    self.debug('doAppStep() sampled state of %s' % stateOfEvent['ident'])

                    return 1 # successfully performed a step by pushing an Event

                reachedEnd = True

            except StopIteration:
                reachedEnd = True
                self.info('hist-read: end of playback')
            except Exception as ex:
                self.logexception(ex)

        # if self.__sampleIdx >0 :
        #     self.__saveFrame(self.__sampleFrm[:self.__sampleIdx])

        # this test should be done if reached here
        self.info('doAppStep() episode[%s] finished: %d steps, KO[%s] end-of-history[%s]' % (self.episodeId, self._stepNoInEpisode, self._bGameOver, reachedEnd))
        self.OnEpisodeDone(reachedEnd)
        
        # exit(0) # ShortSwingScanner is not supposed to run forever, just exit instead of return

    def __saveEventsOfDay(self, eventsOfDay):

        for ev in eventsOfDay:
            stateD4f = ev['stateD4f']
            if not stateD4f or len(stateD4f) <=0:
                continue

            ohlc  = ev['ohlc']
            price = ohlc[3]
            if price <=0.01:
                continue

            dailizedGainRates = []
            for i in range(0, self._daysLong+1):
                label = 'priceIn%02dd' % i
                dgr = (ev[label]-price) *100 / price
                if i >0: dgr /= i
                dailizedGainRates.append(dgr)

            # sample code on how to classify the grainRates
            gainRateL, gainRateS = dailizedGainRates[self._daysLong-1], dailizedGainRates[self._daysShort-1]
            gainClassL, gainClassS= 0, 0
            for redge in ShortSwingScanner.DAILIZED_GAIN_PCTS:
                if gainRateL >= redge:
                    gainClassL += 1
                if gainRateS >= redge:
                    gainClassS += 1
            
            gainClass = [0] * (1 + len(ShortSwingScanner.DAILIZED_GAIN_PCTS))*2
            gainClass[gainClassL] =1
            gainClass[len(ShortSwingScanner.DAILIZED_GAIN_PCTS) +1 + gainClassS] =1

            self.__sampleIdx = self.__sampleIdx % self.__sampleFrmSize
            if 0 == self.__sampleIdx and not None in self.__sampleFrm :
                # frame full, output it into a HDF5 file
                self.__saveFrame(self.__sampleFrm)

            self.__sampleFrm[self.__sampleIdx] = (stateD4f, dailizedGainRates, ohlc)
            self.__sampleIdx +=1

    def __saveFrame(self, rangedFrame):
        metrix  = np.array(rangedFrame)
        lenF    = len(rangedFrame)

        stateshape = np.array(metrix[0][0]).shape
        col_state  = np.concatenate(metrix[:, 0]).reshape(lenF, *stateshape).astype('float16')

        col_gainRates  = np.concatenate(metrix[:, 1]).reshape(lenF, *np.array(metrix[0][1]).shape).astype('float16')
        col_ohlc       = np.concatenate(metrix[:, 2]).reshape(lenF, *np.array(metrix[0][2]).shape).astype('float16')  # col_price = metrix[:, 2].astype('float16')

        normalizedId = 'FclzD4X%dR%dBy%s' %(col_state.shape[1], self._daysLong, self._byEvent[len(MARKETDATE_EVENT_PREFIX):])

        h5args =copy.copy(hist.H5DSET_DEFAULT_ARGS)
        if self._h5compression and len(self._h5compression)>0:
            h5args['compression'] = self._h5compression

        if not self._h5filename or len(self._h5filename) <=0:
            self._h5filename = os.path.join(self.wkTrader.outdir, '%s_%s.h5' % (normalizedId, self._tradeSymbol) )
        with h5py.File(self._h5filename, 'a') as h5file:
            frameId = '%s.frm%s' % (self._tradeSymbol, str(self.__frameNo).zfill(3))
            self.__frameNo += 1

            desc = '%s.%s: dailized gain-rates of %s in future %s days by %s' % (normalizedId, frameId, self._tradeSymbol, self._daysLong, self.ident)
            if frameId in h5file.keys():
                del h5file[frameId]
                
            g = h5file.create_group(frameId)
            g.attrs['desc']     = desc
            g.attrs['state']     = 'market state'
            g.attrs[u'default']     = 'state'
            g.attrs['normalizedId'] = normalizedId
            g.attrs['size']         = col_state.shape[0]
            g.attrs['futureDays']   = self._daysLong
            g.attrs['signature']    = self.ident

            # g.create_dataset(u'title',      data= desc)
            st = g.create_dataset('state',  data= col_state, **h5args)
            st.attrs['f4schema'] = str(self._f4schema)
            ac = g.create_dataset('gainRates', data= col_gainRates, **h5args)
            ac.attrs['desc'] = 'grain-rate(%%) in up to %d days, 0-means close-of-today' % col_gainRates.shape[1]
            pr = g.create_dataset('ohlc', data= col_ohlc, **h5args)
            pr.attrs['desc'] = 'open-high-low-price so far in the day'
            
        self.info('saved %s with %s samples into file %s with sig[%s]' % (frameId, len(col_state), self._h5filename, self.ident))

