# encoding: UTF-8

'''
BackTest inherits from Trader
'''
from __future__ import division

from Application import *
from Account import *
from Trader import *
import HistoryData as hist
from MarketData import TickData, KLineData, EVENT_TICK, EVENT_KLINE_1MIN, EVENT_KLINE_5MIN, EVENT_KLINE_1DAY

from Perspective import *
import vn.VnTrader as vn

from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import multiprocessing
import copy

# import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import codecs
import math

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

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

        self._btStartDate  = datetime.strptime(self.getConfig('startDate', '2000-01-01'), '%Y-%m-%d').replace(hour=0, minute=0, second=0) # 回测数据开始日期，datetime对象
        self._btEndDate    = datetime.strptime(self.getConfig('endDate', '2999-12-31'), '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        self._startBalance = self.getConfig('startBalance', 100000)
        self._episodes     = self.getConfig('episodes', 1)
        self._plotReport   = self.getConfig('plotReport', 'False').lower() in BOOL_STRVAL_TRUE
        self._pctMaxDrawDown = self.getConfig('pctMaxDrawDown', 21) # we allow 30% lost during a episode
        self._warmupDays     = self.getConfig('warmupDays', 5) # observe for a week by default to make the market state not so empty

        self._episodeNo = 1 # count start from 1 to ease reading
        self._stepNoInEpisode =0
        self.__execStamp_appStart = datetime.now()
        self.__execStamp_episodeStart = self.__execStamp_appStart

        self._dailyCapCost = float(self._startBalance) * self._initTrader._annualCostRatePcnt /220 / 100  # assuming 220 opendays every year
        self._maxBalance = self._startBalance

        # backtest will always clear the datapath
        self._initTrader._outDir = '%s%s%s' % (self.dataRoot, self.ident, self.program.progId)
        try :
            shutil.rmtree(self._initTrader._outDir)
        except:
            pass
        try :
            os.makedirs(self._initTrader._outDir)
        except:
            pass

        self.__dtLastData = self._btStartDate
        self._bGameOver = False
        self._openDays = 0

    @property
    def episodeId(self) :
        return 'E' + str(self._episodeNo).zfill(6)

    @property
    def wkTrader(self) :
        return self.__wkTrader

    @property
    def outdir(self) :
        return self._initTrader._outDir

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
        return '%s/%s' % (self.__class__.__name__, (self.__wkTrader.ident if self.__wkTrader else self._id))

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

        # step 1. wrapper the Trader
        if not self._initTrader :
            return False
        
        self.info('doAppInit() taking trader-template[%s]' % (self._initTrader.ident))
        self._program.removeApp(self._initTrader.ident)
        self._program.addApp(self)
        if not self._initTrader.doAppInit() :
            self.info('doAppInit() failed to initialize trader-template[%s]' % (self._initTrader.ident))
            return False

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
                    self.debug('hist-read: symbol[%s]%s asof[%s] lastPrice[%s] OHLC%s' % (s, ev.type[len(MARKETDATE_EVENT_PREFIX):], self._marketState.getAsOf(s).strftime('%Y%m%dT%H%M'), self._marketState.latestPrice(s), self._marketState.dailyOHLC_sofar(s)))
                    self.OnEvent(ev) # call Trader
                    self._stepNoInEpisode += 1
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
        with codecs.open('%s/%s_summary.txt' %(self._initTrader._outDir, self.episodeId), "w","utf-8") as rptfile:
            rptfile.write(strReport)
            self.debug('doAppStep() episode[%s] summary report generated' %(self.episodeId))

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

    def onDayOpen(self, symbol, date):
        return self.__wkTrader.onDayOpen(symbol, date)

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
            # csvfile = '%s/%s_DR.csv' %(self._initTrader._outDir, self.episodeId)
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
        # self._recorder = self._program.createApp(hist.TaggedCsvRecorder, filepath ='%s/BT_%s.tcsv' % (self._initTrader._outDir, self.episodeId))

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
        self._program.addApp(self.__wkTrader)
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
           
        self._dataBegin_date = None
        self._dataBegin_openprice = 0.0
        
        self._dataEnd_date = None
        self._dataEnd_closeprice = 0.0

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
        self.subscribeEvent(Account.EVENT_ORDER)
        self.subscribeEvent(Account.EVENT_TRADE)

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

    #----------------------------------------------------------------------
    def plotResult(self, tradeDays):
        # 绘图
        
        filename = '%s%s_DR.png' %(self._initTrader._outDir, self.episodeId)
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
       
''' 
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
##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--
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

        self._dataBegin_date = None
        self._dataBegin_openprice = 0.0
        self._dataEnd_date = None
        self._dataEnd_closeprice = 0.0

        self.setRecorder(self.__wkTrader.recorder)

        # attributes of virtual account
        # -----------------------------------------
        self._startBalance = 100000      # 10w
        self._startBalance = self.getConfig('startBalance', 100000)

        self.__execStamp_appStart = datetime.now()
        self.__dtLastData = None
        self._maxBalance = self._startBalance

        # backtest will always clear the datapath
        self.__wkTrader._outDir = '%s%s%s' % (self.dataRoot, self.ident, self.program.progId)
        try :
            shutil.rmtree(self.__wkTrader._outDir)
        except:
            pass

        try :
            os.makedirs(self.__wkTrader._outDir)
        except:
            pass

        self._openDays = 0

    @property
    def wkTrader(self) :
        return self.__wkTrader

    @property
    def outdir(self) :
        return self.__wkTrader._outDir

    def setRecorder(self, recorder) :
        self._recorder = recorder
        if self._recorder :
            self._recorder.registerCategory(Account.RECCATE_ORDER,       params= {'columns' : OrderData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_TRADE,       params= {'columns' : TradeData.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYPOS,    params= {'columns' : DailyPosition.COLUMNS})
            self._recorder.registerCategory(Account.RECCATE_DAILYRESULT, params= {'columns' : DailyResult.COLUMNS})

        if self.__wkTrader :
            self.__wkTrader._recorder = self._recorder
            
        return self._recorder

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    @property
    def ident(self) :
        return '%s/%s' % (self.__class__.__name__, (self.__wkTrader.ident if self.__wkTrader else self._id))

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

        prevState = self.program.loadObject('%s/marketState' % 'OnlineSimulator') # '%s/marketState' % self.__class__)
        if prevState:
            self.__wkTrader._marketState = prevState
        self._marketState = self.__wkTrader._marketState

        prevAccount = self.program.loadObject('%s/account' % 'OnlineSimulator') # '%s/marketState' % self.__class__)
        if prevAccount:
            self.__wkTrader.account = prevAccount
        self._originAcc = self.__wkTrader.account

        # # ADJ_1. adjust the Trader._dictObjectives to append suffix MarketData.TAG_BACKTEST
        # for obj in self._dictObjectives.values() :
        #     if len(obj["dsTick"]) >0 :
        #         obj["dsTick"] += MarketData.TAG_BACKTEST
        #     if len(obj["ds1min"]) >0 :
        #         obj["ds1min"] += MarketData.TAG_BACKTEST

        # step 2. connects the trader and account 
        if self._account :
            self._program.removeApp(self._account)
            self._account =None
        
        # step 3. wrapper the broker drivers of the accounts
        self._maxBalance = self._startBalance
        if self._originAcc and not isinstance(self._originAcc, AccountWrapper):
            self._program.removeApp(self._originAcc)
            self._account = AccountWrapper(self._program, btTrader =self, account=copy.copy(self._originAcc)) # duplicate the original account for test espoches
            self._account.hostTrader(self) # adopt the account by pointing its._trader to self
            if not prevAccount:
                self._account.setCapital(self._startBalance, True)
                self.__wkTrader._dailyCapCost = 0

            self._program.addApp(self._account)
            self._account._marketState = self._marketState
            self.__wkTrader._account = self._account
            
            cashAvail, cashTotal, positions = self._account.positionState()
            _, posvalue = self._account.summrizeBalance(positions, cashTotal)
            capitalTotal = cashTotal + posvalue

            if capitalTotal > self._maxBalance:
                self._maxBalance = capitalTotal
            self.info('doAppInit() wrappered account[%s] restored[%s] to [%s] with capitalTotal[%s=%s+%s] max[%s]' % (self._originAcc.ident, self._account.ident, 'T' if prevState else 'F', capitalTotal, cashTotal, posvalue, self._maxBalance))

        if len(self.__wkTrader._dictObjectives) <=0:
            sl = self._marketState.listOberserves()
            for symbol in sl:
                self.__wkTrader.openObjective(symbol)

        # step 4. subscribe events
        self.subscribeEvent(Account.EVENT_ORDER)
        self.subscribeEvent(Account.EVENT_TRADE)

        self.subscribeEvent(EVENT_TICK)
        self.subscribeEvent(EVENT_KLINE_1MIN)
        self.subscribeEvent(EVENT_KLINE_5MIN)
        self.subscribeEvent(EVENT_KLINE_1DAY)

        self.info('doAppInit() done, obj-in-program: %s' % (self._program.listByType(MetaObj)))
        return True

    def doAppStep(self):
        super(OnlineSimulator, self).doAppStep()
        self._account.doAppStep()

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
        self.program.saveObject(self._originAcc, '%s/account' % 'OnlineSimulator')
        return self.__wkTrader.eventHdl_Order(ev)
            
    def eventHdl_Trade(self, ev):
        self.program.saveObject(self._originAcc, '%s/account' % 'OnlineSimulator')
        return self.__wkTrader.eventHdl_Trade(ev)

    def onDayOpen(self, symbol, date):
        self.program.saveObject(self._originAcc, '%s/account' % 'OnlineSimulator')
        return self.__wkTrader.onDayOpen(symbol, date)


##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--

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

        with self._nest._lock:
            outgoingOrders = copy.deepcopy(list(self._nest._dictOutgoingOrders.values()))

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

        if (len(ordersToCancel) + len(outgoingOrders)) >0:
            self.debug('step() cancelled %d orders, placed %d orders'% (len(ordersToCancel), len(outgoingOrders)))

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
    # def sendOrder(self, vtSymbol, orderType, price, volume, strategy): return self._nest.sendOrder(vtSymbol, orderType, price, volume, strategy)
    def cancelOrder(self, brokerOrderId): return self._nest.cancelOrder(brokerOrderId)
    def batchCancel(self, brokerOrderIds): return self._nest.batchCancel(brokerOrderIds)
    def cancelAllOrders(self): return self._nest.cancelAllOrders()
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy): return self._nest.sendStopOrder(vtSymbol, orderType, price, volume, strategy)
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
    def roundToPriceTick(self, price): return self._nest.roundToPriceTick(price)
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
    def sendOrder(self, symbol, orderType, price, volume, strategy):
        source = 'ACCOUNT'
        if strategy:
            source = strategy.id

        orderData = OrderData(self)
        # 代码编号相关
        orderData.symbol      = symbol
        orderData.exchange    = self._exchange
        orderData.price       = self.roundToPriceTick(price) # 报单价格
        orderData.totalVolume = volume    # 报单总数量
        orderData.source      = source
        orderData.datetime    = self.datetimeAsOfMarket()

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
            bestPrice          = round(((kldata.open + kldata.close) *4 + kldata.high + kldata.low) /10, 2)

            buyCrossPrice      = kldata.low        # 若买入方向限价单价格高于该价格，则会成交
            sellCrossPrice     = kldata.high      # 若卖出方向限价单价格低于该价格，则会成交
            maxCrossVolume     = kldata.volume
            buyBestCrossPrice  = bestPrice       # 在当前时间点前发出的买入委托可能的最优成交价
            sellBestCrossPrice = bestPrice       # 在当前时间点前发出的卖出委托可能的最优成交价
            
            # 张跌停封板
            if buyCrossPrice <= kldata.open*0.9 :
                buyCrossPrice =0
            if sellCrossPrice >= kldata.open*1.1 :
                sellCrossPrice =0

        if not symbol :
            return # ignore those non-tick/kline events

        # 先撮合限价单
        self.__crossLimitOrder(symbol, dtEvent, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3), maxCrossVolume)
        # 再撮合停止单
        self.__crossStopOrder(symbol, dtEvent, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3), maxCrossVolume)

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
                
                # 如果发生了成交
                if not buyCross and not sellCross:
                    pendingOrders.append(order)
                    continue

                # 推送成交数据
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

        if len(finishedOrders)>0:
            strPendings = ''
            for o in pendingOrders:
                strPendings += 'O[%s],' % o.desc
            self.info('crossLimitOrder() crossed %d orders:%s; %d pendings:%s'% (len(finishedOrders), strCrossed, len(pendingOrders), strPendings))

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


if __name__ == '__main__':
    print('-'*20)

    # oldprogram()

    # new program:
    sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_AShare.json']
    p = Program()
    p._heartbeatInterval =-1
    SYMBOL = '000001' # '000001' # '000540' '000001'

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder')
    csvdir = '/mnt/e/AShareSample' # '/mnt/m/AShareSample'
    ps = Perspective('AShare', SYMBOL)
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder='%s/%s' % (csvdir, SYMBOL), fields='date,time,open,high,low,close,volume,ammount')
    histdata = PerspectiveGenerator(ps)
    histdata.adaptReader(csvreader, EVENT_KLINE_1MIN)
    
    marketstate = hist.PlaybackDay('AShare') # = PerspectiveState('AShare')
    p.addObj(marketstate)

    # btdr = p.createApp(BaseTrader, configNode ='backtest', account=acc)
    vntdr = p.createApp(vn.VnTrader, configNode ='backtest', account=acc)
    
    p.info('all objects registered piror to BackTestApp: %s' % p.listByType(MetaObj))
    
    p.createApp(BackTestApp, configNode ='backtest', trader=vntdr, histdata=csvreader) #histdata = histdata)

    # # cta.loadSetting()
    # # logger.info(u'CTA策略载入成功')
    
    # # cta.initAll()
    # # logger.info(u'CTA策略初始化成功')
    
    # # cta.startAll()
    # # logger.info(u'CTA策略启动成功')

    p.start()
    p.loop()
    p.stop()

