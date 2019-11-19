# encoding: UTF-8

'''
BackTest inherits from Trader
'''
from __future__ import division

from Application import *
from Account import *
from Trader import *
from HistoryData import *
from Perspective import *

from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import multiprocessing
import copy

# import pymongo
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import shutil

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

########################################################################
class BackTestApp(MetaTrader):
    '''
    BackTest is a wrapprer of Trader, with same interface of Trader
    '''
    
    #----------------------------------------------------------------------
    def __init__(self, program, trader, jsettings):
        """Constructor"""

        super(BackTestApp, self).__init__(program, settings)
        self._nestTrader = trader
        self._nestTraderClz = type(trader) # class of the nested Trader
        self._account = None

        # 回测相关属性
        # -----------------------------------------
        tmpstr = settings.startDate('2000-01-01')
        self._btStartDate = datetime.strptime(tmpstr, '%Y-%m-%d') # 回测数据开始日期，datetime对象
        self.__dtData = self._btStartDate
        self._btEndDate   = None         # 回测数据结束日期，datetime对象, None to today or late data
        tmpstr = settings.endDate('')
        if len(tmpstr) >8:
            self._btEndDate = datetime.strptime(tmpstr, '%Y-%m-%d')
        self._startBalance = settings.startBalance(100000)

        # backtest will always clear the datapath
        if len(self._dataPath) <=0:
            self._dataPath = '/tmp'

        self._dataPath += '/%s_%s' % (self.ident, datetime.now().strftime('%Y%m%dT%H%M%S'))
        try :
            shutil.rmtree(self._dataPath)
            os.makedirs(self._dataPath)
        except:
            pass

    @property
    def ident(self) :
        if not self._nestTrader :
            return super(BackTestApp, self).ident
        
        return '%s.%s' % (self.__class__.__name__, self._nestTrader.ident)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def debug(self, message):
        """输出内容"""
        if self.__dtData:
            message = str(self.__dtData) + ' ' + message
        super(BackTestApp, self).debug(message)
    
    def log(self, level, msg):
        if self.__dtData:
            msg = str(self.__dtData) + ' ' + msg
        super(BackTestApp, self).log(level, msg)

    def doAppInit(self): # return True if succ
        if not super(BackTestApp, self).init() :
            return False

        # step 1. wrapper the Trader
        if not self._nestTrader :
            self._program.removeApp(self._nestTrader.ident)
            self._program.addApp(self)
            self._account = self._nestTrader.account

        # step 1. wrapper the broker drivers of the accounts
        if not self._account and not isinstance(self._account, AccountWrapper):
            originAcc = self._account
            self._program.removeApp(originAcc.ident)
            self._initAcc = AccountWrapper(self, copy(originAcc)) # duplicate the original account for test espoches
            self._initAcc.setCapital(self._startBalance, True)
            # MOVED to resetTest():
            # self._account = wrapper
            # self._program.addApp(self._account)

        # ADJ_1. adjust the Trader._dictObjectives to append suffix MarketData.TAG_BACKTEST
        for obj in self._dictObjectives.values() :
            if len(obj["dsTick"]) >0 :
                obj["dsTick"] += MarketData.TAG_BACKTEST
            if len(obj["ds1min"]) >0 :
                obj["ds1min"] += MarketData.TAG_BACKTEST

        # end of adjust the Trader

        self._execDate_Begin = None
        self._execClosePrice_Begin = 0

        self.initData = []          # 初始化用的数据
        self.resetTest()

    def doAppStep(self):
        super(BaseTrader, self).doAppStep()

        try :
            ev = next(self.__pg)
            if not ev : return

            self.OnEvent(ev) # call Trader
            return # successfully pushed an Event
        except StopIteration:
            pass

        # this test should be done if reached here
        self.info('test-round[%s] done, generating report', self.__testRoundId)
        self.generateReport()

        self.__testRoundId +=1
        if (self.__testRoundId < self.__testRounds) :
            self.info('test-round[%s] starts', self.__testRoundId)
            self.resetTest()
            return

        # all tests have been done
        self.stop()

    def OnEvent(self, ev): 
        # step 2. 收到行情后，在启动策略前的处理
        if EVENT_TICK == ev.type:
            self.__tradeMatchingByTick(ev.data)
        elif EVENT_KLINE_PREFIX == evtype[:len(EVENT_KLINE_PREFIX)] :
            self.__tradeMatchingByKLine(ev.data)

        return self._nestTrader.OnEvent(ev)

    # end of BaseApplication routine
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    # Overrides of Events handling
    def eventHdl_Order(self, ev):
        return self._nestTrader.eventHdl_Order(ev)
            
    def eventHdl_Trade(self, ev):
        return self._nestTrader.eventHdl_Trade(ev)

    def onDayOpen(self, symbol, date):
        return self._nestTrader.onDayOpen(symbol, date)

    def proc_MarketEvent(self, evtype, symbol, data):
        self.error('proc_MarketEvent() should not be here')

    #------------------------------------------------
    # 数据回放结果计算相关

    def resetTest(self) :
#        self._account = self._accountClass(None, tdBackTest, self._settings.account)
#        self._account._dvrBroker._backtest= self
#        self._account._id = "BT.%s:%s" % (self.strategyBT, self.symbol)
        
        if self._account :
            self._program.removeApp(self._account)
            self._account =None
        
        if self._initAcc :
            self._account = copy.deepcopy(self._initAcc)
            self._program.addApp(self._account)
        
        self._account.setCapital(self._startBalance, True) # 回测时的起始本金（默认10万）

        self._execClosePrice_Begin = 0.0
        self._execEndClose = 0.0

        self._execDate_Begin = None
        self._execEnd = None
        self._execClosePrice_Begin = 0.0

        self.initData = []          # 初始化用的数据

        self.resultList = []             # 交易结果列表
        self.posList = [0]               # 每笔成交后的持仓情况        
        self.tradeTimeList = []          # 每笔成交时间戳

        # 清空限价单相关
        self.tdDriver.limitOrderCount = 0
        self.tdDriver.limitOrderDict.clear()
        self._dictLimitOrders.clear()        
        
        # 清空停止单相关
        self.tdDriver.stopOrderCount = 0
        self.tdDriver.stopOrderDict.clear()
        self._dictStopOrders.clear()
        
        # 清空成交相关
        self.tdDriver.tradeCount = 0
        self._dictTrades.clear()

        # 当前最新数据，用于模拟成交用
        self.tick = None
        self.bar  = None
        self.__dtData   = None      # 最新数据的时间

        self._marketstat = PerspectiveDict(self._account.exchange)
        self.__pg = None

        ##########################
        for ob in self._dictObjectives.keys() :
            self._marketstat.addMonitor(ob)
            ps = Perspective(self._account.exchange, ob)
            self.__pg = PerspectiveGenerator(ps)
            hpb = hist.CsvPlayback(symbol=ob, folder=self._dataPath, fields='date,time,open,high,low,close,volume,ammount')
            self.__pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)
            break;

    def __tradeMatchingByKLine(self, kldata):
        """收到行情后处理本地停止单（检查是否要立即发出）"""

        # TODO check if received the END signal of backtest data

        # if self.mode != self.BAR_MODE:
        #     return

        if self._execDate_Begin ==None:
            self._execClosePrice_Begin = kldata.close
            self._execDate_Begin = kldata.date

        self.__dtData = kldata.datetime

        # 先确定会撮合成交的价格
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

        # 先撮合限价单
        self._account.crossLimitOrder(kldata.symbol, kldata.datetime, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3), maxCrossVolume)
        # 再撮合停止单
        self._account.crossStopOrder(kldata.symbol, kldata.datetime, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3), maxCrossVolume)

    def __tradeMatchingByTick(self, tkdata):
        """收到行情后，在启动策略前的处理
        通常处理本地停止单（检查是否要立即发出）"""

        if not self._execDate_Begin:
            self._execDate_Begin = tkdata.date
            self._execClosePrice_Begin = tkdata.priceTick

        self.__dtData = tkdata.datetime

        # 先确定会撮合成交的价格
        buyCrossPrice      = tkdata.a1P
        sellCrossPrice     = tkdata.b1P
        buyBestCrossPrice  = tkdata.a1P
        sellBestCrossPrice = tkdata.b1P

        # 先撮合限价单
        self._account.crossLimitOrder(tkdata.symbol, self.__dtData, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3)) # to determine maxCrossVolume from Tick, maxCrossVolume)
        # 再撮合停止单
        self._account.crossStopOrder(tkdata.symbol, self.__dtData, buyCrossPrice, sellCrossPrice, round(buyBestCrossPrice,3), round(sellBestCrossPrice,3)) # to determine maxCrossVolume from Tick, maxCrossVolume)

    #----------------------------------------------------------------------
    def generateReport(self, df=None, result=None):
        """显示按日统计的交易结果"""
        if df is None:
            df, result = sumupDailyResults(self.startBalance, self._account.dailyResultDict)

        df.to_csv(self._account._id+'.csv')
            
        originGain = 0.0
        if self._execClosePrice_Begin >0 :
            originGain = (self._execEndClose - self._execClosePrice_Begin)*100/self._execClosePrice_Begin

        # 输出统计结果
        self.debug('-' * 30)
        self.debug(u'回放日期：\t%s(close:%.2f)~%s(close:%.2f): %s%%'  %(self._execDate_Begin, self._execClosePrice_Begin, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.debug(u'交易日期：\t%s(close:%.2f)~%s(close:%.2f)' % (result['startDate'], self._execClosePrice_Begin, result['endDate'], self._execEndClose))
        
        self.debug(u'交易日数：\t%s (盈利%s,亏损%s)' % (result['totalDays'], result['profitDays'], result['lossDays']))
        
        self.debug(u'起始资金：\t%s' % formatNumber(self._startBalance))
        self.debug(u'结束资金：\t%s' % formatNumber(result['endBalance']))
    
        self.debug(u'总收益率：\t%s%%' % formatNumber(result['totalReturn']))
        self.debug(u'年化收益：\t%s%%' % formatNumber(result['annualizedReturn']))
        self.debug(u'总盈亏：\t%s' % formatNumber(result['totalNetPnl']))
        self.debug(u'最大回撤: \t%s' % formatNumber(result['maxDrawdown']))   
        self.debug(u'百分比最大回撤: %s%%' % formatNumber(result['maxDdPercent']))   
        
        self.debug(u'总手续费：\t%s' % formatNumber(result['totalCommission']))
        self.debug(u'总滑点：\t%s' % formatNumber(result['totalSlippage']))
        self.debug(u'总成交金额：\t%s' % formatNumber(result['totalTurnover']))
        self.debug(u'总成交笔数：\t%s' % formatNumber(result['totalTradeCount'],0))
        
        self.debug(u'日均盈亏：\t%s' % formatNumber(result['dailyNetPnl']))
        self.debug(u'日均手续费：\t%s' % formatNumber(result['dailyCommission']))
        self.debug(u'日均滑点：\t%s' % formatNumber(result['dailySlippage']))
        self.debug(u'日均成交金额：\t%s' % formatNumber(result['dailyTurnover']))
        self.debug(u'日均成交笔数：\t%s' % formatNumber(result['dailyTradeCount']))
        
        self.debug(u'日均收益率：\t%s%%' % formatNumber(result['dailyReturn']))
        self.debug(u'收益标准差：\t%s%%' % formatNumber(result['returnStd']))
        self.debug(u'夏普率：\t%s' % formatNumber(result['sharpeRatio']))
        
        self.plotResult(df)

    #----------------------------------------------------------------------
    def plotResult(self, df):
        # 绘图
        plt.rcParams['agg.path.chunksize'] =10000

        fig = plt.figure(figsize=(10, 16))
        
        pBalance = plt.subplot(4, 1, 1)
        pBalance.set_title(self._id + ' Balance')
        df['balance'].plot(legend=True)
        
        pDrawdown = plt.subplot(4, 1, 2)
        pDrawdown.set_title('Drawdown')
        pDrawdown.fill_between(range(len(df)), df['drawdown'].values)
        
        pPnl = plt.subplot(4, 1, 3)
        pPnl.set_title('Daily Pnl') 
        df['netPnl'].plot(kind='bar', legend=False, grid=False, xticks=[])

        pKDE = plt.subplot(4, 1, 4)
        pKDE.set_title('Daily Pnl Distribution')
        df['netPnl'].hist(bins=50)
        
        plt.savefig('DR-%s.png' % self._account._id, dpi=400, bbox_inches='tight')
        plt.show()
        plt.close()
       
    # #----------------------------------------------------------------------
    # def runOptimization(self, strategyClass, optimizationSetting):
    #     """优化参数"""
    #     # 获取优化设置        
    #     settingList = optimizationSetting.generateSetting()
    #     targetName = optimizationSetting.optimizeTarget
        
    #     # 检查参数设置问题
    #     if not settingList or not targetName:
    #         self.debug(u'优化设置有问题，请检查')
        
    #     # 遍历优化
    #     self.resultList =[]
    #     for setting in settingList:
    #         self.clearBackTesting()
    #         self.debug('-' * 30)
    #         self.debug('setting: %s' %str(setting))
    #         self.initStrategy(strategyClass, setting)
    #         self.runBacktesting()

    #         df, d = sumupDailyResults(self.startBalance, self._account.dailyResultDict)
    #         try:
    #             targetValue = d[targetName]
    #         except KeyError:
    #             targetValue = 0
    #         self.resultList.append(([str(setting)], targetValue, d))
        
    #     # 显示结果
    #     self.resultList.sort(reverse=True, key=lambda result:result[1])
    #     self.debug('-' * 30)
    #     self.debug(u'优化结果：')
    #     for result in self.resultList:
    #         self.debug(u'参数：%s，目标：%s' %(result[0], result[1]))    
    #     return self.resultList
            
    # #----------------------------------------------------------------------
    # def runParallelOptimization(self, strategyClass, optimizationSetting):
    #     """并行优化参数"""
    #     # 获取优化设置        
    #     settingList = optimizationSetting.generateSetting()
    #     targetName = optimizationSetting.optimizeTarget
        
    #     # 检查参数设置问题
    #     if not settingList or not targetName:
    #         self.debug(u'优化设置有问题，请检查')
        
    #     # 多进程优化，启动一个对应CPU核心数量的进程池
    #     pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #     l = []

    #     for setting in settingList:
    #         l.append(pool.apply_async(optimize, (strategyClass, setting,
    #                                              targetName, self.mode, 
    #                                              self.startDate, self.initDays, self.endDate,
    #                                              self.dbName, self.symbol)))
    #     pool.close()
    #     pool.join()
        
    #     # 显示结果
    #     resultList = [res.get() for res in l]
    #     resultList.sort(reverse=True, key=lambda result:result[1])
    #     self.debug('-' * 30)
    #     self.debug(u'优化结果：')
    #     for result in resultList:
    #         self.debug(u'参数：%s，目标：%s' %(result[0], result[1]))    
            
    #     return resultList

''' 
    # Transaction-based
    #----------------------------------------------------------------------
    def plotBacktestingResult(self, d):
        # 绘图
        plt.rcParams['agg.path.chunksize'] =10000
        fig = plt.figure(figsize=(10, 16))
        
        pCapital = plt.subplot(4, 1, 1)
        pCapital.set_ylabel("capital")
        pCapital.plot(d['capitalList'], color='r', lw=0.8)
        
        pDD = plt.subplot(4, 1, 2)
        pDD.set_ylabel("DD")
        pDD.bar(range(len(d['drawdownList'])), d['drawdownList'], color='g')
        
        pPnl = plt.subplot(4, 1, 3)
        pPnl.set_ylabel("pnl")
        pPnl.hist(d['pnlList'], bins=50, color='c')

        pPos = plt.subplot(4, 1, 4)
        pPos.set_ylabel("Position")
        if d['posList'][-1] == 0:
            del d['posList'][-1]
        tradeTimeIndex = [item.strftime("%m/%d %H:%M:%S") for item in d['tradeTimeList']]
        xindex = np.arange(0, len(tradeTimeIndex), np.int(len(tradeTimeIndex)/10))
        tradeTimeIndex = map(lambda i: tradeTimeIndex[i], xindex)
        pPos.plot(d['posList'], color='k', drawstyle='steps-pre')
        pPos.set_ylim(-1.2, 1.2)
        plt.sca(pPos)
        plt.tight_layout()
        plt.xticks(xindex, tradeTimeIndex, rotation=30)  # 旋转15
        
        plt.savefig('BT-%s.png' % self._id, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()

    #----------------------------------------------------------------------
    def showBacktestingResult(self):
        """显示回测结果"""

        d = self.calculateTransactions()
        originGain = 0.0
        if self._execClosePrice_Begin >0 :
            originGain = (self._execEndClose - self._execClosePrice_Begin)*100/self._execClosePrice_Begin

        # 输出
        self.debug('-' * 30)
        self.debug(u'回放日期：\t%s(close:%.2f)~%s(close:%.2f): %s%%'  %(self._execDate_Begin, self._execClosePrice_Begin, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.debug(u'交易日期：\t%s(close:%.2f)~%s(close:%.2f)' % (d['timeList'][0], self._execClosePrice_Begin, d['timeList'][-1], self._execEndClose))
        
        self.debug(u'总交易次数：\t%s' % formatNumber(d['totalResult'],0))        
        self.debug(u'总盈亏：\t%s' % formatNumber(d['capital']))
        self.debug(u'最大回撤: \t%s' % formatNumber(min(d['drawdownList'])))                
        
        self.debug(u'平均每笔盈利：\t%s' %formatNumber(d['capital']/d['totalResult']))
        self.debug(u'平均每笔滑点：\t%s' %formatNumber(d['totalSlippage']/d['totalResult']))
        self.debug(u'平均每笔佣金：\t%s' %formatNumber(d['totalCommission']/d['totalResult']))
        
        self.debug(u'胜率\t\t%s%%' %formatNumber(d['winningRate']))
        self.debug(u'盈利交易平均值\t%s' %formatNumber(d['averageWinning']))
        self.debug(u'亏损交易平均值\t%s' %formatNumber(d['averageLosing']))
        self.debug(u'盈亏比：\t%s' %formatNumber(d['profitLossRatio']))

        # self.plotBacktestingResult(d)
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
            result = TradingResult(trade.price, trade.dt, self._execEndClose, self.__dtData, 
                                   trade.volume, self._ratePer10K, self._slippage, self._account.size)
            self.resultList.append(result)
            txnstr += '%+dx%.2f' % (trade.volume, trade.price)
            
        for trade in sellTrades:
            result = TradingResult(trade.price, trade.dt, self._execEndClose, self.__dtData, 
                                   -trade.volume, self._ratePer10K, self._slippage, self._account.size)
            self.resultList.append(result)
            txnstr += '%-dx%.2f' % (trade.volume, trade.price)

        # return resultList;
        return self.settleResult()
        
########################################################################
SEE BackTestApp.calculateTransactions()
class TradingResult(object):
    """每笔交易的结果
     """

   #----------------------------------------------------------------------
    def __init__(self, entryPrice, entryDt, exitPrice, 
                 exitDt, volume, rate, slippage, size):
        """Constructor"""
        self.entryPrice = entryPrice    # 开仓价格
        self.exitPrice = exitPrice      # 平仓价格
        
        self.entryDt = entryDt          # 开仓时间datetime    
        self.exitDt = exitDt            # 平仓时间
        
        self.volume = volume    # 交易数量（+/-代表方向）
        
        self.turnover   = (self.entryPrice + self.exitPrice) *size*abs(volume)   # 成交金额
        entryCommission = self.entryPrice *size*abs(volume) *rate
        if entryCommission < 2.0:
            entryCommission =2.0

        exitCommission = self.exitPrice *size*abs(volume) *rate
        if exitCommission < 2.0:
            exitCommission =2.0

        self.commission = entryCommission + exitCommission
        self.slippage   = slippage*2*size*abs(volume)                            # 滑点成本
        self.pnl        = ((self.exitPrice - self.entryPrice) * volume * size 
                            - self.commission - self.slippage)                   # 净盈亏

'''

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
def sumupDailyResults(startBalance, dayResultDict):
    '''
    @param dayResultDict - OrderedDict of account DailyResult during the date-window
    @return panda dataframe of formated-'DailyResult', summary-stat
    '''

    # step 1 convert OrderedDict of DailyResult to DataFrame
    if not dayResultDict or len(dayResultDict) <=0:
        return None, 'NULL dayResultDict'

    columns ={}
    for k in dayResultDict[0].__dict__.keys() :
        if k == 'tradeList' : # to exclude some columns
            continue
        columns[k] =[]

    for dr in dayResultDict.values():
        for k, v in dr.__dict__.items() :
            if k in columns :
                columns[k].append(v)
            
    df = pd.DataFrame.from_dict(columns)

    # step 2. append the DataFrame with new columns [balance, return, highlevel, drawdown, ddPercent]
    df['balance'] = df['netPnl'].cumsum() + startBalance
    df['return'] = (np.log(df['balance']) - np.log(df['balance'].shift(1))).fillna(0)
    df['highlevel'] = df['balance'].rolling(min_periods=1,window=len(df),center=False).max()
    df['drawdown'] = df['balance'] - df['highlevel']
    df['ddPercent'] = df['drawdown'] / df['highlevel'] * 100
    
    # step 3. calculate the overall performance summary
    startDate = df.index[0]
    endDate = df.index[-1]

    totalDays = len(df)
    profitDays = len(df[df['netPnl']>0])
    lossDays = len(df[df['netPnl']<0])
    
    endBalance   = round(df['balance'].iloc[-1],2)
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
    
    totalTradeCount = df['tcBuy'].sum() + df['tcSell'].sum()
    dailyTradeCount = totalTradeCount / totalDays
    
    totalReturn = (endBalance/startBalance - 1) * 100
    annualizedReturn = totalReturn / totalDays * 240
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
        'endBalance': endBalance,
        'maxDrawdown': maxDrawdown,
        'maxDdPercent': maxDdPercent,
        'totalNetPnl': totalNetPnl,
        'dailyNetPnl': dailyNetPnl,
        'totalCommission': totalCommission,
        'dailyCommission': dailyCommission,
        'totalSlippage': totalSlippage,
        'dailySlippage': dailySlippage,
        'totalTurnover': totalTurnover,
        'dailyTurnover': dailyTurnover,
        'totalTradeCount': totalTradeCount,
        'dailyTradeCount': dailyTradeCount,
        'totalReturn': totalReturn,
        'annualizedReturn': annualizedReturn,
        'dailyReturn': dailyReturn,
        'returnStd': returnStd,
        'sharpeRatio': sharpeRatio
    }
    
    return df, summary

#----------------------------------------------------------------------
def formatNumber(n, dec=2):
    """格式化数字到字符串"""
    rn = round(n, dec)      # 保留两位小数
    return format(rn, ',')  # 加上千分符
    

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
    
    df, d = sumupDailyResults(startBalance, account.dailyResultDict)
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
    def __init__(self, program, accountId):
        """Constructor"""

        super(AccountWrapper, self).__init__(program, account.id, account.exchange, None)

        # self._btTrader = btTrader             # refer to the BackTest engine
        self._nest  = account
        self._tradeCount = 0

        # 日线回测结果计算用
        self.__dailyResultDict = OrderedDict()
        self.__previousClose = 0
        self.__openPosition = 0

    @property
    def dailyResultDict(self):
        return self.__dailyResultDict

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def doAppInit(self): 
        return self._nest.doAppInit()

    @abstractmethod
    def doAppStep(self):
        ''' 
        this is a 'duplicated' impl of Account in order to call BackTestAcc._broker_xxxx() 
        instead of those of Account
        '''
        outgoingOrders = []
        ordersToCancel = []
        cStep =0

        with self._nest._lock:
            outgoingOrders = copy.deepcopy(self._nest._dictOutgoingOrders.values())

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

        if cStep<=0 and self._nest._recorder :
            cStep += self._nest._recorder.step()

        return cStep

    #----------------------------------------------------------------------
    # most of the methods are just forward to the self._nest
    @property
    def priceTick(self): return self._nest.priceTick
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
    def getAllPositions(self): return self._nest.getAllPositions() # returns PositionData
    def cashAmount(self): return self._nest.cashAmount() # returns (avail, total)
    def cashChange(self, dAvail=0, dTotal=0): return self._nest.cashChange(dAvail, dTotal)
    def insertData(self, dbName, collectionName, data): return self._nest.insertData(dbName, collectionName, data)
    def postEvent_Order(self, orderData): return self._nest.postEvent_Order(orderData)
    def sendOrder(self, vtSymbol, orderType, price, volume, strategy): return self._nest.sendOrder(vtSymbol, orderType, price, volume, strategy)
    def cancelOrder(self, brokerOrderId): return self._nest.cancelOrder(brokerOrderId)
    def batchCancel(self, brokerOrderIds): return self._nest.batchCancel(brokerOrderIds)
    def sendStopOrder(self, vtSymbol, orderType, price, volume, strategy): return self._nest.sendStopOrder(vtSymbol, orderType, price, volume, strategy)
    def findOrdersOfStrategy(self, strategyId, symbol=None): return self._nest.findOrdersOfStrategy(strategyId, symbol)
    
    def _broker_datetimeAsOf(self): return self._nest._broker_datetimeAsOf()
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
        self.__dailyResultDict[self._nest._datePrevClose] = self._nest._todayResult
        self.__previousClose = self._nest._todayResult.closePrice
        self.__openPosition = self._nest._todayResult.openPosition

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
    def _broker_placeOrder(self, orderData):
        """发单"""
        orderData.brokerOrderId = "$" + orderData.reqId
        orderData.status = OrderData.STATUS_SUBMITTED

        # redirectly simulate a place ok
        self._broker_onOrderPlaced(orderData)

    def _broker_cancelOrder(self, orderData) :
        # simuate a cancel by orderData
        orderData.status = OrderData.STATUS_CANCELLED
        orderData.cancelTime = self._broker_datetimeAsOf().strftime('%H:%M:%S.%f')[:3]
        self._broker_onCancelled(orderData)


    def onDayOpen(self, newDate):
        # instead that the true Account is able to sync with broker,
        # Backtest should perform cancel to restore available/frozen positions
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
            self._nest.debug('BT.onDayOpen() batchCancelled: %s' % clist)
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
     
    #----------------------------------------------------------------------
    def insertData(self, dbName, collectionName, data):
        """考虑到回测中不允许向数据库插入数据，防止实盘交易中的一些代码出错"""
        pass

    #----------------------------------------------------------------------
    def crossLimitOrder(self, symbol, dt, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume=-1):
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
                trade.dt        = self._btTrader._dtData
                # trade.tradeTime = self._btTrader._dtData.strftime('%H:%M:%S')
                # trade.dt = self.__dtData
                if buyCross:
                    trade.price = min(order.price, buyBestCrossPrice)
                else:
                    trade.price  = max(order.price, sellBestCrossPrice)

                trades.append(trade)

                order.tradedVolume = trade.volume
                order.status = OrderData.STATUS_ALLTRADED
                if order.tradedVolume < order.totalVolume :
                    order.status = OrderData.STATUS_PARTTRADED
                finishedOrders.append(order)

                self._nest.info('crossLimitOrder() crossed order[%s] to trade[%s]'% (order.desc, trade.desc))

        for o in finishedOrders:
            self._broker_onOrderDone(o)

        for t in trades:
            self._broker_onTrade(t)

    #----------------------------------------------------------------------
    def crossStopOrder(self, symbol, dt, buyCrossPrice, sellCrossPrice, buyBestCrossPrice, sellBestCrossPrice, maxCrossVolume=-1): 
        """基于最新数据撮合停止单
            A stop order is an order to buy or sell a security when its price moves past
            a particular point, ensuring a higher probability of achieving a predetermined 
            entry or exit price, limiting the investor's loss or locking in a profit. Once 
            the price crosses the predefined entry/exit point, the stop order becomes a
            market order.
        """
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
        #         trade.tradeTime = self.__dtData.strftime('%H:%M:%S')
        #         trade.dt = self.__dtData
                    
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

    def onTestEnd(self) :
        # ---------------------------
        # 结算日
        # ---------------------------
        # step 1 到最后交易日尚未平仓的交易，则以最后价格平仓
        self._btTrader.debug('onTestEnd() faking trade to flush out all positions')
        currentPositions = self.getAllPositions()
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
            trade.price  = self._btTrader.latestPrice(trade.symbol)
            trade.volume = pos.position
            trade.dt     = self._btTrader._dtData

            self._broker_onTrade(trade)
            self._btTrader.debug('onTestEnd() faked trade: %s' % trade.desc)

        # step 2 enforce a day-close
        self._btTrader.debug('onTestEnd() enforcing a day-close')
        self.onDayClose()


'''
def oldprogram() :
    # dirname(dirname(abspath(file)))
    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_AShare.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        return

    me = MainRoutine(settings)

    # me.addMarketData(mdHuobi, settings['marketdata'][0])
    me.addMarketData(mdOffline, settings['marketdata'][0])

    me.createApp(BackTestApp, settings['backtest'])
    # logger.info(u'主引擎创建成功')

    me.start()
    # logger.info(u'MainRoutine starts')

    # cta.loadSetting()
    # logger.info(u'CTA策略载入成功')
    
    # cta.initAll()
    # logger.info(u'CTA策略初始化成功')
    
    # cta.startAll()
    # logger.info(u'CTA策略启动成功')
    
    me.loop()
    me.stop()
'''

if __name__ == '__main__':
    print('-'*20)

    # oldprogram()

    # new program:
    sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_AShare.json']
    # p = Program(sys.argv)
    p = Program()
    p._heartbeatInterval =-1

    p.createApp(Account_AShare, configNode ='account', accountId='sadfasfd', ratePer10K =30)
    pdict = PerspectiveDict('AShare')
    p.addObj(pdict)
    print('listed all Objects: %s\n' % p.listByType(MetaObj))
    p.createApp(BackTestApp, p.jsettings('backtest'))

    p.start()
    p.loop()
    p.stop()

    # me.addMarketData(mdHuobi, jsettings['marketdata'][0])
    me.addMarketData(mdOffline, jsettings['marketdata'][0])

    me.createApp(BackTestApp, jsettings['backtest'])
    # logger.info(u'主引擎创建成功')

    me.start()
    # logger.info(u'MainRoutine starts')

    # cta.loadSetting()
    # logger.info(u'CTA策略载入成功')
    
    # cta.initAll()
    # logger.info(u'CTA策略初始化成功')
    
    # cta.startAll()
    # logger.info(u'CTA策略启动成功')
    
    me.loop()
    me.stop()
