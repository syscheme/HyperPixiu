# encoding: UTF-8

'''
本文件中包含的是vnApp模块的回测引擎，回测引擎的API和CTA引擎一致，
可以使用和实盘相同的代码进行回测。
'''
from __future__ import division

from datetime import datetime, timedelta
from collections import OrderedDict
from itertools import product
import multiprocessing
import copy

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

from vnpy.trader.vtGlobal import globalSetting
# from vnpy.trader.vtObject import VtTickData, KLineData
# from vnpy.trader.vtConstant import *
# from vnpy.trader.vtGateway import VtOrderData, VtTradeData

from .MainRoutine import *
from .Account import *
from .Trader import *
from .marketdata.mdBacktest import *

########################################################################
class DateRange(object):
    """
    回测Trader
    函数接口和Trader保持一样，
    """
    
    #----------------------------------------------------------------------
    def __init__(self, accountId, size=1):
        """Constructor"""

        super(DateRange, self).__init__()

        self.size= size  # 资金
        self.capital= 0  # 资金
        self.maxCapital= 0 # 资金 最高净值
        self.drawdown= 0 # 回撤
        self.totalResult= 0 # 总成交数量
        self.totalTurnover= 0 # 总成交金额（合约面值）
        self.totalCommission= 0 # 总手续费
        self.totalSlippage= 0 # 总滑点
        self.winningResult= 0 # 盈利次数
        self.losingResult= 0 # 亏损次数
        self.totalWinning= 0 # 总盈利金额		
        self.totalLosing= 0 # 总亏损金额        

        self._timeList = []           # 时间序列
        self._pnlList = []            # 每笔盈亏序列
        self._capitalList = []        # 盈亏汇总的时间序列
        self._drawdownList = []       # 回撤的时间序列
        self._dictDailyPos= {}        # symbol to [dpos]
        self.tradeTimeList= 0
        self.resultList= 0


    #------------------------------------------------
    # 参数设置相关
    #------------------------------------------------

    #------------------------------------------------
    # 结果计算相关
    #------------------------------------------------
    #----------------------------------------------------------------------
    def pushDailyPosition(self, dpos):
        symbol = dpos.symbol
        if symbol not in self._dictDailyPos:
            self._dictDailyPos[symbol] = []
        
        self._dictDailyPos[symbol].append(dpos.__dict__)
        
    def pushDailyKline(self, kline):
        pass

    def loadDays(self, dbConn, dbName, accountId, dateStart, dateEnd):

        flt = {'date':{'$gte': dateStart}}   # 数据过滤条件
        if self._dateEnd:
            flt = {'date':{'$gte': dateStart, '$lte': dateEnd }}
        
        coltn_dPos = dbConn[dbName]['dPos.' + accountId]
        cursor_dPos = coltn_dPos.find(flt).sort('datetime')

        data = None
        while True:
            try :
                if not data :
                    data = next(cursor_dPos, None)
            except :
                pass
            
            if not data: break
            
            dpos = Account.DailyPoistion()
            dpos.__dict__ = data

            self.pushDailyPosition(dpos)

    def report():

        # merge all symbols into a big datafrom with index date
        symbols = self._dictDailyPos.keys()
        df = None
        for symbol in symbols:
            if len(self._dictDailyPos[symbol]) <=0:
                continue

            sdf = pd.DataFrame(self._dictDailyPos[symbol])
            del sdf['symbol']
            for k in sdf.columns: # append the column name with symbol
                if k in ['date']:
                    continue
                sdf.rename(columns = {k : symbol +'-' +k}, inplace = True)
            if not df:
                df = sdf
            else:
                df = pd.merge(df, sdf, on='date', how='outer')
 
        list = pd.DataFrame(columes=dpos.__dict__.keys(), index='date')
        df.add

        dayfields = {}
        for k in dpos.__dict__.keys() :
            if k in ['date', 'symbol']:
                continue
            nk = symbol + '-' + k
            if nk in dayfields:
                break
            dayfields[nk] = []

        key = dpos.date + '^' + dpos.symbol
        self._dictDailyPos[key] = dpos
        if not dpos.symbol in self._symbolSet:
            self._symbolSet.add(dpos.symbol)


        dkeylist = self._dictDailyPos.keys().sort()
        if len(dkeylist) <=0:
            return

        startDate = self._dictDailyPos[dkeylist[0]].date
        endDate = self._dictDailyPos[dkeylist[:-1]].date

        currentDate = startDate
        MarketValueOfDay =0

        for dkey in dkeylist :
            dpos = self._dictDailyPos[dkey]
            if dpos.date == currentDate:
                MarketValue += dpos.recentPrice * dpos.recentPos *self.size


                pass
            
            if not startDate:
                startDate = 
            startDate = 


        # 检查是否有交易
        # {'recentPos': 20.0, 'cBuy': 1, 'recentPrice': 7.59, 'prevPos': 40.0, 'symbol': 'A601567', 'posAvail': 0.0, 'calcPos': 20.0, 'commission': 143.46, 
        # 'netPnl': -599.5, 'avgPrice': 7.694, 'prevClose': 7.67, 'calcMValue': 15180.0, 'positionPnl': -416.04, 'dailyPnl': -456.04, 'cSell': 1, 
        # 'slippage': 0.0, 'date': u'20120829', 'tradingPnl': -40.0, 'asof': [datetime.datetime(2012, 8, 29, 11, 24), 0], 'txnHist': '+20x7.67-40x7.62',
        # 'turnover': 45820.0}
        if not self.resultList:
            self.stdout(u'无交易结果')
            return {}
        
        # 然后基于每笔交易的结果，我们可以计算具体的盈亏曲线和最大回撤等        
        # capital = 0            # 资金
        # maxCapital = 0          # 资金 最高净值
        # drawdown = 0            # 回撤
        
        # totalResult = 0         # 总成交数量
        # totalTurnover = 0       # 总成交金额（合约面值）
        # totalCommission = 0     # 总手续费
        # totalSlippage = 0       # 总滑点
        
        
        # winningResult = 0       # 盈利次数
        # losingResult = 0        # 亏损次数		
        # totalWinning = 0        # 总盈利金额		
        # totalLosing = 0         # 总亏损金额        
        
        for result in self.resultList:
            capital += result.pnl
            maxCapital = max(capital, maxCapital)
            drawdown = capital - maxCapital
            
            pnlList.append(result.pnl)
            timeList.append(result.exitDt)      # 交易的时间戳使用平仓时间
            capitalList.append(capital)
            drawdownList.append(drawdown)
            
            totalResult += 1
            totalTurnover += result.turnover
            totalCommission += result.commission
            totalSlippage += result.slippage
            
            if result.pnl >= 0:
                winningResult += 1
                totalWinning += result.pnl
            else:
                losingResult += 1
                totalLosing += result.pnl
                
        # 计算盈亏相关数据
        winningRate = winningResult/totalResult*100         # 胜率
        
        averageWinning = 0                                  # 这里把数据都初始化为0
        averageLosing = 0
        profitLossRatio = 0
        
        if winningResult:
            averageWinning = totalWinning/winningResult     # 平均每笔盈利
        if losingResult:
            averageLosing = totalLosing/losingResult        # 平均每笔亏损
        if averageLosing:
            profitLossRatio = -averageWinning/averageLosing # 盈亏比

        # 返回回测结果
        d = {}
        d['capital'] = capital
        d['maxCapital'] = maxCapital
        d['drawdown'] = drawdown
        d['totalResult'] = totalResult
        d['totalTurnover'] = totalTurnover
        d['totalCommission'] = totalCommission
        d['totalSlippage'] = totalSlippage
        d['timeList'] = timeList
        d['pnlList'] = pnlList
        d['capitalList'] = capitalList
        d['drawdownList'] = drawdownList
        d['winningRate'] = winningRate
        d['averageWinning'] = averageWinning
        d['averageLosing'] = averageLosing
        d['profitLossRatio'] = profitLossRatio
        d['posList'] = self.posList
        d['tradeTimeList'] = self.tradeTimeList
        d['resultList'] = self.resultList
        
        return d
        
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
        if self._execStartClose >0 :
            originGain = (self._execEndClose - self._execStartClose)*100/self._execStartClose

        # 输出
        self.stdout('-' * 30)
        self.stdout(u'回放日期：\t%s(close:%.2f)~%s(close:%.2f): %s%%'  %(self._execStart, self._execStartClose, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.stdout(u'交易日期：\t%s(close:%.2f)~%s(close:%.2f)' % (d['timeList'][0], self._execStartClose, d['timeList'][-1], self._execEndClose))
        
        self.stdout(u'总交易次数：\t%s' % formatNumber(d['totalResult'],0))        
        self.stdout(u'总盈亏：\t%s' % formatNumber(d['capital']))
        self.stdout(u'最大回撤: \t%s' % formatNumber(min(d['drawdownList'])))                
        
        self.stdout(u'平均每笔盈利：\t%s' %formatNumber(d['capital']/d['totalResult']))
        self.stdout(u'平均每笔滑点：\t%s' %formatNumber(d['totalSlippage']/d['totalResult']))
        self.stdout(u'平均每笔佣金：\t%s' %formatNumber(d['totalCommission']/d['totalResult']))
        
        self.stdout(u'胜率\t\t%s%%' %formatNumber(d['winningRate']))
        self.stdout(u'盈利交易平均值\t%s' %formatNumber(d['averageWinning']))
        self.stdout(u'亏损交易平均值\t%s' %formatNumber(d['averageLosing']))
        self.stdout(u'盈亏比：\t%s' %formatNumber(d['profitLossRatio']))

        # self.plotBacktestingResult(d)
    
    
    #----------------------------------------------------------------------
    def clearBackTesting(self):
        """清空之前回测的结果"""

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

        self.clearResult()
        self._id = ""

    #----------------------------------------------------------------------
    def batchBacktesting(self, strategyList, d):
        """批量回测结果"""

        # self.loadHistoryData()

        for strategy in strategyList:
            if strategy ==None :
                continue

            self.clearBackTesting()
            self.initStrategy(strategy, d)
            self.runBacktesting()
            # self.showBacktestingResult()
            self.showDailyResult()
        
    #----------------------------------------------------------------------
    def runOptimization(self, strategyClass, optimizationSetting):
        """优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.stdout(u'优化设置有问题，请检查')
        
        # 遍历优化
        self.resultList =[]
        for setting in settingList:
            self.clearBackTesting()
            self.stdout('-' * 30)
            self.stdout('setting: %s' %str(setting))
            self.initStrategy(strategyClass, setting)
            self.runBacktesting()
            df = self.calculateDailyResult()
            df, d = self.calculateDailyStatistics(df)            
            try:
                targetValue = d[targetName]
            except KeyError:
                targetValue = 0
            self.resultList.append(([str(setting)], targetValue, d))
        
        # 显示结果
        self.resultList.sort(reverse=True, key=lambda result:result[1])
        self.stdout('-' * 30)
        self.stdout(u'优化结果：')
        for result in self.resultList:
            self.stdout(u'参数：%s，目标：%s' %(result[0], result[1]))    
        return self.resultList
            
    #----------------------------------------------------------------------
    def runParallelOptimization(self, strategyClass, optimizationSetting):
        """并行优化参数"""
        # 获取优化设置        
        settingList = optimizationSetting.generateSetting()
        targetName = optimizationSetting.optimizeTarget
        
        # 检查参数设置问题
        if not settingList or not targetName:
            self.stdout(u'优化设置有问题，请检查')
        
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
        self.stdout('-' * 30)
        self.stdout(u'优化结果：')
        for result in resultList:
            self.stdout(u'参数：%s，目标：%s' %(result[0], result[1]))    
            
        return resultList

    #----------------------------------------------------------------------
    def updateDailyClose(self, dt, price):
        """更新每日收盘价"""
        date = dt.date()

        if date not in self.dailyResultDict:
            self.dailyResultDict[date] = DailyResult(date, price)
        else:
            self.dailyResultDict[date].closePrice = price
            
    #----------------------------------------------------------------------
    def calculateDailyResult(self):
        """计算按日统计的交易结果"""
        self.stdout(u'计算按日统计结果')

        if self._dictTrades ==None or len(self._dictTrades) <=0:
            return None
        
        # 将成交添加到每日交易结果中
        for trade in self._dictTrades.values():
            date = trade.dt.date()
            dailyResult = self.dailyResultDict[date]
            dailyResult.addTrade(trade)
            
        # 遍历计算每日结果
        previousClose = 0
        openPosition = 0
        for dailyResult in self.dailyResultDict.values():
            dailyResult.previousClose = previousClose
            previousClose = dailyResult.closePrice
            
            dailyResult.calculatePnl(self.account, openPosition)
            openPosition = dailyResult.closePosition
            
        # 生成DataFrame
        resultDict ={}
        for k in dailyResult.__dict__.keys() :
            if k == 'tradeList' : # to exclude some columns
                continue
            resultDict[k] =[]

        for dailyResult in self.dailyResultDict.values():
            for k, v in dailyResult.__dict__.items() :
                if k in resultDict :
                    resultDict[k].append(v)
                
        resultDf = pd.DataFrame.from_dict(resultDict)
        
        # 计算衍生数据
        resultDf = resultDf.set_index('date')

        return resultDf
    
    #----------------------------------------------------------------------
    def calculateDailyStatistics(self, df):
        """计算按日统计的结果"""

        # if df ==None:
        #     self.stdout(u'计算按日统计结果')
        #     return None, None

        df['balance'] = df['netPnl'].cumsum() + self._startBalance
        df['return'] = (np.log(df['balance']) - np.log(df['balance'].shift(1))).fillna(0)
        df['highlevel'] = df['balance'].rolling(min_periods=1,window=len(df),center=False).max()
        df['drawdown'] = df['balance'] - df['highlevel']
        df['ddPercent'] = df['drawdown'] / df['highlevel'] * 100
        
        # 计算统计结果
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
        
        totalReturn = (endBalance/self.capital - 1) * 100
        annualizedReturn = totalReturn / totalDays * 240
        dailyReturn = df['return'].mean() * 100
        returnStd = df['return'].std() * 100
        
        if returnStd:
            sharpeRatio = dailyReturn / returnStd * np.sqrt(240)
        else:
            sharpeRatio = 0
            
        # 返回结果
        result = {
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
        
        return df, result
    
    #----------------------------------------------------------------------
    def showDailyResult(self, df=None, result=None):
        """显示按日统计的交易结果"""
        if df is None:
            df = self.calculateDailyResult()
            df, result = self.calculateDailyStatistics(df)

        df.to_csv(self.account._id+'.csv')
            
        originGain = 0.0
        if self._execStartClose >0 :
            originGain = (self._execEndClose - self._execStartClose)*100/self._execStartClose

        # 输出统计结果
        self.stdout('-' * 30)
        self.stdout(u'回放日期：\t%s(close:%.2f)~%s(close:%.2f): %s%%'  %(self._execStart, self._execStartClose, self._execEnd, self._execEndClose, formatNumber(originGain)))
        self.stdout(u'交易日期：\t%s(close:%.2f)~%s(close:%.2f)' % (result['startDate'], self._execStartClose, result['endDate'], self._execEndClose))
        
        self.stdout(u'交易日数：\t%s (盈利%s,亏损%s)' % (result['totalDays'], result['profitDays'], result['lossDays']))
        
        self.stdout(u'起始资金：\t%s' % formatNumber(self._startBalance))
        self.stdout(u'结束资金：\t%s' % formatNumber(result['endBalance']))
    
        self.stdout(u'总收益率：\t%s%%' % formatNumber(result['totalReturn']))
        self.stdout(u'年化收益：\t%s%%' % formatNumber(result['annualizedReturn']))
        self.stdout(u'总盈亏：\t%s' % formatNumber(result['totalNetPnl']))
        self.stdout(u'最大回撤: \t%s' % formatNumber(result['maxDrawdown']))   
        self.stdout(u'百分比最大回撤: %s%%' % formatNumber(result['maxDdPercent']))   
        
        self.stdout(u'总手续费：\t%s' % formatNumber(result['totalCommission']))
        self.stdout(u'总滑点：\t%s' % formatNumber(result['totalSlippage']))
        self.stdout(u'总成交金额：\t%s' % formatNumber(result['totalTurnover']))
        self.stdout(u'总成交笔数：\t%s' % formatNumber(result['totalTradeCount'],0))
        
        self.stdout(u'日均盈亏：\t%s' % formatNumber(result['dailyNetPnl']))
        self.stdout(u'日均手续费：\t%s' % formatNumber(result['dailyCommission']))
        self.stdout(u'日均滑点：\t%s' % formatNumber(result['dailySlippage']))
        self.stdout(u'日均成交金额：\t%s' % formatNumber(result['dailyTurnover']))
        self.stdout(u'日均成交笔数：\t%s' % formatNumber(result['dailyTradeCount']))
        
        self.stdout(u'日均收益率：\t%s%%' % formatNumber(result['dailyReturn']))
        self.stdout(u'收益标准差：\t%s%%' % formatNumber(result['returnStd']))
        self.stdout(u'Sharpe Ratio：\t%s' % formatNumber(result['sharpeRatio']))
        
        self.plotDailyResult(df)

    #----------------------------------------------------------------------
    def plotDailyResult(self, df):
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
        
        plt.savefig('DR-%s.png' % self.account._id, dpi=400, bbox_inches='tight')
        plt.show()
        plt.close()
       
        
########################################################################
class TradingResult(object):
    """每笔交易的结果"""

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


########################################################################
class DailyResult(object):
    """每日交易的结果"""

    #----------------------------------------------------------------------
    def __init__(self, date, closePrice):
        """Constructor"""

        self.date = date                # 日期
        self.closePrice = closePrice    # 当日收盘价
        self.previousClose = 0          # 昨日收盘价
        
        self.tradeList = []             # 成交列表
        self.tcBuy = 0             # 成交数量
        self.tcSell = 0             # 成交数量
        
        self.openPosition = 0           # 开盘时的持仓
        self.closePosition = 0          # 收盘时的持仓
        
        self.tradingPnl = 0             # 交易盈亏
        self.positionPnl = 0            # 持仓盈亏
        self.totalPnl = 0               # 总盈亏
        
        self.turnover = 0               # 成交量
        self.commission = 0             # 手续费
        self.slippage = 0               # 滑点
        self.netPnl = 0                 # 净盈亏
        
        self.txnHist = ""

    #----------------------------------------------------------------------
    def addTrade(self, trade):
        """添加交易"""
        self.tradeList.append(trade)

    #----------------------------------------------------------------------
    def calculatePnl(self, account, openPosition=0):
        """
        计算盈亏
        size: 合约乘数
        rate：手续费率
        slippage：滑点点数
        """
        # 持仓部分
        self.openPosition = openPosition
        self.positionPnl = round(self.openPosition * (self.closePrice - self.previousClose) * account.size, 3)
        self.closePosition = self.openPosition
        
        # 交易部分
        self.tcBuy = 0
        self.tcSell = 0
        
        for trade in self.tradeList:
            if trade.direction == OrderData.DIRECTION_LONG:
                posChange = trade.volume
                self.tcBuy += 1
            else:
                posChange = -trade.volume
                self.tcSell += 1
                
            self.txnHist += "%+dx%s" % (posChange, trade.price)

            self.tradingPnl += round(posChange * (self.closePrice - trade.price) * account.size, 2)
            self.closePosition += posChange
            turnover, commission, slippagefee = account.calcAmountOfTrade(trade.symbol, trade.price, trade.volume)
            self.turnover += turnover
            self.commission += commission
            self.slippage += slippagefee
        
        # 汇总
        self.totalPnl = round(self.tradingPnl + self.positionPnl, 2)
        self.netPnl = round(self.totalPnl - self.commission - self.slippage, 2)

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
            print u'参数起始点必须不大于终止点'
            return
        
        if step <= 0:
            print u'参数布进必须大于0'
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
def formatNumber(n, dec=2):
    """格式化数字到字符串"""
    rn = round(n, dec)      # 保留两位小数
    return format(rn, ',')  # 加上千分符
    

#----------------------------------------------------------------------
def optimize(strategyClass, setting, targetName,
             mode, startDate, initDays, endDate,
             dbName, symbol):

    """多进程优化时跑在每个进程中运行的函数"""
    account = BTAccount_AShare()
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
    
    df = account.calculateDailyResult()
    df, d = account.calculateDailyStatistics(df)
    try:
        targetValue = d[targetName]
    except KeyError:
        targetValue = 0
                    
    return (str(setting), targetValue, d)    
    
hs300s= [
        "600000","600008","600009","600010","600011","600015","600016","600018","600019","600023",
        "600025","600028","600029","600030","600031","600036","600038","600048","600050","600061",
        "600066","600068","600085","600089","600100","600104","600109","600111","600115","600118",
        "600153","600157","600170","600176","600177","600188","600196","600208","600219","600221",
        "600233","600271","600276","600297","600309","600332","600339","600340","600346","600352",
        "600362","600369","600372","600373","600376","600383","600390","600398","600406","600415",
        "600436","600438","600482","600487","600489","600498","600516","600518","600519","600522",
        "600535","600547","600549","600570","600583","600585","600588","600606","600637","600660",
        "600663","600674","600682","600688","600690","600703","600704","600705","600739","600741",
        "600795","600804","600809","600816","600820","600837","600867","600886","600887","600893",
        "600900","600909","600919","600926","600958","600959","600977","600999","601006","601009",
        "601012","601018","601021","601088","601099","601108","601111","601117","601155","601166",
        "601169","601186","601198","601211","601212","601216","601225","601228","601229","601238",
        "601288","601318","601328","601333","601336","601360","601377","601390","601398","601555",
        "601600","601601","601607","601611","601618","601628","601633","601668","601669","601688",
        "601718","601727","601766","601788","601800","601808","601818","601828","601838","601857",
        "601866","601877","601878","601881","601888","601898","601899","601901","601919","601933",
        "601939","601958","601985","601988","601989","601991","601992","601997","601998","603160",
        "603260","603288","603799","603833","603858","603993","000001","000002","000060","000063",
        "000069","000100","000157","000166","000333","000338","000402","000413","000415","000423",
        "000425","000503","000538","000540","000559","000568","000623","000625","000627","000630",
        "000651","000671","000709","000723","000725","000728","000768","000776","000783","000786",
        "000792","000826","000839","000858","000876","000895","000898","000938","000959","000961",
        "000963","000983","001965","001979","002007","002008","002024","002027","002044","002050",
        "002065","002074","002081","002085","002142","002146","002153","002202","002230","002236",
        "002241","002252","002294","002304","002310","002352","002385","002411","002415","002450",
        "002456","002460","002466","002468","002470","002475","002493","002500","002508","002555",
        "002558","002572","002594","002601","002602","002608","002624","002625","002673","002714",
        "002736","002739","002797","002925","300003","300015","300017","300024","300027","300033",
        "300059","300070","300072","300122","300124","300136","300144","300251","300408","300433"
        ]
########################################################################
from .BrokerDriver import *

class AccountWrapper(object):
    """
    回测BrokerDriver
    函数接口和BrokerDriver保持一样，
    从而实现同一套代码从回测到实盘。
    """

    #----------------------------------------------------------------------
    def __init__(self, btTrader, account):
        """Constructor"""

        super(AccountWrapper, self).__init__()

        self._btTrader = btTrader             # refer to the BackTest engine
        self._nest  = account
        self._tradeCount = 0

    #----------------------------------------------------------------------
    # most of the methods are just forward to the self._nest
    @property
    def priceTick(self): return self._nest.priceTick
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
    def onDayClose(self): return self._nest.onDayClose()
    def onTimer(self, dt): return self._nest.onTimer(dt)
    def saveDB(self): return self._nest.saveDB()
    def loadDB(self, since =None): return self._nest.loadDB(since =None)
    def calcDailyPositions(self): return self._nest.calcDailyPositions()
    def log(self, message): return self._nest.log(message)
    def stdout(self, message): return self._nest.stdout(message)
    def loadStrategy(self, setting): return self._nest.loadStrategy(setting)
    def getStrategyNames(self): return self._nest.getStrategyNames()
    def getStrategyVar(self, name): return self._nest.getStrategyVar(name)
    def getStrategyParam(self, name): return self._nest.getStrategyParam(name)
    def initStrategy(self, name): return self._nest.initStrategy(name)
    def startStrategy(self, name): return self._nest.startStrategy(name)
    def stopStrategy(self, name): return self._nest.stopStrategy(name)
    def callStrategyFunc(self, strategy, func, params=None): return self._nest.callStrategyFunc(strategy, func, params)
    def initAll(self): return self._nest.initAll()
    def startAll(self): return self._nest.startAll()
    def stop(self): return self._nest.stop()
    def stopAll(self): return self._nest.stopAll()
    def saveSetting(self): return self._nest.saveSetting()
    def updateDailyStat(self, dt, price): return self._nest.updateDailyStat(dt, price)
    def evaluateDailyStat(self, startdate, enddate): return self._nest.evaluateDailyStat(startdate, enddate)

    #------------------------------------------------
    # overwrite of Account
    #------------------------------------------------    
    def _broker_placeOrder(self, orderData):
        """发单"""
        orderData.brokerOrderId = "$" + orderData.reqId
        orderData.status = OrderData.STATUS_NOTTRADED

        # redirectly simulate a place ok
        self._broker_onOrderPlaced(orderData)

    def _broker_cancelOrder(self, brokerOrderId) :
        orderData = None

        # find out he orderData by brokerOrderId
        with self._nest._lock :
            try :
                if OrderData.STOPORDERPREFIX in brokerOrderId :
                    orderData = self._nest._dictStopOrders[brokerOrderId]
                else :
                    orderData = self._nest._dictLimitOrders[brokerOrderId]
            except KeyError:
                pass

            if not orderData :
                return

            orderData = copy.copy(orderData)

        # orderData found
        orderData.status = OrderData.STATUS_CANCELLED
        orderData.cancelTime = self._broker_datetimeAsOf().strftime('%H:%M:%S.%f')[:3]
        self._broker_onCancelled(orderData)

    def step(self) :
        outgoingOrders = []
        ordersToCancel = []

        with self._nest._lock:
            outgoingOrders = copy.deepcopy(self._nest._dictOutgoingOrders.values())
            ordersToCancel = copy.copy(self._nest._lstOrdersToCancel)
            self._nest._lstOrdersToCancel = []

        for boid in ordersToCancel:
            self._broker_cancelOrder(boid)
        for o in outgoingOrders:
            self._broker_placeOrder(o)

        if (len(ordersToCancel) + len(outgoingOrders)) >0:
            self._nest.debug('step() cancelled %d orders, placed %d orders'% (len(ordersToCancel), len(outgoingOrders)))

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
                    order.status = OrderData.STATUS_NOTTRADED

                # 判断是否会成交
                buyCross = (order.direction == OrderData.DIRECTION_LONG and 
                            order.price>=buyCrossPrice and
                            buyCrossPrice > 0)      # 国内的tick行情在涨停时askPrice1为0，此时买无法成交
                
                sellCross = (order.direction == OrderData.DIRECTION_SHORT and 
                            order.price<=sellCrossPrice and
                            sellCrossPrice > 0)    # 国内的tick行情在跌停时bidPrice1为0，此时卖无法成交
                
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
                # trade.dt = self._dtData
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
        #         trade.tradeTime = self._dtData.strftime('%H:%M:%S')
        #         trade.dt = self._dtData
                    
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
                    
        #         self.account.onTrade(trade)

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
            if symbol == Account.SYMBOL_CASH:
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
