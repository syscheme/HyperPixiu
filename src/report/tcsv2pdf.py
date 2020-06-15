# encoding: UTF-8

'''
BackTest reports
'''
from __future__ import division

from HistoryData import TcsvFilter

import pandas as pd
import numpy as np
import shutil
import codecs
import math

import matplotlib as mpl # pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import warnings; warnings.simplefilter('ignore')
# import pypdfplot as pdfplt # pip install pypdfplot
mpl.use('Agg')

import os, re
import bz2

# 如果安装了seaborn则设置为白色风格
try:
    import seaborn as sns       
    sns.set_style('whitegrid')  
except ImportError:
    pass

def testTcsvToPandas():
    srcdatahome = '/mnt/e/temp/sim_offline_BAK05101013'
    # filter = TcsvFilter('/mnt/e/AShareSample/sina/test_Crawler_848_0320', 'evmdKL5m', 'SH601390')
    filter = TcsvFilter('%s/SH510050_P9334.tcsv' % srcdatahome, 'DRes', 'SH510050')
    
    # for l in filter:
    #     print('%s' %l)
    dfSample = pd.read_csv(filter, error_bad_lines=False)
    # dfSample.set_index('datetime', inplace = True)
    # del dfSample['exchange']
    # dfSample['prdDIR']= dfSample[['dirNONE', 'dirLONG','dirSHORT']].apply(lambda x: (x[1]-x[2])*0.5 +0.5, axis=1)
    print(dfSample)

from fpdf import FPDF # pip install fpdf

def testPDF():
    pdf = FPDF(orientation='P', unit='mm', format='A4') #orientation='{P|L}'
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
    pdf.set_font("Arial", size=8)
    pdf.cell(200, 30, align="L", txt="Why no automatic new line: The first item that we need to talk about is the import. Here we import the FPDF class from the fpdf package. The defaults for this class are to create the PDF in Portrait mode, use millimeters for its measurement unit and to use the A4 page size. If you wanted to be explicit, you could write the instantiation line like this:")
    pdf.add_page(orientation='L')
    for x in range(10):
        y=2
        # for y in range(10):
        pdf.cell(1*x, 10*y, align="L", txt="X:%d0,Y:%d0" % (x,y))
    pdf.add_page() # back to P
    for x in range(10):
        for y in range(10):
            pdf.cell(10*x, 10*y, align="L", txt="X:%d0,Y:%d0" % (x,y))
    
    x = np.arange(-10,20,0.1)
    y = x**2

    plt.plot(x,y,'r')
    plt.show()

    pdf.output("/mnt/e/temp/simple_demo.pdf")

if __name__ == '__main__':
    # testTcsvToPandas()
    # testPDF()

'''
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
