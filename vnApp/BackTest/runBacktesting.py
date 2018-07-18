# encoding: UTF-8

"""
展示如何执行策略回测。
"""
from __future__ import division
from ahsApp.AShare.Strategy.Backtesting import BacktestingEngine, MINUTE_DB_NAME
import ahsApp.AShare.Strategy.strategy as tg
import os
import gc
from time import sleep

# symbols= ["601000", "601001", "601002", "601003", "601005", "601006", "601007", "601008", "601009", "601010", "601011", "601012", "601018", "601028", "601038", "601058", "601088", "601098", "601099", "601100", "601101", "601106", "601107", "601111", "601113", "601116", "601117", "601118", "601126", "601137", "601139", "601158", "601166", "601168", "601169", "601177", "601179", "601186", "601188", "601199", "601208", "601216", "601218", "601222", "601231", "601233", "601238", "601258", "601268", "601288", "601299", "601311", "601313", "601318", "601328", "601333", "601336", "601339", "601369", "601377", "601388", "601390", "601398", "601515", "601518", "601555", "601558", "601566", "601567", "601588", "601599", "601600", "601601", "601607", "601608", "601616", "601618", "601628", "601633", "601636", "601666", "601668", "601669", "601677", "601678", "601688", "601699", "601700", "601717", "601718", "601727", "601766", "601777", "601788", "601789", "601798", "601799", "601800", "601801", "601808", "601818", "601857", "601866", "601872", "601877", "601880", "601886", "601888", "601890", "601898", "601899", "601901", "601908", "601918", "601919", "601928", "601929", "601933", "601939", "601958", "601965", "601988", "601989", "601991", "601992", "601996", "601998", "601999"]
# symbols= ["601188", "601199", "601208", "601216", "601218", "601222", "601231", "601233", "601238", "601258", "601268", "601288", "601299", "601311", "601313", "601318", "601328", "601333", "601336", "601339", "601369", "601377", "601388", "601390", "601398", "601515", "601518", "601555", "601558", "601566", "601567", "601588", "601599", "601600", "601601", "601607", "601608", "601616", "601618", "601628", "601633", "601636", "601666", "601668", "601669", "601677", "601678", "601688", "601699", "601700", "601717", "601718", "601727", "601766", "601777", "601788", "601789", "601798", "601799", "601800", "601801", "601808", "601818", "601857", "601866", "601872", "601877", "601880", "601886", "601888", "601890", "601898", "601899", "601901", "601908", "601918", "601919", "601928", "601929", "601933", "601939", "601958", "601965", "601988", "601989", "601991", "601992", "601996", "601998", "601999"]
symbols= ["601000"]

#----------------------------------------------------------------------
def backTestSymbol(symbol, startDate, endDate=''):
    """将Multicharts导出的csv格式的历史数据插入到Mongo数据库中"""
    from ahsApp.AShare.Strategy.strategy.strategyKingKeltner import KkStrategy
    from ahsApp.AShare.Strategy.strategy.strategyBollChannel import BollChannelStrategy

    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置引擎的回测模式为K线
    engine.setBacktestingMode(engine.BAR_MODE)

    # 设置产品相关参数
    # engine.setSlippage(0.2)     # 股指1跳
    engine.setRate(30/10000)   # 万30
    engine.setSize(100)         # 股指合约大小 
    engine.setPriceTick(0.2)    # 股指最小价格变动
    
    # 设置回测用的数据起始日期
    engine.setStartDate(startDate)
    engine.setEndDate(endDate)
    engine.setDatabase('VnTrader_1Min_Db', symbol)
    
    # 在引擎中创建策略对象
    d = {}
    strategyList = [BollChannelStrategy] # , KkStrategy]
    engine.batchBacktesting(strategyList, d)
    # engine.initStrategy(KkStrategy, d)
    
    # # 开始跑回测
    # engine.runBacktesting()
    
    # # 显示回测结果
    # engine.showBacktestingResult()

#----------------------------------------------------------------------
def backTestSymbolByAllStategy(symbol, startDate):

    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置引擎的回测模式为K线
    engine.setBacktestingMode(engine.BAR_MODE)

    # 设置产品相关参数
    engine.setSlippage(0.2)     # 股指1跳
    engine.setRate(0.3/10000)   # 万0.3
    engine.setSize(100)         # 股指合约大小 
    engine.setPriceTick(0.2)    # 股指最小价格变动
    
    # 设置回测用的数据起始日期
    engine.setStartDate(startDate)
    engine.setDatabase(MINUTE_DB_NAME, symbol)
    
    # 在引擎中创建策略对象
    strategyList = []
    for k in tg.STRATEGY_CLASS :
        strategyList.append(tg.STRATEGY_CLASS[k])

    d = {}
    engine.batchBacktesting(strategyList, d)

#----------------------------------------------------------------------

stoppedChilds = len(symbols)

if __name__ == '__main__':

    for s in symbols :
        try:
            gc.collect()
            backTestSymbol('A'+s, '20120101') #, '20120906')
            #backTestSymbolByAllStategy('A'+s, '20110101')
        except OSError, e:
            pass

