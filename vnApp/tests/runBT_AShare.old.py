# encoding: UTF-8

"""
展示如何执行策略回测。
cd ${workspaceFolder} ; env "PYTHONPATH=${workspaceFolder}:${workspaceFolder}/kits/vnpy" "PYTHONIOENCODING=UTF-8" "PYTHONUNBUFFERED=1" /usr/bin/python2.7 vnApp/BackTest/runBacktesting.py
"""
from __future__ import division

import vnApp.BackTest as bt
import vnApp.strategies as tg
import vnApp.Account as acnt

import os
import jsoncfg # pip install json-cfg
import gc
from time import sleep

# symbols= ["601000", "601001", "601002", "601003", "601005", "601006", "601007", "601008", "601009", "601010", "601011", "601012", "601018", "601028", "601038", "601058", "601088", "601098", "601099", "601100", "601101", "601106", "601107", "601111", "601113", "601116", "601117", "601118", "601126", "601137", "601139", "601158", "601166", "601168", "601169", "601177", "601179", "601186", "601188", "601199", "601208", "601216", "601218", "601222", "601231", "601233", "601238", "601258", "601268", "601288", "601299", "601311", "601313", "601318", "601328", "601333", "601336", "601339", "601369", "601377", "601388", "601390", "601398", "601515", "601518", "601555", "601558", "601566", "601567", "601588", "601599", "601600", "601601", "601607", "601608", "601616", "601618", "601628", "601633", "601636", "601666", "601668", "601669", "601677", "601678", "601688", "601699", "601700", "601717", "601718", "601727", "601766", "601777", "601788", "601789", "601798", "601799", "601800", "601801", "601808", "601818", "601857", "601866", "601872", "601877", "601880", "601886", "601888", "601890", "601898", "601899", "601901", "601908", "601918", "601919", "601928", "601929", "601933", "601939", "601958", "601965", "601988", "601989", "601991", "601992", "601996", "601998", "601999"]
# symbols= ["601188", "601199", "601208", "601216", "601218", "601222", "601231", "601233", "601238", "601258", "601268", "601288", "601299", "601311", "601313", "601318", "601328", "601333", "601336", "601339", "601369", "601377", "601388", "601390", "601398", "601515", "601518", "601555", "601558", "601566", "601567", "601588", "601599", "601600", "601601", "601607", "601608", "601616", "601618", "601628", "601633", "601636", "601666", "601668", "601669", "601677", "601678", "601688", "601699", "601700", "601717", "601718", "601727", "601766", "601777", "601788", "601789", "601798", "601799", "601800", "601801", "601808", "601818", "601857", "601866", "601872", "601877", "601880", "601886", "601888", "601890", "601898", "601899", "601901", "601908", "601918", "601919", "601928", "601929", "601933", "601939", "601958", "601965", "601988", "601989", "601991", "601992", "601996", "601998", "601999"]
symbols= ["601000"]

#---------------------------------------------------------------------
def backTestSymbol(symbol, startDate, endDate=''):
    """将Multicharts导出的csv格式的历史数据插入到Mongo数据库中"""
    # from vnApp.strategies.stgKingKeltner import stgKingKeltner
    from vnApp.strategies.stgBBand import stgBBand

    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_AShare.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        return

    # 创建回测引擎
    engine = bt.BackTest(acnt.Account_AShare, settings)
    
    # # 设置引擎的回测模式为K线
    # engine.setBacktestingMode(engine.BAR_MODE)

    # # 设置产品相关参数
    # # account.setSlippage(0.2)     # 股指1跳
    # engine.setRate(30/10000)   # 万30
    # engine.setSize(100)         # 股指合约大小 
    # engine.setPriceTick(0.2)    # 股指最小价格变动
    
    # # 设置回测用的数据起始日期
    # engine.setStartDate(startDate)
    # engine.setEndDate(endDate)
    # engine.setDatabase('VnTrader_1Min_Db', symbol)
    
    # 在引擎中创建策略对象
    d = {}
    strategyList = [stgBBand] # , stgKingKeltner]
    engine.batchBacktesting(strategyList, d)
    # engine.initStrategy(KkStrategy, d)
    
    # # 开始跑回测
    # engine.runBacktesting()
    
    # # 显示回测结果
    # engine.showBacktestingResult()

# #----------------------------------------------------------------------
# def backTestSymbolByAllStategy(symbol, startDate):

#     # 创建回测引擎
#     account = BTAccount_AShare()
    
#     # 设置引擎的回测模式为K线
#     account.setBacktestingMode(account.BAR_MODE)

#     # 设置产品相关参数
#     account.setSlippage(0.2)     # 股指1跳
#     account.setRate(0.3/10000)   # 万0.3
#     account.setSize(100)         # 股指合约大小 
#     account.setPriceTick(0.2)    # 股指最小价格变动
    
#     # 设置回测用的数据起始日期
#     account.setStartDate(startDate)
#     account.setDatabase(MINUTE_DB_NAME, symbol)
    
#     # 在引擎中创建策略对象
#     strategyList = []
#     for k in tg.STRATEGY_CLASS :
#         strategyList.append(tg.STRATEGY_CLASS[k])

#     d = {}
#     account.batchBacktesting(strategyList, d)

#----------------------------------------------------------------------

stoppedChilds = len(symbols)

if __name__ == '__main__':

    for s in symbols :
        try:
            gc.collect()
            backTestSymbol('A'+s, '2012-01-01','2012-09-06')
#            backTestSymbolByAllStategy('A'+s, '2011-01-01')
        except OSError, e:
            pass

