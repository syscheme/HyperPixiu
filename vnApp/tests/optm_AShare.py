# encoding: UTF-8

"""
展示如何执行参数优化。
cd ${workspaceFolder} ; env "PYTHONPATH=${workspaceFolder}:${workspaceFolder}/kits/vnpy" "PYTHONIOENCODING=UTF-8" "PYTHONUNBUFFERED=1" /usr/bin/python2.7 vnApp/BackTest/runOptimization.py
"""

from __future__ import division
from __future__ import print_function


import vnApp.BackTest as bt
import vnApp.Account as acnt

if __name__ == '__main__':
    from vnApp.strategies.strategyAtrRsi import AtrRsiStrategy
    from vnApp.strategies.strategyBollChannel import BollChannelStrategy
    
    settings = acnt.loadSettings('vnApp/conf/BT_AShare.json')
    # 创建回测引擎
    engine = bt.BackTest(acnt.Account_AShare, settings)
    
    # 设置引擎的回测模式为K线
    engine.setBacktestingMode(engine.BAR_MODE)

    # 设置产品相关参数
    # account.setSlippage(0.2)     # 股指1跳
    engine.setRate(30/10000)   # 万30
    engine.setSize(100)         # 股指合约大小 
    engine.setPriceTick(0.2)    # 股指最小价格变动
    
    # 设置回测用的数据起始日期
    engine.setDatabase('vnDB_1Min', 'A601000')
    engine.setStartDate('20121001')

    optSetting = bt.OptimizationSetting()                 # 新建一个优化任务设置对象
    optSetting.setOptimizeTarget('endBalance')            # 设置优化排序的目标是策略净盈利

    # 跑优化
    runStrategy = BollChannelStrategy
    optSetting.addParameter('initDays', 10, 15, 1)
    
    # runStrategy = AtrRsiStrategy
    # setting.addParameter('atrLength', 12, 13, 1)    # 增加第一个优化参数atrLength，起始12，结束20，步进2
    # setting.addParameter('atrMa', 20, 24, 2)        # 增加第二个优化参数atrMa，起始20，结束30，步进5
    # optSetting.addParameter('rsiLength', 9, 16, 1)            # 增加一个固定数值的参数
    
    # 性能测试环境：I7-3770，主频3.4G, 8核心，内存16G，Windows 7 专业版
    # 测试时还跑着一堆其他的程序，性能仅供参考
    import time    
    start = time.time()
    
    # 运行单进程优化函数，自动输出结果，耗时：359秒
    engine.runOptimization(runStrategy, optSetting)            

    # 多进程优化，耗时：89秒
    # engine.runParallelOptimization(runStrategy, optSetting)
    
    print(u'耗时：%s' %(time.time()-start))