# encoding: UTF-8

'''
动态载入所有的策略类
'''

import os
import importlib
import traceback

# 用来保存策略类的字典
STRATEGY_CLASS = {}
STRATEGY_PREFFIX = 'stg'
STRATEGY_PREFFIX_LEN = len(STRATEGY_PREFFIX)

#----------------------------------------------------------------------
def __loadStrategyModule(moduleName):
    """使用importlib动态载入模块"""
    try:
        module = importlib.import_module(moduleName)
        
        # only the class with STRATEGY_PREFFIX in the module is StrategyClass
        for k in dir(module):
            if len(k)<= STRATEGY_PREFFIX_LEN or STRATEGY_PREFFIX != k[:STRATEGY_PREFFIX_LEN]:
                continue
            v = module.__getattribute__(k)
            STRATEGY_CLASS[k[STRATEGY_PREFFIX_LEN:]] = v

    except BaseException as ex:
        print ('failed to import strategy module[%s] %s: %s' % (moduleName, ex, traceback.format_exc()))

# populate all strategies under the current vn package
path = os.path.abspath(os.path.dirname(__file__))
for root, subdirs, files in os.walk(path):
    for name in files:
        # 只有文件名strategy preffix且以.py结尾的文件，才是策略文件
        if STRATEGY_PREFFIX != name[:STRATEGY_PREFFIX_LEN] or '.py' != name[-3:] :
            continue

        # 模块名称需要模块路径前缀
        moduleName = __name__ + '.' + name[:-3]
        __loadStrategyModule(moduleName)
