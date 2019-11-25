# encoding: UTF-8

'''
load all agent class
'''

import os
import importlib
import traceback

# dict of agents
GYMAGENT_CLASS = {}
GYMAGENT_PREFIX = 'agent'
GYMAGENT_PREFIX_LEN = len(GYMAGENT_PREFIX)

#----------------------------------------------------------------------
def __loadAgentModule(moduleName):
    try:
        module = importlib.import_module(moduleName)
        
        # only the class with GYMAGENT_PREFIX in the module is AgentClass
        for k in dir(module):
            if len(k)<= GYMAGENT_PREFIX_LEN or GYMAGENT_PREFIX != k[:GYMAGENT_PREFIX_LEN]:
                continue
            v = module.__getattribute__(k)
            GYMAGENT_CLASS[k[GYMAGENT_PREFIX_LEN:]] = v

    except BaseException as ex:
        print ('failed to import gym agent module[%s] %s: %s' % (moduleName, ex, traceback.format_exc()))

# populate all strategies under the current vn package
path = os.path.abspath(os.path.dirname(__file__))
for root, subdirs, files in os.walk(path):
    for name in files:
        # agent prefix且以.py结尾的文件，才是策略文件
        if GYMAGENT_PREFIX != name[:GYMAGENT_PREFIX_LEN] or '.py' != name[-3:] :
            continue

        # 模块名称需要模块路径前缀
        moduleName = __name__ + '.' + name[:-3]
        __loadAgentModule(moduleName)
