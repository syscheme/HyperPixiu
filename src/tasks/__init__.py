# encoding: UTF-8

'''
dynamically load all tasks
'''

import os
import importlib
import traceback

# 用来保存Task类的字典
TASK_CLASS = {}
TASK_PREFIX = 'task_'
TASK_PREFIX_LEN = len(TASK_PREFIX)

#----------------------------------------------------------------------
def __loadTaskModule(moduleName):
    """使用importlib动态载入模块"""
    try:
        module = importlib.import_module(moduleName)
        
        # only the class with TASK_PREFIX in the module is TaskClass
        for k in dir(module):
            if len(k)<= TASK_PREFIX_LEN or TASK_PREFIX != k[:TASK_PREFIX_LEN]:
                continue
            v = module.__getattribute__(k)
            TASK_CLASS[k[TASK_PREFIX_LEN:]] = v

    except BaseException as ex:
        print ('failed to import task module[%s] %s: %s' % (moduleName, ex, traceback.format_exc()))

# # populate all tasks under the current vn package
# path = os.path.abspath(os.path.dirname(__file__))
# for root, subdirs, files in os.walk(path):
#     for name in files:
#         # 只有文件名TASK_PREFIX且以.py结尾的文件，才是TaskModule
#         if TASK_PREFIX != name[:TASK_PREFIX_LEN] or '.py' != name[-3:] :
#             continue

#         # 模块名称需要模块路径前缀
#         moduleName = __name__ + '.' + name[:-3]
#         __loadTaskModule(moduleName)
