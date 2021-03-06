# encoding: UTF-8

from __future__ import absolute_import, unicode_literals

# This will make sure the app is always imported

# import dapp.sinaCrawler.worker as celery
from .worker import worker as app
# __all__ =('app', 'celery',)

'''
from ..celery import app as celery_app

__all__ = ['celery_app'] # , 'basic']

TASK_PREFIX = 'task_'
TASK_PREFIX_LEN = len(TASK_PREFIX)

# celery_app.tasks.register(basic.add)
# celery_app.tasks.register(basic.mul)

from ..celery import app as celery_app

__all__ = ['celery_app',]

import os
import importlib
import traceback

# 用来保存Task类的字典
TASK_CLASS = {}

#----------------------------------------------------------------------
def __loadTaskModule(moduleName):
    """使用importlib动态载入模块"""
    try:
        module = importlib.import_module(__name__ + '.' + TASK_PREFIX + moduleName)
        
        # only the class with TASK_PREFIX in the module is TaskClass
        for k in dir(module):
            # if len(k)<= TASK_PREFIX_LEN or TASK_PREFIX != k[:TASK_PREFIX_LEN]:
            #     contvinue
            # v = module.__getattribute__(k)
            # TASK_CLASS[k[TASK_PREFIX_LEN:]] = v

            v = module.__getattribute__(k)
            try:
                isWorkerTask = getattr(v, 'Worker.task', False)
                if isWorkerTask: #     if isinstance(v, celery_app.Task) :
                    TASK_CLASS['%s.%s' % (moduleName, k)] = v
                    continue
            except Exception as ex:
                print ('issubclass(%s) failed %s: %s' % (k, ex, traceback.format_exc()))

    except BaseException as ex:
        print ('failed to import task module[%s] %s: %s' % (moduleName, ex, traceback.format_exc()))

# populate all tasks under the current vn package
path = os.path.abspath(os.path.dirname(__file__))
for root, subdirs, files in os.walk(path):
    for name in files:
        # 只有文件名TASK_PREFIX且以.py结尾的文件，才是TaskModule
        if TASK_PREFIX != name[:TASK_PREFIX_LEN] or '.py' != name[-3:] :
            continue

        # 模块名称需要模块路径前缀
        __loadTaskModule(name[TASK_PREFIX_LEN :-3])

__all__ += list(TASK_CLASS.keys())
__all__.append('TASK_CLASS')

'''
