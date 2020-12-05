# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks

from __future__ import absolute_import
from celery import Celery, shared_task, Task
from Application import Program

import os

TASK_PREFIX = 'tasks_'
TASK_PREFIX_LEN = len(TASK_PREFIX)
MAPPED_HOME = '/mnt/s'

#----------------------------------------------------------------------
def populateTaskModules(dirOfParent, moduleParent =''):
    taskModules = []
    if moduleParent and len(moduleParent) >0 :
        if '.' != moduleParent[-1]:
            moduleParent += '.'
    else:
        moduleParent =''

    for root, subdirs, files in os.walk(dirOfParent):
        for name in files:
            # 只有文件名TASK_PREFIX且以.py结尾的文件，才是TaskModule
            if '.py' != name[-3:] :
                continue

            name = name[:-3]
            if 'tasks' != name and TASK_PREFIX != name[:TASK_PREFIX_LEN] :
                continue

            taskModules.append('%s%s' % (moduleParent, name))

    return taskModules

#----------------------------------------------------------------------
_thePROG = None
def createWorkerProgram(appName, taskModules = []):
    worker = Worker(appName,
        broker='redis://:hpxwkr@tc2.syscheme.com:15379/0',
        backend='redis://:hpxwkr@tc2.syscheme.com:15379/1',
        include=taskModules)

    worker.conf.update(
            result_expires=3600,
            )

    global _thePROG
    if not _thePROG:
        _thePROG = Program(name=appName, argvs=[])
        _thePROG._heartbeatInterval =-1

    return worker, _thePROG

#----------------------------------------------------------------------
class RetryableError(Exception):
    def __init__(self, errCode, message):
        self.errCode = errCode
        self.message = message
        super().__init__(self.message)

class Retryable(Task):
    autoretry_for = (RetryableError,)
    retry_kwargs = {'max_retries': 5, 'countdown': 60,}
    retry_backoff = True
    
#----------------------------------------------------------------------
class Worker(Celery) :
    def gen_task_name(self, name, module):
        tokens = module.split('.')
        if 'tasks' == tokens[-1]:
            module = '.'.join(tokens[:-1])
        elif tokens[-1].startswith(TASK_PREFIX):
            module = '.'.join(tokens[:-1]) + '.%s' % tokens[-1][TASK_PREFIX_LEN:]
        
        return super(Worker, self).gen_task_name(name, module)

    # # override the decorator task
    # def task(self, func):
    #     t = super(Worker, self).task(func)
    #     setattr(t, '$Worker_task', True)
    #     return t
            
#----------------------------------------------------------------------
# app = Worker('HPXWorker',
#     broker='redis://tc2.syscheme.com/0',
#     backend='redis://tc2.syscheme.com/1',
#     include=[
#         'dist.sinaMaster.task_basic',
#         'dist.sinaMaster.tasks',
#         # 'tasks.sina.task_basic',
#         ])

# @shared_task
# def hello():
#     return 'hello world'

#----------------------------------------------------------------------
def getMappedAs(homeDir = MAPPED_HOME) :
    accLogin = None

    try :
        if not homeDir or len(homeDir) <=1:
            homeDir = os.path.join(os.environ['HOME'], 'wkspaces/hpx_publish/..')
            homeDir = os.path.realpath(homeDir)

        with open(os.path.join(homeDir, '.ssh', 'id_rsa.pub'), 'r') as fkey:
            line = fkey.readline().strip()
            accLogin = line.split(' ')[-1]
    except Exception as ex:
        pass
        
    return accLogin, homeDir
    

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.
    app.conf.update( result_expires=3600,)

    # app.autodiscover_tasks()
    app.start()

