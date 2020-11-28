# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks

from __future__ import absolute_import
from celery import Celery, shared_task, Task

TASK_PREFIX = 'tasks_'
TASK_PREFIX_LEN = len(TASK_PREFIX)

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
if __name__ == '__main__':
    # Optional configuration, see the application user guide.
    app.conf.update( result_expires=3600,)

    # app.autodiscover_tasks()
    app.start()

