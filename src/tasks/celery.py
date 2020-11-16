# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks

from __future__ import absolute_import
from celery import Celery

class Worker(Celery) :
    def gen_task_name(self, name, module):
        if module.endswith('.tasks'):
            module = module[:-6]
        
        return super(Worker, self).gen_task_name(name, module)

    # # override the decorator task
    # def task(self, func):
    #     t = super(Worker, self).task(func)
    #     setattr(t, '$Worker_task', True)
    #     return t
            
app = Worker('HPXWorker',
    broker='redis://tc2.syscheme.com:6379/0',
    backend='redis://tc2.syscheme.com:6379/1', 
    include=[
        'tasks.task_basic',
        # 'tasks.sina.task_basic',
        ])

@app.task
def hello():
    return 'hello world'

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.
    app.conf.update( result_expires=3600,)

    # app.autodiscover_tasks()
    app.start()

