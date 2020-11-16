# encoding: UTF-8
# DO NOT rename this to other than celery.py

from __future__ import absolute_import
from celery import Celery

class Worker(Celery) :
    def gen_task_name(self, name, module):
        if module.endswith('.tasks'):
            module = module[:-6]
        
        return super(Worker, self).gen_task_name(name, module)

    # override the decorator task
    def task(self, func):
        t = super(Worker, self).task(func)
        setattr(t, 'Worker.task', True)
        return t
            
app = Worker('HPXWorker',
    broker='redis://',
    backend='redis://') # , include=['proj.tasks'])

# Optional configuration, see the application user guide.
app.conf.update( result_expires=3600,)

#----------------------------------------------------------------------
if __name__ == '__main__':
    app.start()

