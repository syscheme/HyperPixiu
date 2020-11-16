# encoding: UTF-8
# DO NOT rename this to other than celery.py

from __future__ import absolute_import
from ..celery import Worker
from tasks.sina import task_basic as basic

app = Worker('HPXWorker',
    broker='redis://tc.syscheme.com:6379/0',
    backend='redis://tc.syscheme.com:6379/1') # , include=['proj.tasks'])

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.
    app.conf.update( result_expires=3600,)
    app.autodiscover_tasks()
    app.start()

