# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks

from __future__ import absolute_import
from celery import shared_task
from ..CeleryDefs import Worker

app = Worker('HPXWorker',
    broker='redis://tc2.syscheme.com/0',
    backend='redis://tc2.syscheme.com/1',
    include=[
        'dist.sinaMaster.task_basic',
        'dist.sinaMaster.tasks',
        ])

app.conf.update( result_expires=3600,)

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    app.start()

