# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks
# start this worker:
#   celery -A distApp.sinaMaster worker -l INFO
# invocation:
# >>> import dapp.sinaMaster.tasks as m
# >>> import dapp.sinaMaster.celery.app as app
# >>> m.add.delay(5,5).get()

from __future__ import absolute_import
from celery import shared_task
from ..CeleryDefs import Worker
from Application import Program

app = Worker('HPX_sinaMaster',
    broker='redis://tc2.syscheme.com/0',
    backend='redis://tc2.syscheme.com/1',
    include=[
        'dapp.sinaMaster.task_basic',
        'dapp.sinaMaster.tasks',
        ])

app.conf.update( result_expires=3600,)

theProg = Program(name="sinaMaster", argvs=[])

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    app.start()

