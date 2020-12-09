# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks
# start this worker:
#   celery -A distApp.sinaMaster worker -l INFO
# invocation:
# >>> import dapp.sinaMaster.tasks as m
# >>> import dapp.sinaMaster.celery.app as app
# >>> m.add.delay(5,5).get()

from __future__ import absolute_import, unicode_literals
from dapps.celeryCommon import populateTaskModules, createWorkerProgram
import os

APP_NAME = '.'.join(__name__.split('.')[:-1])

taskMods = populateTaskModules(os.path.dirname(__file__), APP_NAME)

if not APP_NAME or len(APP_NAME) <=0:
    APP_NAME = os.path.abspath(os.path.dirname(__file__)).split('/')[-1]

worker, thePROG = createWorkerProgram(APP_NAME, taskMods)
worker.conf.update(
    result_expires=7200, # extend the result of crawling to 2hr
    )


#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    worker.start()

