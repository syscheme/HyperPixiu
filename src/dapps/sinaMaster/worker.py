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

from celery.schedules import crontab
APP_NAME = '.'.join(__name__.split('.')[:-1])

taskMods = populateTaskModules(os.path.dirname(__file__), APP_NAME)

# the master also do the Crawler works on the local machine
taskCrawler = populateTaskModules(os.path.dirname(__file__) + "/../sinaCrawler", 'dapps.sinaCrawler')

if not APP_NAME or len(APP_NAME) <=0:
    APP_NAME = os.path.abspath(os.path.dirname(__file__)).split('/')[-1]

worker, thePROG = createWorkerProgram(APP_NAME, taskMods) #  + taskCrawler)

worker.conf.beat_schedule = {
    "checkResult-every-5min":{
        "task":"dapps.sinaMaster.Archive.schOn_Every5min",
        "schedule":crontab(minute="*/5"),
        "args":(),
        # "options":{'queue':'hipri'}
    },

    "every_TradeDayClose":{
        "task":"dapps.sinaMaster.Archive.schOn_TradeDayClose",
    #    "schedule":crontab(minute="*/5"),
        'schedule': crontab(hour=16, minute=30, day_of_week='1-5'),
        "args":(),
        # "options":{'queue':'hipri'}
    },

    # # Executes every weekday at 16:30
    # 'every-weekday-end': {
    #     'task': 'listSymbols',
    #     'schedule': crontab(hour=16, minute=30, day_of_week='1-5'),
    #     # 'args': (16, 16),
    # },
    # "add-every-minute":{
    #     "task":"dapps.sinaMaster.add",
    #     "schedule":crontab(minute="*/1"),
    #     "args":(3, 4),
    #     # "options":{'queue':'hipri'}
    # },
}

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    worker.start()

