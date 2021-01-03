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
        "task":"dapps.sinaMaster.Archive.schChkRes_Crawlers",
        "schedule":crontab(minute="*/5"),
        "args":(),
        # "options":{'queue':'hipri'}
    },

    # "every_TradeDayClose":{
    #     "task":"dapps.sinaMaster.Archive.schDo_pitchArchiedFiles",
    #     'schedule': crontab(hour=16, minute=5, day_of_week='1-5'),
    #     "args":(),
    # },

    "every_Satday6am":{
        "task":"dapps.sinaMaster.Archive.schDo_ZipWeek",
        'schedule': crontab(hour=6, minute=5, day_of_week='6'),
        "args":(),
    },

    "every_1hrAfterTradeDayClose":{
        "task":"dapps.sinaMaster.Archive.schKickOff_DownloadToday",
        'schedule': crontab(hour=16, minute=45, day_of_week='1-5'),
        "args":(),
    },

}
'''
    TODO
    "reload_beforeDayClose":
    {
        crawlers = wkr.control.ping(timeout=2.0, queue='crawler')
        crawlers = [ list(c.keys())[0] for c in crawlers ]
        # for c in crawlers
        'schedule': crontab(hour=16, minute=15, day_of_week='1-5'),
        wkr.control.pool_restart(reload=Rrue, destination=['hpx'])

        'schedule': crontab(hour=16, minute=35, day_of_week='1-5'),
        wkr.control.pool_restart(reload=Rrue, destination=['hpx53'])

        'schedule': crontab(hour=16, minute=40, day_of_week='1-5'),
        wkr.control.pool_restart(reload=Rrue, destination=['hpx79'])

        'schedule': crontab(hour=16, minute=50, day_of_week='1-5'),
        wkr.control.pool_restart(reload=Rrue, destination=['hpx50'])
    }
'''

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    worker.start()

