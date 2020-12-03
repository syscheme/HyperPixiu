# encoding: UTF-8
# DO NOT rename this to other than celery.py
# task routine: https://www.celerycn.io/yong-hu-zhi-nan/lu-you-ren-wu-routing-tasks
# start this worker:
#   celery -A distApp.sinaMaster worker -l INFO
# invocation:
# >>> import dapp.sinaMaster.tasks as m
# >>> import dapp.sinaMaster.celery.app as app
# >>> m.add.delay(5,5).get()

from dapps.CeleryDefs import Worker, TASK_PREFIX, TASK_PREFIX_LEN
from Application import Program
import os
from celery.schedules import crontab

TASK_MODELS=[]

APP_NAME = '.'.join(__name__.split('.')[:-1])

path = os.path.abspath(os.path.dirname(__file__))
for root, subdirs, files in os.walk(path):
    for name in files:
        # 只有文件名TASK_PREFIX且以.py结尾的文件，才是TaskModule
        if '.py' != name[-3:] :
            continue
        name = name[:-3]
        if 'tasks' != name and TASK_PREFIX != name[:TASK_PREFIX_LEN] :
            continue

        TASK_MODELS.append('%s.%s' % (APP_NAME, name) if len(APP_NAME)>0 else name)

if not APP_NAME or len(APP_NAME) <=0:
    APP_NAME = path.split('/')[-1]

worker = Worker(APP_NAME,
    broker='redis://:hpxwkr@tc2.syscheme.com:15379/0',
    backend='redis://:hpxwkr@tc2.syscheme.com:15379/1',
    include=TASK_MODELS)

worker.conf.update(
    result_expires=3600,
    )

worker.conf.beat_schedule = {
    "add-every-minute":{
        "task":"dapps.sinaMaster.add",
        "schedule":crontab(minute="*/1"),
        "args":(3, 4),
        # "options":{'queue':'hipri'}
    },

    # # Executes every weekday at 16:30
    # 'every-weekday-end': {
    #     'task': 'listSymbols',
    #     'schedule': crontab(hour=16, minute=30, day_of_week='1-5'),
    #     # 'args': (16, 16),
    # },
}

thePROG = Program(name=APP_NAME, argvs=[])
thePROG._heartbeatInterval =-1

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Optional configuration, see the application user guide.

    # app.autodiscover_tasks()
    worker.start()

