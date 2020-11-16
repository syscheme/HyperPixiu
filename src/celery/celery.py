# encoding: UTF-8

from __future__ import absolute_import
from celery import Celery

app = Celery('proj',
    broker='redis://',
    backend='redis://') # , include=['proj.tasks'])

# Optional configuration, see the application user guide.
app.conf.update( result_expires=3600,)

#----------------------------------------------------------------------
if __name__ == '__main__':
    app.start()

