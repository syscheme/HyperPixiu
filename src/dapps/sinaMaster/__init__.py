# encoding: UTF-8

from __future__ import absolute_import, unicode_literals
# This will make sure the app is always imported

# import dapp.sinaCrawler.worker as celery
from .worker import worker as app
