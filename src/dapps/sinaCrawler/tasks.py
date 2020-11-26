# encoding: UTF-8

from __future__ import absolute_import, unicode_literals

from celery import shared_task

@shared_task # (name='sina.basic.add')
def add(x, y):
    return x + y

@shared_task #(name='sina.basic.mul')
def mul(x, y):
    return x * y

@shared_task # (name='sina.basic.xsum')
def xsum(numbers):
    return sum(numbers)
