# encoding: UTF-8

from __future__ import print_function


import multiprocessing
from time import sleep
from datetime import datetime, time

from vnApp.MainRoutine import MainRoutine
from vnApp.marketdata.mdHuobi import mdHuobi
from vnApp.marketdata.mdBacktest import mdBacktest
from vnApp.DataRecorder import *
from vnApp.EventChannel import EventChannel
from vnApp.BackTest import *

from vnpy.trader.vtEvent import EVENT_LOG, EVENT_ERROR

import os
import jsoncfg # pip install json-cfg

#----------------------------------------------------------------------
def processErrorEvent(event):
    """
    处理错误事件
    错误信息在每次登陆后，会将当日所有已产生的均推送一遍，所以不适合写入日志
    """
    error = event.dict_['data']
    print(u'错误代码：%s，错误信息：%s' %(error.errorID, error.errorMsg))

#----------------------------------------------------------------------
def runChildProcess():
    """子进程运行函数"""
    print('-'*20)

    # dirname(dirname(abspath(file)))
    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_app.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        return

    # 创建日志引擎
    # logger = Logger()
    # logger.setLogLevel(logger.LEVEL_INFO)
    # logger.addConsoleHandler()
    # logger.info(u'Huobi交易子进程')
    
    # ee = EventChannel()
    # logger.info(u'事件引擎创建成功')
    
    me = MainRoutine(settings)

    # me.addMarketData(mdHuobi, settings['marketdata'][0])
    me.addMarketData(mdBacktest, settings['marketdata'][0])

    me.addApp(BackTestApp, settings['backtest'])
    # logger.info(u'主引擎创建成功')

    me.start()
    # logger.info(u'MainRoutine starts')

    # cta.loadSetting()
    # logger.info(u'CTA策略载入成功')
    
    # cta.initAll()
    # logger.info(u'CTA策略初始化成功')
    
    # cta.startAll()
    # logger.info(u'CTA策略启动成功')
    
    me.loop()
    me.stop()

#----------------------------------------------------------------------
def runParentProcess():
    """父进程运行函数"""
    # 创建日志引擎
    logger = Logger()
    logger.setLogLevel(logger.LEVEL_INFO)
    logger.addConsoleHandler()
    logger.info(u'启动行情记录守护父进程')
    
    DAY_START = time(8, 57)         # 日盘启动和停止时间
    DAY_END = time(15, 18)
    NIGHT_START = time(20, 57)      # 夜盘启动和停止时间
    NIGHT_END = time(2, 33)
    
    p = None        # 子进程句柄

    while True:
        currentTime = datetime.now().time()
        recording = False

        # 判断当前处于的时间段
        if ((currentTime >= DAY_START and currentTime <= DAY_END) or
            (currentTime >= NIGHT_START) or
            (currentTime <= NIGHT_END)):
            recording = True
            
        # 过滤周末时间段：周六全天，周五夜盘，周日日盘
        if ((datetime.today().weekday() == 6) or 
            (datetime.today().weekday() == 5 and currentTime > NIGHT_END) or 
            (datetime.today().weekday() == 0 and currentTime < DAY_START)):
            recording = False

        # 记录时间则需要启动子进程
        if recording and p is None:
            logger.info(u'启动子进程')
            p = multiprocessing.Process(target=runChildProcess)
            p.start()
            logger.info(u'子进程启动成功')

        # 非记录时间则退出子进程
        if not recording and p is not None:
            logger.info(u'关闭子进程')
            p.terminate()
            p.join()
            p = None
            logger.info(u'子进程关闭成功')

        sleep(5)


if __name__ == '__main__':
    runChildProcess()
    # runParentProcess()
