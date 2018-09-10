# encoding: UTF-8

from __future__ import print_function


import multiprocessing
from time import sleep
from datetime import datetime, time

from vnApp.MainRoutine import MainRoutine
from vnApp.marketdata.mdHuobi import mdHuobi
from vnApp.DataRecorder import *
from vnApp.EventChannel import EventChannel

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
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/DR_huobi.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        return

    me = MainRoutine(settings)

    me.addMarketData(mdHuobi, settings['marketdata'][0])
    # recorder = me.addApp(DataRecorder, settings['datarecorder'])
    recorder = me.addApp(CsvRecorder, settings['datarecorder'])
    me.addApp(Zipper, settings['datarecorder'])
    me.info(u'主引擎创建成功')

    me.start(False)
    startDate=datetime.now() - timedelta(60)
    data = recorder.loadRecentMarketData('ethusdt', startDate)
    data = recorder.loadRecentMarketData('ethusdt', startDate, MarketData.EVENT_TICK)
    me.loop()
    # input()
    

#----------------------------------------------------------------------
def runParentProcess():
    """父进程运行函数"""
    # 创建日志引擎
    le = Logger()
    le.setLogLevel(le.LEVEL_INFO)
    le.addConsoleHandler()
    le.info(u'启动行情记录守护父进程')
    
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
            le.info(u'启动子进程')
            p = multiprocessing.Process(target=runChildProcess)
            p.start()
            le.info(u'子进程启动成功')

        # 非记录时间则退出子进程
        if not recording and p is not None:
            le.info(u'关闭子进程')
            p.terminate()
            p.join()
            p = None
            le.info(u'子进程关闭成功')

        sleep(5)


if __name__ == '__main__':
    runChildProcess()
    # runParentProcess()
