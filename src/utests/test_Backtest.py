# encoding: UTF-8

from __future__ import print_function

import unittest

from Application import Program
import HistoryData as hist
import Perspective as psp
import MarketData as md
# from vnApp.marketdata.mdBacktest import mdBacktest
# from vnApp.DataRecorder import *
# from vnApp.EventChannel import EventChannel
from BackTest import *

import os

#----------------------------------------------------------------------
def processErrorEvent(event):
    """
    处理错误事件
    错误信息在每次登陆后，会将当日所有已产生的均推送一遍，所以不适合写入日志
    """
    error = event.data
    print(u'错误代码：%s，错误信息：%s' %(error.errorID, error.errorMsg))

#----------------------------------------------------------------------
def runChildProcess():
    """子进程运行函数"""
    print('-'*20)

    # dirname(dirname(abspath(file)))
    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/BT_AShare.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        return

    me = Program(settings)

    # me.addMarketData(mdHuobi, settings['marketdata'][0])
    me.addMarketData(mdOffline, settings['marketdata'][0])

    me.createApp(BackTestApp, settings['backtest'])
    # logger.info(u'主引擎创建成功')

    me.start()
    # logger.info(u'Program starts')

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


########################################################################
class TestBacktest(unittest.TestCase):

    def _test_AShare(self):
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        for i in hpb :
            print('Row: %s\n' % i.desc)

    def _test_PerspectiveGenerator(self):
        ps = psp.Perspective('AShare', '000001')
        pg = psp.PerspectiveGenerator(ps)
        hpb = hist.CsvPlayback(symbol='000001', folder='/mnt/e/AShareSample/000001', fields='date,time,open,high,low,close,volume,ammount')
        pg.adaptReader(hpb, md.EVENT_KLINE_1MIN)

        for i in pg :
            print('Psp: %s\n' % i.desc)

    def test_AShareBT(self):
        print('-'*20)

        prog = Program(progName=os.path.basename(__file__)[0:-3], 
                    setting_filename=None) #, setting_filename=os.path.dirname(os.path.abspath(__file__)) + '/../../conf/BT_AShare.json')
        prog._heartbeatInterval =-1

        # prog.addMarketData(mdOffline, settings['marketdata'][0])
        
        prog.createApp(BackTestApp, None)

        prog.start()
        prog.loop()
        prog.stop()

########################################################################
if __name__ == '__main__':
    # runChildProcess()
    # runParentProcess()
    unittest.main()
