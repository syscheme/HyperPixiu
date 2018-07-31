# encoding: UTF-8

'''
本文件中实现了行情数据记录引擎，用于汇总TICK数据，并生成K线插入数据库。

使用DR_setting.json来配置需要收集的合约，以及主力合约代码。
'''
from __future__ import division

import json
import csv
import os
import copy
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from Queue import Queue, Empty
from threading import Thread
from pymongo.errors import DuplicateKeyError

from .DataSubscriber import *
from vnpy.trader.vtEvent import *
from vnpy.trader.vtFunction import todayDate, getJsonPath
from vnpy.trader.vtObject import VtSubscribeReq, VtLogData, VtBarData, VtTickData

# DB names for recording
SETTING_DB_NAME = 'vnRec_Db'
TICK_DB_NAME   = 'recDB_Tick'
DAILY_DB_NAME  = 'recDB_Daily'
MINUTE_DB_NAME = 'recDB_1Min'

# 行情记录模块事件
EVENT_DATARECORDER_LOG = 'eRec_LOG'     # 行情记录日志更新事件

from vnpy.trader.vtConstant import EMPTY_UNICODE, EMPTY_STRING, EMPTY_FLOAT, EMPTY_INT

from .language import text

########################################################################
class DataRecorder(object):
    """数据记录引擎"""
     
    className = 'DataRecorder'
    displayName = 'DataRecorder'
    typeName = 'DataRecorder'
    appIco  = 'aaa.ico'

    settingFileName = 'DR_setting.json'
    settingFilePath = getJsonPath(settingFileName, __file__)  

    #----------------------------------------------------------------------
    def __init__(self, mainEngine, settings):
        """Constructor"""
        self._engine = mainEngine
        self._settings = settings
        self._dbNameTick = settings.dbNameTick(TICK_DB_NAME)
        self._dbName1Min = settings.dbName1Min(MINUTE_DB_NAME)
        
        # 当前日期
        self._today = todayDate()
        
        # K线合成器字典
        self._dictKLineMerge = {}
        
        # 配置字典
        self._dictTicks = OrderedDict()
        self._dict1mins = OrderedDict()
        
        # 负责执行数据库插入的单独线程相关
        self.active = False                     # 工作状态
        self.queue = Queue()                    # 队列
        self.thread = Thread(target=self.run)   # 线程
        
        # 载入设置，订阅行情
        self.subscriber()
        
        # 启动数据插入线程
        # self.start()
    
        # 注册事件监听
        # self.registerEvent()  
    
    #----------------------------------------------------------------------
    def subscriber(self):
        """加载配置"""

        # Tick记录配置
        for i in self._settings.ticks :
            try:
                symbol = i.symbol('')
                ds = i.ds('')
                if len(symbol) <=3 or len(ds)<=0:
                    continue

                self._engine.getDataSubscriber(ds).subscribe(symbol, EVENT_TICK)
                if len(self._dictTicks) <=0:
                    self._engine._eventChannel.register(EVENT_TICK, self.procecssTickEvent)

                # 保存到配置字典中
                if symbol not in self._dictTicks:
                    d = {
                        'symbol': symbol,
                        'ds': ds,
                    }

                    self._dictTicks[symbol] = d
                else:
                    d = self._dictTicks[symbol]
                    d['tick'] = True
            except Exception as e:
                print(e)
                continue

        return
        # 分钟线记录配置
        for i in self._settings.kline1min :
            try:
                symbol = i.symbol
                ds = i.ds
                if len(symbol) <=3:
                    continue

                self._engine.getDataSubscriber(ds).subscribe(symbol, EVENT_KLINE_1MIN)
                if len(self._dict1mins) <=0:
                    self._engine._eventChannel.register(EVENT_KLINE_1MIN, self.procecssKLineEvent)
                
                # 保存到配置字典中
                if symbol not in self._dict1mins:
                    d = {
                        'symbol': symbol,
                        'ds': ds,
                    }

                    self._dict1mins[symbol] = d
                else:
                    d = self._dict1mins[symbol]
                    d['bar'] = True
                        
                # 创建BarManager对象
                self._dictKLineMerge[symbol] = BarGenerator(self.onBar)
            except Exception :
                pass

    #----------------------------------------------------------------------
    def getSetting(self):
        """获取配置"""
        return self._dictSettings

    #----------------------------------------------------------------------
    def procecssTickEvent(self, event):
        """处理行情事件"""
        tick = event.dict_['data'] # this is a vtTickData
        symbol = tick.vtSymbol
        
        # 生成datetime对象
        #if not tick.datetime:
        #    tick.datetime = datetime.strptime(' '.join([tick.date, tick.time]), '%Y%m%d %H:%M:%S.%f')            

        self.onTick(tick)
        
    #----------------------------------------------------------------------
    def onTick(self, tick):
        """Tick更新"""
        vtSymbol = tick.vtSymbol
        if not vtSymbol in self._dictTicks :
            return

        tblName = vtSymbol
        if len(tick.exchange) >0:
            tblName += '_'+tick.exchange

        self.insertData(self._dbNameTick, tblName, tick)
        if not 'bar' in confSymbol or not confSymbol['bar']:
            return
            
        self.writeDrLog(text.TICK_LOGGING_MESSAGE.format(symbol=tick.vtSymbol,
                                                             time=tick.time, 
                                                             last=tick.lastPrice, 
                                                             bid=tick.bidPrice1, 
                                                             ask=tick.askPrice1))
    
    #----------------------------------------------------------------------
    def onBar(self, bar):
        """分钟线更新"""
        vtSymbol = bar.vtSymbol
        if not vtSymbol in self._dict1mins:
            return

        tblName = vtSymbol
        if len(bar.exchange) >0:
            tblName += '_'+bar.exchange
        
        self.insertData(MINUTE_DB_NAME, tblName, bar)
        
        self.writeDrLog(text.BAR_LOGGING_MESSAGE.format(symbol=bar.vtSymbol, 
                                                        time=bar.time, 
                                                        open=bar.open, 
                                                        high=bar.high, 
                                                        low=bar.low, 
                                                        close=bar.close))        

    #----------------------------------------------------------------------
    def insertData(self, dbName, collectionName, data):
        """插入数据到数据库（这里的data可以是VtTickData或者VtBarData）"""
        self.queue.put((dbName, collectionName, data.__dict__))
        
    #----------------------------------------------------------------------
    def run(self):
        """运行插入线程"""
        while self.active:
            try:
                dbName, collectionName, d = self.queue.get(block=True, timeout=1)
                
                # 这里采用MongoDB的update模式更新数据，在记录tick数据时会由于查询
                # 过于频繁，导致CPU占用和硬盘读写过高后系统卡死，因此不建议使用
                #flt = {'datetime': d['datetime']}
                #self._engine.dbUpdate(dbName, collectionName, d, flt, True)
                
                # 使用insert模式更新数据，可能存在时间戳重复的情况，需要用户自行清洗
                try:
                    self._engine.dbInsert(dbName, collectionName, d)
                except DuplicateKeyError:
                    self.writeDrLog(u'键值重复插入失败，报错信息：%s' %traceback.format_exc())
            except Empty:
                pass
            
    #----------------------------------------------------------------------
    def start(self):
        """启动"""
        self.active = True
        self.thread.start()
        
    #----------------------------------------------------------------------
    def stop(self):
        """退出"""
        if self.active:
            self.active = False
            self.thread.join()
        
    #----------------------------------------------------------------------
    def writeDrLog(self, content):
        """快速发出日志事件"""
        log = VtLogData()
        log.logContent = content
        event = Event(type_=EVENT_DATARECORDER_LOG)
        event.dict_['data'] = log
        self._engine.mainEngine._eventChannel.put(event)   
    