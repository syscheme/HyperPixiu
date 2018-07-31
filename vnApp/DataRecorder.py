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

from vnApp.MainRoutine import *

from .MarketData import *
from vnpy.trader.vtEvent import *
from vnpy.trader.vtFunction import todayDate, getJsonPath
from vnpy.trader.vtObject import VtSubscribeReq, VtLogData, VtBarData, VtTickData

# DB names for recording
SETTING_DB_NAME = 'vnRec_Db'
TICK_DB_NAME   = 'drTick'
DAILY_DB_NAME  = 'drDaily'
MINUTE_DB_NAME = 'dr1Min'

# 行情记录模块事件
EVENT_DATARECORDER_LOG = 'eRec_LOG'     # 行情记录日志更新事件

from vnpy.trader.vtConstant import EMPTY_UNICODE, EMPTY_STRING, EMPTY_FLOAT, EMPTY_INT

from .language import text

########################################################################
class DataRecorder(BaseApplication):
    """数据记录引擎"""
     
    className = 'DataRecorder'
    displayName = 'DataRecorder'
    typeName = 'DataRecorder'
    appIco  = 'aaa.ico'

    #----------------------------------------------------------------------
    def __init__(self, mainEngine, settings):
        """Constructor"""
        super(DataRecorder, self).__init__(mainEngine, settings)

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
        self.queue = Queue()                    # 队列
        self.thread = Thread(target=self.run)   # 线程
        
        # 载入设置，订阅行情
        self.subscriber()
        
        # 启动数据插入线程
        # self.start()
    
        # 注册事件监听
        # self.registerEvent()  
    
    #----------------------------------------------------------------------
    def _subscribeMarketData(self, settingnode, event, dict, cbFunc) :
        if len(settingnode({})) <=0:
            return

        for i in settingnode :
            try:
                symbol = i.symbol('')
                ds = i.ds('')
                if len(symbol) <=3 or len(ds)<=0:
                    continue

                self._engine.getMarketData(ds).subscribe(symbol, event)
                if len(dict) <=0:
                    self.subscribeEvent(event, cbFunc)

                # 保存到配置字典中
                if symbol not in dict:
                    d = {
                        'symbol': symbol,
                        'ds': ds,
                    }

                    dict[symbol] = d
            except Exception as e:
                print(e)

    def subscriber(self):
        """加载配置"""

        # Tick记录配置
        self._subscribeMarketData(self._settings.ticks, MarketData.EVENT_TICK, self._dictTicks, self.procecssTickEvent)

        # 分钟线记录配置
        self._subscribeMarketData(self._settings.kline1min, MarketData.EVENT_KLINE_1MIN, self._dict1mins, self.procecssKLineEvent)

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
    def procecssKLineEvent(self, event):
        """处理行情事件"""
        bar = event.dict_['data'] # this is a vtBarData
        symbol = bar.vtSymbol
        
        # 生成datetime对象
        #if not tick.datetime:
        #    tick.datetime = datetime.strptime(' '.join([tick.date, tick.time]), '%Y%m%d %H:%M:%S.%f')            

        self.onBar(bar)

    #----------------------------------------------------------------------
    def onTick(self, tick):
        """Tick更新"""
        if tick.sourceType != MarketData.DATA_SRCTYPE_MARKET:
            return

        vtSymbol = tick.vtSymbol
        if not vtSymbol in self._dictTicks :
            return

        tblName = vtSymbol
        if len(tick.exchange) >0:
            tblName += '_'+tick.exchange

        self.enqueue(self._dbNameTick, tblName, tick)
        return #TODO: merge the Kline data by Tick if KLine data is not available 

        if not 'bar' in confSymbol or not confSymbol['bar']:
            return
            
        # self.logEvent(EVENT_DATARECORDER_LOG, text.TICK_LOGGING_MESSAGE.format(symbol=libIceUtil.so.3.2.1tick.vtSymbol,
        #                                                      time=tick.time, 
        #                                                      last=tick.lastPrice, 
        #                                                      bid=tick.bidPrice1, 
        #                                                      ask=tick.askPrice1))
    
    #----------------------------------------------------------------------
    def onBar(self, bar):
        """分钟线更新"""
        if bar.sourceType != MarketData.DATA_SRCTYPE_MARKET:
            return

        vtSymbol = bar.vtSymbol
        if not vtSymbol in self._dict1mins:
            return

        tblName = vtSymbol
        if len(bar.exchange) >0:
            tblName += '.'+bar.exchange
        
        self.enqueue(MINUTE_DB_NAME, tblName, bar)
        
        # self.logEvent(EVENT_DATARECORDER_LOG, text.BAR_LOGGING_MESSAGE.format(symbol=bar.vtSymbol, 
        #                                                 time=bar.time, 
        #                                                 open=bar.open, 
        #                                                 high=bar.high, 
        #                                                 low=bar.low, 
        #                                                 close=bar.close))        

    #----------------------------------------------------------------------
    def enqueue(self, dbName, collectionName, data):
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
                    self.dbInsert(dbName, collectionName, d)
                except DuplicateKeyError:
                    self.logEvent(EVENT_DATARECORDER_LOG, u'键值重复插入失败，报错信息：%s' %traceback.format_exc())
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
        
