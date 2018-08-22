# encoding: UTF-8

'''
本文件中实现了行情数据记录引擎，用于汇总TICK数据，并生成K线插入数据库。

使用DR_setting.json来配置需要收集的合约，以及主力合约代码。
'''
from __future__ import division

from .MainRoutine import *
from .MarketData import *
from .language import text

import json
import csv
import os
import copy
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta
from Queue import Queue, Empty
from pymongo.errors import DuplicateKeyError

########################################################################
class DataRecorder(BaseApplication):
    """数据记录引擎"""
     
    className = 'DataRecorder'
    displayName = 'DataRecorder'
    typeName = 'DataRecorder'
    appIco  = 'aaa.ico'

    #----------------------------------------------------------------------
    def __init__(self, mainRoutine, settings):
        """Constructor"""
        super(DataRecorder, self).__init__(mainRoutine, settings)

        self._dbNamePrefix = settings.dbNamePrefix('dr')
        self._dbNameTick = settings.dbNameTick(TICK_DB_NAME)
        self._dbName1Min = settings.dbName1Min(MINUTE_DB_NAME)
        
        # 配置字典
        self._dictDR = OrderedDict()


        self._dictTicks = OrderedDict()
        self._dict1mins = OrderedDict()
        # K线合成器字典
        self._dictKLineMerge = {}
        
        # 负责执行数据库插入的单独线程相关
        self.queue = Queue()                    # 队列
        # self.thread = Thread(target=self.run)   # 线程

    #----------------------------------------------------------------------
    def _subscribeMarketData(self, settingnode, eventType): # , dict, cbFunc) :
        if len(settingnode({})) <=0:
            return

        mdEvent = eventType[len(EVENT_NAME_PREFIX):]
        if not mdEvent in self._dictDR.keys() :
            self._dictDR[mdEvent] = OrderedDict()
        dict = self._dictDR[mdEvent]

        for i in settingnode :
            try:
                symbol = i.symbol('')
                ds = i.ds('')
                if len(symbol) <=3 or len(ds)<=0 or symbol in dict:
                    continue

                self._engine.getMarketData(ds).subscribe(symbol, eventType)
                if len(dict) <=0:
                    self.subscribeEvent(eventType, self.onMarketEvent)

                if symbol in dict:
                    continue

                # 保存到配置字典中
                d = {
                        'symbol': symbol,
                        'ds': ds,
                    }

                dict[symbol] = d
            except Exception as e:
                self.logexception(e)

    def subscribe(self):
        """加载配置"""

        # Tick记录配置
        self._subscribeMarketData(self._settings.ticks, MarketData.EVENT_TICK)

        # 分钟线记录配置
        self._subscribeMarketData(self._settings.kline1min, MarketData.EVENT_KLINE_1MIN)

    #----------------------------------------------------------------------
    def onMarketEvent(self, event):
        """处理行情事件"""
        eventType = event.type_

        if  EVENT_NAME_PREFIX != eventType[:len(EVENT_NAME_PREFIX)] :
            return

        mdEvent = eventType[len(EVENT_NAME_PREFIX):]
        eData = event.dict_['data'] # this is a TickData or KLineData
        # if tick.sourceType != MarketData.DATA_SRCTYPE_REALTIME:
        if MarketData.TAG_BACKTEST in eData.exchange :
            return

        if not mdEvent in self._dictDR or not eData.symbol in self._dictDR[mdEvent]:
            return

        if not eData.datetime : # 生成datetime对象
            try :
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S')
                eData.datetime = datetime.strptime(' '.join([eData.date, eData.time]), '%Y%m%d %H:%M:%S.%f')
            except:
                pass

        self.debug('On%s: %s' % (mdEvent, eData.desc))

        # collectionName = eData.symbol
        # if len(eData.exchange) >0:
        #     collectionName += '.'+ eData.exchange
        # else collectionName += '.'+ self._dictDR[mdEvent][eData.symbol]['ds']
        collectionName = '%s.%s' % (eData.symbol, self._dictDR[mdEvent][eData.symbol]['ds'])

        self.queue.put((self._dbNamePrefix + mdEvent, collectionName, eData.__dict__))

    #----------------------------------------------------------------------
    # impl of BaseApplication
    #----------------------------------------------------------------------
    @abstractmethod
    def init(self): # return True if succ
        return super(DataRecorder, self).init()

    @abstractmethod
    def start(self):
        # 载入设置，订阅行情
        self.debug('start() subcribing')
        self.subscribe()

        for e in self._dictDR.keys() :
            dict = self._dictDR[e]
            for symbol in dict.keys() :
                collName = '%s.%s' %(symbol, dict[symbol]['ds'])
                self.dbEnsureIndex(collName, [('date', ASCENDING), ('time', ASCENDING)], True, self._dbNamePrefix +e)

    @abstractmethod
    def step(self):
        try:
            dbName, collectionName, d = self.queue.get(block=True, timeout=1)
            
            # 这里采用MongoDB的update模式更新数据，在记录tick数据时会由于查询
            # 过于频繁，导致CPU占用和硬盘读写过高后系统卡死，因此不建议使用
            #flt = {'datetime': d['datetime']}
            #self._engine.dbUpdate(dbName, collectionName, d, flt, True)
            
            # 使用insert模式更新数据，可能存在时间戳重复的情况，需要用户自行清洗
            try:
                self.dbInsert(collectionName, d, dbName)
                self.debug('DB %s[%s] inserted: %s' % (dbName, collectionName, d))
            except DuplicateKeyError:
                self.error('键值重复插入失败：%s' %traceback.format_exc())
        except Empty:
            pass
    
