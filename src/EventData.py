# encoding: UTF-8

# 系统模块
from queue import Queue, Empty
from threading import Thread
from time import sleep
from collections import defaultdict
from datetime import datetime
import traceback
from abc import ABCMeta, abstractmethod

__dtEpoch = datetime.utcfromtimestamp(0)
def datetime2float(dt):
    total_seconds =  (dt - __dtEpoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

EVENT_NAME_PREFIX = 'ev'    # 事件名字前缀

# 系统相关
EVENT_TIMER = EVENT_NAME_PREFIX + 'Timer'   # 计时器事件，每隔1秒发送一次
EVENT_LOG   = EVENT_NAME_PREFIX + 'Log'     # 日志事件，全局通用
EVENT_ERROR = EVENT_NAME_PREFIX + 'Error'   # 错误回报事件

########################################################################
class Event:
    """事件对象"""

    #----------------------------------------------------------------------
    def __init__(self, type_=None):
        """Constructor"""
        self.type_ = type_      # 事件类型
        self.dict_ = {}         # 字典用于保存具体的事件数据

########################################################################
class EventData(object):
    """回调函数推送数据的基础类，其他数据类继承于此"""
    EMPTY_STRING = ''
    EMPTY_UNICODE = u''
    EMPTY_INT = 0
    EMPTY_FLOAT = 0.0

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        # self.gatewayName = EventData.EMPTY_STRING  # Gateway名称        
        # self.rawData = None                     # 原始数据

    @property
    def desc(self) :
        return self.__class__.__name__


########################################################################
class ErrorData(EventData):
    """错误数据类"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(ErrorData, self).__init__()
        
        self.errorID = EventData.EMPTY_STRING             # 错误代码
        self.errorMsg = EventData.EMPTY_UNICODE           # 错误信息
        self.additionalInfo = EventData.EMPTY_UNICODE     # 补充信息
        
        self.errorTime = time.strftime('%X', time.localtime())    # 错误生成时间


########################################################################
class LogData(EventData):
    """日志数据类"""
    EVENT_TAG = 'eLog'                      # 日志事件，全局通用

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(LogData, self).__init__()
        
        self.logTime = time.strftime('%X', time.localtime())    # 日志生成时间
        self.logContent = EventData.EMPTY_UNICODE                         # 日志信息
        self.logLevel = LOGLEVEL_INFO                                    # 日志级别

