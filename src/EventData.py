# encoding: UTF-8

# 系统模块
from threading import Thread
from time import sleep
from collections import defaultdict
from datetime import datetime
import traceback
from abc import ABCMeta, abstractmethod

EVENT_NAME_PREFIX = 'ev'    # 事件名字前缀

# 系统相关
EVENT_SYS_CLOCK = EVENT_NAME_PREFIX + 'SysClk'   # all application attached to program will receive this event as heartbeat, its data is basic EventData
EVENT_ERROR     = EVENT_NAME_PREFIX + 'Error'   # 错误回报事件
#EVENT_START  = EVENT_NAME_PREFIX + 'START'   # not used
# EVENT_EXIT   = EVENT_NAME_PREFIX + 'EXIT'    # not used 程序退出
EVENT_LOG    = EVENT_NAME_PREFIX + 'Log'     # 日志事件，全局通用

DT_EPOCH = datetime.utcfromtimestamp(0)

########################################################################
# utilities
def datetime2float(dt):
    total_seconds =  (dt - DT_EPOCH).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

########################################################################
class Event:
    """事件对象"""

    #----------------------------------------------------------------------
    def __init__(self, type_=None):
        """Constructor"""
        self.__type = type_      # 事件类型
        self.dict_ = {}         # 字典用于保存具体的事件数据

    def setData(self, data=None):
        """Constructor"""
        self.dict_['data'] = data

    @property
    def type(self) : return self.__type

    @property
    def data(self) :
        return self.dict_['data'] if 'data' in self.dict_.keys() else None

    @property
    def desc(self) :
        return '%s@%s/%s' % (self.__type, self.data.asof.strftime('%Y-%m-%dT%H:%M:%S'), self.data.desc)

########################################################################
class EventData(object):
    """回调函数推送数据的基础类，其他数据类继承于此
    minimally carry a AsOf datetime 
    """
    EMPTY_STRING = ''
    EMPTY_UNICODE = u''
    EMPTY_INT = 0
    EMPTY_FLOAT = 0.0

    #the columns or data-fields that wish to be saved, their name must match the member var in the EventData
    COLUMNS = 'datetime'

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        # self.gatewayName = EventData.EMPTY_STRING  # Gateway名称        
        # self.rawData = None                     # 原始数据
        self.datetime = datetime.now()

    @property
    def desc(self) :
        return self.__class__.__name__

    @property
    def asof(self) :
        return self.datetime

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

