# encoding: UTF-8

'''
本文件中包含了CTA模块中用到的一些基础设置、类和常量等。
期货术语 https://www.douban.com/group/topic/71316036/
'''


# CTA引擎中涉及的数据类定义
from vnpy.trader.vtConstant import EMPTY_UNICODE, EMPTY_STRING, EMPTY_FLOAT, EMPTY_INT

# 常量定义
# CTA引擎中涉及到的交易方向类型
CTAORDER_BUY   = u'BUY'   # u'买开' 是指投资者对未来价格趋势看涨而采取的交易手段，买进持有看涨合约，意味着帐户资金买进合约而冻结
CTAORDER_SELL  = u'SELL'  # u'卖平' 是指投资者对未来价格趋势不看好而采取的交易手段，而将原来买进的看涨合约卖出，投资者资金帐户解冻
CTAORDER_SHORT = u'SHORT' # u'卖开' 是指投资者对未来价格趋势看跌而采取的交易手段，卖出看跌合约。卖出开仓，帐户资金冻结
CTAORDER_COVER = u'COVER' # u'买平' 是指投资者将持有的卖出合约对未来行情不再看跌而补回以前卖出合约，与原来的卖出合约对冲抵消退出市场，帐户资金解冻

# 本地停止单状态
STOPORDER_WAITING   = u'WAITING'   #u'等待中'
STOPORDER_CANCELLED = u'CANCELLED' #u'已撤销'
STOPORDER_TRIGGERED = u'TRIGGERED' #u'已触发'

# 本地停止单前缀
STOPORDERPREFIX = 'AshStopOrder.'

# 数据库名称
SETTING_DB_NAME = 'Ash_Setting_Db'
POSITION_DB_NAME = 'Ash_Position_Db'

TICK_DB_NAME   = 'Ash_Tick_Db'
DAILY_DB_NAME  = 'Ash_Daily_Db'
MINUTE_DB_NAME = 'Ash_1Min_Db'

# 引擎类型，用于区分当前策略的运行环境
ENGINETYPE_BACKTESTING = 'backtesting'  # 回测
ENGINETYPE_TRADING = 'trading'          # 实盘

# CTA模块事件
EVENT_CTA_LOG      = 'eAShLog'          # CTA相关的日志事件
EVENT_CTA_STRATEGY = 'eAShStrategy.'    # CTA策略状态变化事件


########################################################################
class StopOrder(object):
    """本地停止单"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.vtSymbol = EMPTY_STRING
        self.orderType = EMPTY_UNICODE
        self.direction = EMPTY_UNICODE
        self.offset = EMPTY_UNICODE
        self.price = EMPTY_FLOAT
        self.volume = EMPTY_INT
        
        self.strategy = None             # 下停止单的策略对象
        self.stopOrderID = EMPTY_STRING  # 停止单的本地编号 
        self.status = EMPTY_STRING       # 停止单状态