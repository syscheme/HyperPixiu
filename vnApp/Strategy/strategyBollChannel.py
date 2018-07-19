# encoding: UTF-8

"""
感谢Darwin Quant贡献的策略思路。
知乎专栏原文：https://zhuanlan.zhihu.com/p/24448511

策略逻辑：
1. 布林通道（信号）
2. CCI指标（过滤）
3. ATR指标（止损）

适合品种：螺纹钢
适合周期：15分钟

这里的策略是作者根据原文结合vn.py实现，对策略实现上做了一些修改，仅供参考。

1,观察CCI范围
当CCI从0~+100的正常范围内，由下往上突破+100时，股指或股价有可能出现强势上涨，是买入的时机；当CCI从+100之上，由上往下跌破+100，股指或股价短线有可能出现回调，是卖出的时机。当CCI从0~-100的正常范围内，由上往下跌破-100时，股指或股价有可能出现弱势下跌，是抛出的时机。当CCI从-100的下方，由下往上突破-100时，有可能出现反弹，可逢低买入。
2, CCI运用也可以用顶背离来判断短线头部的出现，用底背离来判断短线底部的到来
当股指或股价创出新高，而CCI没有同步创出新高时，顶背离出现，短线股指或股价有可能出现回挡，可逢高卖出；当股指或股价创出新低，而CCI没有同步创出新低时，底背离出现，短线股指或股价有可能出现反弹，可逢低买入。
应用技巧编辑
1、如果CCI指标一直上行突破了100的话,表示此时的股市进入了异常波动的阶段,可能伴随着较大的成交量,可以进行中短线的投资者,此时的买入信号比较明显.
2、反之如果CCI指标向下突破了-100,则代表此时的股市进入了新一轮的下跌趋势,此时可以选择不要操作,保持观望的态度面对市场.
3、如果CCI指标从上行突破100又回到100之内的正常范围,则代表股价这一阶段的上涨行情已经疲软,投资者可以在此时选择卖出.反之CCI突破-100又回到正常范围,则代表下跌趋势已经结束,观察一段时间可能有转折的信号出现,可以先少量买入.
注意CCI指标主要用来判断100到-100范围之外的行情趋势,在这之间的趋势分析应用 CCI指标没有作用和意义,可以选择KDJ指标来分析.另外CCI指标是进行短线操作的投资者比较实用的武器,可以很快帮助交易者找到准确的交易信号. [2] 


"""

from __future__ import division

from vnpy.trader.vtObject import VtBarData
from vnpy.trader.vtConstant import *
from vnApp.Strategy.Template import (AShTemplate, 
                                                     BarGenerator, 
                                                     ArrayManager)


########################################################################
class BollChannelStrategy(AShTemplate):
    """基于布林通道的交易策略"""

    className = 'BollChannel'
    author = u'用Python的交易员'

    # 策略参数
    bollWindow = 18                     # 布林通道窗口数
    bollDev = 3.4                       # 布林通道的偏差
    cciWindow = 10                      # CCI窗口数
    atrWindow = 30                      # ATR窗口数
    slMultiplier = 5.2                  # 计算止损距离的乘数
    initDays = 10                       # 初始化数据所用的天数
    fixedSize = 20                      # 每次交易的数量/手

    # 策略变量
    bollUp = 0                          # 布林通道上轨
    bollDown = 0                        # 布林通道下轨
    cciValue = 0                        # CCI指标数值
    atrValue = 0                        # ATR指标数值
    
    intraTradeHigh = 0                  # 持仓期内的最高点
    intraTradeLow = 0                   # 持仓期内的最低点
    longStop = 0                        # 多头止损
    shortStop = 0                       # 空头止损

    # 参数列表，保存了参数的名称
    paramList = ['name',
                 'className',
                 'author',
                 'vtSymbol',
                 'bollWindow',
                 'bollDev',
                 'cciWindow',
                 'atrWindow',
                 'slMultiplier',
                 'initDays',
                 'fixedSize']    

    # 变量列表，保存了变量的名称
    varList = ['inited',
               'trading',
               'pos',
               'bollUp',
               'bollDown',
               'cciValue',
               'atrValue',
               'intraTradeHigh',
               'intraTradeLow',
               'longStop',
               'shortStop']  
    
    # 同步列表，保存了需要保存到数据库的变量名称
    syncList = ['pos',
                'intraTradeHigh',
                'intraTradeLow']    

    #----------------------------------------------------------------------
    def __init__(self, account, setting):
        """Constructor"""
        super(BollChannelStrategy, self).__init__(account, setting)
        
        self.bg    = BarGenerator(self.onBar, 15, self.onXminBar)        # 创建K线合成器对象
        self.bg_L2 = BarGenerator(self.onBar, 60, self.onBar_L2)
        self.am = ArrayManager()
        
    #----------------------------------------------------------------------
    def onBar_L2(self, bar):
        """"""
        
        
    #----------------------------------------------------------------------
    def onInit(self):
        """初始化策略（必须由用户继承实现）"""
        self.logBT(u'%s策略初始化' %self.name)
        
        # 载入历史数据，并采用回放计算的方式初始化策略数值
        initData = self.loadBar(self.initDays)
        for bar in initData:
            self.onBar(bar)

        self.putEvent()

    #----------------------------------------------------------------------
    def onStart(self):
        """启动策略（必须由用户继承实现）"""
        self.logBT(u'%s策略启动' %self.name)
        self.putEvent()

    #----------------------------------------------------------------------
    def onStop(self):
        """停止策略（必须由用户继承实现）"""
        self.logBT(u'%s策略停止' %self.name)
        self.putEvent()

    #----------------------------------------------------------------------
    def onTick(self, tick):
        """收到行情TICK推送（必须由用户继承实现）""" 
        self.bg.updateTick(tick)

    #----------------------------------------------------------------------
    def onBar(self, bar):
        """收到Bar推送（必须由用户继承实现）"""
        self.bg.updateBar(bar)
    
    #----------------------------------------------------------------------
    def onXminBar(self, bar):
        """收到X分钟K线"""

        # 全撤之前发出的委托
        self.cancelAll()

        if self.account.capital <0:
            return
    
        # 保存K线数据
        am = self.am
        
        am.updateBar(bar)
        
        if not am.inited:
            return

        # 计算指标数值
        self._lastCCI = self.cciValue
        self._lastATR = self.atrValue

        if self.pos ==0:
           self.intraTradeLow = self.intraTradeHigh =0

        self.intraTradeHigh = max(bar.high, self.intraTradeHigh)
        if self.intraTradeLow ==0 :
            self.intraTradeLow = bar.low
        else:
            self.intraTradeLow = min(bar.low, self.intraTradeLow)

        toBuy=0
        toSell=0

        bollUp, bollDown = am.boll(self.bollWindow, self.bollDev)
        bollMean = (self.bollUp + self.bollDown) /2
        
        # https://baike.baidu.com/item/%E5%B8%83%E6%9E%97%E7%BA%BF%E6%8C%87%E6%A0%87
        if bollUp >self.bollUp and bollDown > self.bollDown : 
            # 当布林线的上、中、下轨线同时向上运行时，表明股价强势特征非常明显，股价短期内将继续上涨，投资者应坚决持股待涨或逢低买入
            toBuy +=1
        elif bollUp <self.bollUp and bollDown < self.bollDown :
            # 当布林线的上、中、下轨线同时向下运行时，表明股价的弱势特征非常明显，股价短期内将继续下跌，投资者应坚决持币观望或逢高卖出。
            toSell +=1

        # 开口型喇叭口形态的形成必须具备两个条件。其一，是股价要经过长时间的中低位横盘整理，整理时间越长、上下轨之间的距离越小则未来涨升的幅度越大；其二，是布林线开始开口时要有明显的大的成交量出现。
        if (bollUp -bollDown) >(self.bollUp-self.bollDown)*1.1 :
            toBuy +=1 

        # 收口型喇叭口形态的形成虽然对成交量没有要求，但它也必须具备一个条件，即股价经过前期大幅的短线拉升，拉升的幅度越大、上下轨之间的距离越大则未来下跌幅度越大。
        if (bollUp -bollDown) <(self.bollUp-self.bollDown)*0.9 :
            toBuy =0
            toSell +=1

        if bar.close <self.intraTradeHigh *0.95:
            toSell = self.pos

        (self.bollUp, self.bollDown) = (bollUp, bollDown)

        self.cciValue = am.cci(self.cciWindow)
        self.atrValue = am.atr(self.atrWindow)

        dCCI = self.cciValue - self._lastCCI
        dATR = self.atrValue - self._lastATR

        if dCCI >0 and self.cciValue > 100:
            toBuy +=1
            toSell -=1
        if dCCI <0 and self.cciValue < 100:
            toSell +=1
        if dCCI <0 and self.cciValue < 0:
            toBuy  =0
        

        # 判断是否要进行交易
        cash = self.account.getCashAvailable()
    
        posDesc ='%s/%s,ca%.2f' % (self._posAvail, self.pos, cash)
        barDesc = '%.2f,%.2f,%.2f,%.2f' % (bar.open, bar.close, bar.high, bar.low)
        measureDesc = 'boll[%.2f~%.2f] cci[%d->%d] atr:%.2f' % (self.bollDown, self.bollUp, self._lastCCI, self.cciValue, self.atrValue)

        # determine buy ability according to the available cash
        maxBuy = (bar.close*1.01) * self.fixedSize * self.account.size
        if maxBuy >0:
            maxBuy = cash /maxBuy
        else:
            maxBuy =0

        if (toBuy - toSell) >0 and maxBuy >1 :
            vol = self.fixedSize*int(min(toBuy, maxBuy))
            
            self.logBT(u'onXminBar() pos[%s] bar[%s] %s => BUY(%d)' %(posDesc, barDesc, measureDesc, vol))
            self.buy(bar.close+0.01, vol, False)

        elif self._posAvail >0 and (toSell - toBuy) >0 :
            vol = min(self._posAvail, self.fixedSize*toSell*10)
            
            self.logBT(u'onXminBar() pos[%s] bar[%s] %s => SELL(%d)' %(posDesc, barDesc, measureDesc, vol))
            self.sell(bar.close-0.01, vol, False) # abs(self.pos), False)


        # # 当前无仓位，发送Buy委托
        # if self.pos <= 0:
        #     # self.intraTradeHigh = bar.high
        #     # self.intraTradeLow  = bar.low            
            
        #     if dATR >0 and self.cciValue > 100:
        #         self.logBT(u'onXminBar() pos[%s] bar[%s] %s => issuing buy' %(self.pos, barDesc, measureDesc))
        #         # self.buy(self.bollUp, self.fixedSize, False)
        #         self.buy(bar.close+0.01, self.fixedSize, False)
            
        # else : # 持有仓位
        #     self.intraTradeHigh = max(self.intraTradeHigh, bar.high)
        #     self.intraTradeLow  = bar.low
        #     self.longStop       = self.intraTradeHigh - self.atrValue * self.slMultiplier
                
        #     if dATR <0 and self.cciValue < 100 and self._lastCCI >100:
        #         self.logBT(u'onXminBar() pos[%s] bar[%s] %s => issuing sell' %(self.pos, barDesc, measureDesc))
        #         # self.sell(self.longStop, abs(self.pos), False)
        #         self.sell(bar.close-0.01, abs(self.pos), False)
    
        # 同步数据到数据库
        self.saveSyncData()        
    
        # 发出状态更新事件
        self.putEvent()        

    #----------------------------------------------------------------------
    def onDayOpen(self, date):
        """收到交易日开始推送"""

    #----------------------------------------------------------------------
    def onOrder(self, order):
        """收到委托变化推送（必须由用户继承实现）"""
        if order.status == STATUS_ALLTRADED :
            # self.totalCash is comfirmed reduced
            pass
        elif order.status == STATUS_PARTTRADED : # order.tradedVolume < order.totalVolume :
            # partially traded
            pass
        elif order.status == STATUS_CANCELLED :
            # TODO: cash on hold goes to available
            pass

    #----------------------------------------------------------------------
    def onTrade(self, trade):
        # 发出状态更新事件
        # TODO update self.pos
        if trade.direction == DIRECTION_SHORT:
            self._posAvail -= trade.volume

        self.putEvent()

    #----------------------------------------------------------------------
    def onStopOrder(self, so):
        """停止单推送"""
        pass
    
