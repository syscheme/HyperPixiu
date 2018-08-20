# encoding: UTF-8

from vnApp.Account import *
from vnApp.Trader import *

from Queue import Queue, Empty
from datetime import datetime
from threading import Thread
from multiprocessing.dummy import Pool
import traceback
from copy import copy

import urllib
import hmac
import base64
import hashlib
import requests 
import json
import zlib
# retrieve package: sudo pip install websocket websocket-client pathlib
from websocket import create_connection, _exceptions


########################################################################
class Huobi(Account):
    """交易API"""
    # 常量定义
    HUOBI = 'huobi'
    HADAX = 'hadax'
    HUOBI_API_HOST = "api.huobi.pro"
    HADAX_API_HOST = "api.hadax.com"
    LANG = 'zh-CN'
    TIMEOUT = 5

    DEFAULT_GET_HEADERS = {
        "Content-type": "application/x-www-form-urlencoded",
        'Accept': 'application/json',
        'Accept-Language': LANG,
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'
    }

    DEFAULT_POST_HEADERS = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Accept-Language': LANG,
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'    
    }

    #----------------------------------------------------------------------
    def createSign(self, params, method, host, path, secretKey):
        """创建签名"""
        sortedParams = sorted(params.items(), key=lambda d: d[0], reverse=False)
        encodeParams = urllib.urlencode(sortedParams)
        
        payload = [method, host, path, encodeParams]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='UTF8')

        secretKey = secretKey.encode(encoding='UTF8')

        digest = hmac.new(secretKey, payload, digestmod=hashlib.sha256).digest()

        signature = base64.b64encode(digest)
        signature = signature.decode()
        return signature    

    #----------------------------------------------------------------------
    def __init__(self, trader, settings):
        """Constructor"""
        super(Huobi, self).__init__(trader, settings)

        self._mode        = Account.BROKER_API_ASYNC
        self._queRequests = Queue()        # queue of request ids
        self._dictRequests = {}            # dict from request Id to request
        
        # self.pool = None            # 线程池

        # about the proxy
        self._proxies = {}
        prx = settings.httpproxy('')
        if prx != '' :
            self._proxies['http']  = prx
            self._proxies['https'] = prx

        # about the exchange
        if self._exchange == self.HADAX:
            self._hostname = self.HADAX_API_HOST
        else:
            self._exchange = self.HUOBI
            self._hostname = self.HUOBI_API_HOST
            
        self._hosturl = 'https://%s' %self._hostname
        
    @property
    def accessKey(self) :
        return self._settings.accessKey('')

    @property
    def secretKey(self) :
        return self._settings.secretKey('')

    #----------------------------------------------------------------------
    # most of the methods are just forward to the self._nest
    # def _broker_datetimeAsOf(self): return self._nest._broker_datetimeAsOf()
    # def _broker_onOrderPlaced(self, orderData): return self._nest._broker_onOrderPlaced(orderData)
    # def _broker_onCancelled(self, orderData): return self._nest._broker_onCancelled(orderData)
    # def _broker_onOrderDone(self, orderData): return self._nest._broker_onOrderDone(orderData)
    # def _broker_onTrade(self, trade): return self._nest._broker_onTrade(trade)
    # def _broker_onGetAccountBalance(self, data, reqid): return self._nest._broker_onGetAccountBalance(data, reqid)
    # def _broker_onGetOrder(self, data, reqid): return self._nest._broker_onGetOrder(data, reqid)
    # def _broker_onGetOrders(self, data, reqid): return self._nest._broker_onGetOrders(data, reqid)
    # def _broker_onGetMatchResults(self, data, reqid): return self._nest._broker_onGetMatchResults(data, reqid)
    # def _broker_onGetMatchResult(self, data, reqid): return self._nest._broker_onGetMatchResult(data, reqid)
    # def _broker_onGetTimestamp(self, data, reqid): return self._nest._broker_onGetTimestamp(data, reqid)

    # def calcAmountOfTrade(self, symbol, price, volume): return self._nest.calcAmountOfTrade(symbol, price, volume)
    # def maxOrderVolume(self, symbol, price): return self._nest.maxOrderVolume(symbol, price)
    # def roundToPriceTick(self, price): return self._nest.roundToPriceTick(price)
    # def onStart(self): return self._nest.onStart()
    # def onDayClose(self): return self._nest.onDayClose()
    # def onTimer(self, dt): return self._nest.onTimer(dt)
    # def saveDB(self): return self._nest.saveDB()
    # def loadDB(self, since =None): return self._nest.loadDB(since =None)
    # def calcDailyPositions(self): return self._nest.calcDailyPositions()
    # def log(self, message): return self._nest.log(message)
    # def stdout(self, message): return self._nest.stdout(message)
    # def loadStrategy(self, setting): return self._nest.loadStrategy(setting)
    # def getStrategyNames(self): return self._nest.getStrategyNames()
    # def getStrategyVar(self, name): return self._nest.getStrategyVar(name)
    # def getStrategyParam(self, name): return self._nest.getStrategyParam(name)
    # def initStrategy(self, name): return self._nest.initStrategy(name)
    # def startStrategy(self, name): return self._nest.startStrategy(name)
    # def stopStrategy(self, name): return self._nest.stopStrategy(name)
    # def callStrategyFunc(self, strategy, func, params=None): return self._nest.callStrategyFunc(strategy, func, params)
    # def initAll(self): return self._nest.initAll()
    # def startAll(self): return self._nest.startAll()
    # def stop(self): return self._nest.stop()
    # def stopAll(self): return self._nest.stopAll()
    # def saveSetting(self): return self._nest.saveSetting()
    # def updateDailyStat(self, dt, price): return self._nest.updateDailyStat(dt, price)
    # def evaluateDailyStat(self, startdate, enddate): return self._nest.evaluateDailyStat(startdate, enddate)
    #----------------------------------------------------------------------
    def enqueue(self, reqId, req):
        with self._lock :
            self._dictRequests[reqId] = req
        self._queRequests.put(reqId)

    def dequeue(self, secTimeout=3):
        try :
            reqId = self._queRequests.get(timeout=secTimeout)
            with self._lock :
                if reqId in self._dictRequests.keys():
                    req = self._dictRequests[reqId]
                    req = copy(req)
                    del self._dictRequests[reqId]
                    return req
        except Empty:
            pass

        return None

    def pushRequest(self, path, reqId, httpParams, reqData, funcMethod, callbackResp):
        """添加请求"""       
        # 同步模式
        if self._mode != Account.BROKER_API_ASYNC:
            return funcMethod(path, httpParams)

        # 异步模式
        if not reqId:
            reqId = self.nextOrderReqId

        req = (reqId, path, httpParams, reqData, funcMethod, callbackResp)

        self.enqueue(reqId, req)
        return reqId

    def procRequest(self, secTimeout):
        """处理请求"""
        req = self.dequeue(secTimeout)
        if not req:
            return

        (reqId, path, httpParams, reqData, funcMethod, callbackResp) = req

        result, resp = funcMethod(path, httpParams)
        
        if result:
            if resp['status'] == 'ok':
                callbackResp(reqData, resp['data'])
            else:
                msg = u'错误代码：%s，错误信息：%s' %(resp['err-code'], resp['err-msg'])
                self.onError(msg, reqData)
        else:
            self.onError(resp, reqData)
            # 失败的请求重新放回队列，等待下次处理
            self.enqueue(reqId, req)
    
    #------------------------------------------------
    # overwrite of Account
    #------------------------------------------------    
    def _broker_placeOrder(self, orderData):
        """下单"""
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/order/orders/place'
        else:
            path = '/v1/hadax/order/orders/place'

        params = {
            'account-id': self._id,
            'amount': orderData.volume * self.size,
            'symbol': orderData.symbol,
            'type'  : orderData.direction
        }
        
        if orderData.price >0:
            params['price'] = orderData.price
        if len(orderData.source) >0:
            params['source'] = orderData.source

        return self.pushRequest(path, orderData.reqId, params, orderData, self.onResp_placeOrder)

    def onResp_placeOrder(self, req, resp):
        """下单的response"""
        orderData = req['reqData']

        #TODO fill response data into orderData

        self._broker_onOrderPlaced(orderData) # if succ, else _broker_onCancelled() ??

    def _broker_cancelOrder(self, orderData) :
        """撤单"""
        path = '/v1/order/orders/%s/submitcancel' % orderData.brokerOrderId
        
        params = {}
        return self.pushRequest(path, orderData.reqId, {}, orderData, self.onResp_cancelOrder)
        
    def onResp_cancelOrder(self, req, resp):
        """撤单的response"""
        orderData = req['reqData']

        #TODO fill response data into orderData

        self._broker_onCancelled(orderData) # if succ, else _broker_onCancelled() ??

    def step(self) :

        try :
            nextSleep = super(Huobi, self).step()
            if nextSleep <0:
                nextSleep = 0.01
        except Exception as ex:
            self.error('ThreadedHuobi::step() excepton: %s' % ex)

        try :
            self.procRequest(nextSleep)
        except Exception as ex:
            self.error('ThreadedHuobi::step() proc excepton: %s' % ex)

    def onDayOpen(self, newDate):
        return super(Huobi, self).onDayOpen(newDate)

    #----------------------------------------------------------------------
    def generateSignParams(self):
        """生成签名参数"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        d = {
            'AccessKeyId': self.accessKey,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': timestamp
        }    
        
        return d

    def apiGet(self, path, params):
        """API GET"""
        method = 'GET'
        
        params.update(self.generateSignParams())
        params['Signature'] = self.createSign(params, method, self._hostname, path, self.secretKey)
        
        url = self._hosturl + path
        #print("url=%s, param:%s" % (url, params))
        
        headers = copy(self.DEFAULT_GET_HEADERS)
        postdata = urllib.urlencode(params)
        
        try:
            response = requests.get(url, postdata, headers=headers, proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

    def apiPost(self, path, params, add_to_headers=None):
        """API POST"""
        method = 'POST'
        
        signParams = self.generateSignParams()
        signParams['Signature'] = self.createSign(signParams, method, self._hostname, path, self.secretKey)
        
        url = self._hosturl + path + '?' + urllib.urlencode(signParams)

        headers = copy(self.DEFAULT_POST_HEADERS)
        postdata = json.dumps(params)
        
        try:
            response = requests.post(url, postdata, headers=headers, proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, u'POST请求失败，返回信息：%s' %response.json()
        except Exception as e:
            return False, u'POST请求触发异常，原因：%s' %e
    
    #----------------------------------------------------------------------
    def getSymbols(self):
        """查询合约代码"""
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/common/symbols'
        else:
            path = '/v1/hadax/common/symbols'

        return self.pushRequest(path, None, {}, None, self.apiGet, self.onGetSymbols)
    
    #----------------------------------------------------------------------
    def getCurrencys(self):
        """查询支持货币"""
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/common/currencys'
        else:
            path = '/v1/hadax/common/currencys'

        return self.pushRequest(path, None, {}, None, self.apiGet, self.onGetCurrencys)
    
    #----------------------------------------------------------------------
    def getTimestamp(self):
        """查询系统时间"""
        return self.pushRequest('/v1/common/timestamp', None, {}, None, self.apiGet, self.onGetTimestamp)
    
    #----------------------------------------------------------------------
    def getAccounts(self):
        """查询账户"""
        return self.pushRequest('/v1/account/accounts', None, {}, None, self.apiGet, self.onGetAccounts)
    
    #----------------------------------------------------------------------
    def getBalance(self):
        """查询余额"""
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/account/accounts/%s/balance' % self._id
        else:
            path = '/v1/hadax/account/accounts/%s/balance' %accountid
            
        return self.pushRequest(path, None, {}, None, self.apiGet, self.onGetAccountBalance)
    
    #----------------------------------------------------------------------
    def getOrders(self, symbol, states, types=None, startDate=None, 
                  endDate=None, from_=None, direct=None, size=None):
        """查询委托"""
        path = '/v1/order/orders'
        
        params = {
            'symbol': symbol,
            'states': states
        }
        
        if types:
            params['types'] = types
        if startDate:
            params['start-date'] = startDate
        if endDate:
            params['end-date'] = endDate        
        if from_:
            params['from'] = from_
        if direct:
            params['direct'] = direct
        if size:
            params['size'] = size        
    
        return self.pushRequest(path, None, params, None, self.apiGet, self.onGetOrders)

    #----------------------------------------------------------------------
    def getOpenOrders(self, accountId=None, symbol=None, side=None, size=None):
        """查询当前帐号下未成交订单
            “account_id” 和 “symbol” 需同时指定或者二者都不指定。如果二者都不指定，返回最多500条尚未成交订单，按订单号降序排列。
        """
        path = '/v1/order/openOrders'
        
        params = { # initial with default required params
            #'account_id': accountId,
            #'symbol': symbol,
        }
        
        if symbol:
            params['symbol'] = symbol
            params['account_id'] = accountId

        if side:
            params['side'] = side

        if size:
            params['size'] = size        
    
        return self.pushRequest(path, None, params, None, self.apiGet, self.onGetOrders)
    
    #----------------------------------------------------------------------
    def getMatchResults(self, symbol, types=None, startDate=None, 
                  endDate=None, from_=None, direct=None, size=None):
        """查询委托"""
        path = '/v1/order/matchresults'

        params = {
            'symbol': symbol
        }

        if types:
            params['types'] = types
        if startDate:
            params['start-date'] = startDate
        if endDate:
            params['end-date'] = endDate        
        if from_:
            params['from'] = from_
        if direct:
            params['direct'] = direct
        if size:
            params['size'] = size        

        func = self.apiGet
        callback = self.onGetMatchResults

        return self.pushRequest(path, None, params, None, self.apiGet, self.onGetMatchResults)
    
    #----------------------------------------------------------------------
    def getOrder(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s' %orderid
        return self.pushRequest(path, None, {}, None, self.apiGet, self.onGetOrder)
    
    #----------------------------------------------------------------------
    def getMatchResult(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s/matchresults' %orderid
        return self.pushRequest(path, None, {}, None, self.apiGet, self.onGetMatchResult)
    
    #----------------------------------------------------------------------
    def batchCancel(self, orderids):
        """批量撤单"""
        path = '/v1/order/orders/batchcancel'
    
        params = {
            'order-ids': orderids
        }
    
        func = self.apiPost
        callback = self.onBatchCancel
    
        return self.pushRequest(path, None, params, None, self.apiPost, self.onPlaceOrder)
        return self.addReq(path, params, func, callback)     
        
    #----------------------------------------------------------------------
    def onError(self, msg, reqid):
        """错误回调"""
        print(msg, reqid)
        
    #----------------------------------------------------------------------
    def onGetSymbols(self, data, reqid):
        """查询代码回调"""
        #print reqid, data 
        for d in data:
            print(d)
    
    #----------------------------------------------------------------------
    def onGetCurrencys(self, data, reqid):
        """查询货币回调"""
        print (reqid, data)
    
    #----------------------------------------------------------------------
    def onGetTimestamp(self, data, reqid):
        """查询时间回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onGetAccounts(self, data, reqid):
        """查询账户回调"""
        print (reqid, data)
    
    #----------------------------------------------------------------------
    def onGetAccountBalance(self, data, reqid):
        """查询余额回调"""
        print (reqid, data)
        for d in data['data']['list']:
            print (d)
        
    #----------------------------------------------------------------------
    def onGetOrders(self, data, reqid):
        """查询委托回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onGetMatchResults(self, data, reqid):
        """查询成交回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onGetOrder(self, data, reqid):
        """查询单一委托回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onGetMatchResult(self, data, reqid):
        """查询单一成交回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onPlaceOrder(self, data, reqid):
        """委托回调"""
        print (reqid, data)
    
    #----------------------------------------------------------------------
    def onCancelOrder(self, data, reqid):
        """撤单回调"""
        print (reqid, data)
        
    #----------------------------------------------------------------------
    def onBatchCancel(self, data, reqid):
        """批量撤单回调"""
        print (reqid, data)

########################################################################
class ThreadedHuobi(Huobi):
    #----------------------------------------------------------------------
    def __init__(self, trader, settings):
        """Constructor"""
        super(ThreadedHuobi, self).__init__(trader, settings)
        self.thread = Thread(target=self._run)

    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    @abstractmethod
    def start(self, n=10):
        """启动"""
        ret = self.onStart()
        
        if ret and self._mode == self.BROKER_API_ASYNC:
            self.pool = Pool(n)
            self.pool.map_async(self._run, range(n))
        
        self.debug('ThreadedHuobi starts')
        return ret

    def _run(self):
        """执行连接 and receive"""
        while self._trader._active:
            self.step()
                
        self.info('ThreadedHuobi exit')

    @abstractmethod
    def stop(self):
        self._trader.stop()
        self.pool.close()
        self.pool.join()
        self.info('ThreadedHuobi stopped')


# ########################################################################
# class tdHuobi_sim(tdHuobi):
#     ''' 
#     simulate the account ordres
#     '''
#     #----------------------------------------------------------------------
#     def __init__(self, account, settings, mode=None):
#         """Constructor"""
#         super(tdHuobi_virtual, self).__init__(account, settings, mode)

#     #----------------------------------------------------------------------
#     def getOrders(self, symbol, states, types=None, startDate=None, 
#                   endDate=None, from_=None, direct=None, size=None):
#         """查询委托"""
#         path = '/v1/order/orders'
        
#         params = {
#             'symbol': symbol,
#             'states': states
#         }
        
#         if types:
#             params['types'] = types
#         if startDate:
#             params['start-date'] = startDate
#         if endDate:
#             params['end-date'] = endDate        
#         if from_:
#             params['from'] = from_
#         if direct:
#             params['direct'] = direct
#         if size:
#             params['size'] = size        
    
#         func = self.apiGet
#         callback = self.onGetOrders
    
#         return self.addReq(path, params, func, callback)     

#     #----------------------------------------------------------------------
#     def getOpenOrders(self, accountId=None, symbol=None, side=None, size=None):
#         """查询当前帐号下未成交订单
#             “account_id” 和 “symbol” 需同时指定或者二者都不指定。如果二者都不指定，返回最多500条尚未成交订单，按订单号降序排列。
#         """
#         path = '/v1/order/openOrders'
        
#         params = { # initial with default required params
#             #'account_id': accountId,
#             #'symbol': symbol,
#         }
        
#         if symbol:
#             params['symbol'] = symbol
#             params['account_id'] = accountId

#         if side:
#             params['side'] = side

#         if size:
#             params['size'] = size        
    
#         func = self.apiGet
#         callback = self.onGetOrders
    
#         return self.addReq(path, params, func, callback)     
    
#     #----------------------------------------------------------------------
#     def getMatchResults(self, symbol, types=None, startDate=None, 
#                   endDate=None, from_=None, direct=None, size=None):
#         """查询委托"""
#         path = '/v1/order/matchresults'

#         params = {
#             'symbol': symbol
#         }

#         if types:
#             params['types'] = types
#         if startDate:
#             params['start-date'] = startDate
#         if endDate:
#             params['end-date'] = endDate        
#         if from_:
#             params['from'] = from_
#         if direct:
#             params['direct'] = direct
#         if size:
#             params['size'] = size        

#         func = self.apiGet
#         callback = self.onGetMatchResults

#         return self.addReq(path, params, func, callback)   
    
#     #----------------------------------------------------------------------
#     def getOrder(self, orderid):
#         """查询某一委托"""
#         path = '/v1/order/orders/%s' %orderid
    
#         params = {}
    
#         func = self.apiGet
#         callback = self.onGetOrder
    
#         return self.addReq(path, params, func, callback)             
    
#     #----------------------------------------------------------------------
#     def getMatchResult(self, orderid):
#         """查询某一委托"""
#         path = '/v1/order/orders/%s/matchresults' %orderid
    
#         params = {}
    
#         func = self.apiGet
#         callback = self.onGetMatchResult
    
#         return self.addReq(path, params, func, callback)     
    
#     #----------------------------------------------------------------------
#     def placeOrder(self, amount, symbol, type_, price=None, source=None):
#         """下单"""
#         if self._hostname == self.HUOBI_API_HOST:
#             path = '/v1/order/orders/place'
#         else:
#             path = '/v1/hadax/order/orders/place'
        
#         params = {
#             'account-id': accountid,
#             'amount': amount,
#             'symbol': symbol,
#             'type': type_
#         }
        
#         if price:
#             params['price'] = price
#         if source:
#             params['source'] = source     

#         func = self.apiPost
#         callback = self.onPlaceOrder

#         return self.addReq(path, params, func, callback)           
    
#     #----------------------------------------------------------------------
#     def cancelOrder(self, orderid):
#         """撤单"""
#         path = '/v1/order/orders/%s/submitcancel' %orderid
        
#         params = {}
        
#         func = self.apiPost
#         callback = self.onCancelOrder

#         return self.addReq(path, params, func, callback)          
    
#     #----------------------------------------------------------------------
#     def batchCancel(self, orderids):
#         """批量撤单"""
#         path = '/v1/order/orders/batchcancel'
    
#         params = {
#             'order-ids': orderids
#         }
    
#         func = self.apiPost
#         callback = self.onBatchCancel
    
#         return self.addReq(path, params, func, callback)     


########################################################################
# API测试程序    
########################################################################

#----------------------------------------------------------------------
class TestTrader(Trader):
   
    def __init__(self, mainRoutine, settings):
        """Constructor"""

        super(TestTrader, self).__init__(mainRoutine, settings)

        for ak in self._dictAccounts.keys() :
            account = Huobi(self, settings.account)
            self._dictAccounts[self._defaultAccId] = account


if __name__ == '__main__':
    import os
    import jsoncfg # pip install json-cfg
    from vnApp.MainRoutine import *

    """测试交易"""

    settings= None
    try :
        conf_fn = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/conf/TD_huobi.json'
        settings= jsoncfg.load_config(conf_fn)
    except Exception as e :
        print('failed to load configure[%s]: %s' % (conf_fn, e))
        exit(-1)

    me = MainRoutine(settings)

    trader = me.addApp(TestTrader, settings['trader'])
    acc = trader.account
    # acc.start()

    # 查询
    # print (acc.getSymbols())
    print (acc.getCurrencys())
    print (acc.getTimestamp())
    print (acc.getAccounts())
    print (acc.getBalance())
    print (acc.getOpenOrders('eosusdt', 'sell'))
    me.loop()

    #online unicode converter
    # symbol = str(setting['symbols'][0])
    # symbol = str(symbols[0]) # 'eop':eos to udtc

    input()
    exit(0)
    
    print (acc.getAccounts())
    print (acc.getOpenOrders(accountId, symbol, 'sell'))
#    print (acc.getOrders(symbol, 'pre-submitted,submitted,partial-filled,partial-canceled,filled,canceled'))
#    print (acc.getOrders(symbol, 'filled'))
    print (acc.getMatchResults(symbol))
    
    print (acc.getOrder('2440401255'))


