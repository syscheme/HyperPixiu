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
    #----------------------------------------------------------------------
    def __init__(self, trader, settings):
        """Constructor"""
        super(Huobi, self).__init__(trader, settings)

        self._mode        = Account.BROKER_API_ASYNC
        self._queRequests = Queue()        # queue of request ids
        self._dictRequests = {}            # dict from request Id to request

        # cash of broker data before merge into Account
        self._dictHbOrders = {}
        self._dictHbPositions = {}
        
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
        }
        
        if orderData.price >0:
            params['price'] = orderData.price
            params['type'] = 'buy-limit' if order.direction == OrderData.DIRECTION_LONG else 'sell-limit' #限价
        else :
            params['type'] = 'buy-market' if order.direction == OrderData.DIRECTION_LONG else 'buy-market' #市价
            #TODO buy-ioc：IOC买单, sell-ioc：IOC卖单, buy-limit-maker, sell-limit-maker

        if len(orderData.source) >0 and 'margin' in orderData.source:
            params['source'] = 'margin-api'

        return self.pushRequest(path, orderData.reqId, params, orderData, self.apiPOST, self.brokerCBplaceOrder)

    def brokerCBplaceOrder(self, reqData, resp):
        """
        下单的response
        i.e. { "status": "ok", "data": "59378" }
        """
        orderData = reqData
        orderData.brokerOrderId = str(resp)

        self._broker_onOrderPlaced(orderData) # if succ, else _broker_onCancelled() ??

    def _broker_cancelOrder(self, orderData) :
        """撤单"""
        path = '/v1/order/orders/%s/submitcancel' % orderData.brokerOrderId
        
        return self.pushRequest(path, orderData.reqId, {}, orderData, self.apiPOST, self.brokerCBcancelOrder)
        
    def brokerCBcancelOrder(self, reqData, resp):
        """ 撤单的response
        ie. { "status": "ok", "data": "59378" }
        """
        orderData = reqData
        # double check: if orderData.brokerOrderId == str(resp)
        self._broker_onCancelled(orderData) # if succ, else _broker_onCancelled() ??

    def _broker_listOpenOrders(self, symbol=None, side=None, size=None):
        """查询当前帐号下未成交订单
        """
        path = '/v1/order/openOrders'
        
        params = { # initial with default required params
            #'account_id': accountId,
            #'symbol': symbol,
        }
        
        if symbol: # “account_id” 和 “symbol” 需同时指定或者二者都不指定。如果二者都不指定，返回最多500条尚未成交订单，按订单号降序排列。
            params['symbol'] = symbol
            params['account_id'] = self._id

        if side:
            params['side'] = side

        if size:
            params['size'] = size        
    
        return self.pushRequest(path, None, params, None, self.apiGET, self.brokerCBlistOpenOrders)

    def brokerCBlistOpenOrders(self, reqData, resp):
        """ 查询当前帐号下未成交订单的response
        ie. { "status": "ok", "data": [
             { "id": 5454937, "symbol": "ethusdt", "account-id": 30925, "amount": "1.000000000000000000", 
               "price": "0.453000000000000000", "created-at": 1530604762277, "type": "sell-limit",  
               "filled-amount": "0.0", "filled-cash-amount": "0.0",  "filled-fees": "0.0", "source": "web",
               "state": "submitted" }
            ] }
        """
        dictOrders = {}
        for o in resp :
            order = OrderData(self)
            order.reqid= ''
            if o['account-id'] != self._id:
                continue
            order.brokerOrderId = o['id']
            order.totalVolume = o['amount']
            order.price = o['price']
            order.status = o['state']
            dictOrders[order.brokerOrderId] =order

        # double check: if orderData.brokerOrderId == str(resp)
        self._broker_onOpenOrders(dictOrders) # if succ, else _broker_onCancelled() ??


    def step(self) :
        nextSleep = 5.0
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

    @abstractmethod
    def _brocker_procSyncData(self):
        with self._lock :
            if len(self._dictHbOrders) >0:
                #TODO
                self._dictHbOrders = {}
            
            if len(self._dictHbPositions) >0:
                #TODO
                for pos in self._dictHbPositions.values() :
                    if not pos.symbol in self._dictPositions.keys():
                        self._dictPositions[pos.symbol] =pos
                    else :
                        self._dictPositions[pos.symbol].posAval = pos.posAvail
                        self._dictPositions[pos.symbol].position = pos.position
                        self._dictPositions[pos.symbol].stampByBroker = pos.stampByBroker
                    print(self._dictPositions[pos.symbol].__dict__)
                self._dictHbPositions = {}

    @abstractmethod
    def _brocker_triggerSync(self):
        self.getBalance()  # 查询余额 and positions

        symbols = []
        with self._lock:
            symbols = self._dictPositions.keys()

        # 查询最近2day的order
        ORDERSTATES_TO_SYNC = ['filled', 'partial-canceled']
        dateStart= (self._broker_datetimeAsOf() - timedelta(days=2)).strftime('%Y-%m-%d')
        for s in symbols:
            self.listOrders(symbol=s, states = ORDERSTATES_TO_SYNC, startDate=dateStart)


    #----------------------------------------------------------------------
    # most of the methods are just forward to the self._nest
    # def _broker_datetimeAsOf(self): return self._nest._broker_datetimeAsOf()
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
    # def calcDailyPositions(self): return self._nest.calcDailyPositions()
    # def log(self, message): return self._nest.log(message)
    #----------------------------------------------------------------------
    '''
    reference: https://github.com/huobiapi/API_Docs/wiki/REST_api_reference
    '''
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
    
    #----------------------------------------------------------------------
    def signParams(self, method, params, path):
        """创建签名"""

        d = {
            'AccessKeyId': self.accessKey,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        }
        
        params.update(d)

        sortedParams = sorted(params.items(), key=lambda d: d[0], reverse=False)
        encodeParams = urllib.urlencode(sortedParams)
        
        payload = [method, self._hostname, path, encodeParams]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='UTF8')

        secretKey = self.secretKey.encode(encoding='UTF8')
        digest = hmac.new(secretKey, payload, digestmod=hashlib.sha256).digest()

        signature = base64.b64encode(digest)
        signature = signature.decode()
        params['Signature'] = signature

        return params    

    def apiGET(self, path, params):
        """API GET"""

        params = self.signParams('GET', params, path)
        url = self._hosturl + path # + '?' + urllib.urlencode(params)
        
        try:
            response = requests.get(url, urllib.urlencode(params), headers=copy(self.DEFAULT_GET_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e

    def apiPOST(self, path, params, add_to_headers=None):
        """API POST"""

        params = self.signParams('POST', params, path)
        url = self._hosturl + path + '?' + urllib.urlencode(params) # look like Huobi server has a bug that require param to be present on a POST url

        try:
            response = requests.post(url, json.dumps(params), headers=copy(self.DEFAULT_POST_HEADERS), proxies=self._proxies, timeout=self.TIMEOUT)
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

        return self.pushRequest(path, None, {}, None, self.apiGET, self.onGetSymbols)
    
    #----------------------------------------------------------------------
    def getCurrencys(self):
        """查询支持货币
        ([u'hb10', u'usdt', u'btc', u'bch', u'eth', u'xrp', u'ltc', u'ht', u'ada', u'eos', u'iota', u'xem', u'xmr', u'dash', u'neo', u'trx', u'icx', u'lsk', u'qtum', u'etc', u'btg', u'omg', u'hsr', u'zec', u'dcr', u'steem', u'bts', u'waves', u'snt', u'salt', u'gnt', u'cmt', u'btm', u'pay', u'knc', u'powr', u'bat', u'dgd', u'ven', u'qash', u'zrx', u'gas', u'mana', u'eng', u'cvc', u'mco', u'mtl', u'rdn', u'storj', u'chat', u'srn', u'link', u'act', u'tnb', u'qsp', u'req', u'rpx', u'appc', u'rcn', u'smt', u'adx', u'tnt', u'ost', u'itc', u'lun', u'gnx', u'ast', u'evx', u'mds', u'snc', u'propy', u'eko', u'nas', u'bcd', u'wax', u'wicc',u'topc', u'swftc', u'dbc', u'elf', u'aidoc', u'qun', u'iost', u'yee', u'dat', u'theta', u'let', u'dta', u'utk', u'meet', u'zil', u'soc', u'ruff', u'ocn', u'ela', u'bcx', u'sbtc', u'etf', u'bifi', u'zla', u'stk', u'wpr', u'mtn', u'mtx', u'edu', u'blz', u'abt', u'ont', u'ctxc', u'bft', u'wan', u'kan', u'lba', u'poly', u'pai', u'wtc', u'box', u'dgb', u'gxs', u'bix', u'xlm', u'xvg', u'hit', u'bt1', u'bt2', u'a5b5', u'x5z5', u'xzc', u'vet'], None)
        """
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/common/currencys'
        else:
            path = '/v1/hadax/common/currencys'

        return self.pushRequest(path, None, {}, None, self.apiGET, self.onGetCurrencys)
    
    #----------------------------------------------------------------------
    def getTimestamp(self):
        """查询系统时间
        2018-08-21 08:16:40,811 (1534810600660, None)
        """
        return self.pushRequest('/v1/common/timestamp', None, {}, None, self.apiGET, self.onGetTimestamp)
    
    #----------------------------------------------------------------------
    def getAccounts(self):
        """查询账户"""
        return self.pushRequest('/v1/account/accounts', None, {}, None, self.apiGET, self.onGetAccounts)
    
    #----------------------------------------------------------------------
    def getBalance(self):
        """查询余额
        """
        if self._hostname == self.HUOBI_API_HOST:
            path = '/v1/account/accounts/%s/balance' % self._id
        else:
            path = '/v1/hadax/account/accounts/%s/balance' % self._id
            
        return self.pushRequest(path, None, {}, None, self.apiGET, self.hbCbBalance)

    def hbCbBalance(self, reqData, resp):
        """查询余额回调
        {u'list': [
            ...
            {u'currency': u'usdt', u'balance': u'0.000572974', u'type': u'trade'},
            {u'currency': u'usdt', u'balance': u'0', u'type': u'frozen'},
            {u'currency': u'btc', u'balance': u'0', u'type': u'trade'},
            {u'currency': u'btc', u'balance': u'0', u'type': u'frozen'},
            {u'currency': u'eth', u'balance': u'0', u'type': u'trade'},
            {u'currency': u'eth', u'balance': u'0', u'type': u'frozen'},
            ...
            {u'currency': u'eos', u'balance': u'8.763676', u'type': u'trade'}, 
            {u'currency': u'eos', u'balance': u'0', u'type': u'frozen'}, 
            ...
            {u'currency': u'theta', u'balance': u'516.76', u'type': u'trade'},
            {u'currency': u'theta', u'balance': u'100', u'type': u'frozen'},
            ...
            {u'currency': u'ctxc', u'balance': u'49.7', u'type': u'trade'},
            {u'currency': u'ctxc', u'balance': u'0', u'type': u'frozen'},
            ...
        ], u'state': u'working', u'type': u'spot', u'id': nnnnnn}
        """
        with self._lock :
            stampNow = self._broker_datetimeAsOf()
            for b in resp['list']:
                symbol = b[u'currency'].encode('utf-8')
                bal = float(b[u'balance'])
                if bal <=0:
                    continue
                if symbol in self._dictHbPositions.keys() :
                    pos = self._dictHbPositions[symbol]
                else :
                    pos = PositionData()
                    pos.symbol = symbol
                    pos.exchange = self._exchange
                    pos.vtSymbol = '%s.%s' % (symbol, self._exchange)

                if b[u'type'] == u'trade' :
                    pos.posAvail = bal
                else :
                    pos.position = pos.posAvail + bal
                
                pos.stampByBroker = stampNow
                self._dictHbPositions[symbol] = pos

    #----------------------------------------------------------------------
    def listOrders(self, symbol, states, types=None, startDate=None, 
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
    
        return self.pushRequest(path, None, params, None, self.apiGET, self.hbCblistOrders)

    def hbCblistOrders(self, reqData, resp):
        """ 查询当前帐号下已成交订单的response
            {
            "status": "ok",
            "data": [
                {
                "id": 59378,
                "symbol": "ethusdt",
                "account-id": 100009,
                "amount": "10.1000000000",
                "price": "100.1000000000",
                "created-at": 1494901162595,
                "type": "buy-limit",
                "field-amount": "10.1000000000",
                "field-cash-amount": "1011.0100000000",
                "field-fees": "0.0202000000",
                "finished-at": 1494901400468,
                "user-id": 1000,
                "source": "api",
                "state": "filled",
                "canceled-at": 0,
                "exchange": "huobi",
                "batch": ""
                }
            ]
            }
        """
        with self._lock :
            for o in resp :
                order = OrderData(self)
                order.reqid= ''
                if o['account-id'] != self._id:
                    continue
                order.brokerOrderId = o['id']
                order.totalVolume = o['amount']
                order.price = o['price']
                order.status = o['state']
                # double check: if orderData.brokerOrderId == str(resp)
                self._dictHbOrders[order.brokerOrderId] =order

    #----------------------------------------------------------------------
    
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

        func = self.apiGET
        callback = self.onGetMatchResults

        return self.pushRequest(path, None, params, None, self.apiGET, self.onGetMatchResults)
    
    #----------------------------------------------------------------------
    def getOrder(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s' %orderid
        return self.pushRequest(path, None, {}, None, self.apiGET, self.onGetOrder)
    
    #----------------------------------------------------------------------
    def getMatchResult(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s/matchresults' %orderid
        return self.pushRequest(path, None, {}, None, self.apiGET, self.onGetMatchResult)
    
    #----------------------------------------------------------------------
    def batchCancel(self, orderids):
        """批量撤单"""
        path = '/v1/order/orders/batchcancel'
    
        params = {
            'order-ids': orderids
        }
    
        func = self.apiPOST
        callback = self.onBatchCancel
    
        return self.pushRequest(path, None, params, None, self.apiPOST, self.onPlaceOrder)
        
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
    
#         func = self.apiGET
#         callback = self.onGetOrders
    
#         return self.addReq(path, params, func, callback)     

#     #----------------------------------------------------------------------
#     def _broker_listOpenOrders(self, accountId=None, symbol=None, side=None, size=None):
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
    
#         func = self.apiGET
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

#         func = self.apiGET
#         callback = self.onGetMatchResults

#         return self.addReq(path, params, func, callback)   
    
#     #----------------------------------------------------------------------
#     def getOrder(self, orderid):
#         """查询某一委托"""
#         path = '/v1/order/orders/%s' %orderid
    
#         params = {}
    
#         func = self.apiGET
#         callback = self.onGetOrder
    
#         return self.addReq(path, params, func, callback)             
    
#     #----------------------------------------------------------------------
#     def getMatchResult(self, orderid):
#         """查询某一委托"""
#         path = '/v1/order/orders/%s/matchresults' %orderid
    
#         params = {}
    
#         func = self.apiGET
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

#         func = self.apiPOST
#         callback = self.onPlaceOrder

#         return self.addReq(path, params, func, callback)           
    
#     #----------------------------------------------------------------------
#     def cancelOrder(self, orderid):
#         """撤单"""
#         path = '/v1/order/orders/%s/submitcancel' %orderid
        
#         params = {}
        
#         func = self.apiPOST
#         callback = self.onCancelOrder

#         return self.addReq(path, params, func, callback)          
    
#     #----------------------------------------------------------------------
#     def batchCancel(self, orderids):
#         """批量撤单"""
#         path = '/v1/order/orders/batchcancel'
    
#         params = {
#             'order-ids': orderids
#         }
    
#         func = self.apiPOST
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
    od = OrderData(acc)
    od.brokerOrderId='1234445'
    print (acc._broker_cancelOrder(od))
    print (acc._broker_listOpenOrders('eosusdt', 'sell'))

    me.loop()

    #online unicode converter
    # symbol = str(setting['symbols'][0])
    # symbol = str(symbols[0]) # 'eop':eos to udtc

    input()
    exit(0)
    
    print (acc.getAccounts())
    print (acc._broker_listOpenOrders(accountId, symbol, 'sell'))
#    print (acc.getOrders(symbol, 'pre-submitted,submitted,partial-filled,partial-canceled,filled,canceled'))
#    print (acc.getOrders(symbol, 'filled'))
    print (acc.getMatchResults(symbol))
    
    print (acc.getOrder('2440401255'))


