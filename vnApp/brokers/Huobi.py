# encoding: UTF-8

from ..Account import *

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
    
    #----------------------------------------------------------------------
    def __init__(self, trader, settings):
        """Constructor"""
        super(Account_AShare, self).__init__(trader, settings)

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
    def pushRequest(self, path, reqId, httpParams, data, funcMethod, callbackResp):
        """添加请求"""       
        # 同步模式
        if self.mode != self.ASYNC_MODE:
            return funcMethod(path, params)

        # 异步模式
        req = {
                'reqId': reqId,
                'path': path, 
                'httpparams': httpParams, 
                'reqData': data, 
                'apiFunc': funcMethod, 
                'callback': callback
            }

        with self._lock
            self._dictRequests[reqId] = req
            self._queRequests.append(reqId)

        return reqId

    #------------------------------------------------
    # overwrite of Account
    #------------------------------------------------    
    def _broker_placeOrder(self, orderData):
        """下单"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/order/orders/place'
        else:
            path = '/v1/hadax/order/orders/place'

        params = {
            'account-id': self._id,
            'amount': orderData.volume * self.size,
            'symbol': orderData.symbol,
            'type'  : orderData.direction
        }
        
        if price:
            params['price'] = price
        if source:
            params['source'] = source

        return self.pushRequest(path, orderData.reqId, params, orderData, onResp_placeOrder)

    def onResp_placeOrder(self, req, resp):
        """下单的response"""
        orderData = req['reqData']

        #TODO fill response data into orderData

        self._broker_onOrderPlaced(orderData) # if succ, else _broker_onCancelled() ??

    def _broker_cancelOrder(self, brokerOrderId) :
        orderData = None

        # find out he orderData by brokerOrderId
        with self._nest._lock :
            try :
                if OrderData.STOPORDERPREFIX in brokerOrderId :
                    orderData = self._nest._dictStopOrders[brokerOrderId]
                else :
                    orderData = self._nest._dictLimitOrders[brokerOrderId]
            except KeyError:
                pass

            if not orderData :
                return

            orderData = copy.copy(orderData)

        # orderData found
        orderData.status = OrderData.STATUS_CANCELLED
        orderData.cancelTime = self._broker_datetimeAsOf().strftime('%H:%M:%S.%f')[:3]
        self._broker_onCancelled(orderData)

    def step(self) :
        outgoingOrders = []
        ordersToCancel = []

        with self._nest._lock:
            outgoingOrders = copy.deepcopy(self._nest._dictOutgoingOrders.values())
            ordersToCancel = copy.copy(self._nest._lstOrdersToCancel)
            self._nest._lstOrdersToCancel = []

        for boid in ordersToCancel:
            self._broker_cancelOrder(boid)
        for o in outgoingOrders:
            self._broker_placeOrder(o)

        if (len(ordersToCancel) + len(outgoingOrders)) >0:
            self._nest.debug('step() cancelled %d orders, placed %d orders'% (len(ordersToCancel), len(outgoingOrders)))

    def onDayOpen(self, newDate):
        # instead that the true Account is able to sync with broker,
        # Backtest should perform cancel to restore available/frozen positions
        clist = []
        with self._nest._lock :
            # A share will not keep yesterday's order alive
            self._nest._dictOutgoingOrders.clear()
            for o in self._nest._dictLimitOrders.values():
                clist.append(o.brokerOrderId)
            for o in self._nest._dictStopOrders.values():
                clist.append(o.brokerOrderId)

        if len(clist) >0:
            self.batchCancel(clist)
            self._nest.debug('BT.onDayOpen() batchCancelled: %s' % clist)
        self._nest.onDayOpen(newDate)


########################################################################################################

    #----------------------------------------------------------------------
    def start(self, n=10):
        """启动"""
        self.active = True
        
        if self.mode == self.ASYNC_MODE:
            self.pool = Pool(n)
            self.pool.map_async(self.run, range(n))
        
    #----------------------------------------------------------------------
    def close(self):
        """停止"""
        self.active = False
        self.pool.close()
        self.pool.join()
        
    #----------------------------------------------------------------------
    def httpGet(self, url, params):
        """HTTP GET"""        
        headers = copy(DEFAULT_GET_HEADERS)
        postdata = urllib.urlencode(params)
        
        try:
            response = requests.get(url, postdata, headers=headers, proxies=self._proxies, timeout=TIMEOUT)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, u'GET请求失败，状态代码：%s' %response.status_code
        except Exception as e:
            return False, u'GET请求触发异常，原因：%s' %e
    
    #----------------------------------------------------------------------    
    def httpPost(self, url, params, add_to_headers=None):
        """HTTP POST"""       
        headers = copy(DEFAULT_POST_HEADERS)
        postdata = json.dumps(params)
        
        try:
            response = requests.post(url, postdata, headers=headers, proxies=self._proxies, timeout=TIMEOUT)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, u'POST请求失败，返回信息：%s' %response.json()
        except Exception as e:
            return False, u'POST请求触发异常，原因：%s' %e
        
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
        
    #----------------------------------------------------------------------
    def apiGet(self, path, params):
        """API GET"""
        method = 'GET'
        
        params.update(self.generateSignParams())
        params['Signature'] = createSign(params, method, self._hostname, path, self.secretKey)
        
        url = self._hosturl + path
        #print("url=%s, param:%s" % (url, params))
        
        return self.httpGet(url, params)
    
    #----------------------------------------------------------------------
    def apiPost(self, path, params):
        """API POST"""
        method = 'POST'
        
        signParams = self.generateSignParams()
        signParams['Signature'] = createSign(signParams, method, self._hostname, path, self.secretKey)
        
        url = self._hosturl + path + '?' + urllib.urlencode(signParams)

        return self.httpPost(url, params)
    
    #----------------------------------------------------------------------
    def addReq(self, path, params, func, callback):
        """添加请求"""       
        # 异步模式
        if self.mode != self.ASYNC_MODE:
            return func(path, params)

        # 同步模式
        self.reqid += 1
        req = (path, params, func, callback, self.reqid)
        self.queue.put(req)
        return self.reqid
    
    #----------------------------------------------------------------------
    def processReq(self, req):
        """处理请求"""
        path, params, func, callback, reqid = req
        result, data = func(path, params)
        
        if result:
            if data['status'] == 'ok':
                callback(data['data'], reqid)
            else:
                msg = u'错误代码：%s，错误信息：%s' %(data['err-code'], data['err-msg'])
                self.onError(msg, reqid)
        else:
            self.onError(data, reqid)
            
            # 失败的请求重新放回队列，等待下次处理
            self.queue.put(req)
    
    #----------------------------------------------------------------------
    def run(self, n):
        """连续运行"""
        while self.active:    
            try:
                req = self.queue.get(timeout=1)
                self.processReq(req)
            except Empty:
                pass
    
    #----------------------------------------------------------------------
    def getSymbols(self):
        """查询合约代码"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/common/symbols'
        else:
            path = '/v1/hadax/common/symbols'

        params = {}
        func = self.apiGet
        callback = self.onGetSymbols
        
        return self.addReq(path, params, func, callback)
    
    #----------------------------------------------------------------------
    def getCurrencys(self):
        """查询支持货币"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/common/currencys'
        else:
            path = '/v1/hadax/common/currencys'

        params = {}
        func = self.apiGet
        callback = self.onGetCurrencys
        
        return self.addReq(path, params, func, callback)   
    
    #----------------------------------------------------------------------
    def getTimestamp(self):
        """查询系统时间"""
        path = '/v1/common/timestamp'
        params = {}
        func = self.apiGet
        callback = self.onGetTimestamp
        
        return self.addReq(path, params, func, callback) 
    
    #----------------------------------------------------------------------
    def getAccounts(self):
        """查询账户"""
        path = '/v1/account/accounts'
        params = {}
        func = self.apiGet
        callback = self.onGetAccounts
    
        return self.addReq(path, params, func, callback)         
    
    #----------------------------------------------------------------------
    def getAccountBalance(self, accountid):
        """查询余额"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/account/accounts/%s/balance' %accountid
        else:
            path = '/v1/hadax/account/accounts/%s/balance' %accountid
            
        params = {}
        func = self.apiGet
        callback = self.onGetAccountBalance
    
        return self.addReq(path, params, func, callback) 
    
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
    
        func = self.apiGet
        callback = self.onGetOrders
    
        return self.addReq(path, params, func, callback)     

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
    
        func = self.apiGet
        callback = self.onGetOrders
    
        return self.addReq(path, params, func, callback)     
    
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

        return self.addReq(path, params, func, callback)   
    
    #----------------------------------------------------------------------
    def getOrder(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s' %orderid
    
        params = {}
    
        func = self.apiGet
        callback = self.onGetOrder
    
        return self.addReq(path, params, func, callback)             
    
    #----------------------------------------------------------------------
    def getMatchResult(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s/matchresults' %orderid
    
        params = {}
    
        func = self.apiGet
        callback = self.onGetMatchResult
    
        return self.addReq(path, params, func, callback)     
    
    #----------------------------------------------------------------------
    def placeOrder(self, amount, symbol, type_, price=None, source=None):
        """下单"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/order/orders/place'
        else:
            path = '/v1/hadax/order/orders/place'
        
        params = {
            'account-id': accountid,
            'amount': amount,
            'symbol': symbol,
            'type': type_
        }
        
        if price:
            params['price'] = price
        if source:
            params['source'] = source     

        func = self.apiPost
        callback = self.onPlaceOrder

        return self.addReq(path, params, func, callback)           
    
    #----------------------------------------------------------------------
    def cancelOrder(self, orderid):
        """撤单"""
        path = '/v1/order/orders/%s/submitcancel' %orderid
        
        params = {}
        
        func = self.apiPost
        callback = self.onCancelOrder

        return self.addReq(path, params, func, callback)          
    
    #----------------------------------------------------------------------
    def batchCancel(self, orderids):
        """批量撤单"""
        path = '/v1/order/orders/batchcancel'
    
        params = {
            'order-ids': orderids
        }
    
        func = self.apiPost
        callback = self.onBatchCancel
    
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
class tdHuobi_sim(tdHuobi):
    ''' 
    simulate the account ordres
    '''
    #----------------------------------------------------------------------
    def __init__(self, account, settings, mode=None):
        """Constructor"""
        super(tdHuobi_virtual, self).__init__(account, settings, mode)

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
    
        func = self.apiGet
        callback = self.onGetOrders
    
        return self.addReq(path, params, func, callback)     

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
    
        func = self.apiGet
        callback = self.onGetOrders
    
        return self.addReq(path, params, func, callback)     
    
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

        return self.addReq(path, params, func, callback)   
    
    #----------------------------------------------------------------------
    def getOrder(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s' %orderid
    
        params = {}
    
        func = self.apiGet
        callback = self.onGetOrder
    
        return self.addReq(path, params, func, callback)             
    
    #----------------------------------------------------------------------
    def getMatchResult(self, orderid):
        """查询某一委托"""
        path = '/v1/order/orders/%s/matchresults' %orderid
    
        params = {}
    
        func = self.apiGet
        callback = self.onGetMatchResult
    
        return self.addReq(path, params, func, callback)     
    
    #----------------------------------------------------------------------
    def placeOrder(self, amount, symbol, type_, price=None, source=None):
        """下单"""
        if self._hostname == HUOBI_API_HOST:
            path = '/v1/order/orders/place'
        else:
            path = '/v1/hadax/order/orders/place'
        
        params = {
            'account-id': accountid,
            'amount': amount,
            'symbol': symbol,
            'type': type_
        }
        
        if price:
            params['price'] = price
        if source:
            params['source'] = source     

        func = self.apiPost
        callback = self.onPlaceOrder

        return self.addReq(path, params, func, callback)           
    
    #----------------------------------------------------------------------
    def cancelOrder(self, orderid):
        """撤单"""
        path = '/v1/order/orders/%s/submitcancel' %orderid
        
        params = {}
        
        func = self.apiPost
        callback = self.onCancelOrder

        return self.addReq(path, params, func, callback)          
    
    #----------------------------------------------------------------------
    def batchCancel(self, orderids):
        """批量撤单"""
        path = '/v1/order/orders/batchcancel'
    
        params = {
            'order-ids': orderids
        }
    
        func = self.apiPost
        callback = self.onBatchCancel
    
        return self.addReq(path, params, func, callback)     
