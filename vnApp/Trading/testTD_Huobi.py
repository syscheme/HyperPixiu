# encoding: utf-8

import json
from vnpy.api.huobi import *

#----------------------------------------------------------------------
def testTrade():
    """测试交易"""
    try:
        f = file('vnApp/conf/Huobi.json')
    except IOError:
        return
        
    # 解析json文件
    setting = json.load(f)
    try:
        accessKey = str(setting['accessKey'])
        secretKey = str(setting['secretKey'])
        accountId = str(setting['accountId'])
    except KeyError:
       return            
   
    # 创建API对象并初始化
    api = TradeApi()

    # api.init(api.HADAX, accessKey, secretKey, mode=api.SYNC_MODE)
    api.init(api.HUOBI, accessKey, secretKey, mode=api.SYNC_MODE)
    api.start()

    # 查询
    # print (api.getSymbols())
    print (api.getCurrencys())
    print (api.getTimestamp())

    #online unicode converter
    symbol = str(setting['symbols'][0])
    # symbol = str(symbols[0]) # 'eop':eos to udtc
    
    print (api.getAccounts())
    print (api.getAccountBalance(accountId))
    print (api.getOpenOrders(accountId, symbol, 'sell'))
#    print (api.getOrders(symbol, 'pre-submitted,submitted,partial-filled,partial-canceled,filled,canceled'))
#    print (api.getOrders(symbol, 'filled'))
    print (api.getMatchResults(symbol))
    
    print (api.getOrder('2440401255'))
    #api.getMatchResult('2440401255')
    
    #api.placeOrder(accountid, '2', symbol, 'sell-market', source='api')
    #api.cancelOrder('2440451757')
    #api.batchCancel(['2440538580', '2440537853', '2440536765'])
    
    # input()    


    
    
if __name__ == '__main__':
    testTrade()