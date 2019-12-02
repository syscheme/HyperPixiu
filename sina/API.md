# List the symbols

There are over 3000 symbols, so must be listed by pages

```
http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?'
    url += 'page=' + str( pageNum )
    url += '&num=' + str( rows )
    url += '&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=init'
```
请求样例:
```
http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=init&num=100&page=3
```
```
http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=init&num=50&page=70
```
返回样例:
```
[{symbol:"sz300542",code:"300542",name:"新晨科技",trade:"16.070",pricechange:"0.020",changepercent:"0.125",buy:"16.070",sell:"16.080",settlement:"16.050",open:"16.140",high:"16.300",low:"15.620",volume:6757950,amount:107650851,ticktime:"14:05:27",per:64.28,pb:5.852,mktcap:372390.277128,nmc:136116.584851,turnoverratio:7.97847},
{symbol:"sz300543",code:"300543",name:"朗科智能",trade:"24.380",pricechange:"-0.020",changepercent:"-0.082",buy:"24.370",sell:"24.380",settlement:"24.400",open:"24.480",high:"24.570",low:"24.020",volume:1745766,amount:42370586,ticktime:"14:05:24",per:65.892,pb:4.203,mktcap:292560,nmc:221797.05,turnoverratio:1.91895},
{symbol:"sz300545",code:"300545",name:"联得装备",trade:"28.560",pricechange:"0.880",changepercent:"3.179",buy:"28.560",sell:"28.580",settlement:"27.680",open:"27.440",high:"28.800",low:"27.440",volume:3860742,amount:109306646,ticktime:"14:05:24",per:47.6,pb:6.542,mktcap:411513.820032,nmc:138701.959872,turnoverratio:7.94962},
{symbol:"sz300546",code:"300546",name:"雄帝科技",trade:"27.190",pricechange:"0.390",changepercent:"1.455",buy:"27.180",sell:"27.190",settlement:"26.800",open:"26.800",high:"27.420",low:"26.800",volume:963471,amount:26172829,ticktime:"14:05:27",per:33.568,pb:4.734,mktcap:367283.8795,nmc:181020.315297,turnoverratio:1.44717},
...
```

# 历史数据
## day line：
```
http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz002095&scale=240&ma=no&datalen=250
```
值得一提的是改成scale=240就变成日K了，scale=1200变成周K，分钟级别的还支持5、15和30分钟。
- 5分钟(scale=5),  最近一周
- 15分钟(scale=15), 约最近二周
- 30分钟(scale=30), 约最近一月
- 1小时(scale=60), 约最近二月
- 日线(scale=240&datalen=4000), 最近十几年

然后去掉ma=no参数还可以获得5、10和30日均价均值

请求样例:
```
http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh600036&scale=240&datalen=100
```
返回样例:
```
[{day:"2019-04-26",open:"35.160",high:"35.200",low:"34.340",close:"34.360",volume:"51734443",ma_price5:35.124,ma_volume5:54055057,ma_price10:35.377,ma_volume10:56977681,ma_price30:34.098,ma_volume30:59737401},
{day:"2019-04-29",open:"34.570",high:"35.730",low:"34.420",close:"35.330",volume:"55033176",ma_price5:35.23,ma_volume5:50745098,ma_price10:35.428,ma_volume10:53231789,ma_price30:34.22,ma_volume30:59585364},
```
loop sample:
```
for i in `cat numsymb.txt`; do curl -o ${i}.json "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=${i}&scale=5&ma=no&datalen=1000" ; done
```

## 复权数据
关于复权数据，
请求样例:
以龙力生物2015年6月5日复权为例，当日该股下跌2.26%
```
http://finance.sina.com.cn/realstock/newcompany/sz002604/phfq.js
```
返回样例:
```
_2015_06_05:"52.5870",
_2015_06_04:"53.8027",
52.5870 / 53.8027 = 0.9774，和下跌2.26%一致
```


# 当日数据

## 5min
```
http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sz002095&scale=5&ma=no&datalen=250
```
分钟级别的支持5、15和30分钟

请求样例:
```
http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh600036&scale=240&datalen=100
```
返回样例:
```
[{day:"2019-04-26",open:"35.160",high:"35.200",low:"34.340",close:"34.360",volume:"51734443",ma_price5:35.124,ma_volume5:54055057,ma_price10:35.377,ma_volume10:56977681,ma_price30:34.098,ma_volume30:59737401},
{day:"2019-04-29",open:"34.570",high:"35.730",low:"34.420",close:"35.330",volume:"55033176",ma_price5:35.23,ma_volume5:50745098,ma_price10:35.428,ma_volume10:53231789,ma_price30:34.22,ma_volume30:59585364},
```

## 分时数据
请求样例:
```
http://vip.stock.finance.sina.com.cn/quotes_service/view/vML_DataList.php?asc=j&symbol=sh600036&num=10

```
返回样例:
```
['14:59:00', '18.59', '3778240']
```
数据：时间，价格，成交量（股）

## 个股当日资金流向接口
```
http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj?daima=sz300543
```
返回样例:
```
({r0_in:"0.0000",r0_out:"0.0000",r0:"0.0000",r1_in:"3851639.0000",r1_out:"4794409.0000",r1:"9333936.0000",r2_in:"8667212.0000",r2_out:"10001938.0000",r2:"18924494.0000",r3_in:"7037186.0000",r3_out:"7239931.2400",r3:"15039741.2400",curr_capital:"9098",name:"朗科智能",trade:"24.4200",changeratio:"0.000819672",volume:"1783866.0000",turnover:"196.083",r0x_ratio:"0",netamount:"-2480241.2400"})
```

### QQ 数据
请求样例:
```
http://qt.gtimg.cn/q=ff_sz300543
```
返回样例:
```
v_ff_sz300543="sz300543~0.00~0.00~0.00~0.00~2242.79~2332.08~-89.28~-3.83~47820000.00~228.39~1214.94~朗科智能~20190919~20190918^0.00^72.96~20190917^81.29^405.17~20190916^67.60^321.23~20190912^79.50^415.58~0.00~1883300.00~20190919141120";
```

## 逐笔交易明细数据
请求样例:
```
http://vip.stock.finance.sina.com.cn/quotes_service/view/CN_TransListV2.php?num=10&symbol=sh600036
http://vip.stock.finance.sina.com.cn/quotes_service/view/CN_TransListV2.php?symbol=sh600036
```
数据：时间，成交量(股)，价格，类型（UP-买，DOWN-卖，EQUAL-平）
返回样例:
```
var trade_item_list = new Array(); 
trade_item_list[0] = new Array('15:00:00', '722600', '34.510', 'UP'); 
trade_item_list[1] = new Array('14:57:00', '200', '34.510', 'UP'); 
trade_item_list[2] = new Array('14:56:57', '2200', '34.500', 'DOWN'); 
trade_item_list[3] = new Array('14:56:54', '3100', '34.500', 'DOWN'); 
trade_item_list[4] = new Array('14:56:51', '30000', '34.510', 'UP');
trade_item_list[5] = new Array('14:56:48', '5200', '34.500', 'UP');
trade_item_list[6] = new Array('14:56:45', '4200', '34.500', 'UP');
...
trade_item_list[9] = new Array('14:56:36', '64200', '34.500', 'UP');
```
尾部额外带了当日总买入股数和总卖出股数
```
var trade_INVOL_OUTVOL=[27793246.5,26052078.5];
```

## 当日资金流向
请求样例:
```
http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj?daima=sz300543
```
数据：???
返回样例:
```
({r0_in:"0.0000",r0_out:"0.0000",r0:"0.0000",r1_in:"3851639.0000",r1_out:"4794409.0000",r1:"9333936.0000",r2_in:"8667212.0000",r2_out:"10001938.0000",r2:"18924494.0000",r3_in:"7037186.0000",r3_out:"7239931.2400",```
r3:"15039741.2400",curr_capital:"9098",name:"朗科智能",trade:"24.4200",changeratio:"0.000819672",volume:"1783866.0000",turnover:"196.083",r0x_ratio:"0",netamount:"-2480241.2400"})

others req(sz000988/sz002547):
({r0_in:"7429627.0000",r0_out:"16185897.0300",r0:"26018214.0300",r1_in:"46907640.0900",r1_out:"69787281.6100",r1:"118430649.7000",r2_in:"49811265.7600",r2_out:"55028477.3100",r2:"105451116.0700",r3_in:"21042859.7400",r3_out:"24161035.0300",r3:"45598029.7700",curr_capital:"76289",name:"春兴精工",trade:"7.7900",changeratio:"-0.0126743",volume:"37763445.0000",turnover:"495.003",r0x_ratio:"-113.174",netamount:"-39971298.3900"})
({r0_in:"41516334.0000",r0_out:"7692945.0000",r0:"51993879.0000",r1_in:"68831362.0400",r1_out:"68346240.5900",r1:"138288710.6300",r2_in:"53052223.2200",r2_out:"58870981.9900",r2:"114373675.8700",r3_in:"17638991.0100",r3_out:"21055345.8000",r3:"39940694.0900",curr_capital:"100518",name:"华工科技",trade:"18.5900",changeratio:"0.0242424",volume:"18610355.0000",turnover:"185.145",r0x_ratio:"76.163",netamount:"25073396.8900"})
```
QQ请求样例:
```
http://qt.gtimg.cn/q=ff_sz300543
```
数据：???
返回样例:
```
v_ff_sz300543="sz300543~0.00~0.00~0.00~0.00~2242.79~2332.08~-89.28~-3.83~47820000.00~228.39~1214.94~朗科智能~20190919~20190918^0.00^72.96~20190917^81.29^405.17~20190916^67.60^321.23~20190912^79.50^415.58~0.00~1883300.00~20190919141120";
```

# References

https://github.com/HarrisonXi/AStock/blob/master/%E5%AE%9E%E6%97%B6%E8%A1%8C%E6%83%85API.md
