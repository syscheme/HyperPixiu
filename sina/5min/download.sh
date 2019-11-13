for i in `cat numsymb.txt`; do curl -o ${i}.json "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=${i}&scale=5&ma=no&datalen=1000" ; done
