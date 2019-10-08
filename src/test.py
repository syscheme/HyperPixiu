from marketdata.mdBackEnd import MarketData
from marketdata.mdSina import mdSina
from datetime import datetime

if __name__ == '__main__':
    md = mdSina(None, None);
    result = md.searchKLines("000002", MarketData.EVENT_KLINE_5MIN, datetime.strptime("2019-09-20 10:00", "%Y-%m-%d %H:%M"))
    print(result)

