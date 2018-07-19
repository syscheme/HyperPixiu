# encoding: UTF-8

"""
导入MC导出的CSV历史数据到MongoDB中
"""

from   vnApp.Strategy.Base import MINUTE_DB_NAME
import vnApp.Strategy.HistoryData as hd


if __name__ == '__main__':
    # loadMcCsv('examples/AShBacktesting/IF0000_1min.csv', MINUTE_DB_NAME, 'IF0000')
    # hd.loadMcCsvBz2('examples/AShBacktesting/IF0000_1min.csv.bz2', MINUTE_DB_NAME, 'IF0000')
    # loadMcCsv('rb0000_1min.csv', MINUTE_DB_NAME, 'rb0000')
    
    srcDataHome=u'/bigdata/sourcedata/shop37077890.taobao/股票1分钟csv'
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2012.7-2012.12/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2012.1-2012.6/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2011.7-2011.12/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    # hd.loadTaobaoCsvBz2(srcDataHome +'/2011.1-2011.6/SH601519.csv.bz2', MINUTE_DB_NAME, 'A601519')
    folders = ['2012H2', '2012H1', '2011H2', '2011H1']
    # symbols= ["601607","601611","601618","601628","601633","601668","601669","601688",
    #     "601718","601727","601766","601788","601800","601808","601818","601828","601838","601857",
    #     "601866","601877","601878","601881","601888","601898","601899","601901","601919","601933",
    #     "601939","601958","601985","601988","601989","601991","601992","601997","601998","603160",
    #     "603260","603288","603799","603833","603858","603993" ]

    symbols= ["601000"]

    for s in symbols :
        for f in folders :
            try :
                csvfn = '%s/SH%s.csv.bz2' % (f, s)
                sym = 'A%s' % s
                hd.loadTaobaoCsvBz2(srcDataHome +'/'+csvfn, MINUTE_DB_NAME, sym)
            except :
                pass



