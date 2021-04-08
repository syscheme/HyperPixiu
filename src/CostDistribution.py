# encoding: UTF-8
from __future__ import division

import pandas as pd
import copy
import csv

########################################################################
class CostDistribution():

    def __init__(self, priceTick=0.01, decayByTurnover=1):
        self._priceBucketsRecent = {} # the recent volume-in-priceBuckets
        self._priceBucketsSofar = {}  # map of date to pricebuckets

        self._priceTick = priceTick
        self._decayByTurnover = decayByTurnover
        self._vTrim = -1

    def __decayLatest(self, decayRates):
        # decay the volume in the previous price-buckets
        for i in self._priceBucketsRecent.keys():
            for j in range(len(decayRates)):
                self._priceBucketsRecent[i][j] = int(self._priceBucketsRecent[i][j] * (1 -decayRates[j]))

    def __calcByMA(self, dateT, pricebuckets, volT, volR0, decayRate):
        '''
        以当日的换手筹码在当日的最高价和最低价之间平均分布来计算
        '''
        volPerBucket, vR0PerBucket = volT / len(pricebuckets), volR0 / len(pricebuckets)

        # decay the volume in the previous price-buckets
        self.__decayLatest([decayRate, decayRate*2])

        # append the recent price-buckets
        for i in pricebuckets:
            if i not in self._priceBucketsRecent or self._priceBucketsRecent[i][0] <0:
                self._priceBucketsRecent[i] = [0, 0]

            for j in range(2):
                self._priceBucketsRecent[i][j] += int(volPerBucket) # += int(volPerBucket * decayRate)

    def __calcByTriangle(self, dateT, pricebuckets, volT, volR0, avgT, decayRate):
        '''
        以当日的换手筹码在当日的最高价、最低价和平均价之间三角形分布来计算
        '''
        lowT, highT = pricebuckets[0], round(pricebuckets[-1] + self._priceTick, 3)
        if avgT - lowT < self._priceTick or highT - avgT <=self._priceTick:
            return self.__calcByMA(dateT, pricebuckets, volT, volR0, decayRate)

        # 计算今日的筹码分布, 极限法分割去逼近
        volPerBucket, vR0PerBucket = volT / len(pricebuckets), volR0 / len(pricebuckets)
        tmpChip = {}
        pdUp, pdDown = round((highT - lowT) / (avgT - lowT), 3), round((highT - lowT) / (highT - avgT), 3)

        for i in pricebuckets:
            if i < avgT:
                y1 = pdUp * (i - lowT)
                y2 = pdUp * (i + self._priceTick - lowT)
            else:
                y1 = pdDown *(highT - i)
                y2 = pdDown *(highT - i - self._priceTick)

            s = self._priceTick *(y1 + y2)
            tmpChip[i] = [s*volT, s*volR0]

        # decay the volume in the previous price-buckets
        self.__decayLatest([decayRate, decayRate*2])

        for i in tmpChip.keys():
            if i not in self._priceBucketsRecent.keys() or len(self._priceBucketsRecent[i]) <=0:
                self._priceBucketsRecent[i] = [0,0]

            for j in range(2) :
                self._priceBucketsRecent[i][j] += int(tmpChip[i][j]) # +=int(tmpChip[i][j] *(decayRate))

    def buildup(self, pdata, method='triangle') : # triangle
        '''
        AC 衰减系数, 既当日被移动的成本的权重: 
            如果今天的换手率是t，衰减系数是d，那么我们计算昨日的被移动的筹码的总量是t*d,
            如果d取值为1，就是一般意义上理解的今天换手多少，就有多少筹码被从昨日的成本分布中被搬移；
            如果d是2，那么我们就放大了作日被移动的筹码的总量.. 这样的目的在于突出“离现在越近的筹码分布其含义越明显”
        '''
        method = str(method).lower()

        date = pdata['date']
        low = pdata['low']
        high = pdata['high']
        vol = pdata['volume']
        turnoverRate = pdata['turnoverRate'] # /=100 # 东方财富的小数位要注意，兄弟萌。我不除100懵逼了
        avg = pdata['price']

        vr0 = vol * pdata['ratioR0']

        for i in range(len(date)):
            dateT = date[i]
            # print(dateT)

            # if i < 90: continue
            highT = high[i]
            lowT = low[i]
            volT = vol[i]
            avgT = avg[i]
            turnoverRateT = turnoverRate[i]
            volR0 = vr0[i]
            if self._vTrim <0 and turnoverRateT >0 and volT >0:
                self._vTrim = int(volT * max(0.001, turnoverRateT/20000))

            steps = int((highT - lowT) / self._priceTick)
            pricebuckets = [round(lowT + i * self._priceTick, 3) for i in range(steps)] if steps >0 else [round(lowT, 3)]

            if method in ['ma', 'movingaverage']:
                self.__calcByMA(dateT, pricebuckets, volT, volR0, turnoverRateT *self._decayByTurnover)
            else: # default triangle, elif 'triangle' == method:
                self.__calcByTriangle(dateT, pricebuckets, volT, volR0, avgT, turnoverRateT *self._decayByTurnover)

            # trim at both ends
            if self._vTrim >0:
                pricebuckets = list(self._priceBucketsRecent.keys())
                pricebuckets.sort()
                for p in pricebuckets:
                    if p in self._priceBucketsRecent and self._priceBucketsRecent[p][0] >= self._vTrim:
                        break
                    del self._priceBucketsRecent[p]

                pricebuckets.reverse()
                for p in pricebuckets:
                    if p in self._priceBucketsRecent and self._priceBucketsRecent[p][0] >= self._vTrim:
                        break
                    del self._priceBucketsRecent[p]

            self._priceBucketsSofar[dateT] = copy.deepcopy(self._priceBucketsRecent)

    def saveCsv(self, filename): 
        with open(filename, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(['asof', 'price', 'v0', 'v1'])
            for asof, disb in self._priceBucketsSofar.items():
                for p, vs in disb.items():
                    row = [asof, p] +vs
                    write.writerow(row)        

    def calcRatioInCostRange(self, costHigh, costLow=None):
        result = []

        if isinstance(costHigh, list):
            costHigh =  [ float(x) for x in costHigh]
        else: costHigh = [float(costHigh)]

        if not costLow:
            costLow = [0.0]
        elif isinstance(costLow, list):
            costLow =  [ float(x) for x in costLow]
        else : costLow = [float(costLow)]

        for dt, buckets in self._priceBucketsSofar.items():
            # 计算目前的比例
            vol_total = 0
            vol_hit = [ 0, 0]

            costH = costHigh[len(result)] if len(result) <len(costHigh) else costHigh[-1]
            costL = costLow[len(result)] if len(result) <len(costLow) else 0.0
            for k, v in buckets.items():
                vol_total += v[0]
                if k <= costH and k >=costL: 
                    for i in range(len(vol_hit)):
                        vol_hit[i] += v[i]

            if vol_total > 0:
                ratio_hit = [ round(x/vol_total, 5) if x>0 else 0 for x in vol_hit ]
            else:
                ratio_hit = [0.0] * len(vol_hit)

            result.append([dt] + ratio_hit)

        # import matplotlib.pyplot as plt
        # dates = list(self._priceBucketsSofar.keys())
        # plt.plot(dates[len(dates) - 200:-1], result[len(dates) - 200:-1])
        # plt.show()
        return result

    def costOfLowers(self, lowerPercent, idxDist=0):
        if lowerPercent > 1.0:
            lowerPercent = lowerPercent / 100.0  # 转换成百分比
        
        if not idxDist or idxDist <0: idxDist=0
        result = []

        dates = list(self._priceBucketsSofar.keys())
        dates.sort()
        for dt in dates:
            priceBuckets = self._priceBucketsSofar[dt]
            prices = sorted(priceBuckets.keys())  # 排序
            sumOfDay = 0    # 所有筹码的总和
            
            # calc the sumOfDay
            sumOfDay = sum([ x[idxDist] for x in priceBuckets.values()])
            volLower = sumOfDay * lowerPercent
            vol = 0

            for j in prices:
                vol += priceBuckets[j][idxDist]
                if vol > volLower:
                    result.append(j)
                    break

        import matplotlib.pyplot as plt
        plt.plot(dates[len(dates) - 1000:-1], result[len(dates) - 1000:-1])
        plt.show()
        return result

if __name__ == "__main__":
    FOLDER='/mnt/e/AShareSample/SZ002008/'
    pdata = pd.read_csv( FOLDER + 'SZ002008_1d20210129.csv')
    cd= CostDistribution()
    cd.buildup(pdata) #计算
    # cd.saveCsv(FOLDER + 'SZ002008_1d20210129_dist.csv')
    win_ratios = cd.calcRatioInCostRange(pdata['close'].tolist()) #获利
    close2pct_ratios = cd.calcRatioInCostRange((pdata['close']*1.02).tolist(), (pdata['close']*0.98).tolist())
    with open(FOLDER + 'close2pct.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(['asof', 'H0', 'H1'])
        for row in close2pct_ratios:
            write.writerow(row)

    # cd.calcRatioInCostRange(pdata['close'].iat[-1]) #获利
    cd.costOfLowers(90) #成本分布
