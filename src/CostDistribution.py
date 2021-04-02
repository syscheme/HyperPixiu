# encoding: UTF-8
from __future__ import division

import pandas as pd
import copy

########################################################################
class CostDistribution():

    def __init__(self):
        self._priceBucketsRecent = {} # the recent volume-in-priceBuckets
        self._priceBucketsSofar = {}  # map of date to pricebuckets

        self._priceBucketsRecent_R0 = {} # about the R0
        self._priceBucketsSofar_R0 = {}

    def __calcByMA(self, dateT, pricebuckets, volT, volR0, turnoverRateT, decayRate, priceTick):
        '''
        以当日的换手筹码在当日的最高价和最低价之间平均分布来计算
        '''
        volPerBucket, vR0PerBucket = volT / len(pricebuckets), volR0 / len(pricebuckets)

        # decay the volume in the previous price-buckets
        for i in self._priceBucketsRecent.keys():
            self._priceBucketsRecent[i] *= (1 -turnoverRateT * decayRate)

        # append the recent price-buckets
        for i in pricebuckets:
            if i not in self._priceBucketsRecent or self._priceBucketsRecent[i] <0:
                self._priceBucketsRecent[i] = 0
            
            self._priceBucketsRecent[i] = volPerBucket * turnoverRateT * decayRate

        self._priceBucketsSofar[dateT] = copy.deepcopy(self._priceBucketsRecent)

    def __calcByTriangle(self, dateT, pricebuckets, volT, volR0, avgT, turnoverRateT, decayRate, priceTick):
        '''
        以当日的换手筹码在当日的最高价、最低价和平均价之间三角形分布来计算
        '''
        # 计算今日的筹码分布, 极限法分割去逼近
        volPerBucket, vR0PerBucket = volT / len(pricebuckets), volR0 / len(pricebuckets)
        tmpChip = {}
        lowT, highT = pricebuckets[0], pricebuckets[-1]
        h = 2 / (highT - lowT)
        pdLow, pdHigh = (avgT - lowT), (highT - avgT)
        for i in pricebuckets:
            x1 = i
            x2 = i + priceTick
            if i < avgT:
                y1 = h /pdLow * (x1 - lowT)
                y2 = h /pdLow * (x2 - lowT)
            else:
                y1 = h /pdHigh *(highT - x1)
                y2 = h /pdHigh *(highT - x2)

            s = priceTick *(y1 + y2) /2
            tmpChip[i] = [s*volT, s*volR0]

        # decay the volume in the previous price-buckets
        for i in self._priceBucketsRecent.keys():
            for j in range(2):
                self._priceBucketsRecent[i][0] *= (1 -turnoverRateT * decayRate)

        for i in tmpChip.keys():
            if i not in self._priceBucketsRecent.keys() or len(self._priceBucketsRecent[i]) <=0:
                self._priceBucketsRecent[i] = [0,0]

            for j in range(2):
                self._priceBucketsRecent[i][j] += tmpChip[i][j] *(turnoverRateT * decayRate)

            # if self._priceBucketsRecent[i][1] <0:
            #     self._priceBucketsRecent[i][1] =0

        self._priceBucketsSofar[dateT] = copy.deepcopy(self._priceBucketsRecent)

    def buildup(self, pdata, method='triangle', decayRate=1, priceTick =0.01): #triangle
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

            pricebuckets =[]
            steps = (highT - lowT) / priceTick
            for i in range(int(steps)):
                pricebuckets.append(round(lowT + i * priceTick, 3))
            if len(pricebuckets) <=0 or highT > pricebuckets[-1]:
                pricebuckets.append(highT)

            if method in ['ma', 'movingaverage']:
                self.__calcByMA(dateT, pricebuckets, volT, volR0, turnoverRateT, decayRate=decayRate, priceTick =priceTick)
            else: # default triangle, elif 'triangle' == method:
                self.__calcByTriangle(dateT, pricebuckets, volT, volR0, avgT, turnoverRateT, decayRate=decayRate, priceTick =priceTick)

    def calcWinRatios(self, priceBy):
        winRatios = []

        if isinstance(priceBy, list):
            priceBy =  [ float(x) for x in priceBy]
        else: priceBy = [float(priceBy)]

        for dt, buckets in self._priceBucketsSofar.items():
            # 计算目前的比例
            vol_total = 0
            vol_win = 0

            price = priceBy[len(winRatios)] if len(winRatios) <len(priceBy) else priceBy[-1]
            for k, v in buckets.items():
                vol_total += v
                if k < price: 
                    vol_win += v

            if vol_total > 0:
                win_ratio = vol_win / vol_total
            else:
                win_ratio = 0

            winRatios.append(win_ratio)

        # import matplotlib.pyplot as plt
        # dates = list(self._priceBucketsSofar.keys())
        # plt.plot(dates[len(dates) - 200:-1], winRatios[len(dates) - 200:-1])
        # plt.show()
        return winRatios

    def costOfLowers(self, lowerPercent):
        if lowerPercent > 1.0:
            lowerPercent = lowerPercent / 100  # 转换成百分比
        
        result = []

        dates = list(self._priceBucketsSofar.keys())
        dates.sort()
        for dt in dates:
            priceBuckets = self._priceBucketsSofar[dt]
            prices = sorted(priceBuckets.keys())  # 排序
            sumOfDay = 0    # 所有筹码的总和
            
            # calc the sumOfDay
            sumOfDay = sum(list(priceBuckets.values()))
            volLower = sumOfDay * lowerPercent
            vol = 0

            for j in prices:
                vol += priceBuckets[j]
                if vol > volLower:
                    result.append(j)
                    break

        import matplotlib.pyplot as plt
        plt.plot(dates[len(dates) - 1000:-1], result[len(dates) - 1000:-1])
        plt.show()
        return result

if __name__ == "__main__":
    pdata = pd.read_csv('/mnt/e/AShareSample/SZ002008/SZ002008_1d20210129.csv')
    cd= CostDistribution()
    cd.buildup(pdata, decayRate=1) #计算
    cd.calcWinRatios(pdata['close'].tolist()) #获利
    # cd.calcWinRatios(pdata['close'].iat[-1]) #获利
    cd.costOfLowers(90) #成本分布
