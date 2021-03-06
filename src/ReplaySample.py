from __future__ import division
from abc import ABC, abstractmethod

from Application import listAllFiles
from datetime import datetime, timedelta

import sys, os, re, random
import h5py, bz2
import numpy as np

SAMPLE_FLOAT = 'float16'
# float32(single-preccision) -3.4e+38 ~ 3.4e+38, float16(half~) 5.96e-8 ~ 6.55e+4, float64(double-preccision)
CLASSIFY_INT = 'int8'

RFGROUP_PREFIX = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

H5DSET_DEFAULT_ARGS={ 'compression': 'lzf' } # note lzf is good at speed, but HDFExplorer doesn't support it. Change it to gzip if want to view

########################################################################
def classifyGainRates_level6(gain_rates, interestDays=[0,1,2,4]) : # [0,1,2,4] to measure day0,day1,day2,day4 within a week, interestDays=[0,1,2,-1]) :
    '''
    @param gain_rates: a 2d metrix: [[gr_day0, gr_day1, gr_day2 ... gr_dayN], ...]
    @return np.array of gain-classes
    '''
    gainRates = np.array(gain_rates).astype(SAMPLE_FLOAT) #  'gain_rates' is a list here
    days = gainRates.shape[1]
    daysOfcol2 =2 # = days-1
    gainRates = gainRates[:, interestDays] # = gainRates[0,1, daysOfcol2]] # we only interest day0, day1 and dayN
    # dailize the gain rate, by skipping day0 and day1
    for i in range(len(interestDays)):
        if interestDays[i] < 0: interestDays[i] += days # covert last(-1) to real index
        if interestDays[i] > 1:
            gainRates[:, i] = gainRates[:, i] /daysOfcol2
    
    # # scaling the gain rate to fit in [0,1) : 0 maps -2%, 1 maps +8%
    # SCALE, OFFSET =10, 0.02
    # gainRates = (gainRates + OFFSET) *SCALE
    # gainRates.clip(0.0, 1.0)``

    LC = [-1000, -2.0, 0.5, 1.0, 3.0, 5.0, 100 ] # by %, -1000 means -INF, +1000= +INF
    gainClasses = np.zeros(shape=(gainRates.shape[0], (len(LC) -1) *len(interestDays))).astype(CLASSIFY_INT) # 3classes for day0: <1%, 1~5%, >5%
    for i in range(len(LC) -1):
        for j in range(len(interestDays)):
            d = interestDays[j]
            C = np.where((gainRates[:, j] > LC[i]/100.0) & (gainRates[:, j] <= LC[i+1]/100.0))
            gainClasses[C, j*(len(LC)-1) +i] =1

    return gainClasses

# -------------------------------------------------------------------
def classifyGainRates_screeningTplus1(gain_rates) : # just for screening after day-close
    '''
    @param gain_rates: a 2d metrix: [[gr_day0, gr_day1, gr_day2 ... gr_dayN], ...]
    @return np.array of gain-classes
    '''
    gainRates = np.array(gain_rates).astype(SAMPLE_FLOAT) #  'gain_rates' is a list here
    days = gainRates.shape[1]
    gainRates = gainRates[:, [0, 1, 2]] # we only interest day0, day1 and day2

    gainClasses = np.zeros(shape=(gainRates.shape[0], 8)).astype(CLASSIFY_INT) # reserved for 8 classes/attrs
    
    # attr-0~2: no profit cases that should eliminate or sell positions
    # attr-0. day1 gr<=-0.05%
    C = np.where(gainRates[:, 1] <= -0.005)
    gainClasses[C, 0] =1
    # attr-1. day2 <day1
    C = np.where(gainRates[:, 2] < gainRates[:, 1])
    gainClasses[C, 1] =1
    # attr-2. day2 <=1%
    C = np.where(gainRates[:, 2] <= 0.01)
    gainClasses[C, 2] =1

    # # class-3~6: maybe good to buy tomorrow
    # # class-3: 1% < day2 <=3% 
    # C = np.where((gainRates[:, 2] > 0.01) & (gainRates[:, 2] <=0.03))
    # gainClasses[C, 3] =1
    # # class-4. 3%< day2 <=5%
    # C = np.where((gainRates[:, 2] > 0.03) & (gainRates[:, 2] <=0.05))
    # gainClasses[C, 4] =1
    # # class-5. 5%< day2 <=8%
    # C = np.where((gainRates[:, 2] > 0.05) & (gainRates[:, 2] <=0.08))
    # gainClasses[C, 5] =1
    # # class-6. day2 >8%
    # C = np.where(gainRates[:, 2] > 0.08)
    # gainClasses[C, 6] =1
    
    # version2, simplized
    C = np.where((gainRates[:, 1] > 0.01) & (gainRates[:, 1] <=0.03))
    gainClasses[C, 3] =1
    C = np.where((gainRates[:, 1] > 0.03))
    gainClasses[C, 4] =1
    C = np.where((gainRates[:, 2] > 0.02) & (gainRates[:, 2] <=0.05))
    gainClasses[C, 5] =1
    C = np.where(gainRates[:, 2] > 0.05)
    gainClasses[C, 6] =1

    # attr-7: optional about today for in-day-trade
    C = np.where((gainRates[:, 0] >=0.01) & ((gainRates[:, 0] + gainRates[:, 1]) >0.03))
    gainClasses[C, 7] =1

    return gainClasses

########################################################################
def balanceSamples(frameDict, nameSample, nameClassifyBy, maxOverMin =-1.0):
    '''
        balance the samples, usually reduce some action=HOLD, which appears too many
    '''
    chunk_Classes = np.array(frameDict[nameClassifyBy])

    AD = np.where(chunk_Classes >=0.99) # to match 1 because action is float read from RFrames
    kI = [np.count_nonzero(AD[1] ==i) for i in range(chunk_Classes.shape[1])] # counts of each actions in frame
    idxToDel = []
    if maxOverMin >0.0:
        kImax = int(min(kI) *(1 + maxOverMin))
        for i in range(len(kI)):
            cToReduce = kI[i] -kImax
            if cToReduce <=0: continue
            
            idxItems = np.where(AD[1] ==i)[0].tolist()
            random.shuffle(idxItems)
            del idxItems[cToReduce:]
            idxToDel += [int(x) for x in idxItems] 
    else:
        kImax = max(kI)
        idxMax = kI.index(kImax)
        cToReduce = kImax - int(1.6*(sum(kI) -kImax))
        if cToReduce >0:
            idxItems = np.where(AD[1] ==idxMax)[0].tolist()
            random.shuffle(idxItems)
            del idxItems[cToReduce:]
            idxToDel = [int(i) for i in idxItems]

    if len(idxToDel) >0:
        frameDict[nameClassifyBy] = np.delete(frameDict[nameClassifyBy], idxToDel, axis=0)
        frameDict[nameSample] = np.delete(frameDict[nameSample], idxToDel, axis=0)
    return len(frameDict[nameClassifyBy])

# -------------------------------------------------------------------
def concateH5Samples(filenameOut, filenameIns, compress=True, balancing=False, maxOverMin=1.0, samplesPerFrame=2*1024, skipFirsts=0, stateChannels=-1) :
    if isinstance(filenameIns, str):
        filenameIns = filenameIns.split(',')

    if not filenameOut and len(filenameIns)>0:
        filenameOut = filenameIns[0]
        if balancing:
            filenameOut += 'b' if '.h5' == filenameOut[-3:] else '.h5b'
        else:
            filenameOut += '_cat.h5'

    EXPORT_SIGNATURE= 'concated'
    dsargs={}
    if compress :
        dsargs['compression'] = 'lzf' # 'gzip' for HDFExplorer

    listFnIn = []
    for pathname in filenameIns:
        try :
            if os.path.isdir(pathname):
                for fn2 in listAllFiles(pathname, fileOnly=True) :
                    if '.h5b' != fn2[-4:] and '.h5' != fn2[-3:] : continue
                    if balancing and '.h5' != fn2[-3:] : continue
                    listFnIn.append(fn2)
            elif os.path.isfile(pathname):
                if '.h5b' != pathname[-4:] and '.h5' != pathname[-3:] : continue
                if balancing and '.h5' != pathname[-3:] : continue
                listFnIn.append(pathname)
        except :
            pass
    
    listFnIn = list(set(listFnIn))
    listFnIn.sort()
    print("concating h5 files w/ balancing[%s] into %s: %s" % (balancing, filenameOut, ','.join(listFnIn)))

    with h5py.File(filenameOut, 'w') as h5out:
        frmId, frmState, frmAction=0, None, None
        fnIns = []

        def _saveFrameToFile(h5out, col_state, col_action, frmId) :
            AD = np.where(col_action >=0.99)
            kIout = [np.count_nonzero(AD[1] ==i) for i in range(3)]

            frmName ='%s%06d' % (RFGROUP_PREFIX2, frmId)
            g = h5out.create_group(frmName)
            g.create_dataset(u'title', data= 'compressed replay frame[%s]' % (frmId))
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs[u'default'] = 'state'
            g.attrs['size'] = col_state.shape[0]
            g.attrs['signature'] = EXPORT_SIGNATURE

            st = g.create_dataset('state', data= col_state, **dsargs)
            st.attrs['dim'] = col_state.shape[1]
            ac = g.create_dataset('action', data= col_action, **dsargs)
            ac.attrs['dim'] = col_action.shape[1]
            return 'outfrm[%s] %d samples saved, count-of-actions[%s]' % (frmName, len(col_state), ','.join([str(x) for x in kIout]))

        for fn in listFnIn:
            with h5py.File(fn, 'r') as h5f:
                framesInHd5 = []
                frmId_Ins = []
                for name in h5f.keys() :
                    if RFGROUP_PREFIX != name[:len(RFGROUP_PREFIX)] and RFGROUP_PREFIX2 != name[:len(RFGROUP_PREFIX2)] :
                        continue
                    framesInHd5.append(name)

                framesInHd5.sort()
                if skipFirsts>0:
                    del framesInHd5[:skipFirsts]
                print("taking frames in %s: %s" % (fn, ','.join(framesInHd5)))
                frmId_Ins.append('%s:' % fn)
                for name in framesInHd5:
                    frm = h5f[name]
                    frmId_Ins.append('%s(%d),' %(name, len(frm['state'])))
                    col_state  = np.array(list(frm['state']))
                    col_action = np.array(list(frm['action']))

                    if stateChannels <=0 : 
                        stateChannels = col_state.shape[-1]
                    elif stateChannels > col_state.shape[-1] :
                        print("%s skipped %s as its state-chs %d less than requested %s" % (fn, name, col_state.shape[-1], stateChannels))
                        continue
                    elif stateChannels >0 and stateChannels < col_state.shape[-1] :
                        # buildup the shape
                        shapelen = len(col_state.shape)
                        if 5 == shapelen:
                            col_state = col_state[:,:,:,:,:stateChannels]
                        elif 4 == shapelen:
                            col_state = col_state[:,:,:,:stateChannels]
                        elif 3 == shapelen:
                            col_state = [ x[:,:stateChannels] for x in col_state ]
                        elif 2 == shapelen:
                            col_state = [ x[:stateChannels] for x in col_state ]

                    if frmState is None:
                        lenBefore =0
                        frmState  = col_state
                        frmAction = col_action
                    else :
                        lenBefore = len(frmState)
                        frmState  = np.concatenate((frmState,  col_state), axis=0)
                        frmAction = np.concatenate((frmAction, col_action), axis=0)

                    lenAfter = len(frmState)
                    if lenAfter != len(frmAction):
                        print("ERROR: sizeof frmState(%d)/frmAction(%d) doesn't match" % (lenAfter, len(frmAction)))
                        exit(2)

                    if balancing:
                        frameDict = {
                            'state':  frmState,
                            'action': frmAction
                        }

                        balanceSamples(frameDict, 'state', 'action', maxOverMin=maxOverMin)
                        frmState, frmAction = frameDict['state'], frameDict['action']

                    while len(frmState) >= samplesPerFrame :
                        col_state = frmState[:samplesPerFrame]
                        col_action = frmAction[:samplesPerFrame]
                        frmState  = frmState[samplesPerFrame:]
                        frmAction = frmAction[samplesPerFrame:]

                        txndesc = _saveFrameToFile(h5out, col_state, col_action, frmId)
                        print("%s %d-state-chs balanced[%s] pending %s samples, read from: %s" % (txndesc, stateChannels, balancing, len(frmState), ''.join(frmId_Ins)))
                        frmId +=1
                        frmId_Ins =[]

        if not frmState is None and len(frmState) >= 0:
            print("%s, %d-state-chs last frm, read from: %s" % (_saveFrameToFile(h5out, frmState, frmAction, frmId), stateChannels, ''.join(frmId_Ins)))

    print("concated into file %s" % (filenameOut ))

########################################################################
if __name__ == '__main__':

    # sys.argv = [sys.argv[0]] + '-c -l 6 -o /mnt/e/temp/aaa.h5b -b -z /mnt/e/AShareSample/ETF.2013-2019/RFrmD4M1X5/RFrmD4M1X5_SH510050.h5'.split(' ')
    if '-c' in sys.argv :
        idx = sys.argv.index('-c')
        outfn, balancing, compress, skips, statechs = None, False, False, 0, -1
        if '-b' in sys.argv :
            balancing = True
            idx = max(idx, sys.argv.index('-b'))

        if '-s' in sys.argv :
            skips = 1
            idx = max(idx, sys.argv.index('-s'))

        if '-z' in sys.argv :
            compress = True
            idx = max(idx, sys.argv.index('-z'))

        if '-l' in sys.argv :
            i = sys.argv.index('-l')
            idx = max(idx, i)
            if i >0 and i < len(sys.argv) -1:
                statechs = int(sys.argv[i+1])
                idx = max(idx, i+1)

        if '-o' in sys.argv :
            i = sys.argv.index('-o')
            idx = max(idx, i)
            if i >0 and i < len(sys.argv) -1:
                outfn = sys.argv[i+1]
                idx = max(idx, i+1)

        concateH5Samples(filenameOut=outfn, filenameIns=sys.argv[idx+1:], compress=compress, balancing=balancing, skipFirsts=skips, stateChannels=statechs)
        quit()
