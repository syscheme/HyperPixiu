# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from crawler.producesSina import Sina_Tplus1, populateMuxFromArchivedDir
from advisors.dnn import DnnAdvisor

import sys, os, platform
import random

RFGROUP_PREFIX  = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

########################################################################
def concateH5Samples(filenameOut, filenameIns, compress=True, balancing=False, maxOverMin=1.0, samplesPerFrame=4*1024, skipFirsts=0) :
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
                for fn2 in hist.listAllFiles(pathname, fileOnly=True) :
                    if '.h5b' != fn2[-4:] and '.h5' != fn2[-3:] : continue
                    if balancing and '.h5' != fn2[-3:] : continue
                    listFnIn.append(fn2)
            elif os.path.isfile(pathname):
                if '.h5b' != fn2[-4:] and '.h5' != fn2[-3:] : continue
                if balancing and '.h5' != fn2[-3:] : continue
                listFnIn.append(pathname)
        except :
            pass
    
    listFnIn = list(set(listFnIn))
    listFnIn.sort()
    print("concat h5 files w/ balancing[%s] into %s: %s" % (balancing, filenameOut, ','.join(listFnIn)))
    with h5py.File(filenameOut, 'w') as h5out:
        frmId, frmState, frmAction=0, None, None
        fnIns = []

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
                
                for name in framesInHd5:
                    frmId_Ins.append(name)
                    frm = h5f[name]
                    if frmState is None:
                        lenBefore =0
                        frmState  = np.array(list(frm['state']))
                        frmAction = np.array(list(frm['action']))
                    else :
                        lenBefore = len(frmState)
                        frmState  = np.concatenate((frmState,  list(frm['state'])), axis=0)
                        frmAction = np.concatenate((frmAction, list(frm['action'])), axis=0)

                    lenAfter = len(frmState)
                    if lenAfter != len(frmAction):
                        print("ERROR: sizeof frmState(%d)/frmAction(%d) doesn't match" % (lenAfter, len(frmAction)))
                        exit(2)

                    if balancing:
                        frameDict = {
                            'state':  frmState,
                            'action': frmAction
                        }

                        hist.balanceSamples(frameDict, 'state', 'action', maxOverMin=maxOverMin)
                        frmState, frmAction = frameDict['state'], frameDict['action']

                    if len(frmState) >= samplesPerFrame:
                        col_state = frmState[:samplesPerFrame]
                        col_action = frmAction[:samplesPerFrame]
                        frmState  = frmState[samplesPerFrame:]
                        frmAction = frmAction[samplesPerFrame:]

                        balstr = ''
                        if balancing:
                            AD = np.where(col_action >=0.99)
                            kIout = [np.count_nonzero(AD[1] ==i) for i in range(3)]
                            balstr = ' balanced classes[%s]' % ','.join([str(x) for x in kIout])

                        frmName ='%s%06d' % (RFGROUP_PREFIX2, frmId)
                        g = h5out.create_group(frmName)
                        g.create_dataset(u'title', data= 'compressed replay frame[%s]' % (frmId))
                        frmId +=1
                        g.attrs['state'] = 'state'
                        g.attrs['action'] = 'action'
                        g.attrs[u'default'] = 'state'
                        g.attrs['size'] = col_state.shape[0]
                        g.attrs['signature'] = EXPORT_SIGNATURE

                        st = g.create_dataset('state', data= col_state, **dsargs)
                        st.attrs['dim'] = col_state.shape[1]
                        ac = g.create_dataset('action', data= col_action, **dsargs)
                        ac.attrs['dim'] = col_action.shape[1]
                        print("outfrm[%s] %d samples saved,%s pending %s" % (frmName, len(col_state), balstr, len(frmState)))

            if len(frmId_Ins) >0:
                fnIns.append('%s(%d)' % (fn, len(frmId_Ins)))

        if not frmState is None and len(frmState) >= 0:
            col_state = frmState
            col_action = frmAction

            balstr = ''
            if balancing:
                AD = np.where(col_action >=0.99)
                kIout = [np.count_nonzero(AD[1] ==i) for i in range(3)]
                balstr = ' balanced classes[%s]' % ','.join([str(x) for x in kIout])

            frmName ='%s%06d' % (RFGROUP_PREFIX2, frmId)
            g = h5out.create_group(frmName)
            g.create_dataset(u'title', data= 'compressed replay frame[%s]' % (frmId))
            frmId +=1
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs[u'default'] = 'state'
            g.attrs['size'] = col_state.shape[0]
            g.attrs['signature'] = EXPORT_SIGNATURE

            st = g.create_dataset('state', data= col_state, **dsargs)
            st.attrs['dim'] = col_state.shape[1]
            ac = g.create_dataset('action', data= col_action, **dsargs)
            ac.attrs['dim'] = col_action.shape[1]
            print("outfrm[%s] last %d samples saved %s" % (frmName, len(col_state), balstr ))

    print("concated into file %s: %s" % (filenameOut, ','.join(fnIns) ))

########################################################################
if __name__ == '__main__':

    # sys.argv += ['-z', '-b', '/mnt/e/h5_to_h5b/RFrmD4M1X5_SZ159949.h5']

    # concateH5Samples('/mnt/e/tmp.h5', '/mnt/e/AShareSample/RFrm2dImg32x18C8_ETF2013-2020/', compress=False, skipFirsts=1)
    # sys.argv = [sys.argv[0]] + '-c -z -o /tmp/abc.h5b /mnt/e/AShareSample/RFrm2dImg32x18C8_ETF2013-2020/'.split(' ')
    if '-c' in sys.argv :
        idx = sys.argv.index('-c')
        outfn, balancing, compress, skips = None, False, False, 0
        if '-b' in sys.argv :
            balancing = True
            idx = max(idx, sys.argv.index('-b'))

        if '-s' in sys.argv :
            skips = 1
            idx = max(idx, sys.argv.index('-s'))

        if '-z' in sys.argv :
            compress = True
            idx = max(idx, sys.argv.index('-z'))

        if '-o' in sys.argv :
            i = sys.argv.index('-o')
            idx = max(idx, i)
            if i >0 and i < len(sys.argv) -1:
                outfn = sys.argv[i+1]
                idx = max(idx, i+1)

        concateH5Samples(filenameOut=outfn, filenameIns=sys.argv[idx+1:], compress=compress, balancing=balancing, skipFirsts=skips)
        quit()
        
    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/Trader.json']

    p = Program()
    p._heartbeatInterval =-1

    evMdSource  = p.getConfig('marketEvents/source', None) # market data event source
    advisorType = p.getConfig('advisor/type', "remote")
    ideal       = p.getConfig('trader/backTest/ideal', None) # None

    if "remote" != advisorType:
        # this is a local advisor, so the trader's source of market data event must be the same of the local advisor
        pass
        # jsetting = p.jsettings('advisor/eventSource')
        # if not jsetting is None:
        #     evMdSource = jsetting(None)

    # In the case that this utility is started from a shell script, this reads env variables for the symbols
    objectives = None
    if 'SYMBOL' in os.environ.keys():
        objectives = [os.environ['SYMBOL']]

    acc = p.createApp(Account_AShare, configNode ='account', ratePer10K =30)
    tdrCore = p.createApp(BaseTrader, configNode ='trader', objectives=objectives, account=acc)
    objectives = tdrCore.objectives
    SYMBOL = objectives[0]

    rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = os.path.join(p.outdir, '%s_P%s.tcsv' % (SYMBOL, p.pid)))
    revents = None

    # determine the Playback instance
    # evMdSource = '/mnt/e/AShareSample/Sina2021W' # TEST-CODE
    # evMdSource = '/mnt/e/AShareSample/ETF.2013-2019' # TEST-CODE
    evMdSource = Program.fixupPath(evMdSource)
    basename = os.path.basename(evMdSource)
    
    if os.path.isdir(evMdSource) :
        try :
            os.stat(os.path.join(evMdSource, 'h5tar.py'))
            histReader = populateMuxFromArchivedDir(p, evMdSource, symbol=SYMBOL)
        except:
            # csvPlayback can only cover one symbol
            p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
            histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
    elif '.tcsv' in basename :
        p.info('taking TaggedCsvPlayback on %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.TaggedCsvPlayback(program=p, symbol=SYMBOL, tcsvFilePath=evMdSource)
        histReader.setId('%s@%s' % (SYMBOL, basename))
        histReader.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        histReader.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

    elif '.tar.bz2' in basename :
        p.info('taking TaggedCsvInTarball on %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.TaggedCsvInTarball(program=p, symbol=SYMBOL, fnTarball=evMdSource, memberPattern='%s_evmd_*.tcsv' %SYMBOL )
        histReader.setId('%s@%s' % (SYMBOL, basename))
        histReader.registerConverter(EVENT_KLINE_1MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_5MIN, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_KLINE_1DAY, KLineData.hatch, KLineData.COLUMNS)
        histReader.registerConverter(EVENT_TICK,       TickData.hatch,  TickData.COLUMNS)

        histReader.registerConverter(EVENT_MONEYFLOW_1MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_5MIN, MoneyflowData.hatch, MoneyflowData.COLUMNS)
        histReader.registerConverter(EVENT_MONEYFLOW_1DAY, MoneyflowData.hatch, MoneyflowData.COLUMNS)

    tdrWraper = None
    if 'remote' == advisorType :
        p.error('sim_offline only takes local advisor')
        quit()
        # revents = p.createApp(ZmqEE, configNode ='remoteEvents')
        # revs = [EVENT_ADVICE]
        # if not evMdSource:
        #     revs += [EVENT_TICK, EVENT_KLINE_1MIN]
        # revents.subscribe(revs)

    # ideal ='T+1' #TEST-CODE

    if 'T+1' == ideal :
        tdrWraper = p.createApp(IdealTrader_Tplus1, configNode ='trader', trader=tdrCore, histdata=histReader) # ideal trader to generator ReplayFrames
    elif 'SinaT+1' == ideal :
        tdrWraper = p.createApp(Sina_Tplus1, configNode ='trader', trader=tdrCore, symbol='SZ000001', dirOfflineData='/mnt/e/AShareSample/SinaWeek.20200629')
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._enableMStateSS = False # MUST!!!
        advisor._exchange = tdrCore.account.exchange

        p.info('all objects registered piror to simulator: %s' % p.listByType())
        tdrWraper = p.createApp(OfflineSimulator, configNode ='trader', trader=tdrCore, histdata=histReader) # the simulator with brain loaded to verify training result

    tdrWraper.setRecorder(rec)

    p.start()
    if tdrWraper.isActive :
        p.loop()
    p.stop()
