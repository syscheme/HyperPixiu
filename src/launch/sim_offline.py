# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from advisors.dnn import DnnAdvisor_S1548I4A3
from crawler.producesSina import Sina_Tplus1

import sys, os, platform
RFGROUP_PREFIX  = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'
OUTFRM_SIZE = 8*1024
import random

def balanceSamples(filepathRFrm, compress=True) :
    '''
    read a frame from H5 file
    '''
    dsargs={}
    if compress :
        dsargs['compression'] = 'lzf' # 'gzip' for HDFExplorer

    print("balancing samples in %s to %sb" % (filepathRFrm, filepathRFrm))
    with h5py.File(filepathRFrm+'b', 'w') as h5out:
        frmId=0
        frmState=None
        frmAction=None
        frmInName=''
        subtotal = np.asarray([0]*3)

        with h5py.File(filepathRFrm, 'r') as h5f:
            framesInHd5 = []
            for name in h5f.keys() :
                if RFGROUP_PREFIX == name[:len(RFGROUP_PREFIX)] or RFGROUP_PREFIX2 == name[:len(RFGROUP_PREFIX2)] :
                    framesInHd5.append(name)

            framesInHd5.sort()
            print("found frames in %s: %s" % (filepathRFrm, ','.join(framesInHd5)))

            for frmInName in framesInHd5 :
                print("reading frmIn[%s] from %s" % (frmInName, filepathRFrm))
                frm = h5f[frmInName]
                if frmState is None:
                    frmState = np.array(list(frm['state']))
                    frmAction = np.array(list(frm['action']))
                    lenBefore =0
                    lenAfter = len(frmState)
                else :
                    lenBefore = len(frmState)
                    a = np.array(list(frm['state']))
                    frmState = np.concatenate((frmState, a), axis=0)
                    a = np.array(list(frm['action']))
                    frmAction= np.concatenate((frmAction, a), axis=0)
                    lenAfter = len(frmState)

                npActions = frmAction[lenBefore:]
                AD = np.where(npActions >=0.99) # to match 1 because action is float read from RFrames
                kI = [np.count_nonzero(AD[1] ==i) for i in range(3)] # counts of each actions in frame
                kImax = max(kI)
                idxMax = kI.index(kImax)
                cToReduce = kImax - int(1.2*(sum(kI) -kImax))
                if cToReduce >0:
                    print("frmIn[%s] actCounts[%s,%s,%s]->evicting %d samples of max-act[%d]" % (frmInName, kI[0],kI[1],kI[2], cToReduce, idxMax))
                    idxItems = np.where(AD[1] ==idxMax)[0].tolist()
                    random.shuffle(idxItems)
                    del idxItems[cToReduce:]
                    idxToDel = [lenBefore +i for i in idxItems]
                    frmAction = np.delete(frmAction, idxToDel, axis=0)
                    frmState = np.delete(frmState, idxToDel, axis=0)

                # update the stat now
                AD = np.where(frmAction >=0.99) # to match 1 because action is float read from RFrames
                kI = [np.count_nonzero(AD[1] ==i) for i in range(3)] # counts of each actions in frame
                print("frmIn[%s] processed, pending %s actCounts[%s,%s,%s]" % (frmInName, len(frmState), kI[0],kI[1],kI[2]))

                if len(frmState) >= OUTFRM_SIZE:
                    col_state = frmState[:OUTFRM_SIZE]
                    col_action = frmAction[:OUTFRM_SIZE]
                    frmState  = frmState[OUTFRM_SIZE:]
                    frmAction = frmAction[OUTFRM_SIZE:]

                    AD = np.where(col_action >=0.99)
                    kIout = [np.count_nonzero(AD[1] ==i) for i in range(3)]
                    subtotal += np.asarray(kIout)
                    # AD = np.where(frmAction >=0.99)
                    # kI = [np.count_nonzero(AD[1] ==i) for i in range(3)]

                    frmName ='%s%s' % (RFGROUP_PREFIX, frmId)
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
                    print("outfrm[%s] actCounts[%s,%s,%s] saved, pending %s" % (frmName, kIout[0],kIout[1],kIout[2], len(frmState)))

            # the last frame
            if len(frmState) >= 0:
                col_state = frmState
                col_action = frmAction
                AD = np.where(col_action >=0.99)
                kIout = [np.count_nonzero(AD[1] ==i) for i in range(3)]
                subtotal += np.asarray(kIout)
                
                frmName ='%s%s' % (RFGROUP_PREFIX, frmId)
                frmId +=1
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

                print("lastfrm[%s] actCounts[%s,%s,%s] saved, size %s" % (frmName, kIout[0],kIout[1],kIout[2], len(col_action)))

            print("balanced %s to %sb: %s->%d frameOut, actSubtotal%s" % (filepathRFrm, filepathRFrm, frmInName, frmId, list(subtotal)))

if __name__ == '__main__':

    # sys.argv += ['-z', '-b', '/mnt/e/h5_to_h5b/RFrmD4M1X5_SZ159949.h5']

    if '-b' in sys.argv :
        idx = sys.argv.index('-b') +1
        if idx >0 and idx < len(sys.argv):
            h5fn = sys.argv[idx]
            compress = '-z' in sys.argv
            balanceSamples(h5fn, compress)
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
    evMdSource = Program.fixupPath(evMdSource)
    if 'tcsv' in os.path.basename(evMdSource):
        p.info('taking TaggedCsvPlayback on %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.TaggedCsvPlayback(program=p, symbol=SYMBOL, tcsvFilePath=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
    else:
        # csvPlayback can only cover one symbol
        p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
        histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')

    tdrWraper = None
    if 'remote' == advisorType :
        p.error('sim_offline only takes local advisor')
        quit()
        # revents = p.createApp(ZmqEE, configNode ='remoteEvents')
        # revs = [EVENT_ADVICE]
        # if not evMdSource:
        #     revs += [EVENT_TICK, EVENT_KLINE_1MIN]
        # revents.subscribe(revs)

    if 'T+1' == ideal :
        tdrWraper = p.createApp(IdealTrader_Tplus1, configNode ='trader', trader=tdrCore, histdata=histReader) # ideal trader to generator ReplayFrames
    elif 'SinaT+1' == ideal :
        tdrWraper = p.createApp(Sina_Tplus1, configNode ='trader', trader=tdrCore, symbol='SZ000001', dirOfflineData='/mnt/e/AShareSample/SinaWeek.20200629')
    elif 'FuturePrice' == ideal :
        tdrWraper = p.createApp(ShortSwingScanner, configNode ='trader', trader=tdrCore, histdata=histReader) # ShortSwingScanner to classify future price
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor_S1548I4A3, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._enableMStateSS = False # MUST!!!
        advisor._exchange = tdrCore.account.exchange

        p.info('all objects registered piror to simulator: %s' % p.listByType())
        tdrWraper = p.createApp(OfflineSimulator, configNode ='trader', trader=tdrCore, histdata=histReader) # the simulator with brain loaded to verify training result

    tdrWraper.setRecorder(rec)

    p.start()
    if tdrWraper.isActive :
        p.loop()
    p.stop()
