# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from advisors.dnn import DnnAdvisor_S1548I4A3

import sys, os, platform
RFGROUP_PREFIX = 'ReplayFrame:'
OUTFRM_SIZE = 8*1024
import random

def balanceSamples(filepathRFrm, compress=True) :
    '''
    read a frame from H5 file
    '''
    dsargs={}
    if compress :
        dsargs['compression'] = 'gzip'

    print("balancing samples in %s to %sb" % (filepathRFrm, filepathRFrm))
    with h5py.File(filepathRFrm+'b', 'w') as h5out:
        frmId=0
        frmState=None
        frmAction=None

        with h5py.File(filepathRFrm, 'r') as h5f:
            framesInHd5 = []
            for name in h5f.keys() :
                if RFGROUP_PREFIX == name[:len(RFGROUP_PREFIX)] :
                    framesInHd5.append(name)

            framesInHd5.sort()

            for frmName in framesInHd5 :
                print("opening frm[%s]" % frmName)
                frm = h5f[frmName]
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
                cRowToKeep = max(kI[1:]) + sum(kI[1:]) # = max(kI[1:]) *3
                idxHolds = np.where(AD[1] ==0)[0].tolist()
                cHoldsToDel = len(idxHolds) - (cRowToKeep - sum(kI[1:]))
                if cHoldsToDel>0 :
                    random.shuffle(idxHolds)
                    del idxHolds[cHoldsToDel:]
                    idxToDel = [lenBefore +i for i in idxHolds]
                    frmState = np.delete(frmState, idxToDel, axis=0)
                    frmAction = np.delete(frmAction, idxToDel, axis=0)

                print("frm[%s] processed, pending %s" % (frmName, len(frmState)))
                if len(frmState) >= OUTFRM_SIZE:
                    col_state = frmState[:OUTFRM_SIZE]
                    col_action = frmAction[:OUTFRM_SIZE]
                    frmState  = frmState[OUTFRM_SIZE:]
                    frmAction = frmAction[OUTFRM_SIZE:]

                    g = h5out.create_group('%s%s' % (RFGROUP_PREFIX, frmId))
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
                    print("outfrm[%s] saved, pending %s" % (frmId, len(frmState)))
                    frmId +=1

            if len(frmState) >= 0:
                col_state = frmState
                col_action = frmAction
                g = h5out.create_group('%s%s' % (RFGROUP_PREFIX, frmId))
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

                print("lastfrm[%s] saved, size %s" % (frmId, len(frmState)))

if __name__ == '__main__':

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

    evMdSource = Program.fixupPath(evMdSource)
    p.info('taking input dir %s for symbol[%s]' % (evMdSource, SYMBOL))
    # csvPlayback can only cover one symbol
    csvreader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
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
        tdrWraper = p.createApp(IdealTrader_Tplus1, configNode ='trader', trader=tdrCore, histdata=csvreader) # ideal trader to generator ReplayFrames
    else :
        p.info('all objects registered piror to local Advisor: %s' % p.listByType())
        advisor = p.createApp(DnnAdvisor_S1548I4A3, configNode ='advisor', objectives=objectives, recorder=rec)
        advisor._skipMStateStore = True # MUST!!!
        advisor._exchange = tdrCore.account.exchange

        p.info('all objects registered piror to simulator: %s' % p.listByType())
        tdrWraper = p.createApp(OfflineSimulator, configNode ='trader', trader=tdrCore, histdata=csvreader) # the simulator with brain loaded to verify training result

    tdrWraper.setRecorder(rec)

    p.start()
    p.loop()
    p.stop()
