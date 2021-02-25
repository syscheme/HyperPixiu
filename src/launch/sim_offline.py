# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from Simulator import *
from Account import Account_AShare
from Application import *
import HistoryData as hist
from crawler.producesSina import Sina_Tplus1, populateMuxFromWeekDir
from advisors.dnn import DnnAdvisor
import h5tar

import sys, os, platform, re
from io import StringIO
import random

RFGROUP_PREFIX  = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

########################################################################
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
                for fn2 in hist.listAllFiles(pathname, fileOnly=True) :
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
    print("concat h5 files w/ balancing[%s] into %s: %s" % (balancing, filenameOut, ','.join(listFnIn)))

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

                        hist.balanceSamples(frameDict, 'state', 'action', maxOverMin=maxOverMin)
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

    # sys.argv += ['-z', '-b', '/mnt/e/h5_to_h5b/RFrmD4M1X5_SZ159949.h5']

    # concateH5Samples('/mnt/e/tmp.h5', '/tmp/RFrm2dImg32x18C8_SH600019.h5', balancing=True, compress=True, skipFirsts=0) # , stateChannels=4)
    # concateH5Samples('/mnt/e/tmp2.h5', '/mnt/d/wkspaces/HyperPixiu/out/SH60000/RFrm2dImg32x18C8_SH600006.h5', compress=True, skipFirsts=0) # , stateChannels=6)
    # exit(0)
    # sys.argv = [sys.argv[0]] + '-c -l 6 -o /mnt/h/RFrm2dImg32x18C8_0222trial/RFrm2dImg32x18C8_0222trial_C4.h5b -b -z /mnt/h/RFrm2dImg32x18C8_0222trial/RFrm2dImg32x18C8_SH600276.h5'.split(' ')
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
    MF1d_toAdd = []
    # MF1d_toAdd = ['/mnt/e/AShareSample/SinaMF1d_20200620.h5t', '/mnt/e/AShareSample/Sina2021W/Sina2021W01_0104-0108.h5t']
    # MF1d_toAdd = ['/mnt/e/AShareSample/Sina2021W/Sina2021W01_0104-0108.h5t']
    
    if os.path.isdir(evMdSource) :
        try :
            os.stat(os.path.join(evMdSource, 'h5tar.py'))
            histReader = populateMuxFromWeekDir(p, evMdSource, symbol=SYMBOL)
        except Exception as ex:
            # csvPlayback can only cover one symbol
            p.info('taking CsvPlayback on dir %s for symbol[%s]' % (evMdSource, SYMBOL))
            histReader = hist.CsvPlayback(program=p, symbol=SYMBOL, folder=evMdSource, fields='date,time,open,high,low,close,volume,ammount')
            if not MF1d_toAdd or len(MF1d_toAdd) <=0:
                try :
                    fnMF1d = os.path.realpath(os.path.join(evMdSource, 'SinaMF1d.h5t'))
                    os.stat(fnMF1d)
                    MF1d_toAdd = [ fnMF1d ]
                except: pass

            for mf in MF1d_toAdd:
                try :
                    os.stat(mf)
                except: continue

                bn = os.path.basename(mf)
                pb = None
                if 'SinaMF1d_' in bn and 'h5t' == bn.split('.')[-1]:
                    memfn = bn.split('.')[0].split('_')[-1]
                    memfn = '%s_MF1d%s.csv' % (SYMBOL, memfn)
                    lines = h5tar.read_utf8(mf, memfn)
                    if len(lines) <=0: continue

                    pb = hist.CsvStream(SYMBOL, StringIO(lines), MoneyflowData.COLUMNS, evtype=EVENT_MONEYFLOW_1DAY, program=p)
                    pb.setId('%s@%s' % (memfn, mf))
                else:
                    m = re.match(r'Sina([0-9]*)W[0-9]*_([0-9]*)-([0-9]*).h5t', bn)
                    if not m: continue
                    mlst = h5tar.list_utf8(mf)
                    memfn = None
                    for mem in mlst:
                        if 'MF1d' in mem['name'] and SYMBOL in mem['name']:
                            memfn = mem['name']
                            break
                    if not memfn: continue
                    lines = h5tar.read_utf8(mf, memfn)
                    if len(lines) <=0: continue

                    pb = hist.CsvStream(SYMBOL, StringIO(lines), MoneyflowData.COLUMNS, evtype=EVENT_MONEYFLOW_1DAY, program=p)
                    pb.setId('%s@%s' % (memfn, mf))
                
                if not pb: continue
                if not isinstance(histReader, hist.PlaybackMux):
                    mux = hist.PlaybackMux(program=p)
                    mux.addStream(histReader)
                    histReader = mux

                histReader.addStream(pb)
                p.info('mux-ed MF1d[%s]' % (pb.id))

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
