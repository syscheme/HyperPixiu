# encoding: UTF-8
'''
This utility reads csv history data and generate ReplayFrame for offline DQN training
'''

from RemoteEvent import *
from Application import *

import sys, os, platform

if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.realpath(os.path.dirname(os.path.abspath(__file__))+ '/../../conf') + '/EventChannel.json']

    p = Program()
    p._heartbeatInterval =0.5

    chType = None
    try:
        jsetting = p.jsettings('eventChannel/type')
        if not jsetting is None:
            chType = jsetting(None)
    except:
        pass

    # TODO launch EventChannel per specified type
    # if chType == 'ZeroMQ':
    evCh  = p.createApp(ZmqEventChannel, configNode ='eventChannel')

    p.start()
    p.loop()
    
    p.stop()
