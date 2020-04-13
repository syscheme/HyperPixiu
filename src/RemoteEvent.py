# encoding: UTF-8
'''
EventProxy to remote ZeroMQ
'''
from __future__ import division

from EventData    import *
from Application  import BaseApplication

import os
import json   # to save params
from datetime import datetime, timedelta
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
import traceback

########################################################################
class EventProxy(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(EventProxy, self).__init__(program, **kwargs)

        # 事件队列
        self.__queueTo = Queue()
        self.__conn = None
        self.__evTypesToFwd = []
        self.__evTypesToRecv = []

    @abstractmethod
    def send(self, event):
        pass

    @abstractmethod
    def recv(self, secTimeout=0.1):
        return None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev):
        '''
        dispatch the event
        '''
        if not ev.type in self.__evTypesToFwd:
            return
        
        self.__queueTo.put(ev)

    def doAppStep(self):
        if not super(TradeAdvisor, self).doAppInit() :
            return False
        
        # step 1. forward the outgoing events
        ev = True # dummy
        cSent, cRecv =0, 0
        while ev and cSent <10:
            ev = None
            try :
                ev = self.__queueTo.get(block = enabledHB, timeout = 0.1)  # 获取事件的阻塞时间设为0.1秒
                if ev :
                    self.send(ev)
            except Empty:
                break
            except KeyboardInterrupt:
                self._bRun = False
                self.error("quit per KeyboardInterrupt")
                break
            except Exception as ex:
                self.logexception(ex)
            
        # step 2. receive the incomming events
        ev = self.recv(0.1)
        if ev and ev.type in self.__evTypesToRecv:
            self.postEvent(ev)
            cRecv +=1

        return cRecv + cSent;


    def doAppInit(self): # return True if succ
        if not super(EventProxy, self).doAppInit() :
            return False

        if len(self.__evTypesToFwd) + len(self.__evTypesToRecv) <=0 :
            return False

        for et in self.__evTypesToFwd:
            self.subscribeEvent(et)
        
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------


import zmq
__ctxZerMQ = zmq.Context()
ZMQPORT_PUB = 1818
ZMQPORT_SUB = 1819


########################################################################
class ZeroMqProxy(EventProxy):
    def __init__(self, advisorId, symbol, exchange):
        """Constructor"""
        super(ZeroMqProxy, self).__init__(program, **kwargs)

        self._endPointEventCh = "localhost"
        self.__soPub = __ctxZerMQ.socket(zmq.PUB)
        self.__soSub = __ctxZerMQ.socket(zmq.SUB)
        self.__soPub.setsockopt(zmq.LINGER, 0)
        self.__soSub.setsockopt(zmq.LINGER, 0)

        self.__poller = zmq.Poller()
        self.__poller.register(self.__soSub, zmq.POLLIN)

    def connect(self):
        self.__soPub.connect('tcp://%s:%d' % (self._endPointEventCh, ZMQPORT_PUB))
        self.__soSub.connect('tcp://%s:%d' % (self._endPointEventCh, ZMQPORT_SUB))

    def send(self, ev):
        if not ev: return
        pklstr = pickle.dumps(ev)
        self.__soPub.send('%s>%s' % (ev.type, pklstr))

    def recv(self, secTimeout=0.1):
        ev = None
        if not self.__poller.poll(int(secTimeout*1000)): # 10s timeout in milliseconds
            return None

        msg = self.__soPub.recv()
        evtype, pklstr = msg.split('>')
        if evtype in self.__evTypesToRecv:
            ev = pickle.loads(pklstr)

        return ev

########################################################################
class ZeroMqEventChannel(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(ZeroMqEventChannel, self).__init__(program, **kwargs)
        self.__msTimeout = 200 # 200msec
        self.__soPub = __ctxZerMQ.socket(zmq.PUB)
        self.__soSub = __ctxZerMQ.socket(zmq.SUB)

        # event channel's portNum is reverse
        self.__soSub.bind("tcp://*:%d" % ZMQPORT_SUB)
        self.__soPub.bind("tcp://*:%d" % ZMQPORT_PUB)

        self.__poller = zmq.Poller()
        self.__poller.register(self.__soSub, zmq.POLLIN)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        return super(ZeroMqEventChannel, self).doAppInit()

    def doAppStep(self):
        # step 1. forward the outgoing events
        fNow = datetime2float(datetime.now()) *1000
        fExp = fNow + self.__msTimeout

        c =0
        ev = True # dummy
        while fNow < fExp:
            if not self.__poller.poll(int(fExp - fNow)): # 10s timeout in milliseconds
                break

            try:
                msg = self.__soPub.recv()
                if not msg : continue
                c +=1
                self.__soPub.send(ev)
            except Exception as ex:
                self.logexception(ex)
                break
        return c
