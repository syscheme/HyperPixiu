# encoding: UTF-8
'''
EventProxy to remote ZeroMQ
'''
from __future__ import division

from EventData    import *
from Application  import BaseApplication

import os
import pickle, json   # to save params
from datetime import datetime, timedelta
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
import traceback

import sys
if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty

########################################################################
class EventProxy(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(EventProxy, self).__init__(program, **kwargs)

        # 事件队列
        self.__queOutgoing = Queue(maxsize=100)
        self.__conn = None

        self.__evTypesOutgoing  = self.getConfig('outgoing', [])
        self.__evTypesIncoming  = self.getConfig('incoming', [])

    @abstractmethod
    def send(self, event):
        pass

    @abstractmethod
    def recv(self, secTimeout=0.1):
        return None

    def registerOutgoing(self, evTypes):
        if isinstance(evTypes, list) :
            for et in evTypes:
                if not et in self.__evTypesOutgoing:
                    self.__evTypesOutgoing.append(et)
        elif not evTypes in self.__evTypesOutgoing:
            self.__evTypesOutgoing.append(et)

    def subscribeIncoming(self, evTypes):
        if isinstance(evTypes, list) :
            for et in evTypes:
                if not et in self.__evTypesIncoming:
                    self.__evTypesIncoming.append(et)
        elif not evTypes in self.__evTypesIncoming:
            self.__evTypesIncoming.append(et)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev):
        '''
        dispatch the event
        '''
        if not ev.type in self.__evTypesOutgoing:
            return
        
        self.__queOutgoing.put(ev)

    def doAppStep(self):
        super(EventProxy, self).doAppStep()
        
        # step 1. forward the outgoing events
        ev = True # dummy
        cSent, cRecv =0, 0
        while ev and cSent <10:
            ev = None
            try :
                ev = self.__queOutgoing.get(block =False, timeout = 0.1)  # 获取事件的阻塞时间设为0.1秒
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
        if ev and ev.type in self.__evTypesIncoming:
            self.postEvent(ev)
            cRecv +=1

        return cRecv + cSent;


    def doAppInit(self): # return True if succ
        if not super(EventProxy, self).doAppInit() :
            return False

        if len(self.__evTypesOutgoing) + len(self.__evTypesIncoming) <=0 :
            return False

        for et in self.__evTypesOutgoing:
            self.subscribeEvent(et)
        
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------


import zmq
import socket
ZMQPORT_PUB = 1818
ZMQPORT_SUB = 1819

########################################################################
class ZeroMqProxy(EventProxy):

    def __init__(self, program, **kwargs) :
        super(ZeroMqProxy, self).__init__(program, **kwargs)

        self._endPointEventCh = self.getConfig('endpoint', "localhost")

        ctxZMQ = zmq.Context()
        self.__soPub = ctxZMQ.socket(zmq.PUB)
        self.__soSub = ctxZMQ.socket(zmq.SUB)
        self.__soPub.setsockopt(zmq.LINGER, 0)
        self.__soSub.setsockopt(zmq.LINGER, 0)

        self.__poller = zmq.Poller()
        self.__poller.register(self.__soSub, zmq.POLLIN)

    def __connect(self):
        self.__soPub.connect('tcp://%s:%d' % (self._endPointEventCh, ZMQPORT_PUB))
        self.__soSub.connect('tcp://%s:%d' % (self._endPointEventCh, ZMQPORT_SUB))

    def send(self, ev):
        if not ev: return
        pklstr = pickle.dumps(ev)
        self.__soPub.send_string('%s>%s' % (ev.type, pklstr))

    def recv(self, secTimeout=0.1):
        ev = None
        if not self.__poller.poll(int(secTimeout*1000)): # 10s timeout in milliseconds
            return None

        msg = self.__soPub.recv_string()
        evtype, pklstr = msg.split('>')
        if evtype in self.__evTypesIncoming:
            ev = pickle.loads(pklstr)

        return ev

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZeroMqProxy, self).doAppInit() :
            return False

        try:
            self.__connect()
            return True
        except Exception as ex:
            self.logexception(ex)
        
        return True #???
    # end of BaseApplication routine
    #----------------------------------------------------------------------

########################################################################
class ZeroMqEventChannel(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(ZeroMqEventChannel, self).__init__(program, **kwargs)

        self._bind = self.getConfig('bind', '*') # anyaddr
        self.__msTimeout = self.getConfig('timeout', 200) # 200msec by default
        if not self._id or len(self._id) <=0:
            self._id = self.program.hostname

        if not self._bind or len(self._bind) <=0:
            self._bind = "*"
        if self._bind !="*":
            self._bind = socket.gethostbyname(self._bind) # zmq only take IP address instead of hostname

        ctxZMQ = zmq.Context()
        self.__soPub = ctxZMQ.socket(zmq.PUB)
        self.__soSub = ctxZMQ.socket(zmq.SUB)

        # event channel's portNum is reverse
        self.__soSub.bind("tcp://%s:%d" % (self._bind, ZMQPORT_SUB))
        self.__soPub.bind("tcp://%s:%d" % (self._bind, ZMQPORT_PUB))

        self.__poller = zmq.Poller()
        self.__poller.register(self.__soSub, zmq.POLLIN)
        self.info('bind on %s' % self._bind)

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

        if c >0 :
            self.debug('delivered %s events' % c)

        return c

    def OnEvent(self, ev): pass

    # end of BaseApplication routine
    #----------------------------------------------------------------------
