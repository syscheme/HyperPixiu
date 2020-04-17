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

        self._topicsOutgoing  = self.getConfig('outgoing', [])
        self._topicsIncomming  = self.getConfig('incoming', ['*'])

    @abstractmethod
    def send(self, event):
        pass

    @abstractmethod
    def recv(self, secTimeout=0.1):
        return None

    def topicOfEvent(self, ev):
        return ev.type

    def registerOutgoing(self, evTypes):
        if isinstance(evTypes, list) :
            for et in evTypes:
                if not et in self._topicsOutgoing:
                    self._topicsOutgoing.append(et)
        elif not evTypes in self._topicsOutgoing:
            self._topicsOutgoing.append(et)

    def subscribeIncoming(self, evTypes):
        if isinstance(evTypes, list) :
            for et in evTypes:
                if not et in self._topicsIncomming:
                    self._topicsIncomming.append(et)
        elif not evTypes in self._topicsIncomming:
            self._topicsIncomming.append(et)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev):
        '''
        dispatch the event
        '''
        if not ev.type in self._topicsOutgoing:
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
        if ev and ev.type in self._topicsIncomming:
            self.postEvent(ev)
            cRecv +=1

        return cRecv + cSent;


    def doAppInit(self): # return True if succ
        if not super(EventProxy, self).doAppInit() :
            return False

        if len(self._topicsOutgoing) + len(self._topicsIncomming) <=0 :
            return False

        for et in self._topicsOutgoing:
            self.subscribeEvent(et)
        
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------


import zmq
import socket
ZMQPORT_PUB = 1818
ZMQPORT_SUB = ZMQPORT_PUB +1

########################################################################
class ZeroMqProxy(EventProxy):

    def __init__(self, program, **kwargs) :
        super(ZeroMqProxy, self).__init__(program, **kwargs)

        self._endPointEventCh = self.getConfig('endpoint', "localhost")
        portPUB   = self.getConfig('portPUB', ZMQPORT_PUB)
        portSUB   = self.getConfig('portPUB', 0)

        if portPUB <=0:
            portPUB = ZMQPORT_PUB
        if portSUB<=0 or portSUB == portPUB :
            portSUB = portPUB +1

        self.__epPUB = "tcp://%s:%d" % (self._endPointEventCh, portPUB)
        self.__epSUB = "tcp://%s:%d" % (self._endPointEventCh, portSUB)

        self.__soPub, self.__soSub = None, None
        self.__ctxZMQ = zmq.Context()
        self.__poller = zmq.Poller()

    def send(self, ev):
        if not ev: return
        if not self.__soPub:
            self.__soPub = self.__ctxZMQ.socket(zmq.PUB)
            self.__soPub.connect(self.__epPUB)

        pklstr = pickle.dumps(ev)
        msg = '%s %s' % (self.topicOfEvent(ev), pklstr)
        self.__soPub.send(msg.encode())

    def recv(self, secTimeout=0.1):
        ev = None
        if not self.__soSub:
            self.__soSub = self.__ctxZMQ.socket(zmq.SUB)
            self.__soSub.connect(self.__epSUB)
            for s in self._topicsIncomming:
                topicfilter = '%s' % s
                socket.setsockopt(zmq.SUBSCRIBE, topicfilter.encode())

            self.__poller.register(self.__soSub, zmq.POLLIN)

        if not self.__poller.poll(int(100)): # 100msec timeout
            return None

        msg = self.__soSub.recv().decode()
        topic, pklstr = msg.split('>')

        # necessary to filter arrivals as topicfilter covered it: 
        # if topic in self._topicsIncomming:
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
        self._threadWished = True # always thread-ed because zmq.device(zmq.FORWARDER) is blocking

        self._bind    = self.getConfig('bind', '*') # anyaddr
        self._portIN  = self.getConfig('portIN', ZMQPORT_PUB)
        self._portOUT = self.getConfig('portOUT', 0)

        if not self._id or len(self._id) <=0:
            self._id = self.program.hostname

        if not self._bind or len(self._bind) <=0:
            self._bind = "*"
        if self._bind !="*":
            self._bind = socket.gethostbyname(self._bind) # zmq only take IP address instead of hostname

        if self._portIN <=0:
            self._portIN= ZMQPORT_PUB
        if self._portOUT<=0 or self._portOUT == self._portIN :
            self._portOUT = self._portIN +1

        self.__endIN, self.__endOUT = None, None
        self.__ctxZMQ = zmq.Context(1)

        # Socket facing PUBs
        self.__endIN = self.__ctxZMQ.socket(zmq.SUB)
        self.__endIN.bind("tcp://*:%d" % self._portIN)
        self.__endIN.setsockopt(zmq.SUBSCRIBE, "".encode())

        # Socket facing SUBs
        self.__endOUT = self.__ctxZMQ.socket(zmq.PUB)
        self.__endOUT.bind("tcp://*:%d" % self._portOUT)

        self.info('bind on %s:(%s->%s)' % (self._bind, self._portIN, self._portOUT))

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZeroMqEventChannel, self).doAppInit():
            return False
        
        if not self.__endIN or not self.__endOUT:
            return False
        
        return True

    def doAppStep(self):
        self.info('starting channel %s:(%s->%s)' % (self._bind, self._portIN, self._portOUT))
        zmq.device(zmq.FORWARDER, self.__endIN, self.__endOUT) # this is a blocking call
        if self.__endIN:  self.__endIN.close()
        if self.__endOUT: self.__endOUT.close()
        if self.__ctxZMQ: self.__ctxZMQ.term()
        self.info('stopped channel %s:(%s->%s)' % (self._bind, self._portIN, self._portOUT))

    def OnEvent(self, ev): pass

    # end of BaseApplication routine
    #----------------------------------------------------------------------
