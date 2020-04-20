# encoding: UTF-8
'''
EventProxy to remote ZeroMQ
'''
from __future__ import division

from EventData    import *
from Application  import BaseApplication, BOOL_STRVAL_TRUE

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
        self.__selfstamp = '%s@%s' % (self.program.progId, self.program.hostname)

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
        if ev.publisher or not ev.type in self._topicsOutgoing: # ignore those undeclared type and were captured from remote
            return
        
        ev = copy(ev)
        ev.sign(self.__selfstamp)
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
        if ev and self.topicOfEvent(ev) in self._topicsIncomming:
            if ev.publisher and not self.__selfstamp in ev.publisher:
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
ZMQ_DELIMITOR_TOPIC='>'

########################################################################
class ZmqProxy(EventProxy):

    def __init__(self, program, **kwargs) :
        super(ZmqProxy, self).__init__(program, **kwargs)

        self._endPointEventCh = self.getConfig('endpoint', "localhost")
        portPUB   = self.getConfig('portPUB', ZMQPORT_PUB)
        portSUB   = self.getConfig('portSUB', 0)

        if portPUB <=0:
            portPUB = ZMQPORT_PUB
        if portSUB<=0 or portSUB == portPUB :
            portSUB = portPUB +1

        self.__epPUB = "tcp://%s:%d" % (self._endPointEventCh, portPUB)
        self.__epSUB = "tcp://%s:%d" % (self._endPointEventCh, portSUB)

        self.__soPub, self.__soSub = None, None
        self.__ctxZMQ = zmq.Context()
        self.__poller = zmq.Poller()

    @property
    def myStamp(self):
        return self.__epSUB

    def send(self, ev):
        if not ev: return
        if not self.__soPub:
            self.__soPub = self.__ctxZMQ.socket(zmq.PUB)
            self.__soPub.connect(self.__epPUB)
            self.info('connected pub to ech[%s]'% (self.__epPUB))

        pklstr = pickle.dumps(ev) # this is bytes
        msg = self.topicOfEvent(ev).encode() + ZMQ_DELIMITOR_TOPIC.encode() + pklstr # this is bytes
        self.__soPub.send(msg) # send must take bytes
        self.debug('sent to ech[%s]: %s'% (self.__epPUB, ev.desc))

    def recv(self, secTimeout=0.1):
        ev = None
        if not self.__soSub:
            self.__soSub = self.__ctxZMQ.socket(zmq.SUB)
            self.__soSub.connect(self.__epSUB)
            subscribedTopics = []
            for s in self._topicsIncomming:
                topicfilter = '%s' % s
                if len(topicfilter) <=0: continue
                # if '*' == topicfilter: topicfilter ='' # all topics
                self.__soSub.setsockopt(zmq.SUBSCRIBE, topicfilter.encode())
                subscribedTopics.append(topicfilter)

            self.__poller.register(self.__soSub, zmq.POLLIN)
            self.info('subscribed from ech[%s] for %d-topic %s' % (self.__epSUB, len(subscribedTopics), ','.join(subscribedTopics)))

        if not self.__poller.poll(int(100)): # 100msec timeout
            return None

        msg = self.__soSub.recv()
        pos = msg.find(ZMQ_DELIMITOR_TOPIC.encode())
        if pos <=0: return None

        topic, pklstr = msg[:pos].decode(), msg[pos+1:]

        # necessary to filter arrivals as topicfilter covered it: 
        # if topic in self._topicsIncomming:
        ev = pickle.loads(pklstr)
        self.debug('recv from ech[%s]: %s'% (self.__epSUB, ev.desc))
        return ev

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZmqProxy, self).doAppInit() :
            return False

        return True #???

    # end of BaseApplication routine
    #----------------------------------------------------------------------

########################################################################
import threading
from time import sleep
class ZmqEventChannel(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(ZmqEventChannel, self).__init__(program, **kwargs)
        self._threadWished = True # always thread-ed because zmq.device(zmq.FORWARDER) is blocking

        self._bind    = self.getConfig('bind', '*') # anyaddr
        self._portIN  = self.getConfig('portIN', ZMQPORT_PUB)
        self._portOUT = self.getConfig('portOUT', 0)
        self._monitor = self.getConfig('monitor', 'False').lower() in BOOL_STRVAL_TRUE

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

        self.__thread = threading.Thread(target=self.__forwarder)

        if self._monitor:
            self.__monIN = self.__ctxZMQ.socket(zmq.PAIR)
            self.__endIN.monitor('inproc://zmqchIN', zmq.EVENT_ALL)
            self.__monIN.connect('inproc://zmqchIN')

            self.__monOUT = self.__ctxZMQ.socket(zmq.PAIR)
            self.__endOUT.monitor('inproc://zmqchOUT', zmq.EVENT_ALL)
            self.__monOUT.connect('inproc://zmqchOUT')
        
            self.__poller = zmq.Poller()
            self.__poller.register(self.__monIN, zmq.POLLIN)
            self.__poller.register(self.__monOUT, zmq.POLLIN)

        self.info('bind on %s:(%s->%s)' % (self._bind, self._portIN, self._portOUT))

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZmqEventChannel, self).doAppInit():
            return False
        
        if not self.__endIN or not self.__endOUT or not self.__thread:
            return False

        self.__thread.start()
        return True

    def __forwarder(self):
        self.info('starting channel %s:(%s->%s)' % (self._bind, self._portIN, self._portOUT))
        zmq.device(zmq.FORWARDER, self.__endIN, self.__endOUT) # this is a blocking call
        if self.__endIN:  self.__endIN.close()
        if self.__endOUT: self.__endOUT.close()
        if self.__ctxZMQ: self.__ctxZMQ.term()
        self.info('channel[%s] stopped: %s->%s' % (self._bind, self._portIN, self._portOUT))

    def doAppStep(self):

        c =0
        if not self._monitor:
            sleep(0.2)
            return c

        while True:
            socks = self.__poller.poll(int(10)) # 10msec timeout
            if not socks: return None

            socks = dict(socks)
            if self.__monIN in socks and socks[self.__monIN] == zmq.POLLIN:
                msg = self.__monIN.recv()
                # ev = zmq.utils.monitor.parse_monitor_message(msg)
                # if not ev: return 0
                self.debug('IN: %s' % msg)
                c+=1

            if self.__monOUT in socks and socks[self.__monOUT] == zmq.POLLIN:
                msg = self.__monOUT.recv()
                self.debug('OUT: %s' % msg)
                c+=1

        return c

    def stop(self):
        ''' call to stop this
        '''
        if not self.__thread :
            return
        self.__thread.stop()
        self.__thread.join()

        super(ZmqEventChannel, self).stop()
        self.info('ZmqEventChannel stopped')

    def OnEvent(self, ev): pass # do nothing at this entry

    # end of BaseApplication routine
    #----------------------------------------------------------------------
