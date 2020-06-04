# encoding: UTF-8
'''
EventEnd to remote ZeroMQ
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
class EventEnd(BaseApplication):
    '''
    '''
    def __init__(self, program, **kwargs) :
        super(EventEnd, self).__init__(program, **kwargs)

        # 事件队列
        self._queOutgoing = Queue(maxsize=100)
        self._queIncoming = Queue(maxsize=100)
        self.__selfstamp = '%s@%s' % (self.program.progId, self.program.hostname)

        self._topicsOutgoing   = self.getConfig('outgoing', [])
        self._topicsIncomming  = self.getConfig('incoming', [])
        self._subQuit   = False

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
        forward the local events to the remote eventChannel
        '''
        if ev.publisher or not ev.type in self._topicsOutgoing: # ignore those undeclared type and were captured from remote
            return
        
        ev = copy(ev)
        ev.sign(self.__selfstamp)
        self._queOutgoing.put(ev)

    def doAppStep(self):
        cPrnt = super(EventEnd, self).doAppStep()
        
        # step 1. forward the outgoing events
        cSent, cRecv =0, 0
        ev = True # dummy
        while ev and cSent <10:
            ev = None
            try :
                ev = self._queOutgoing.get(block =False, timeout =0.1)  # 获取事件的阻塞时间设为0.1秒
                if ev :
                    self.send(ev)
                    cSent+=1                    
            except Empty:
                break
            except KeyboardInterrupt:
                self._bRun = False
                self.error("quit per KeyboardInterrupt")
                break
            except Exception as ex:
                self.logexception(ex)
            
        # step 2. receive the incomming events
        ev = True # dummy
        while ev and cRecv <10:
            ev = None
            try :
                ev = self._queIncoming.get(block =False, timeout = 0.1)  # 获取事件的阻塞时间设为0.1秒
                if self.topicOfEvent(ev) in self._topicsIncomming:
                    cRecv +=1
                    if not ev.publisher or self.__selfstamp in ev.publisher:
                        self.debug("self-echo[%s] ignored: %s"%(ev.publisher, ev.desc))
                        continue

                    self.postEvent(ev)
            except Empty:
                break
            except KeyboardInterrupt:
                self._bRun = False
                self.error("quit per KeyboardInterrupt")
                break
            except Exception as ex:
                self.logexception(ex)

        return cRecv + cSent;

    def doAppInit(self): # return True if succ
        if not super(EventEnd, self).doAppInit() :
            return False

        if len(self._topicsOutgoing) + len(self._topicsIncomming) <=0 :
            return False

        # subscribe from local events about to forward
        self.subscribeEvents(self._topicsOutgoing)
        
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------


import zmq
import socket
ZMQPORT_PUB = 1818
ZMQPORT_SUB = ZMQPORT_PUB +1
ZMQ_DELIMITOR_TOPIC='>'

########################################################################
class ZmqEE(EventEnd):

    def __init__(self, program, **kwargs) :
        super(ZmqEE, self).__init__(program, **kwargs)

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
            self.info('connected pub to evch[%s]'% (self.__epPUB))

        pklstr = pickle.dumps(ev) # this is bytes
        #NO such API: self.__soPub.connect()
        msg = self.topicOfEvent(ev).encode() + ZMQ_DELIMITOR_TOPIC.encode() + pklstr # this is bytes
        self.__soPub.send(msg) # send must take bytes
        #NO such API: self.__soPub.close()

        self.debug('sent to evch[%s]: %s'% (self.__epPUB, ev.desc))

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
            self.info('subscribed from evch[%s] for %d-topic: %s' % (self.__epSUB, len(subscribedTopics), ','.join(subscribedTopics)))

        # if not self.__poller.poll(int(100)): # 100msec timeout
        #     return None
        socks = dict(self.__poller.poll(int(secTimeout*1000)))
        if not self.__soSub in socks or socks[self.__soSub] != zmq.POLLIN:
            return None
        
        msg = self.__soSub.recv()

        pos = msg.find(ZMQ_DELIMITOR_TOPIC.encode())
        if pos <=0: return None

        topic, pklstr = msg[:pos].decode(), msg[pos+1:]

        # necessary to filter arrivals as topicfilter covered it: 
        # if topic in self._topicsIncomming:
        ev = pickle.loads(pklstr)
        self.debug('recv from evch[%s]: %s'% (self.__epSUB, ev.desc))
        return ev

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZmqEE, self).doAppInit() :
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

        self._bind    = self.getConfig('bind', '0.0.0.0') # anyaddr
        self._portIN  = self.getConfig('portIN', ZMQPORT_PUB)
        self._portOUT = self.getConfig('portOUT', 0)
        self._portDBG = self.getConfig('portDBG', 0)
        self._monitor = False # self._portDBG >1000

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
        self.__endIN = self.__ctxZMQ.socket(zmq.XSUB)
        self.__endIN.bind("tcp://%s:%d" % (self._bind, self._portIN))
        # self.__endIN.setsockopt(zmq.SUBSCRIBE, b'')

        # Socket facing SUBs
        self.__endOUT = self.__ctxZMQ.socket(zmq.XPUB)
        self.__endOUT.bind("tcp://%s:%d" % (self._bind, self._portOUT))

        self.__endDBG = None
        if self._portDBG > 1000:
            self.__endDBG = ctx.socket(zmq.PUB)
            self.__endDBG.bind("tcp://%s:%d" % (self._bind, self._portDBG))

        if self._monitor:
            self.__thread = threading.Thread(target=self.__loop)

            self.__monIN = self.__ctxZMQ.socket(zmq.PAIR)
            self.__endIN.monitor('inproc://zmqchIN', zmq.EVENT_ALL)
            self.__monIN.connect('inproc://zmqchIN')

            self.__monOUT = self.__ctxZMQ.socket(zmq.PAIR)
            self.__endOUT.monitor('inproc://zmqchOUT', zmq.EVENT_ALL)
            self.__monOUT.connect('inproc://zmqchOUT')
        
            self.__poller = zmq.Poller()
            self.__poller.register(self.__monIN, zmq.POLLIN)
            self.__poller.register(self.__monOUT, zmq.POLLIN)

        self.info('bind on %s:%s->%s' % (self._bind, self._portIN, self._portOUT))

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(ZmqEventChannel, self).doAppInit():
            return False
        
        if not self.__endIN or not self.__endOUT:
            return False

        if self._monitor:
            if not self.__thread: return False

            self.__thread.start()

        return True

    def __loop(self):
        self.info('starting channel tcp://%s:%s->%s' % (self._bind, self._portIN, self._portOUT))
        # zmq.device(zmq.FORWARDER, self.__endIN, self.__endOUT) # this is a blocking call
        try:
            zmq.proxy(self.__endIN, self.__endOUT, self.__endDBG)
        except zmq.error.ContextTerminated:
            pass
        if self.__endIN:  self.__endIN.close()
        if self.__endOUT: self.__endOUT.close()
        if self.__endDBG: self.__endDBG.close()
        if self.__ctxZMQ: self.__ctxZMQ.term()
        self.info('channel[%s] stopped: %s->%s' % (self._bind, self._portIN, self._portOUT))

    def doAppStep(self):

        c =0
        if not self._monitor:
            self.__loop()
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


import redis
########################################################################
class RedisEE(EventEnd):

    def __init__(self, program, **kwargs) :
        super(RedisEE, self).__init__(program, **kwargs)

        self._redisHost    = self.getConfig('host', "localhost")
        self._redisPort    = self.getConfig('port', 6379)
        self._redisPasswd  = self.getConfig('password', None)
        self.__redisConn   = None
        self.__connPool    = redis.ConnectionPool(host=self._redisHost, port=self._redisPort, password=self._redisPasswd, db=0, socket_connect_timeout=1.0)

        # start a background threaded Sub because redis subscribe.listen() is a blocking call
        self.__threadSub   = None

    @property
    def myStamp(self):
        return self._redisHost

    def send(self, ev):
        if not ev: return
        if self.__redisConn is None:
            self.__connect()
        
        if not self.__redisConn:
            return

        pklstr = pickle.dumps(ev) # this is bytes
        try :
            self.__redisConn.publish(ev.type, pklstr)
            self.debug('sent to evch[%s:%s]: %s'% (self._redisHost, self._redisPort, ev.desc))
        except Exception as ex:
            self.logexception(ex)
            self.__redisConn.close()
            self.__redisConn =None

    def recv(self, secTimeout=0.1): return None

    def __connect(self) :
        try :
            if self.__redisConn: self.__redisConn.close()
            self.__redisConn = None
            self.debug('connecting to evch[%s:%s]'% (self._redisHost, self._redisPort))
            self.__redisConn = redis.StrictRedis(connection_pool=self.__connPool)
        except Exception as ex:
            self.debug('failed to connect to evch[%s:%s]'% (self._redisHost, self._redisPort))

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def doAppInit(self): # return True if succ
        if not super(RedisEE, self).doAppInit() :
            return False

        self.__connect()
        if not self.__redisConn : return False

        if len(self._topicsIncomming) >0 :
            self.__threadSub = threading.Thread(target=self.__execSub)

        if self.__threadSub:
            self.__threadSub.start()
        
        return True

    # overwrite of EventEnd due to blocking
    # def doAppStep(self):
    #     cActivitis = super(RedisEE, self).doAppStep()

    #     if not self.__ps or len(self._topicsIncomming) <=0 :
    #         return 0

    #     for msg in self.__ps.listen(): # BLOCKING here
    #         if msg['type'] != 'message':
    #             continue

    #         evType = msg['channel']
    #         pklstr = msg['data'] 
    #         ev = pickle.loads(pklstr)
    #         if ev:
    #             if self.topicOfEvent(ev) in self._topicsIncomming:
    #                 if ev.publisher and not self.__selfstamp in ev.publisher:
    #                     self.postEvent(ev)
    #                 cRecv +=1

    def stop(self):
        ''' call to stop this
        '''
        self._subQuit = True
        if self.__redisConn:
            self.__redisConn.close()

        if self.__threadSub :
            try:
                self.__threadSub.stop()
            except:
                pass

            self.__threadSub.join()
            
        self.debug('RedisEE stopping')

        return super(RedisEE, self).stop()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __execSub(self):
        ps = None
        if not self.__redisConn or len(self._topicsIncomming) <=0 :
            return 0

        # self._topicsIncomming = ['evTAdv', 'evmdKL1m'] # TEST ONLY
        while not self._subQuit:
            try:
                if ps is None:
                    if self.__redisConn is None:
                        self.__connect()

                    if self.__redisConn:
                        ps = self.__redisConn.pubsub()
                        for s in self._topicsIncomming:
                            topicfilter = '%s' % s
                            if len(topicfilter) <=0: continue
                            ps.subscribe(topicfilter)

                    sleep(0.5)
                    continue # to test _subQuit

                for msg in ps.listen(): # BLOCKING here
                    if self._subQuit: break
                    if msg['type'] != 'message':
                        continue

                    evType = msg['channel'].decode()
                    if not evType or len(evType)<=0 or not evType in self._topicsIncomming:
                        continue

                    pklstr = msg['data'] 
                    ev = pickle.loads(pklstr)
                    if not ev : continue

                    self._queIncoming.put(ev)
                    self.debug('remoteEvent from evch[%s:%s]: %s' % (self._redisHost, self._redisPort, ev.desc))
            except Exception as ex:
                self.logexception(ex)
                ps = None
                if self.__redisConn:  self.__redisConn.close()
                self.__redisConn = None
