# encoding: UTF-8

from __future__ import division

from EventData import Event, EventData, EVENT_SYS_CLOCK, DT_EPOCH, datetime2float

import os
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict, defaultdict
from datetime import datetime
import time
from copy import copy
from abc import ABC, abstractmethod
import traceback

import shelve
import jsoncfg # pip install json-cfg
import json
import bz2

########################################################################
# 常量定义
########################################################################
# 日志级别
LOGLEVEL_DEBUG    = logging.DEBUG
LOGLEVEL_INFO     = logging.INFO
LOGLEVEL_WARN     = logging.WARN
LOGLEVEL_ERROR    = logging.ERROR
LOGLEVEL_CRITICAL = logging.CRITICAL
LOGFMT_GENERAL = '%(asctime)s %(levelname)s\t%(message)s'

BOOL_STRVAL_TRUE = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh', True, 1]

########################################################################
class MetaObj(ABC):
    __lastId__  = 10000

    def __init__(self):
        self._id = None
        self._oseqId = MetaObj.__nextOSeqId()

    @property
    def ident(self) :
        if not self._id or len(self._id)<=0 :
            self._id = 'O%d' % self._oseqId

        return '%s.%s' % (self.__class__.__name__, self._id)

    def __nextOSeqId():
        MetaObj.__lastId__ +=1
        return MetaObj.__lastId__

    def __copy__(self):
        result = object.__new__(type(self))
        result.__dict__ = copy(self.__dict__)
        result._oseqId = MetaObj.__nextOSeqId()
        return result

########################################################################
class MetaApp(MetaObj):

    @abstractmethod
    def theApp(self):
        raise NotImplementedError

    @property
    def isActive(self) :
        raise NotImplementedError

    @abstractmethod
    def doAppInit(self): # return True if succ
        return False

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def OnEvent(self, event):
        '''
        process the event
        '''
        pass

########################################################################
class BaseApplication(MetaApp):

    HEARTBEAT_INTERVAL_DEFAULT = 5 # 5sec
    
    #----------------------------------------------------------------------
    def __init__(self, program, **kwargs):
        '''Constructor'''

        super(BaseApplication, self).__init__()
        
        self._program = program
        self.__active = False    # 工作状态
        self._threadWished = False
        self._id =""
        self.__jsettings = None
        self.__dataDir = './out'

        self._kwargs = kwargs
        if 'jsettings' in self._kwargs.keys():
            self.__jsettings = self._kwargs.pop('jsettings', None)
            self._id        = self.__jsettings.id(self._id)
            self.__dataDir  = self.__jsettings.dataPath(self.__dataDir)
            self._threadWished = self.__jsettings.threaded('False').lower() in BOOL_STRVAL_TRUE

        self._id = self._kwargs.pop('id', self._id)
        self.__dataDir  = self._kwargs.pop('dataPath', self.__dataDir)
        self._threadWished = self._kwargs.pop('threaded', str(self._threadWished)).lower() in BOOL_STRVAL_TRUE

        if '/' != self.__dataDir[-1]:
            self.__dataDir +='/'

        self.__gen = self._generator()

    def __deepcopy__(self, other):
        result = object.__new__(type(self))
        result.__dict__ = copy(self.__dict__)
        return result

    #----------------------------------------------------------------------
    @property
    def app(self) : # for the thread wrapper
        return self

    @property
    def dataRoot(self) :
        return self.__dataDir

    @property
    def program(self) :
        return self._program

    @property
    def kwargs(self) :
        return self.__kwargs

    @property
    def isActive(self) :
        return self.__active

    #------Impl of MetaApp --------------------------------------------------
    def theApp(self): return self

    def stop(self):
        # TODO:
        self.__active = False

    #--- pollable step routine for ThreadedAppWrapper -----------------------
    @abstractmethod
    def doAppInit(self): # return True if succ
        self.__active = True
        next(self.__gen)
        return self.__active

    @abstractmethod
    def doAppStep(self):
        '''
        @return True if busy at this step
        '''
        return False

    def _generator(self) :
        while self.isActive :
            event = yield self.ident
            if isinstance(event, Event):
                self.OnEvent(event)

    def _procEvent(self, event) : # called by Program
        return self.__gen.send(event)

    def getConfig(self, configName, defaultVal, pop=False) :
        try :
            if configName in self._kwargs.keys() :
                return self._kwargs.pop(configName, defaultVal) if pop else self._kwargs[configName]

            if self.__jsettings:
                jn = self.__jsettings
                for i in configName.split('/') :
                    jn = jn[i]

                if defaultVal :
                    if isinstance(defaultVal, list):
                        return jsoncfg.expect_array(jn(defaultVal))
                    if isinstance(defaultVal, dict):
                        return jsoncfg.expect_object(jn(defaultVal))

                return jn(defaultVal)
        except:
            pass

        return defaultVal

    def subConfig(self, subName) :
        jn = None
        try :
            if self.__jsettings:
                jn = self.__jsettings
                for i in subName.split('/') :
                    jn = jn[i]
        except:
            jn = None
        return jn

    #---- event operations ---------------------------
    def subscribeEvent(self, ev) :
        if not self._program:
            pass
        
        self._program.subscribe(EnvironmentError, self)

    def postEventData(self, eventType, edata):
        '''发出事件'''
        if not self._program:
            return

        ev = Event(type_= eventType)
        ev.setData(edata)
        self.postEvent(ev)

    def postEvent(self, ev):
        '''发出事件'''
        if not ev or not self._program:
            return

        self._program.publish(ev)
        self.debug('posted event[%s]' % ev.type)

    #---logging -----------------------
    def log(self, level, msg):
        if not self._program: return
        self._program.log(level, 'APP['+self.ident +'] ' + msg)

    def debug(self, msg):
        if not self._program: return
        self._program.debug('APP['+self.ident +'] ' + msg)
        
    def info(self, msg):
        '''正常输出'''
        if not self._program: return
        self._program.info('APP['+self.ident +'] ' + msg)

    def warn(self, msg):
        '''警告信息'''
        if not self._program: return
        self._program.warn('APP['+self.ident +'] ' + msg)
        
    def error(self, msg):
        '''报错输出'''
        if not self._program: return
        self._program.error('APP['+self.ident +'] ' + msg)
        
    def critical(self, msg):
        '''影响程序运行的严重错误'''
        if not self._program: return
        self._program.critical('APP['+self.ident +'] ' + msg)

    def logexception(self, ex):
        '''报错输出+记录异常信息'''
        if not self._program: return
        self._program.logexception('APP['+self.ident +'] %s: %s' % (ex, traceback.format_exc()))

    #----------------------------------------------------------------------
    def logError(self, eventType, content):
        '''处理错误事件'''
        if not self._program: return
    # TODO   error = event.data
    #    self._lstErrors.append(error)
        pass

########################################################################
import threading
class ThreadedAppWrapper(MetaApp):
    #----------------------------------------------------------------------
    def __init__(self, app, maxinterval=BaseApplication.HEARTBEAT_INTERVAL_DEFAULT):
        '''Constructor'''

        super(ThreadedAppWrapper, self).__init__()

        self._app = app
        self._maxinterval = maxinterval
        self.__thread = threading.Thread(target=self.__run)
        self._evWakeup = threading.Event()
        self.__stopped = False

    #----------------------------------------------------------------------
    def __run(self):
        '''执行连接 and receive'''
        while self._app and self._app.isActive:
            nextSleep = self._maxinterval
            try :
                if self._app.doAppStep() :
                    nextSleep =0 # will trigger again if this step busy 
            except Exception as ex:
                self._app.error('ThreadedAppWrapper::step() excepton: %s' % ex)

            if nextSleep >0:
                self._evWakeup.wait(nextSleep)

        if self._app:
            self._app.info('ThreadedAppWrapper exit')
        self.__stopped = True

    def wakeup(self) :
        self._evWakeup.set()

    #------Impl of MetaApp --------------------------------------------------
    def theApp(self): return self._app

    @property
    def isActive(self) :
        return False if self.__stopped or not self._app else self._app.isActive

    def doAppInit(self):
        '''
        return True if started
        '''
        if not isinstance(self._app, BaseApplication):
            self._app = None

        if not self._app or not self._app.doAppInit():
            return False
        
        self.__thread.start()
        self._app.debug('ThreadedAppWrapper started')
        return True

    def stop(self):
        ''' call to stop a app
        '''
        if not self._app :
            return

        self._app.stop()
        self.wakeup()
        self.__thread.join()
        self._app.info('ThreadedAppWrapper stopped')
    
########################################################################
class Singleton(type):
    '''
    单例，应用方式:静态变量 __metaclass__ = Singleton
    '''
    _instances = {}

    #----------------------------------------------------------------------
    def __call__(cls, *args, **kwargs):
        '''调用'''
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            
        return cls._instances[cls]

########################################################################
class Iterable(ABC):
    ''' A metaclass of Iterable
    '''
    def __init__(self, **kwargs):
        '''Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        '''
        super(Iterable, self).__init__()
        self.__gen = None
        self.__program = None
        if 'program' in kwargs.keys():
            self.__program = kwargs['program']

        self._iterableEnd = False

        # 事件队列
        self.__quePending = Queue(maxsize=100)

    def __iter__(self):
        if self.resetRead() : # alway perform reset here
            self.__gen = self.__generate()
            self.__c = 0
            self._iterableEnd = False
        return self

    def __next__(self):
        if not self.__gen and self.resetRead() : # not perform reset here
            self.__gen = self.__generate()
            self.__c = 0
            self._iterableEnd = False

        if not self.__gen :
            raise StopIteration

        return next(self.__gen)

    def __generate(self):
        while not self._iterableEnd :
            try :
                event = self.popPending()
                if event:
                    yield event
                    self.__c +=1
            except Exception:
                pass

            try :
                n = self.readNext()
                if None ==n:
                    continue

                yield n
                self.__c +=1
            except StopIteration:
                break
            except Exception as ex:
                self.logexception(ex)
                self._iterableEnd = True
                break

        self.__gen=None
        raise StopIteration

    @property
    def pendingSize(self) :
        return self.__quePending.qsize() if self.__quePending else 0

    def enquePending(self, ev, block = True):
        self.__quePending.put(ev, block = block)

    def popPending(self, block=False, timeout=0.1):
        return self.__quePending.get(block = block, timeout = timeout)

    #--- new methods  -----------------------
    @abstractmethod
    def resetRead(self):
        '''For this generator, we want to rewind only when the end of the data is reached.
        '''
        pass

    @abstractmethod
    def readNext(self):
        '''
        @return next item, mostlikely expect one of Event()
        '''
        return None

    #---logging -----------------------
    def setProgram(self, program):
        if program and isinstance(program, Program):
            self.__program = program

    def debug(self, msg):
        if not self.__program: return
        self.__program.debug(self.__class__.__name__ +' ' + msg)
        
    def info(self, msg):
        if not self.__program: return
        self.__program.info(self.__class__.__name__ +' ' + msg)

    def warn(self, msg):
        if not self.__program: return
        self.__program.warn(self.__class__.__name__ +' ' + msg)
        
    def error(self, msg):
        if not self.__program: return
        self.__program.error(self.__class__.__name__ +' ' + msg)

    def logexception(self, ex):
        self.error('%s: %s' % (ex, traceback.format_exc()))

########################################################################
import sys, getopt

if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty

class Program(object):
    ''' main program '''

    # 单例模式
    __metaclass__ = Singleton

    #----------------------------------------------------------------------
    def __init__(self, argvs=None) : # setting_filename=None):
        '''Constructor
           usage: Program(sys.argv)
        '''
        if not argvs or len(argvs) <1:
            argvs = sys.argv

        self._pid = os.getpid() # process id
        self._progName = os.path.basename(argvs[0])[0:-3] # cut off the .py extname
        self._outdir = './out' #TODO
        self._heartbeatInterval = BaseApplication.HEARTBEAT_INTERVAL_DEFAULT    # heartbeat间隔（默认1秒）
        self.__daemonize =False
        # dirname(dirname(abspath(file)))
        self.__jsettings = None

        try:
            opts, args = getopt.getopt(argvs[1:], "hf:o:", ["config=","outdir="])
        except getopt.GetoptError :
            print('%s.py -f <config-file> -o <outputdir>' % self._progName)
            sys.exit(2)

        config_filename = None
        for opt, arg in opts:
            if opt == '-h':
                print('%s.py -f <config-file> -o <outputdir>' % self._progName)
                sys.exit()
            elif opt in ("-f", "--ifile"):
                config_filename = arg
            elif opt in ("-o", "--ofile"):
                self._outdir = arg

        if config_filename :
            try :
                config_filename = os.path.abspath(config_filename)
                print('loading configfile: %s' % config_filename)
                self.__jsettings = jsoncfg.load_config(config_filename)
                self.__daemonize = self.__jsettings.daemonize('False').lower() in BOOL_STRVAL_TRUE
                self._heartbeatInterval = int(self.__jsettings.heartbeatInterval(BaseApplication.HEARTBEAT_INTERVAL_DEFAULT))
            except Exception as e :
                print('failed to load configure[%s]: %s' % (config_filename, e))
                sys.exit(3)

        self._shelvefn = '%s/%s.sobj' % (self._outdir, self._progName)
        # 记录今日日期
        self._runStartDate = datetime.now().strftime('%Y-%m-%d')

        # 日志引擎实例
        self.__logger = None
        self.initLogger()
        
        # 事件队列
        self.__queue = Queue()
        self.__dictMetaObjs = OrderedDict()
        self.__activeApps = []
        
        # heartbeat
        self.__stampLastHB = None
            
        # __subscribers字典，用来保存对应的事件到appId的订阅关系
        # 其中每个键对应的值是一个列表，列表中保存了对该事件进行监听的appId
        self.__subscribers = {}

        self.info('='*10 + ' %s(%d) starts ' %(self._progName, self._pid)  + '='*10)
    
    def jsettings(self, nodeName) : 
        if not self.__jsettings : return None
        if not nodeName or len(nodeName) <=0: return self.__jsettings
        n = self.__jsettings
        try :
            for i in nodeName.split('/') :
                n = n[i]
        except:
            n = None
        return n

    @property
    def logger(self) : 
        return self.__logger

    @property
    def settings(self) :
        return self.__jsettings

    #----------------------------------------------------------------------
    def __addMetaObj(self, id, obj):
        '''添加上层应用'''
        # 创建应用实例
        self.__dictMetaObjs[id] = obj
        self.__dict__[id] = self.__dictMetaObjs[id]
        self.debug('obj[%s] added' % id)
        return obj

    def __removeObj(self, id):
        if not id or not id in self.__dictMetaObjs.keys():
            return None

        # del self.__dict__[id]
        o = self.__dictMetaObjs[id]
        del self.__dictMetaObjs[id]
        return o

    def addObj(self, obj):
        '''添加上层应用'''
        if not obj or not isinstance(obj, MetaObj):
            return None
        
        return self.__addMetaObj(obj.ident, obj)

    def removeObj(self, obj):
        if obj and isinstance(obj, MetaObj):
            return self.__removeObj(obj.ident)
        return None

    def getObj(self, objId):
        if objId in self.__dictMetaObjs.keys():
            return self.__dictMetaObjs[objId]
        return None

    def listByType(self, type=MetaObj):
        if issubclass(type, BaseApplication) :
            return self.listApps(type)

        ret = []
        for oid, obj in self.__dictMetaObjs.items():
            if not obj or not isinstance(obj, type):
                continue
            ret.append(oid)
        
        return ret

    def addApp(self, app, displayName=None):
        '''添加上层应用'''
        if not isinstance(app, MetaApp):
            return

        id = app.theApp().ident
        self.__addMetaObj(id, app)
        
        # TODO maybe test isinstance(apptype, MarketData) to ease index and so on
        if self.hasHeartbeat :
            self.subscribe(EVENT_SYS_CLOCK, app.theApp())

        self.debug('app[%s] added' %(id))
        return app

    def createApp(self, appModule, **kwargs):
        '''添加上层应用'''

#        if not isinstance(appModule, BaseApplication) :
#            return None

        jsettings = None
        configNode = kwargs.pop('configNode', '')
        if self.__jsettings and len(configNode)>0:
            jsettings = self.__jsettings
            try :
                for i in configNode.split('/') :
                    jsettings = jsettings[i]
            except:
                jsettings = None

        try :
            if jsettings :
                kwargs =  {**kwargs, 'jsettings': jsettings }
        except:
            pass

        app = appModule(program=self, **kwargs)
        if not app: return
        if app._threadWished :
            app = ThreadedAppWrapper(app)

        return self.addApp(app)
        
    def removeApp(self, app):
        if not app or not isinstance(app, MetaApp):
            return

        for et in self.eventTypes :
            self.unsubscribe(et, app)

        appId = app.theApp().ident
        if not appId in self.__dictMetaObjs.keys():
            return None

        return self.__removeObj(appId)

    def getApp(self, appId):
        '''获取APP对象'''
        if not appId in self.__dictMetaObjs.keys():
            return None
        o = self.__dictMetaObjs[appId]
        return o.theApp() if isinstance(o, MetaApp) else None

    def listApps(self, type=BaseApplication):
        '''list app object of a given type and its children
        @return a list of appId
        '''
        ret = []
        for aid, app in self.__dictMetaObjs.items():
            if not app or not isinstance(app, MetaApp) or not isinstance(app.theApp(), type):
                continue
            ret.append(aid)
        
        return ret

    #----------------------------------------------------------------------
    def start(self, daemonize=False):
        if self.__activeApps and len(self.__activeApps) >0 :
            self.error('already has active apps: %s ' % self.__activeApps)
            return

        if daemonize :
           self.__daemonize = True

        if self.__daemonize :
            self.daemonize()

        self._bRun =True

        self.debug('starting applications')
        for appId in self.listByType(MetaApp) :
            app = self.getObj(appId)
            if app == None:
                continue
            
            self.debug('staring app[%s]' % appId)
            if not app.doAppInit() :
                self.error('failed to initialize app[%s]' % appId)
            else :
                self.__activeApps.append(appId)
                self.info('initialized app[%s]' % appId)

        if len(self.__activeApps) <=0 : 
            self._bRun =False
            self.error('no apps started, quitting')
            return

        self.info('main-program started %d apps: %s' % (len(self.__activeApps), self.__activeApps))

    def stop(self):
        '''退出程序前调用，保证正常退出'''        
        self._bRun = False

        '''
        # 安全关闭所有接口
        for ds in self._dictMarketDatas.values():        
            ds.close()
        
        '''
        
        # 停止上层应用引擎
        for appId in self.__activeApps :
            app = self.getApp(appId)
            if app: app.stop()
        
        # # 保存数据引擎里的合约数据到硬盘
        # self.dataEngine.saveContracts()

    #----------------------------------------------------------------------
    @property
    def hasHeartbeat(self) : 
        return (self._heartbeatInterval > 0) # what's wrong!!!!

    def loop(self):

        self.info(u'Program start looping')
        busy = True

        while self._bRun:
            timeout =0
            enabledHB = self.hasHeartbeat
            # enabledHB = False
            if enabledHB: # heartbeat enabled
                dt = datetime.now()
                stampNow = datetime2float(datetime.now())
            
                if not self.__stampLastHB :
                    self.__stampLastHB = stampNow

                timeout = self.__stampLastHB + self._heartbeatInterval - stampNow
                if timeout < 0:
                    self.__stampLastHB = stampNow
                    timeout = 0.1

                    # inject then event of heartbeat
                    ed = EventData()
                    ed.datetime = dt
                    event = Event(type_= EVENT_SYS_CLOCK)
                    event.setData(ed)
                    self.publish(event)

            # pop the event to dispatch
            bEmpty = False
            while self._bRun and not bEmpty:
                event = None
                try :
                    event = self.__queue.get(block = enabledHB, timeout = timeout)  # 获取事件的阻塞时间设为0.1秒
                    bEmpty = False
                except Empty:
                    bEmpty = True
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    self._bRun = False
                    break

                # do the step only when there is no event
                if not event :
                    # if blocking: # ????
                    #     continue
                    cApps =0
                    for appId in self.__activeApps :
                        app = self.getObj(appId)
                        # if threaded, it has its own trigger to step()
                        # if isinstance(app, ThreadedAppWrapper)
                        #   continue
                        if not isinstance(app, MetaApp):
                            continue

                        if not app.isActive :
                            continue
                        cApps +=1

                        if not isinstance(app, BaseApplication):
                            continue
                        
                        try:
                            app.doAppStep()
                        except KeyboardInterrupt:
                            self.error("quit per KeyboardInterrupt")
                            self._bRun = False
                            break
                        except Exception as ex:
                            self.error("app[%s] step exception %s %s" % (appId, ex, traceback.format_exc()))

                    if cApps <=0:
                        self.info("Program has no more active apps running, update running state")
                        self._bRun = False
                        break
                            
                    continue

                # 检查是否存在对该事件进行监听的处理函数
                if not event.type in self.__subscribers.keys():
                    continue

                # 若存在，则按顺序将事件传递给处理函数执行
                for appId in self.__subscribers[event.type] :
                    app = self.getApp(appId)
                    if not app or not app.isActive:
                        continue

                    try:
                        app._procEvent(event)
                    except KeyboardInterrupt:
                        self.error("quit per KeyboardInterrupt")
                        self._bRun = False
                        break
                    except Exception as ex:
                        self.error("app step exception %s %s" % (ex, traceback.format_exc()))

        self.info(u'Program finish looping')

    #----------------------------------------------------------------------
    def daemonize(self, stdin='/dev/null',stdout='/dev/null',stderr='/dev/null'):
        import sys
        sys.stdin  = open(stdin,'r')
        sys.stdout = open(stdout,'a+')
        sys.stderr = open(stderr,'a+')
        
        try:
            self._pid = os.fork()
            if self._pid > 0:        #parrent
                os._exit(0)
        except OSError as e:
            sys.stderr.write("first fork failed!!"+e.strerror)
            os._exit(1)
    
        # 子进程， 由于父进程已经退出，所以子进程变为孤儿进程，由init收养
        # setsid使子进程成为新的会话首进程，和进程组的组长，与原来的进程组、控制终端和登录会话脱离。
        os.setsid()

        # 防止在类似于临时挂载的文件系统下运行，例如/mnt文件夹下，这样守护进程一旦运行，临时挂载的文件系统就无法卸载了，这里我们推荐把当前工作目录切换到根目录下'''
        os.chdir("/")
        # 设置用户创建文件的默认权限，设置的是权限“补码”，这里将文件权限掩码设为0，使得用户创建的文件具有最大的权限。否则，默认权限是从父进程继承得来的'''
        os.umask(0)
    
        try:
            self._pid = os.fork()     #第二次进行fork,为了防止会话首进程意外获得控制终端
            if self._pid > 0:
                os._exit(0)     #父进程退出
        except OSError as e:
            sys.stderr.write("second fork failed!!"+e.strerror)
            os._exit(1)
    
        # 孙进程
        #   for i in range(3,64):  # 关闭所有可能打开的不需要的文件，UNP中这样处理，但是发现在python中实现不需要。
        #       os.close(i)
        sys.stdout.write("Daemon has been created! with self._pid: %d\n" % os.getpid())
        sys.stdout.flush()  #由于这里我们使用的是标准IO，回顾APUE第五章，这里应该是行缓冲或全缓冲，因此要调用flush，从内存中刷入日志文件。\

    # methods about event subscription
    #----------------------------------------------------------------------
    @property
    def eventTypes(self):
        ret = []
        for t in self.__subscribers.keys() :
            ret.append(t)
        return ret

    def subscribe(self, type_, app):
        '''注册事件处理函数监听'''
        if not isinstance(app, BaseApplication):
            return
        if not type_ in self.__subscribers.keys() :
            self.__subscribers[type_] = []

        # 若要注册的处理器不在该事件的处理器列表中，则注册该事件
        if app.ident not in self.__subscribers[type_]:
            self.__subscribers[type_].append(app.ident)
            
    def unsubscribe(self, type_, app):
        '''注销事件处理函数监听'''
        if not isinstance(app, MetaApp) or not type_ in self.__subscribers.keys():
            return

        appId = app.theApp().ident

        # 如果该函数存在于列表中，则移除
        if appId in self.__subscribers[type_]:
            self.__subscribers[type_].remove(appId)

        # 如果函数列表为空，则从引擎中移除该事件类型
        if not len(self.__subscribers[type_]) <=0:
            del self.__subscribers[type_]  

    def publish(self, event):
        '''向事件队列中存入事件'''
        self.__queue.put(event)

    @property
    def pendingSize(self):
        return self.__queue.qsize()

    def getConfig(self, configName, defaultVal, pop=False) :
        try :
            if configName in self._kwargs.keys() :
                return self._kwargs.pop(configName, defaultVal) if pop else self._kwargs[configName]

            if self.__jsettings:
                jn = self.__jsettings
                for i in configName.split('/') :
                    jn = jn[i]

                if defaultVal :
                    if isinstance(defaultVal, list):
                        return jsoncfg.expect_array(jn(defaultVal))
                    if isinstance(defaultVal, dict):
                        return jsoncfg.expect_object(jn(defaultVal))

                return jn(defaultVal)
        except:
            pass

        return defaultVal

    # methods about logging
    # ----------------------------------------------------------------------
    def initLogger(self):
        '''初始化日志引擎'''

        # 日志级别函数映射
        self._loglevelFunctionDict = {
            LOGLEVEL_DEBUG:    self.debug,
            LOGLEVEL_INFO:     self.info,
            LOGLEVEL_WARN:     self.warn,
            LOGLEVEL_ERROR:    self.error,
            LOGLEVEL_CRITICAL: self.critical,
        }

        STR2LEVEL = {
            'debug' : LOGLEVEL_DEBUG,
            'info' : LOGLEVEL_INFO,
            'warn' : LOGLEVEL_WARN,
            'error' : LOGLEVEL_ERROR,
            'critical' : LOGLEVEL_CRITICAL,
        }

        level = STR2LEVEL['info']
        echoToConsole = True
        logdir = '/tmp'
        filename = '%s.%s.log' % (self._progName, datetime.now().strftime('%Y%m%d'))
        loggingEvent = True

        if self.__jsettings and jsoncfg.node_exists(self.__jsettings.logger):
            level = self.__jsettings.logger.level(level)
            if str(level).lower() in STR2LEVEL.keys():
                level = STR2LEVEL[str(level).lower()]
        
            echoToConsole = self.__jsettings.logger.console(str(echoToConsole)).lower() in BOOL_STRVAL_TRUE
            logdir = self.__jsettings.logger.dir(logdir)
            filename = self.__jsettings.logger.filename(filename)
            loggingEvent = self.__jsettings.logger.loggingEvent(str(loggingEvent)).lower() in BOOL_STRVAL_TRUE
        
        # abbout the logger
        # ----------------------------------------------------------------------
        self.__logger   = logging.getLogger()        
        LOGFMT  = logging.Formatter(LOGFMT_GENERAL)
        self.__loglevel = LOGLEVEL_CRITICAL
        
        self._hdlrConsole = None
        self._hdlrFile = None
        
        # 设置日志级别
        self.setLogLevel(level) # LOGLEVEL_INFO))
        
        # 设置输出
        if echoToConsole and not self._hdlrConsole:
            # 添加终端输出
            self._hdlrConsole = logging.StreamHandler()
            self._hdlrConsole.setLevel(self.__loglevel)
            self._hdlrConsole.setFormatter(LOGFMT)
            self.__logger.addHandler(self._hdlrConsole)
        else :
            # 添加NullHandler防止无handler的错误输出
            nullHandler = logging.NullHandler()
            self.__logger.addHandler(nullHandler)    

        if '/' != logdir[-1]: logdir +='/'

        if filename and len(filename) >0:
            filepath = '%s%s' % (logdir, filename)
            self._hdlrFile = RotatingFileHandler(filepath, maxBytes=50*1024*1024, backupCount=20) # 50MB about to 10MB after bzip2
            # = TimedRotatingFileHandler(filepath, when='W5', backupCount=9) # when='W5' for Satday, 'D' daily, 'midnight' rollover at midnight
            self._hdlrFile.rotator  = self.__rotator
            self._hdlrFile.namer    = self.__rotating_namer
            self._hdlrFile.setLevel(self.__loglevel)
            self._hdlrFile.setFormatter(LOGFMT)
            self.__logger.addHandler(self._hdlrFile)
            
        # 注册事件监听
        self._loggingEvent = True

    def __rotating_namer(self, name):
        return name + ".bz2"

    def __rotator(self, source, dest):
        with open(source, "rb") as sf:
            data = sf.read()
            compressed = bz2.compress(data, 9)
            with open(dest, "wb") as df:
                df.write(compressed)
        os.remove(source)

    def setLogLevel(self, level):
        '''设置日志级别'''
        if self.__logger ==None:
            return

        self.__logger.setLevel(level)
        self.__loglevel = level
    
    @abstractmethod
    def log(self, level, msg):
        if not self._loglevelFunctionDict or not level in self._loglevelFunctionDict : 
            return
        
        function = self._loglevelFunctionDict[level] # 获取日志级别对应的处理函数
        function(msg)

    @abstractmethod
    def debug(self, msg):
        '''开发时用'''
        if self.__logger: 
            self.__logger.debug(msg)
        else:
            print('%s' % msg)
        
    @abstractmethod
    def info(self, msg):
        '''正常输出'''
        if self.__logger: 
            self.__logger.info(msg)
        else:
            print('%s' % msg)

    @abstractmethod
    def warn(self, msg):
        '''警告信息'''
        if self.__logger: 
            self.__logger.warning(msg)
        else:
            print('%s' % msg)
        
    @abstractmethod
    def error(self, msg):
        '''报错输出'''
        if self.__logger: 
            self.__logger.error(msg)
        else:
            print('%s' % msg)
        
    @abstractmethod
    def critical(self, msg):
        '''影响程序运行的严重错误'''
        if self.__logger: 
            self.__logger.critical(msg)
        else:
            print('%s' % msg)

    def logexception(self, ex):
        '''报错输出+记录异常信息'''
        self.error('%s: %s' % (ex, traceback.format_exc()))

    def eventHdlr_Log(self, event):
        '''处理日志事件'''
        if not self.__logger: return
        log = event.data
        function = self._loglevelFunctionDict[log.logLevel]     # 获取日志级别对应的处理函数
        msg = '\t'.join([log.dsName, log.logContent])
        function(msg)

    def eventHdlr_Error(self, event):
        '''
        处理错误事件
        '''
        if not self.__logger: return

        error = event.data
        self.error(u'错误代码：%s，错误信息：%s' %(error.errorID, error.errorMsg))

    #----------------------------------------------------------------------
    def getTempPath(name):
        '''获取存放临时文件的路径'''
        tempPath = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
            
        path = os.path.join(tempPath, name)
        return path

    #-----about shelve -----------------------------------------------------
    @abstractmethod
    def saveObject(self, sobj):
        if not self._shelve :
            return False
        self._shelve[sobj.ident] = sobj
        self._shelve.close()

    @abstractmethod
    def loadObject(self, category, id):
        '''读取对象'''
        try :
            fn = '%s/objects/%s' % (self.dataRoot, category)
            f = shelve.open(fn)
            if id in f :
                return f[id].value
        except Exception as ex:
            print("loadObject() error: %s %s" % (ex, traceback.format_exc()))

        return None
'''
    #----------------------------------------------------------------------
    def addMarketData(self, dsModule, settings):
        # 添加底层接口

        # 创建接口实例
        clsName = dsModule.__class__.__name__
        md = dsModule(self, settings)
        # 保存接口详细信息
        d = {
            'id': md.id,
            'dsDisplayName': settings.displayName(md.id),
            'dsType': clsName,
        }

        self._dictMarketDatas[md.exchange] = md
        self._dlstMarketDatas.append(d)
        self.debug('md[%s] added: %s' %(md.exchange, d))

        return md

    def getMarketData(self, dsName, exchange=None):
        # 获取接口
        if dsName in self._dictMarketDatas:
            return self._dictMarketDatas[dsName]
        else:
            self.error('getMarketData() %s not exist' % dsName)
            return None

    def subscribeMarketData(self, subscribeReq, dsName):
        # 订阅特定接口的行情
        ds = self.getMarketData(dsName)
        
        if ds:
            ds.subscribe(subscribeReq)
  
        
    #----------------------------------------------------------------------
    @property
    def dbConn(self) :
        return self._dbConn

    def dbConnect(self):
        # 连接MongoDB数据库
        if not self._dbConn:
            # 读取MongoDB的设置
            dbhost = self.__jsettings.database.host('localhost')
            dbport = self.__jsettings.database.port(27017)
            if len(dbhost) <=0:
                return

            self.debug('connecting DB[%s :%s]'%(dbhost, dbport))

            try:
                # 设置MongoDB操作的超时时间为0.5秒
                self._dbConn = MongoClient(dbhost, dbport, connectTimeoutMS=500)
                
                # 调用server_info查询服务器状态，防止服务器异常并未连接成功
                self._dbConn.server_info()

                # 如果启动日志记录，则注册日志事件监听函数
                if self.__jsettings.database.logging("") in ['True']:
                    self._eventLoop.register(LogData.EVENT_TAG, self.dbLogging)
                    
                self.info('connected DB[%s :%s] %s'%(dbhost, dbport))
            except ConnectionFailure:
                self.error('failed to connect to DB[%s :%s]' %(dbhost, dbport))
            except:
                self.error('failed to connect to DB[%s :%s]' %(dbhost, dbport))
    
    #----------------------------------------------------------------------
    @abstractmethod
    def configIndex(self, dbName, collectionName, definition, unique=False):
        # 向MongoDB中定义index
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.ensure_index(definition, unique)
        self.debug('configIndex() %s.%s added: %s' % (dbName, collectionName, definition))

    @abstractmethod
    def dbInsert(self, dbName, collectionName, d):
        # 向MongoDB中插入数据，d是具体数据
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        self.debug('dbInsert() %s[%s] adding: %s' % (dbName, collectionName, d))
        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.insert_one(d)
        self.debug('dbInsert() %s[%s] added: %s' % (dbName, collectionName, d))

    @abstractmethod
    def dbQuery(self, dbName, collectionName, d, sortKey='', sortDirection=INDEX_ASCENDING):
        # 从MongoDB中读取数据，d是查询要求，返回的是数据库查询的指针
        if not self._dbConn:
            self.error(text.DATA_QUERY_FAILED)   
            return []

        self.debug('dbQuery() %s[%s] flt: %s' % (dbName, collectionName, d))
        db = self._dbConn[dbName]
        collection = db[collectionName]
            
        if sortKey:
            cursor = collection.find(d).sort(sortKey, sortDirection)    # 对查询出来的数据进行排序
        else:
            cursor = collection.find(d)

        if cursor:
            return list(cursor)

        return []
        
    @abstractmethod
    def dbUpdate(self, dbName, collectionName, d, flt, upsert=True):
        # 向MongoDB中更新数据，d是具体数据，flt是过滤条件，upsert代表若无是否要插入
        if not self._dbConn:
            self.error(text.DATA_UPDATE_FAILED)        
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.replace_one(flt, d, upsert)

    #----------------------------------------------------------------------
    def dbLogging(self, event):
        # 向MongoDB中插入日志
        log = event.data
        d = {
            'content': log.logContent,
            'time': log.logTime,
            'ds': log.dsName
        }
        self.dbInsert(LOG_DB_NAME, self._runStartDate, d)
    
    #----------------------------------------------------------------------
    def getAllGatewayDetails(self):
        # 查询引擎中所有底层接口的信息
        return self._dlstMarketDatas

''' 

# the following is a sample
if __name__ == "__main__":

    class Foo(BaseApplication) :
        def __init__(self, program, settings):
            super(Foo, self).__init__(program, settings)
            self.__step =0

        def doAppInit(self): # return True if succ
            if not super(Foo, self).doAppInit() :
                return False
            return True

        def OnEvent(self, event):
            print("Foo.OnEvent %s" % event)
        
        def doAppStep(self):
            self.__step +=1
            print("Foo.step %d" % self.__step)

    # a = BaseApplication() #wrong
    p = Program(__file__)
    p._heartbeatInterval =-1
    p.createApp(Foo, {'asfs':1, 'aaa':'000'}, aaa='bbb', ccc='ddd')
    p.start()
    p.loop()
    p.stop()
