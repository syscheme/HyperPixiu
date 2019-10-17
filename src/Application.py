# encoding: UTF-8

from __future__ import division

from EventData import Event, EventData

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from collections import OrderedDict, defaultdict
from threading import Thread
from datetime import datetime
import time
from copy import copy
from abc import ABC, abstractmethod
import traceback
import shelve
import jsoncfg # pip install json-cfg
import sys
if sys.version_info <(3,):
    from Queue import Queue, Empty
else:
    from queue import Queue, Empty

########################################################################
# 常量定义
########################################################################
# 日志级别
LOGLEVEL_DEBUG    = logging.DEBUG
LOGLEVEL_INFO     = logging.INFO
LOGLEVEL_WARN     = logging.WARN
LOGLEVEL_ERROR    = logging.ERROR
LOGLEVEL_CRITICAL = logging.CRITICAL

BOOL_TRUE = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

########################################################################
class BaseApplication(ABC):

    __lastId__ =100
    
    #----------------------------------------------------------------------
    def __init__(self, program, settings=None):
        """Constructor"""

        self._program = program
        self._settings = settings
        self._active = False    # 工作状态

        self._pid = os.getpid() # process id

        # 日志级别函数映射
        self._loglevelFunctionDict = {
            LOGLEVEL_DEBUG:    self.debug,
            LOGLEVEL_INFO:     self.info,
            LOGLEVEL_WARN:     self.warn,
            LOGLEVEL_ERROR:    self.error,
            LOGLEVEL_CRITICAL: self.critical,
        }

        # the app instance Id
        if not settings:
            return

        self._id = settings.id("")
        if len(self._id)<=0 :
            BaseApplication.__lastId__ +=1
            self._id = 'P%d' % BaseApplication.__lastId__

        self._dataPath  = settings.dataPath('./data')

    #----------------------------------------------------------------------
    @property
    def ident(self) :
        return self.__class__.__name__ + self._id

    @property
    def app(self) : # for the thread wrapper
        return self

    @property
    def dataRoot(self) :
        return self._dataPath

    @property
    def program(self) :
        return self._program

    @property
    def settings(self) :
        return self._settings

    @property
    def isActive(self) :
        return self._active
    
    #--- pollable step routine for AppThreadedWrapper -----------------------
    def start(self):
        # TODO:
        self._active = True

    def stop(self):
        # TODO:
        self._active = False

    @abstractmethod
    def init(self): # return True if succ
        return True

    @abstractmethod
    def step(self):
        '''
        return sec for next yield/sleep
        '''
        return AppThreadedWrapper.MAX_INTERVAL

    #---- event operations ---------------------------
    def subscribeEvent(self, event, funcCallback) :
        if not self._program or not self._program._eventLoop:
            pass
        
        self._program._eventLoop.register(event, funcCallback)

    def postEventData(self, eventType, edata):
        """发出事件"""
        if not self._program or not self._program._eventLoop:
            return

        event = Event(type_= eventType)
        event.dict_['data'] = edata
        self.postEvent(event)

    def postEvent(self, event):
        """发出事件"""
        if not event or not self._program or not self._program._eventLoop:
            return

        self._program._eventLoop.put(event)
        self.debug('posted event[%s]' % event.dict_['type_'])

    #---logging -----------------------
    def log(self, level, msg):
        if self._program:
            self._program.log(level, 'APP['+self.ident +'] ' + msg)

    def debug(self, msg):
        self._program.debug('APP['+self.ident +'] ' + msg)
        
    def info(self, msg):
        """正常输出"""
        self._program.info('APP['+self.ident +'] ' + msg)

    def warn(self, msg):
        """警告信息"""
        self._program.warn('APP['+self.ident +'] ' + msg)
        
    def error(self, msg):
        """报错输出"""
        self._program.error('APP['+self.ident +'] ' + msg)
        
    def critical(self, msg):
        """影响程序运行的严重错误"""
        self._program.critical('APP['+self.ident +'] ' + msg)

    def logexception(self, ex):
        """报错输出+记录异常信息"""
        self._program.logexception('APP['+self.ident +'] %s: %s' % (ex, traceback.format_exc()))

    #----------------------------------------------------------------------
    def logError(self, eventType, content):
        """处理错误事件"""
    # TODO   error = event.dict_['data']
    #    self._lstErrors.append(error)
        pass

########################################################################
class AppThreadedWrapper(object):

    MAX_INTERVAL = 5 # 5sec
    #----------------------------------------------------------------------
    def __init__(self, app):
        """Constructor"""
        self._app = app
        self._thread = Thread(target=self._run)
        self._wakeup = threading.Event()

    #----------------------------------------------------------------------
    def _run(self):
        """执行连接 and receive"""
        while self._app.isActive:
            nextSleep =5
            try :
                nextSleep = min([nextSleep, self._app.step()])
            except Exception as ex:
                self._app.error('AppThreadedWrapper::step() excepton: %s' % ex)

            if nextSleep >0:
                self._wakeup(nextSleep)

        self._app.info('AppThreadedWrapper exit')

    def wakeup(self) :
        self._wakeup.set()

    #----------------------------------------------------------------------
    def start(self):
        '''
        return True if started
        '''

        if not self._app :
            return False
        
        ret = self._app.start()
        self._thread.start()
        self._app.debug('AppThreadedWrapper started')
        return ret

    #----------------------------------------------------------------------
    def stop(self):
        ''' call to stop a app
        '''
        self._app.stop()
        self.wakeup()
        self._thread.join()
        self._app.info('AppThreadedWrapper stopped')
    
########################################################################
class Singleton(type):
    """
    单例，应用方式:静态变量 __metaclass__ = Singleton
    """
    _instances = {}

    #----------------------------------------------------------------------
    def __call__(cls, *args, **kwargs):
        """调用"""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            
        return cls._instances[cls]

########################################################################
class Program(object):
    """主引擎"""

    # 单例模式
    __metaclass__ = Singleton

    #----------------------------------------------------------------------
    def __init__(self, setting_filename=None):
        """Constructor"""
        
        # dirname(dirname(abspath(file)))
        settings= None
        if setting_filename :
            try :
                settings= jsoncfg.load_config(settingfilename)
            except Exception as e :
                print('failed to load configure[%s]: %s' % (conf_fn, e))
                return

        self._settings = settings
        self._threadless = True

        # 记录今日日期
        self.todayDate = datetime.now().strftime('%Y%m%d')

        # 日志引擎实例
        self._logger = None
        self.initLogger()
        
        # 创建EventLoop
        self._eventLoop = EventLoop(self) if self.threadless else AppThreadedWrapper(EventLoop(self))
        self._bRun = True
        
    @property
    def threadless(self) :
        return self._threadless

    #----------------------------------------------------------------------
    def addApp(self, app, displayName=None):
        """添加上层应用"""
        id = app.ident
        
        # 创建应用实例
        self._dictApps[id] = app
        
        # 将应用引擎实例添加到主引擎的属性中
        self.__dict__[id] = self._dictApps[id]
        self.info('app[%s] added' %(id))

        # TODO maybe test isinstance(app, MarketData) to ease index and so on

        return app

    def createApp(self, appModule, settings):
        """添加上层应用"""
        app = appModule(self, settings)
        if not app:
            return None

        return self.addApp(app)
        
    def removeApp(self, appId):
        if not appId in self._dictApps.keys():
            return None
        
        ret = self._dictApps[appId]
        del self._dictApps[appId]
        return ret

    def getApp(self, appName):
        """获取APP引擎对象"""
        return self._dictApps[appName]

    #----------------------------------------------------------------------
    def start(self, daemonize=None):

        if daemonize == None: 
            daemonize = self._settings.daemonize(False)
        if daemonize :
            self.daemonize()

        self.debug('starting event loop')
        if self._eventLoop:
            self._eventLoop.start()
            self.info('started event loop')

        '''
        self.dbConnect()
        
        self.debug('starting market data subscribers')
        for (k, ds) in self._dictMarketDatas.items():
            if ds == None:
                continue

            if self.threadless :
                ds.connect()
                self.debug('market subscriber[%s] connected' % k)
            else :
                ds.start()
                self.debug('market subscriber[%s] started' % k)
        '''

        self.debug('starting applications')
        for (k, app) in self._dictApps.items() :
            if app == None:
                continue
            
            self.debug('staring app[%s]' % k)
            app.start()
            self.info('started app[%s]' % k)

        self.info('main-program started')

    def stop(self):
        """退出程序前调用，保证正常退出"""        
        self._bRun = False

        '''
        # 安全关闭所有接口
        for ds in self._dictMarketDatas.values():        
            ds.close()
        '''
        
        # 停止事件引擎
        self._eventLoop.stop()
        
        # 停止上层应用引擎
        for app in self._dictApps.values():
            app.stop()
        
        # # 保存数据引擎里的合约数据到硬盘
        # self.dataEngine.saveContracts()

    #----------------------------------------------------------------------
    def loop(self):

        self.info(u'Program start looping')
        c=0
        
        busy = True
        while self._bRun:
            if not self.threadless :
                try :
                    time.sleep(1)
                    self.debug(u'MainThread heartbeat')
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    break

                continue

            # loop mode as below
            if not busy:
                time.sleep(0.5)

            busy = False
            for (k, ds) in self._dictMarketDatas.items():
                try :
                    if ds == None:
                        continue
                    if ds.step() >0:
                        busy = True        
                except Exception as ex:
                    self.error("marketdata step exception %s %s" % (ex, traceback.format_exc()))

            for app in self._dictApps.values() :
                try :
                    app.step()
                except Exception as ex:
                    self.error("app step exception %s %s" % (ex, traceback.format_exc()))

            pending = self._eventLoop.pendingSize
            if not busy:
                busy = pending >0

            pending = min(20, pending)
            for i in range(0, pending) :
                try :
                    self._eventLoop.step()
                    c+=1
                except KeyboardInterrupt:
                    self.error("quit per KeyboardInterrupt")
                    exit(-1)
                except Exception as ex:
                    self.error("eventCH step exception %s %s" % (ex, traceback.format_exc()))

            # if c % 10 ==0:
            #     self.debug(u'MainThread heartbeat')

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
    
        ''' 子进程， 由于父进程已经退出，所以子进程变为孤儿进程，由init收养
        setsid使子进程成为新的会话首进程，和进程组的组长，与原来的进程组、控制终端和登录会话脱离。
        '''
        os.setsid()

        '''防止在类似于临时挂载的文件系统下运行，例如/mnt文件夹下，这样守护进程一旦运行，临时挂载的文件系统就无法卸载了，这里我们推荐把当前工作目录切换到根目录下'''
        os.chdir("/")
        '''设置用户创建文件的默认权限，设置的是权限“补码”，这里将文件权限掩码设为0，使得用户创建的文件具有最大的权限。否则，默认权限是从父进程继承得来的'''
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

    #----------------------------------------------------------------------
    # methods about logging
    # ----------------------------------------------------------------------
    def initLogger(self):
        """初始化日志引擎"""
        if not self._settings or not jsoncfg.node_exists(self._settings.logger):
            return
        
        # 创建引擎

        # abbout the logger
        # ----------------------------------------------------------------------
        self._logger   = logging.getLogger()        
        self._logfmtr  = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
        self._loglevel = LOGLEVEL_CRITICAL
        
        self._hdlrConsole = None
        self._hdlrFile = None
        
        # 日志级别函数映射
        self._loglevelFunctionDict = {
            LOGLEVEL_DEBUG:    self.debug,
            LOGLEVEL_INFO:     self.info,
            LOGLEVEL_WARN:     self.warn,
            LOGLEVEL_ERROR:    self.error,
            LOGLEVEL_CRITICAL: self.critical,
        }

        # 设置日志级别
        self.setLogLevel(self._settings.logger.level(LOGLEVEL_DEBUG)) # LOGLEVEL_INFO))
        
        BOOL_TRUE = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

        # 设置输出
        tmpval = self._settings.logger.console('True').lower()
        if tmpval in BOOL_TRUE and not self._hdlrConsole:
            """添加终端输出"""
            self._hdlrConsole = logging.StreamHandler()
            self._hdlrConsole.setLevel(self._loglevel)
            self._hdlrConsole.setFormatter(self._logfmtr)
            self._logger.addHandler(self._hdlrConsole)
        else :
            # 添加NullHandler防止无handler的错误输出
            nullHandler = logging.NullHandler()
            self._logger.addHandler(nullHandler)    

        tmpval = self._settings.logger.file('True').lower()
        if tmpval in BOOL_TRUE and not self._hdlrFile:
            # filepath = getTempPath('vnApp' + datetime.now().strftime('%Y%m%d') + '.log')
            filepath = '/tmp/vnApp%s.log' % datetime.now().strftime('%Y%m%d')
            self._hdlrFile = TimedRotatingFileHandler(filepath, when='W5', backupCount=9) # when='W5' for Satday, 'D' daily, 'midnight' rollover at midnight
           #  = RotatingFileHandler(filepath, maxBytes=100*1024*1024, backupCount=9) # now 100MB*10,  = logging.FileHandler(filepath)
            self._hdlrFile.setLevel(self._loglevel)
            self._hdlrFile.setFormatter(self._logfmtr)
            self._logger.addHandler(self._hdlrFile)
            
        # 注册事件监听
        tmpval = self._settings.logger.event.log('True').lower()
        if tmpval in BOOL_TRUE :
            self._eventLoop.register(EventChannel.EVENT_LOG, self.eventHdlr_Log)

        tmpval = self._settings.logger.event.error('True').lower()
        if tmpval in BOOL_TRUE :
            self._eventLoop.register(EventChannel.EVENT_ERROR, self.eventHdlr_Error)

    def setLogLevel(self, level):
        """设置日志级别"""
        if self._logger ==None:
            return

        self._logger.setLevel(level)
        self._loglevel = level
    
    @abstractmethod
    def log(self, level, msg):
        if not self._loglevelFunctionDict or not level in self._loglevelFunctionDict : 
            return
        
        function = self._loglevelFunctionDict[level] # 获取日志级别对应的处理函数
        function(msg)

    @abstractmethod
    def debug(self, msg):
        """开发时用"""
        if self._logger ==None:
            return

        self._logger.debug(msg)
        
    @abstractmethod
    def info(self, msg):
        """正常输出"""
        if self._logger ==None:
            return

        self._logger.info(msg)

    @abstractmethod
    def warn(self, msg):
        """警告信息"""
        if self._logger ==None:
            return

        self._logger.warn(msg)
        
    @abstractmethod
    def error(self, msg):
        """报错输出"""
        if self._logger ==None:
            return

        self._logger.error(msg)
        
    @abstractmethod
    def critical(self, msg):
        """影响程序运行的严重错误"""
        if self._logger ==None:
            return

        self._logger.critical(msg)

    def logexception(self, ex):
        """报错输出+记录异常信息"""
        if self._logger ==None:
            return

        traceback.format_exc()
        #s elf._logger.exception(msg)

    def eventHdlr_Log(self, event):
        """处理日志事件"""
        if self._logger ==None:
            return

        log = event.dict_['data']
        function = self._loglevelFunctionDict[log.logLevel]     # 获取日志级别对应的处理函数
        msg = '\t'.join([log.dsName, log.logContent])
        function(msg)

    def eventHdlr_Error(self, event):
        """
        处理错误事件
        """
        if self._logger ==None:
            return

        error = event.dict_['data']
        self.error(u'错误代码：%s，错误信息：%s' %(error.errorID, error.errorMsg))

    #----------------------------------------------------------------------
    def getTempPath(name):
        """获取存放临时文件的路径"""
        tempPath = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
            
        path = os.path.join(tempPath, name)
        return path

'''
    #----------------------------------------------------------------------
    def addMarketData(self, dsModule, settings):
        """添加底层接口"""

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
        self.info('md[%s] added: %s' %(md.exchange, d))

        return md

    def getMarketData(self, dsName, exchange=None):
        """获取接口"""
        if dsName in self._dictMarketDatas:
            return self._dictMarketDatas[dsName]
        else:
            self.error('getMarketData() %s not exist' % dsName)
            return None

    def subscribeMarketData(self, subscribeReq, dsName):
        """订阅特定接口的行情"""
        ds = self.getMarketData(dsName)
        
        if ds:
            ds.subscribe(subscribeReq)
  
        
    #----------------------------------------------------------------------
    @property
    def dbConn(self) :
        return self._dbConn

    def dbConnect(self):
        """连接MongoDB数据库"""
        if not self._dbConn:
            # 读取MongoDB的设置
            dbhost = self._settings.database.host('localhost')
            dbport = self._settings.database.port(27017)
            if len(dbhost) <=0:
                return

            self.debug('connecting DB[%s :%s]'%(dbhost, dbport))

            try:
                # 设置MongoDB操作的超时时间为0.5秒
                self._dbConn = MongoClient(dbhost, dbport, connectTimeoutMS=500)
                
                # 调用server_info查询服务器状态，防止服务器异常并未连接成功
                self._dbConn.server_info()

                # 如果启动日志记录，则注册日志事件监听函数
                if self._settings.database.logging("") in ['True']:
                    self._eventLoop.register(LogData.EVENT_TAG, self.dbLogging)
                    
                self.info('connected DB[%s :%s] %s'%(dbhost, dbport))
            except ConnectionFailure:
                self.error('failed to connect to DB[%s :%s]' %(dbhost, dbport))
            except:
                self.error('failed to connect to DB[%s :%s]' %(dbhost, dbport))
    
    #----------------------------------------------------------------------
    @abstractmethod
    def dbEnsureIndex(self, dbName, collectionName, definition, unique=False):
        """向MongoDB中定义index"""
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.ensure_index(definition, unique)
        self.debug('dbEnsureIndex() %s.%s added: %s' % (dbName, collectionName, definition))

    @abstractmethod
    def dbInsert(self, dbName, collectionName, d):
        """向MongoDB中插入数据，d是具体数据"""
        if not self._dbConn:
            self.error(text.DATA_INSERT_FAILED)
            return

        self.debug('dbInsert() %s[%s] adding: %s' % (dbName, collectionName, d))
        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.insert_one(d)
        self.debug('dbInsert() %s[%s] added: %s' % (dbName, collectionName, d))

    @abstractmethod
    def dbQuery(self, dbName, collectionName, d, sortKey='', sortDirection=ASCENDING):
        """从MongoDB中读取数据，d是查询要求，返回的是数据库查询的指针"""
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
        """向MongoDB中更新数据，d是具体数据，flt是过滤条件，upsert代表若无是否要插入"""
        if not self._dbConn:
            self.error(text.DATA_UPDATE_FAILED)        
            return

        db = self._dbConn[dbName]
        collection = db[collectionName]
        collection.replace_one(flt, d, upsert)

    #----------------------------------------------------------------------
    def dbLogging(self, event):
        """向MongoDB中插入日志"""
        log = event.dict_['data']
        d = {
            'content': log.logContent,
            'time': log.logTime,
            'ds': log.dsName
        }
        self.dbInsert(LOG_DB_NAME, self.todayDate, d)
    
    #----------------------------------------------------------------------
    def getAllGatewayDetails(self):
        """查询引擎中所有底层接口的信息"""
        return self._dlstMarketDatas
''' 

########################################################################
class EventLoop(BaseApplication):
    """
    非线程的事件驱动引擎        
    """
    #----------------------------------------------------------------------
    def __init__(self, program, timer=True):
        """初始化事件引擎"""
        super(EventLoop, self).__init__(program)

        # 事件队列
        self.__queue = Queue()
        
        # 计时器，用于触发计时器事件
        self.__timerActive = timer     # 计时器工作状态
        self.__timerStep = 60           # 计时器触发间隔（默认1秒）
        self.__stampTimerLast = None
        
        # 这里的__handlers是一个字典，用来保存对应的事件调用关系
        # 其中每个键对应的值是一个列表，列表中保存了对该事件进行监听的函数功能
        self.__handlers = defaultdict(list)
        
        # __generalHandlers是一个列表，用来保存通用回调函数（所有事件均调用）
        self.__generalHandlers = []        
        
    #--- override of BaseApplication routine for AppThreadedWrapper -----------------------
    def step(self):
        """引擎运行"""
        dt = datetime.now()
        stampNow = datetime2float(datetime.now())
        c =0
        if not self.__stampTimerLast :
            self.__stampTimerLast = stampNow

        if self.__timerActive and self.__stampTimerLast + self.__timerStep < stampNow:
            self.__stampTimerLast = stampNow
                
            # 向队列中存入计时器事件
            edata = edTimer(dt)
            event = Event(type_= EventChannel.EVENT_TIMER)
            event.dict_['data'] = edata
            self.put(event)

        # pop the event to dispatch
        event = None
        try :
            event = self.__queue.get(block = True, timeout = 0.5)  # 获取事件的阻塞时间设为1秒
        except Empty:
            pass

        try:
            if event :
                self.__process(event)
                c+=1
        except Exception as ex:
            self.error("eventCH exception %s %s" % (ex, traceback.format_exc()))

        return 0 # because self.__queue.get() is blockable, we don't wish to sleep outside of step()
            
    #----------------------------------------------------------------------
    def __process(self, event):
        """处理事件"""
        # 检查是否存在对该事件进行监听的处理函数
        if event.type_ in self.__handlers:
            # 若存在，则按顺序将事件传递给处理函数执行
            for handler in self.__handlers[event.type_] :
                try:
                    handler(event)
                except Exception as ex:
                    self.error("eventCH handle(%s) %s: %s %s" % (event.type_, ex, handler, traceback.format_exc()))
            
        # 调用通用处理函数进行处理
        if self.__generalHandlers:
            for handler in self.__generalHandlers :
                try:
                    handler(event)
                except Exception as ex:
                    self.error("eventCH handle %s %s" % (ex, traceback.format_exc()))
               
    #----------------------------------------------------------------------
    def start(self, timer=True):
        # 启动计时器，计时器事件间隔默认设定为1秒
        if timer:
            self.__timerActive = True
        super(EventLoop, self).start()
    
    #----------------------------------------------------------------------
    def stop(self):
        """停止引擎"""
        pass
            
    #----------------------------------------------------------------------
    def register(self, type_, handler):
        """注册事件处理函数监听"""
        # 尝试获取该事件类型对应的处理函数列表，若无defaultDict会自动创建新的list
        handlerList = self.__handlers[type_]
        
        # 若要注册的处理器不在该事件的处理器列表中，则注册该事件
        if handler not in handlerList:
            handlerList.append(handler)
            
    #----------------------------------------------------------------------
    def unregister(self, type_, handler):
        """注销事件处理函数监听"""
        # 尝试获取该事件类型对应的处理函数列表，若无则忽略该次注销请求   
        handlerList = self.__handlers[type_]
            
        # 如果该函数存在于列表中，则移除
        if handler in handlerList:
            handlerList.remove(handler)

        # 如果函数列表为空，则从引擎中移除该事件类型
        if not handlerList:
            del self.__handlers[type_]  
        
    #----------------------------------------------------------------------
    def put(self, event):
        """向事件队列中存入事件"""
        self.__queue.put(event)

    #----------------------------------------------------------------------
    @property
    def pendingSize(self):
        return self.__queue.qsize()

    #----------------------------------------------------------------------
    def registerGeneralHandler(self, handler):
        """注册通用事件处理函数监听"""
        if handler not in self.__generalHandlers:
            self.__generalHandlers.append(handler)
            
    #----------------------------------------------------------------------
    def unregisterGeneralHandler(self, handler):
        """注销通用事件处理函数监听"""
        if handler in self.__generalHandlers:
            self.__generalHandlers.remove(handler)

if __name__ == "__main__":
    p = Program()
    # a = BaseApplication() #wrong
