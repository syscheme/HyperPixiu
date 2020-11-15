# encoding: UTF-8
from Application import *

import sys, os, platform, re
from redlock import Redlock, MultipleRedlockException # redis remote locker https://github.com/SPSCommerce/redlock-py/

########################################################################
class Worker(Program):

    def __init__(self, sshMount='/mnt/s', argvs=None) : # setting_filename=None):
        '''Constructor
           usage: Program(sys.argv)
        '''

        super(Worker, self).__init__(argvs)

        # the local of a worker, assuming /mnt/s is sshfs-mounted to the ${WORKER}@master-host
        # # ls -l ~/wkspaces/*
        # lrwxrwxrwx 1 root root 15 Nov 14 15:10 hpx_archived -> /mnt/s/archived
        # lrwxrwxrwx 1 root root 17 Nov 14 15:16 hpx_publish -> /mnt/s/to_publish
        # -rw-rw-rw- 1 root root 204 Nov 14 15:31 hpx_rsync_excl.txt
        # drwxrwxrwx 1 root root 26 Nov 14 12:59 hpx_template <- rsync -auv --delete --exclude-from /mnt/s/hpx_template/src/dist/rsync_excl.txt /mnt/s/hpx_template .

        self.__locker = {} # dict of name to locker
        self.__redlock = Redlock([{"host": "localhost"}])

    def remoteLock(lockerName, secTTL) :
        try :
            lock=None
            lock = self.__redlock.lock(lockerName, secTTL)
            if lock and lock.resource ==  lockerName:
                self.__locker[lockerName] = locker
                return lock
        except:
            pass

        if not lock:
            self.__redlock.unlock(lock)
        return None

    def remoteUnlock(locker) :
        if locker and isinstance(locker, string) and locker in self.__locker.keys():
            locker = self.__locker[locker]
        
        if locker:
            self.__redlock.unlock(locker)

    # overwrite Program.publish(event) in order to forward heartbeat to the master
    def publish(event) :
        super(Worker, self).publish(event)
        if EVENT_SYS_CLOCK != event.evtype:
            return
        
        # TODO send this to the master by identify self id
        pass

    def publish(files=[]):  # list of (src, dest) that wish to copy the src file to ~/wkspaces/hpx_publish/${dest}
        pass

    def forkSelf(count): # fork a number of self to do the work concurrently
        pass

