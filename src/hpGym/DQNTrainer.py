# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''

########################################################################
# to work as a generator for Keras fit_generator() by reading replay-buffers from HDF5 file
# sample the data as training data
class ReplayHdf5toDqnGenerator(object):

    def __init__(self, h5filepath, **kwargs):
        self._h5filepath = h5filepath
        self.__gen = None
        self._iterableEnd = False

        self.__minSamplesInPool =batchSize *1024

        self.__frameNames =[]
        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._batchesLeftFromPool = 10 # should be something like len(self.__samplePool) / batchSize, when downcounting reached 0, the self.__samplePool should be repopulated from the H5 frames

    while 1:
            f = open(path)
            for line in f:
                # create Numpy arrays of input data
                # and labels, from each line in the file
                x, y = process_line(line)
                yield (x, y)
        f.close()

        self.__gen = None
        self.__program = None
        if 'program' in kwargs.keys():
            self.__program = kwargs['program']

        self._iterableEnd = False

        # 事件队列
        self.__quePending = Queue(maxsize=100)

    def __iter__(self):
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

        self.__frameNames = []
        with h5py.File(fn_frame, 'r') as h5file:
            for name in h5file.keys():
                if 'ReplayFrame:' == name[:len('ReplayFrame:')] :
                    self.__frameNames.append(name)
            
            random.suffle(self.__frameNames)

            # build up self.__samplePool
            self.__samplePool = []
            while len(self.__frameNames) >0 && len(self.__samplePool) < self.__minSamplesInPool:
                frame = h5file[self.__frameNames[0]]
                del self.__frameNames[0]
                dqn_state, dqn_action, dqn_reward, dqn_next_state, dqn_done  = frame['state'], frame['action'], frame['reward'], frame['next_state'], frame['done']




            g = h5file.create_group('ReplayFrame:%s' % frameId)
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs['reward'] = 'reward'
            g.attrs['next_state'] = 'next_state'
            g.attrs['done'] = 'done'
            g.attrs[u'default'] = 'state'

            g.create_dataset(u'title',     data= 'replay frame[%s] of %s for DQN training' % (frameId, self.wkTrader._tradeSymbol))
            g.create_dataset('state',      data= col_state, **dsargs)
            g.create_dataset('action',     data= col_action, **dsargs)
            g.create_dataset('reward',     data= col_reward, **dsargs)
            g.create_dataset('next_state', data= col_next_state, **dsargs)
            g.create_dataset('done',       data= col_done, **dsargs)

        self.info('saved frame[%s] len[%s] to file %s' % (frameId, len(col_state), fn_frame))

        while not self._iterableEnd :
                n = self.readNext()
                if None ==n:
                    continue

                yield n
                self.__c +=1
            except StopIteration:
                self.info('reached the end')
                break
            except Exception as ex:
                self.logexception(ex)
                self._iterableEnd = True
                break

        self.__gen=None
        raise StopIteration

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

