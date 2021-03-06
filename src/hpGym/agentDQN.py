# encoding: UTF-8

'''
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
'''
from .GymTrader import GymTrader, MetaAgent
from MarketData import EXPORT_FLOATS_DIMS

import random

import numpy as np
# from keras.layers import Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D,GlobalAveragePooling1D
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.models import model_from_json
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras import backend as backend
from tensorflow.keras.utils import Sequence

from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
import os, threading
import json
import h5py, tarfile, numpy

# GPUs = backend.tensorflow_backend._get_available_gpus()
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()

if len(GPUs) >1:
    from keras.utils.training_utils import multi_gpu_model

########################################################################
class agentDQN(MetaAgent):
    DEFAULT_BRAIN_ID = 'VGG16d1' # 'VGG16d1' 'Cnn1Dx4R2'

    def __init__(self, gymTrader, **kwargs):
        self.__brainDict = {
            'DQN_DrDrDl'     : self.__dqn_DrDrDl, 
            'DQN_Dr64Dr32x3' : self.__dqn_Dr64Dr32x3, # not good yet
            'DQN_Cnn1Dx4'    : self.__dqn_Cnn1Dx4,
            # TODO: other brains
        }

        super(agentDQN, self).__init__(gymTrader, **kwargs)

        self._masterExportHomeDir = self.getConfig('masterHomeDir', None) # this agent work as the master when configured, usually point to a dir under webroot
        self._epochsPerObservOnGpu = self.getConfig('epochsPerObservOnGpu', 5)
        self._lock = threading.Lock()
        self.__replaySize = self._batchSize *64
        if self.__replaySize <1024:
            self.__replaySize = 1024
        if self.__replaySize >10240:
            self.__replaySize = 10240

        self.__replayCache = [None] # * self._batchSize
        self.__sampleIdx = 0
        self.__realDataNum =0
        self.__frameNum =0
        self._loss = None

        self._brainOutDir = '%s%s/' % (self._outDir, self._wkBrainId)

        #format the desc
        self._brainDesc = '%s created at %s, outdir %s' % (self._wkBrainId, datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'), self._brainOutDir)
        lines = []
        self._brain.summary(print_fn=lambda x: lines.append(x))
        self._brainDesc += '\nsummary:\n%s\n' % "\n".join(lines)
        basicAttrs = {
            'Id': self._wkBrainId,
            'GPUs': len(GPUs),
            'summary': '\n' + '\n'.join(lines),
            'stateSize': self._stateSize,
            'actionSize': self._actionSize,
            'created': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'outdir': self._brainOutDir
        }

        self._statusAttrs = { **self._statusAttrs, **basicAttrs }
        self.__bWarmed = False

        # prepare the callbacks
        self._fitCallbacks=[]
        self._cbNewReplayFrame = [self.OnNewFrame] # lisf of callback(frameId, col_state, col_action, col_reward, col_next_state, col_done) to hook new ReplayFrame prepared

        # don't take update_freq='epoch' or batch as we train almost every gymObserve()
        # cbTB = TensorBoard(log_dir =self._brainOutDir, update_freq=self._batchSize *50000)
        # self._fitCallbacks.append(cbTB) # still too frequent

    def __del__(self):  # the destructor
        # self.saveBrain()
        pass

    @property
    def frameNum(self) : return self.__frameNum

    def enableMaster(self, homeDir):
        self._masterExportHomeDir = homeDir # this agent work as the master when configured, usually point to a dir under webroot

    def buildBrain(self, brainId =None, **feedbacks): #TODO param brainId to load json/HD5 from dataRoot/brainId
        '''Build the agent's brain
        '''
        self._statusAttrs = {**self._statusAttrs, **feedbacks}
        if not brainId or len(brainId) <=0 :
            brainId = agentDQN.DEFAULT_BRAIN_ID
            self._gymTrader.warn('taking default brain[%s]' % (brainId))

        self._brain = self.loadBrain(brainId)
        if not self._brain :
            # build the new brain
            builder = None
            if brainId in self.__brainDict.keys():
                builder = self.__brainDict[brainId]

            if builder:
                self._gymTrader.info('no pre-saved brain[%s], building a new one' % (brainId))
                self._brain = builder()

        if self._brain:
            self._wkBrainId = '%s.S%dA%d' %(brainId, self._stateSize, self._actionSize)
            self._brain.compile(loss='mse', optimizer=Adam(lr=self._learningRate))
            # checkpointPath ='best.h5'
            # checkpoint = ModelCheckpoint(filepath=checkpointPath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
            # self._brain.flt(x, Y...., callbacks=[checkpoint])
            # more additiona, to customize the checkpoint other than max(val_acc), such as max(mean(y_pred))
            #    import keras.backend as K
            #    def mean_pred(y_true, y_pred):
            #        return K.mean(y_pred)
            #    model.compile(..., metrics=['accuracy', mean_pred])
            #    ModelCheckpoint(..., monitor='val_mean_pred', mode='max', period=1)

        return self._brain

    def __dqn_DrDrDl(self):
        '''Build the agent's brain
        '''
        model = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        model.add(Dense(neurons_per_layer,
                        input_dim=self._stateSize,
                        activation=activation))
        model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(self._actionSize, activation='linear'))

        return model

    def __dqn_Dr64Dr32x3(self):
        '''Build the agent's brain
        '''
        model = Sequential()
        neurons_per_layer = 32
        activation = "relu"
        model.add(Dense(256,
                        input_dim=self._stateSize,
                        activation=activation))
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation=activation))
        model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(neurons_per_layer, activation=activation))

        model.add(Dense(self._actionSize, activation='linear'))

        return model

    def __dqn_Cnn1Dx4(self):

        '''
        https://www.codercto.com/a/38746.html
        '''
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        model.add(Conv1D(100, 10, activation='relu', input_shape=(self._stateSize/EXPORT_FLOATS_DIMS, EXPORT_FLOATS_DIMS)))
        model.add(Conv1D(100, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(self._actionSize, activation='linear')) # Q func, the output of DQN, always take float results so DO NOT take activation='softmax', which only result 1 or 0

        return model

    def isReady(self) :
        if not super(agentDQN, self).isReady():
            return False
            
        return not None in self.__replayCache

    @property
    def trainable(self):
        return False # self._learningRate > 0.00001 and self._epsilon > self._epsilonMin

    def saveBrain(self, brainId=None, **feedbacks) :
        ''' save the current brain into the dataRoot
        @param a unique brainId must be given
        '''
        if not brainId or len(brainId) <=0:
            brainId = self._wkBrainId
        if not brainId or len(brainId) <=0:
            return

        if not self._gymTrader.dataRoot or not self._brain :
            raise ValueError("Null brain or Null trader")

        self._gymTrader.debug('saving brain[%s] at %s' % (brainId, self._brainOutDir))
        try :
            os.makedirs(self._brainOutDir)
        except:
            pass

        with self._lock:
            # step 1. save the model file in json
            model_json = self._brain.to_json()
            with open('%smodel.json' % self._brainOutDir, 'w') as mjson:
                mjson.write(model_json)
            
            # step 2. save the weights of the model
            self._brain.save('%smodel.json.h5' % self._brainOutDir)

            # TODO step 3. the status.json
            attrsToUpdate = {
                'loss'    : round(self._loss.history["loss"][0], 6) if self._loss else 'n/a',
                'epsilon' : round(self._epsilon, 6),
                'learningRate' : self._learningRate,
                'saveTime' : datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            }
            self._statusAttrs = {**self._statusAttrs, **attrsToUpdate}

            with open('%sstatus.json' % self._brainOutDir, 'w') as outfile:
                json.dump(self._statusAttrs, outfile)

        self._gymTrader.info('saved brain[%s] with weights, status: %s; fd: %s' % (self._brainOutDir, str(attrsToUpdate), str(feedbacks)))
        
    def loadBrain(self, brainId) :
        ''' load the previous saved brain
        @param a unique brainId must be given
        '''
        if not brainId or len(brainId) <=0:
            raise ValueError("No brainId specified")

        if not self._gymTrader :
            raise ValueError("Null trader")

        brainDir = '%s%s.S%dI%dA%d/' % (self._gymTrader.dataRoot, brainId, self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
        brain = None
        strLoadedParts =''
        try : 
            # step 1. read the model file in json
            self._gymTrader.debug('loading saved brain from %s' %brainDir)
            with open('%smodel.json' % brainDir, 'r') as mjson:
                model_json = mjson.read()
            brain = model_from_json(model_json)
            strLoadedParts +='definition'

            # step 2. read the weights of the model
            self._gymTrader.debug('loading saved brain weights from %s' %brainDir)
            brain.load_weights('%sweights.h5' % brainDir)
            strLoadedParts +=',weights'
        except Exception as ex:
            self._gymTrader.logexception(ex)

            # step 3. if load weight successfully, do not start over to mess-up the trained model by
            # limiting epsilon from status.json
        try :
            with open('%sstatus.json' % brainDir, 'r') as f:
                self._statusAttrs = json.loads(f.read())
                strLoadedParts +=',status'

            continued = self._statusAttrs['continued'] if 'continued' in self._statusAttrs.keys() else []
            continued.append(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'))

            attrsToUpdate = {
                'continued': continued
            }
            self._statusAttrs = {**self._statusAttrs, **attrsToUpdate}

            self._epsilon = float(self._statusAttrs['epsilon']) if 'epsilon' in self._statusAttrs.keys() else self._epsilon*0.7
            self._learningRate = float(self._statusAttrs['learningRate']) if 'learningRate' in self._statusAttrs.keys() else self._learningRate/2
        except:
            self._gymTrader.warn('no status.json under %s' % (brainDir))

        if brain:
            self._gymTrader.info('loaded brain %s from %s by taking initial epsilon[%s] learningRate[%s]' % (brainDir, strLoadedParts, self._epsilon, self._learningRate))
        return brain

    def gymAct(self, state, bObserveOnly=False):
        '''Acting Policy of the agentDQN
        @return one of self.__gymTrader.ACTIONS
        '''
        if bObserveOnly:
            return GymTrader.ACTIONS[GymTrader.ACTION_HOLD]

        action = np.zeros(self._actionSize)
        if self.trainable and np.random.rand() <= self._epsilon:
            action[random.randrange(self._actionSize)] = 1
        else:
            state = state.reshape(1, self._stateSize)
            with self._lock:
                act_values = self._brain.predict(state)
                action[np.argmax(act_values[0])] = 1

        return action.astype(GymTrader.NN_FLOAT)

    def gymObserve(self, state, action, reward, next_state, done, bObserveOnly=False, **feedbacks):
        '''Memory Management and training of the agent
        @param bObserveOnly True if the account is not executable, so only observing, no predicting and/or training
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self._statusAttrs = {**self._statusAttrs, **feedbacks}
        if not self._pushToReplay(state, action, reward, next_state, done) or bObserveOnly or not self.trainable:
            return None

        # this basic DQN also performs training in this step
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._sampleBatches()
        sampleLen = len(state_batch) if not state_batch is None else 0
        if not self.trainable or sampleLen < self._batchSize: 
            return None

        with self._lock:
            Q_next = self._brain.predict(next_state_batch) # arrary(sampleLen, actionSize)
            Q_next_max= np.amax(Q_next, axis=1) # arrary(sampleLen, 1)
            reward_batch += (self._gamma * np.logical_not(done_batch) * Q_next_max) # arrary(sampleLen, 1)

            Q_target = self._brain.predict(state_batch)
            Q_target[action_batch[0], action_batch[1]] = reward_batch # action =arrary(2,sampleLen)

            # x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
            # y：标签，numpy array
            # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
            # epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
            # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
            # callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
            # validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
            # validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
            # shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
            # class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
            # sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode=’temporal’。
            # initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
            epochs =1
            if len(GPUs) > 0 and sampleLen >self._batchSize:
                epochs = self._epochsPerObservOnGpu
            self._loss = self._brain.fit(x=state_batch, y=Q_target, epochs=epochs, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)
        return self._loss

    def _pushToReplay(self, state, action, reward, next_state, done):
        '''record the input tuple(state, action, reward, next_state, done) into the cache, then pick out a batch
           of samples from the cache for training
        @return True if warmed up and ready to train
        '''
        with self._lock:
            self.__sampleIdx = self.__sampleIdx % self.__replaySize
            if not self.__bWarmed and not None in self.__replayCache :
                self.__bWarmed = True
                self.__sampleIdx =0

            samplelen = len(self.__replayCache)
            if 0 == self.__sampleIdx and samplelen > self._batchSize: 
                self.__frameNum +=1
                frameId = 'F%s' % (str(self.__frameNum).zfill(4))
                if len(self._cbNewReplayFrame) >0:
                    metrix = np.array(self.__replayCache)
                    col_state      = np.concatenate(metrix[:, 0]).reshape(samplelen, self._stateSize)
                    col_action     = np.concatenate(metrix[:, 1]).reshape(samplelen, self._actionSize)
                    col_reward     = metrix[:, 2].astype('float32')
                    col_next_state = np.concatenate(metrix[:, 3]).reshape(samplelen, self._stateSize)
                    col_done       = [ 1 if i else 0 for i in metrix[:, 4]]
                    for cb in self._cbNewReplayFrame:
                        try:
                            cb(frameId, col_state, col_action, col_reward, col_next_state, col_done)
                        except Exception as ex:
                            self._gymTrader.logexception(ex)

            if self.__sampleIdx >= samplelen :
                self.__replayCache.append((state, action, reward, next_state, done))
                self.__sampleIdx = samplelen
            else :
                self.__replayCache[self.__sampleIdx] = (state, action, reward, next_state, done)
                self.__sampleIdx +=1

            if not self.__bWarmed:
                self.__realDataNum =0
                return False

            self.__realDataNum += 1
            if self.__realDataNum <self._batchSize or 0 != (self.__sampleIdx % self._trainInterval):
                return False
        
        return True

    def _sampleBatches(self):
        '''Selecting a batch of memory as sample, split it into categorical subbatches
           Process action_batch into a position vector
        @return tuple of the sampled batch:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        TODO: if sampling take loss-of-fit as priority to always keep those have most losses as high priorities to pop for next round,
        the training could be more effictive. To get the loss of each sample within a fit-training, a source exmaple could be like:
        https://stackoverflow.com/questions/57087047/how-to-get-loss-for-each-sample-within-a-batch-in-keras

                model = keras.models.Sequential([
                    keras.layers.Input(shape=(4,)),
                    keras.layers.Dense(1)
                ])

                def examine_loss(y_true, y_pred):
                    result = keras.losses.mean_squared_error(y_true, y_pred)
                    result = K.print_tensor(result, message='losses')
                    return result

                model.compile('adam', examine_loss)
                model.summary()

                X = np.random.rand(100, 4)

                def test_fn(x):
                    return x[0] * 0.2 + x[1] * 5.0 + x[2] * 0.3 + x[3] + 0.6

                y = np.apply_along_axis(test_fn, 1, X)

                model.fit(X[0:4], y[0:4])
                
        You should seem something like the following:
        losses [23.2873611 26.1659927 34.1300354 6.16115761]
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch =None,None,None,None,None
        with self._lock:
            sizeToBatch = self._batchSize
            if None in self.__replayCache or len(self.__replayCache) < sizeToBatch:
                return state_batch, action_batch, reward_batch, next_state_batch, done_batch

            if len(GPUs) > 0 and len(self.__replayCache) > self._batchSize:
                sizeToBatch = min(10, int(len(self.__replayCache) / self._batchSize)) *self._batchSize
            
            batch = np.array(random.sample(self.__replayCache, sizeToBatch))

            # pick up the each part from the batch as we inserted via __replayCache[__sampleIdx] = (state, action, reward, next_state, done)
            state_batch = np.concatenate(batch[:, 0]).reshape(sizeToBatch, self._stateSize)
            action_batch = np.concatenate(batch[:, 1]).reshape(sizeToBatch, self._actionSize) # array(sizeToBatch, self._actionSize)
            reward_batch = batch[:, 2] #array(sizeToBatch, 1)
            next_state_batch = np.concatenate(batch[:, 3]).reshape(sizeToBatch, self._stateSize)
            done_batch = batch[:, 4] #array(sizeToBatch, 1)

            # action processing
            action_batch = np.where(action_batch == 1) # array(sizeToBatch, self._actionSize)=>array(2, sizeToBatch): array[0]=[index of item] arrayp[1]=[2d index of where=1]
            # >>> a = np.array([[0,1,0],[1,0,0],[0,0,1],[1,0,0]])
            # >>> link =np.where(a==1)
            # >>> link
            # (array([0, 1, 2, 3]), array([1, 0, 2, 0]))
            # >>> b =a
            # >>> b[link[0],link[1]] =[1,2,3,4]
            # >>> b
            # array([[0, 1, 0],[2, 0, 0],[0, 0, 3],[4, 0, 0]])
            
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def OnNewFrame(self, frameId, col_state, col_action, col_reward, col_next_state, col_done) :

        if not self._masterExportHomeDir or len(self._masterExportHomeDir) <=0:
            return # not as the master
        
        if not col_state or len(col_state) <=0:
            return
        
        if '/' != self._masterExportHomeDir[-1]: self._masterExportHomeDir +='/'

        try :
            statinfo = os.stat(self._masterExportHomeDir)
        except :
            self._gymTrader.error('agent.OnNewFrame() failed to be the master as exportHome[%s] not exists' % (self._masterExportHomeDir))
            self._masterExportHomeDir = None
            return

        brainInst = '%s.%s' % (self._wkBrainId, datetime.now().strftime('%Y%m%dT%H%M%S%f'))

        # collect all needed files into a tmpdir
        tmpdir = '%sft_tmp.%s/' % (self._outDir, brainInst)
        try :
            os.makedirs(tmpdir)
        except :
            pass

        os.system('cp -f %smodel.* %s' % (self._brainOutDir, tmpdir))
        if not 'model.json.h5' in os.listdir(tmpdir) :
            os.system('cp -f %s%s/model.* %s' % (self._gymTrader.dataRoot, self._wkBrainId, tmpdir))
        
        # output the frame into a HDF5 file
        fn_frame = '%sframes.h5' % tmpdir
        self._gymTrader.debug('agent.OnNewFrame() collected fit task supplemental files, generating frame file %s' % (fn_frame))
        # with h5py.File(fn_frame, 'w') as h5file:
        #     rowShape=numpy.asarray(frame[0]).shape
        #     X = h5file.create_dataset(shape=(rows, rowShape[0],),   # 数据集的维度
        #                         dtype=float, name='fitframe',    # no compression needed
        #                         chunks=(self._batchSize, rowShape[0],))
        #     X[0:rows, :,] = frame

        with h5py.File(fn_frame, 'w') as h5file:
            g = h5file.create_group('ReplayBuffer')
            g.attrs['state'] = 'state'
            g.attrs['action'] = 'action'
            g.attrs['reward'] = 'reward'
            g.attrs['next_state'] = 'next_state'
            g.attrs['done'] = 'done'
            g.attrs[u'default'] = 'state'

            g.create_dataset(u'title',     data= '%s replay buffer for NN training' % brainInst)
            done_col = [ 1 if i else 0 for i in metrix[:, 4]]
            g.create_dataset('state',      data= col_state)
            g.create_dataset('action',     data= col_action)
            g.create_dataset('reward',     data= col_reward)
            g.create_dataset('next_state', data= col_next_state)
            g.create_dataset('done',       data= col_done)

        # now make a tar ball as the task file
        # this is a tar.bz2 including a) model.json, b) current weight h5 file, c) version-number, d) the frame exported as hdf5 file
        fn_fit_task = '%stasks/fit_%s.tak' % (self._masterExportHomeDir, brainInst)
        try :
            os.makedirs(os.path.dirname(fn_fit_task))
        except :
            pass

        with tarfile.open(fn_fit_task, "w:bz2") as tar:
            files = os.listdir(tmpdir)
            for f in files:
                tar.add('%s%s' % (tmpdir, f), f)

        self._gymTrader.debug('agent.OnNewFrame() prepared task-file %s, activating it' % (fn_fit_task))
        os.system('rm -rf %s' % tmpdir)

        # swap into the active task
        target_task_file = '%stasks/fit.tak' % self._masterExportHomeDir
        os.system('rm -rf $(realpath %s) %s' % (target_task_file, target_task_file))
        os.system('ln -sf %s %s' % (fn_fit_task, target_task_file))
        self._gymTrader.info('agent.OnNewFrame() fit-task updated: %s->%s' % (target_task_file, fn_fit_task))

    def OnRemoteResult_fitTask(self, resultFn) :
        ''' merge the fit-training result generated by the remote slave into the current brain
        @resultFn  the full path name of the result file (supporsed to be under self._masterExportHomeDir+'res/'), which is bz2 tarball that consists of
          - model.json.h5 the result weights
          - status.json the description of the training result
          # NO THIS IS FOR SIMTASK - GymTrader.tcsv the records of the training
        '''
        # step 1. extract necessary files into tmpdir
        tmpdir = os.path.join(self._outDir, 'ftr_tmp.%s' % datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        try :
            os.makedirs(tmpdir)
        except :
            pass

        with tarfile.open(resultFn, "r:bz2") as tar:
            tar.extract('model.json.h5', path=tmpdir)
            tar.extract('status.json', path=tmpdir)

        # step 2. read the status.json to verify the brainId and more, determine the w_import (<= 0.5) of this remote result
        w_import =0.5

        # step 3. MAYBE can be smarter: create a dummy brain to load the h5 file then export its weights_to_import

        # step 4. merge the weights of self._brain and weights_to_import
        # importRemoteWeights(self, weights_to_import, w_import)

        # step 4. move the result file to imported folder and clean the tmpdir
        finishedDir = os.path.join(self._outDir, 'imported')
        try :
            os.makedirs(finishedDir)
        except :
            pass
        os.system('mv -f %s %s/' % (resultFn, finishedDir))
        os.system('rm -rf %s' % tmpdir)

    def importRemoteWeights(self, weights, w_import=0.01) :
        w_keep = 1.0 - w_import
        if w_keep < 0.1 or w_keep>=1.0: return False
        w_total = w_keep + w_import
        w_import /= w_total
        w_keep /= w_total

        # merge the weights of self._brain and weights_to_import
        with self._lock:
            pass
            # weights_to_merge = [self._brain.get_weights(), weights_to_import]
            # weightsResult = list()

            # for t in zip(*weights_to_merge):
            #     # weightsResult.append( [numpy.array(item).mean(axis=0) for item in zip(*t)])
            #     weightsResult.append( [numpy.array(item[0]*w_keep + item[1]*w_import) for item in zip(*t)])
            # self._brain.set_weights(weightsResult)

        return True

########################################################################
# DoubleDQN to avoid overestimate, 将动作选择（max操作）和动作估计Q(s’,a’)解耦
class agentDoubleDQN(agentDQN):
    def __init__(self, gymTrader, **kwargs):
        super(agentDoubleDQN, self).__init__(gymTrader, **kwargs)

        # treat the orginal self._brain as the right hemicerebrum to perform predicting, and 
        # additional self._theOther to perform training in an additional thread
        self._theOther = None
        if self._brain : 
            model_json = self._brain.to_json()
            self._theOther = model_from_json(model_json)
            self._theOther.set_weights(self._brain.get_weights()) 
            self._theOther.compile(loss='mse', optimizer=Adam(lr=self._learningRate))

    def isReady(self) :
        if not self._theOther:
            return False

        if not super(agentDoubleDQN, self).isReady():
            return False

    def gymObserve(self, state, action, reward, next_state, done, bObserveOnly=False, **feedbacks):
        '''Memory Management and training of the agent
        @param bObserveOnly True if the account is not executable, so only observing, no predicting and/or training
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self._statusAttrs = {**self._statusAttrs, **feedbacks}
        if not self._pushToReplay(state, action, reward, next_state, done) or bObserveOnly :
            return None

        # this basic DQN also performs training in this step
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._sampleBatches()
        sampleLen = len(state_batch) if state_batch else 0
        if sampleLen < self._batchSize: 
            return None

        with self._lock:
            if np.random.rand() < 0.5:
                brainPred  = self._brain
                brainTrain = self._theOther
            else:
                brainPred  = self._theOther
                brainTrain = self._brain
            
            Q_next = brainPred.predict(next_state_batch) # arrary(sampleLen, actionSize)
            Q_next_max = np.amax(Q_next, axis=1) # arrary(sampleLen, 1)
            reward_batch += (self._gamma * np.logical_not(done_batch) * Q_next_max) # arrary(sampleLen, 1)

            Q_target = brainTrain.predict(state_batch)
            Q_target[action_batch[0], action_batch[1]] = reward_batch # action =arrary(2,sampleLen)

            epochs =1
            if len(GPUs) > 0 and sampleLen >self._batchSize:
                epochs = self._epochsPerObservOnGpu
            self._loss = brainTrain.fit(x=state_batch, y=Q_target, epochs=epochs, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)

        return self._loss

'''
########################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import envTrading as envt

    # from tgym.gens.deterministic import WavySignal
    # Instantiating the environmnent
    generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)
    episodes = 50
    episode_length = 400
    trading_fee = .2
    time_fee = 0
    history_length = 2
    environment = envt.envTrading(spread_coefficients=[1],
                                data_generator=generator,
                                trading_fee=trading_fee,
                                time_fee=time_fee,
                                history_length=history_length,
                                episode_length=episode_length)
                                
    state = environment.gymReset()
    # Instantiating the agent
    memory_size = 3000
    state_size = len(state)
    gamma = 0.96
    epsilon_min = 0.01
    batch_size = 64
    action_size = len(envTrading.ACTIONS)
    train_interval = 10
    learning_rate = 0.001
    agent = agentDQN(state_size=state_size,
                     action_size=action_size,
                     memory_size=memory_size,
                     episodes=episodes,
                     episode_length=episode_length,
                     train_interval=train_interval,
                     gamma=gamma,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     epsilon_min=epsilon_min)
    
    # Warming up the agent
    for _ in range(memory_size):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state, done, warming_up=True)
        # this state is regardless state-stepping, rewards and loss

    # Training the agent
    for ep in range(episodes):
        state = environment.gymReset()
        rew = 0
        for _ in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            state = next_state
            rew += reward

        print("Ep:" + str(ep)
              + "| rew:" + str(round(rew, 2))
              + "| eps:" + str(round(agent.epsilon, 2))
              + "| loss:" + str(round(loss.history["loss"][0], 4)))
    
    # Running the agent
    done = False
    state = environment.gymReset()
    while not done:
        action = agent.act(state)
        state, _, done, info = environment.step(action)
        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            environment.render()
'''


''' How to embeded existing model: The following example appends VGG16 with 5-layers Flatten+Dense+Dropout+Dense+Dense

from keras.applications import VGG16

#-------------加载VGG模型并且添加自己的层----------------------
    #这里自己添加的层需要不断调整超参数来提升结果，输出类别更改softmax层即可
 
    #参数说明：inlucde_top:是否包含最上方的Dense层，input_shape：输入的图像大小(width,height,channel)                                         
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    
    #-----------控制需要FineTune的层数，不FineTune的就直接冻结
    for layer in base_model.layers:
        layer.trainable = False
 
    #----------编译，设置优化器，损失函数，性能指标
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

————————————————
版权声明：本文为CSDN博主「李正浩大魔王」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/m0_37316917/article/details/90485314

'''