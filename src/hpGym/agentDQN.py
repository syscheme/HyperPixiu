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
from keras.layers import Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D,GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend

from abc import ABCMeta, abstractmethod
import os, threading, datetime
import json

GPUs = backend.tensorflow_backend._get_available_gpus()

########################################################################
class agentDQN(MetaAgent):
    DEFAULT_BRAIN_ID = 'DQN_Cnn1Dx4' # 'DQN_DrDrDl' 'DQN_Dr64Dr32x3' 'DQN_Cnn1Dx4'

    def __init__(self, gymTrader, **kwargs):
        self.__brainDict = {
            'DQN_DrDrDl'     : self.__dqn_DrDrDl, 
            'DQN_Dr64Dr32x3' : self.__dqn_Dr64Dr32x3, # not good yet
            'DQN_Cnn1Dx4'    : self.__dqn_Cnn1Dx4,
            # TODO: other brains
        }

        super(agentDQN, self).__init__(gymTrader, **kwargs)

        self._epochsPerObservOnGpu = self.getConfig('epochsPerObservOnGpu', 5)
        self._lock = threading.Lock()
        self.__replaySize = self._batchSize *32
        if self.__replaySize <1024:
            self.__replaySize = 1024
        if self.__replaySize >10240:
            self.__replaySize = 10240

        self.__replayCache = [None] * self._batchSize
        self.__sampleIdx = 0
        self.__realDataNum =0
        self.__frameNum =0

        self._brainOutDir = '%s%s.%s_%s/' % (self._outDir, self._wkBrainId, self._stateSize, self._actionSize)

        #format the desc
        self._brainDesc = '%s created at %s with stateSize[%s] actionSize[%s], outdir %s' % (self._wkBrainId, datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'), self._stateSize, self._actionSize, self._brainOutDir)
        lines = []
        self._brain.summary(print_fn=lambda x: lines.append(x))
        self._brainDesc += '\nsummary:\n%s\n' % "\n".join(lines)
        basicAttrs = {
            'Id': self._wkBrainId,
            'GPUs': len(GPUs),
            'summary': '\n' + '\n'.join(lines),
            'stateSize': self._stateSize,
            'actionSize': self._actionSize,
            'created': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'outdir': self._brainOutDir
        }

        self._statusAttrs = { **self._statusAttrs, **basicAttrs }
        self.__bWarmed = False

        # prepare the callbacks
        self.__callbacks=[]
        # don't take update_freq='epoch' or batch as we train almost every gymObserve()
        # cbTB = TensorBoard(log_dir =self._brainOutDir, update_freq=self._batchSize *50000)
        # self.__callbacks.append(cbTB) # still too frequent

    def __del__(self):  # the destructor
        # self.saveBrain()
        pass

    @property
    def frameNum(self) : return self.__frameNum

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
            self._wkBrainId = brainId
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
        model.add(Dense(self._actionSize, activation='softmax'))

        return model

    def isReady(self) :
        if not super(agentDQN, self).isReady():
            return False
            
        return not None in self.__replayCache

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
                'epsilon' : self._epsilon,
                'learningRate' : self._learningRate,
                'saveTime' : datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            }
            self._statusAttrs = {**self._statusAttrs, **attrsToUpdate}

            with open('%sstatus.json' % self._brainOutDir, 'w') as outfile:
                json.dump(self._statusAttrs, outfile)

        self._gymTrader.info('saved brain[%s] with weights' % (self._brainOutDir))
        
    def loadBrain(self, brainId) :
        ''' load the previous saved brain
        @param a unique brainId must be given
        '''
        if not brainId or len(brainId) <=0:
            raise ValueError("No brainId specified")

        if not self._gymTrader :
            raise ValueError("Null trader")

        brainDir = '%s%s.%s_%s/' % (self._gymTrader.dataRoot, brainId, self._stateSize, self._actionSize)
        brain = None
        try : 
            # step 1. read the model file in json
            self._gymTrader.debug('loading saved brain from %s' %brainDir)
            with open('%smodel.json' % brainDir, 'r') as mjson:
                model_json = mjson.read()
            brain = model_from_json(model_json)

            # step 2. read the weights of the model
            self._gymTrader.debug('loading saved brain weight from %s' %brainDir)
            brain.load_weights('%smodel.json.h5' % brainDir)

            # step 3. if load weight successfully, do not start over to mess-up the trained model by
            # limiting epsilon from status.json
            with open('%sstatus.json' % brainDir, 'r') as f:
                self._statusAttrs = json.loads(f.read())

            continued = self._statusAttrs['continued'] if 'continued' in self._statusAttrs.keys() else []
            continued.append(datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'))

            attrsToUpdate = {
                'continued': continued
            }
            self._statusAttrs = {**self._statusAttrs, **attrsToUpdate}

            self._epsilon = float(self._statusAttrs['epsilon']) if 'epsilon' in self._statusAttrs.keys() else self._epsilon*0.7
            self._learningRate = float(self._statusAttrs['learningRate']) if 'learningRate' in self._statusAttrs.keys() else self._learningRate/2
        except:
            pass

        if brain:
            self._gymTrader.info('loaded brain from %s by taking initial epsilon[%s] learningRate[%s]' % (brainDir, self._epsilon, self._learningRate))
        return brain

    def gymAct(self, state):
        '''Acting Policy of the agentDQN
        @return one of self.__gymTrader.ACTIONS
        '''
        action = np.zeros(self._actionSize)
        if np.random.rand() <= self._epsilon:
            action[random.randrange(self._actionSize)] = 1
        else:
            state = state.reshape(1, self._stateSize)
            with self._lock:
                act_values = self._brain.predict(state)
                action[np.argmax(act_values[0])] = 1

        return action

    def gymObserve(self, state, action, reward, next_state, done, **feedbacks):
        '''Memory Management and training of the agent
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self._statusAttrs = {**self._statusAttrs, **feedbacks}
        if not self._pushToReplay(state, action, reward, next_state, done) :
            return None

        # this basic DQN also performs training in this step
        state, action, reward, next_state, done = self._sampleBatches()

        with self._lock:
            y = self._brain.predict(next_state)
            reward += (self._gamma * np.logical_not(done) * np.amax(y, axis=1))

            q_target = self._brain.predict(state)
            q_target[action[0], action[1]] = reward

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
            if len(GPUs) > 0 and len(state) >self._batchSize:
                epochs = self._epochsPerObservOnGpu
            self._loss = self._brain.fit(x=state, y=q_target, epochs=epochs, batch_size=self._batchSize, verbose=0, callbacks=self.__callbacks)
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
            if None in self.__replayCache :
                return state_batch, action_batch, reward_batch, next_state_batch, done_batch

            if len(GPUs) > 0 and len(self.__replayCache) > self._batchSize:
                sizeToBatch = min(10, int(len(self.__replayCache) / self._batchSize)) *self._batchSize
            
            batch = np.array(random.sample(self.__replayCache, sizeToBatch))

            # pick up the each part from the batch as we inserted via __replayCache[__sampleIdx] = (state, action, reward, next_state, done)
            state_batch = np.concatenate(batch[:, 0]).reshape(sizeToBatch, self._stateSize)
            action_batch = np.concatenate(batch[:, 1]).reshape(sizeToBatch, self._actionSize)
            reward_batch = batch[:, 2]
            next_state_batch = np.concatenate(batch[:, 3]).reshape(sizeToBatch, self._stateSize)
            done_batch = batch[:, 4]

            # action processing
            action_batch = np.where(action_batch == 1)
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def _sampleBatches_TODO(self):
        '''Selecting a batch of memory, split it into categorical subbatches
           Process action_batch into a position vector
        '''
        sizeToBatch = self._batchSize
        if self.__realDataNum <=0:
            sizeToBatch = self._batchSize
            batch = np.array(random.sample(self.__replayCache, sizeToBatch)).astype('float32')    
        else:
            if self.__realDataNum >= self.__replaySize:
                sizeToBatch = self.__replaySize
                self.__realDataNum = self.__replaySize
            else :
                sizeToBatch = int((self.__realDataNum +self._batchSize -1) /self._batchSize) * self._batchSize

            idxStart = (self.__sampleIdx + self.__replaySize - sizeToBatch +1) % self.__replaySize
            if self.__sampleIdx <= idxStart:
                a =self.__replayCache[idxStart :]
                b= self.__replayCache[: self.__sampleIdx+1]
                c= a+b
                batch = np.array(c, dtype=np.float32)
            else : 
                c = self.__replayCache[idxStart : self.__sampleIdx+1]
                batch = np.concatenate(c).astype('float32')
            
        state_batch = np.concatenate(batch[:, 0]).reshape(sizeToBatch, self._stateSize)
        action_batch = np.concatenate(batch[:, 1]).reshape(sizeToBatch, self._actionSize)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]).reshape(sizeToBatch, self._stateSize)
        done_batch = batch[:, 4]

        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


########################################################################
# TODO DoubleDQN to avoid overestimate, 将动作选择（max操作）和动作估计Q(s’,a’)解耦
# refer to https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddqn.py
    # # Step 2: calculate y
    # y_batch = []
    # current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
    # max_action_next = np.argmax(current_Q_batch, axis=1)
    # target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})

    # for i in range(0,BATCH_SIZE):
    #   done = minibatch[i][4]
    #   if done:
    #     y_batch.append(reward_batch[i])
    #   else :
    #     target_Q_value = target_Q_batch[i, max_action_next[i]]
    #     y_batch.append(reward_batch[i] + GAMMA * target_Q_value)

    # self.optimizer.run(feed_dict={
    #     self.y_input:y_batch,
    #     self.action_input:action_batch,
    #     self.state_input:state_batch
    #   })

########################################################################
# TODO AsynchronousDQN to avoid overestimate, 将动作选择（max操作）和动作估计Q(s’,a’)解耦
# refer to https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddqn.py
# AsynchronousDQN
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras?hl=zh-cn
# Keras' MirroredStrategy/MultiWorkerMirroredStrategy https://tf.wiki/zh/appendix/distributed.html

########################################################################
class agentDualHemicerebrum(agentDQN):
    def __init__(self, gymTrader, **kwargs):
        super(agentDualHemicerebrum, self).__init__(gymTrader, **kwargs)

        # treat the orginal self._brain as the right hemicerebrum to perform predicting, and 
        # additional self.__leftHemicerebrum to perform training in an additional thread
        self.__leftHemicerebrum = None
        if self._brain : 
            model_json = self._brain.to_json()
            self.__leftHemicerebrum = model_from_json(model_json)
            self.__leftHemicerebrum.set_weights(self._brain.get_weights()) 
            self.__leftHemicerebrum.compile(loss='mse', optimizer=Adam(lr=self._learningRate))

        self.__evWakeup = threading.Event()
        self.__bQuit = False
        self.__thread = threading.Thread(target=self.__trainLeft)
        if self.__leftHemicerebrum :
            self.__thread.start()

    def __del__(self):  # the destructor
        self._app.stop()
        self.wakeup()
        self.__thread.join()

    def wakeup(self) :
        self.__evWakeup.set()

    def isReady(self) :
        if not self.__leftHemicerebrum:
            return False

        if not super(agentDualHemicerebrum, self).isReady():
            return False

    #----------------------------------------------------------------------
    def __trainLeft(self):
        '''perform training left hemicerebrum in this threaded execution'''
        self.local = {}
        nextSleep = 0
        while not self.__bQuit:
            if nextSleep >0.0001:
                self.__evWakeup.wait(nextSleep)

            nextSleep = 1.0
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._sampleBatches()
            if state_batch is None:
                continue

            y = self.__leftHemicerebrum.predict(next_state_batch)
            reward_batch += (self._gamma * np.logical_not(done_batch) * np.amax(y, axis=1))

            q_target = self.__leftHemicerebrum.predict(state_batch)
            q_target[action_batch[0], action_batch[1]] = reward_batch
            
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
            # loss = self.__leftHemicerebrum.fit(x=state_batch, y=q_target, epochs=1, batch_size=self._batchSize, verbose=0)
            loss = self.__leftHemicerebrum.train_on_batch(x=state_batch, y=q_target, callbacks=self.__callbacks)

            #if loss < self._loss:
            with self._lock: # update the trained left hemicerebrum to the right
                self._brain.set_weights(self.__leftHemicerebrum.get_weights()) 
                self._loss =  loss

    def gymObserve(self, state, action, reward, next_state, done, **feedbacks):
        '''cache update of the agent, simple update the cache in this step
        @return loss take the known recent loss
        '''
        if not self._pushToReplay(state, action, reward, next_state, done, **feedbacks) :
            return None

        self.wakeup()
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