# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''

from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
import HistoryData as hist
from MarketData import EXPORT_FLOATS_DIMS

from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend
from keras.layers import Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D,GlobalAveragePooling1D
from keras.layers import BatchNormalization, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras import regularizers

import sys, os, platform, random, copy
import h5py, tarfile
import numpy as np

########################################################################
# to work as a generator for Keras fit_generator() by reading replay-buffers from HDF5 file
# sample the data as training data
class DQNTrainer(BaseApplication):

    def __init__(self, program, h5filepath, model_json=None, initWeights= None, recorder =None, **kwargs):
        super(DQNTrainer, self).__init__(program, **kwargs)

        self._model_json = model_json
        if not self._model_json :
            # modelfn = os.path.join(self.dataRoot, 'DQN_Cnn1Dx4.1556_3/model.json')
            modelfn = 'e:/AShareSample/ETF/DQN_Cnn1Dx4.1548_3.model.json'
            modelfn = self.getConfig('model_file', modelfn)
            self.debug('loading saved brain from %s' % modelfn)
            with open(modelfn, 'r') as mjson:
                self._model_json = mjson.read()

        if not h5filepath : 
            h5filepath = os.path.join(self.dataRoot, 'RFrames.h5')
            h5filepath = self.getConfig('RFSamples_file', h5filepath)

        self.debug('loading saved ReplaySamples from %s' % h5filepath)
        self._h5file = h5py.File(h5filepath, 'r')

        self._batchSize           = self.getConfig('batchSize', 128)
        self._trainSize           = self.getConfig('batchesPerTrain', 8) * self._batchSize
        self._poolReuses          = self.getConfig('poolReuses', -1)
        self._epochsPerFit        = self.getConfig('epochsPerFit', 2)
        self._gamma               = self.getConfig('gamma', 0.01)
        self._learningRate        = self.getConfig('learningRate', 0.02)
        self._maxLossBeforeStepSamples  = self.getConfig('maxLossBeforeStepSamples', 1000)
        self._maxPctLossDiff      = self.getConfig('maxPctLossDiff', 2)

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]

        self._brain = None
        self._theOther = None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev): pass

    def doAppInit(self): # return True if succ
        if not super(DQNTrainer, self).doAppInit() :
            return False

        if not self._model_json or not self._h5file:
            return False

        self._brain = model_from_json(self._model_json)
        if not self._brain:
            self.error('model_from_json failed')
            return False

        self._brain.compile(loss='mse', optimizer=Adam(lr=self._learningRate))

        #checkpoint = ModelCheckpoint('./weights.best.h5', verbose=0, monitor='loss', mode='min', save_best_only=True)
        #self._fitCallbacks =[checkpoint]
        cbTensorBoard = TensorBoard(log_dir='./out/tb', histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
        self._fitCallbacks =[cbTensorBoard]

        self.__gen = self.__generator(self.__train_DDQN)
        return True

    def doAppStep(self):
        if not self.__gen:
            self.stop()
        else:
            try:
                next(self.__gen)
            except Exception as ex:
                self.stop()
                self.logexception(ex)
                raise StopIteration
        
        return super(DQNTrainer, self).doAppStep()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __generator(self, trainMethod):

        framesInHd5 = []
        for name in self._h5file.keys():
            if 'ReplayFrame:' == name[:len('ReplayFrame:')] :
                framesInHd5.append(name)

        # I'd like to skip frame-0 as it most-likly includes many zero-samples
        if len(framesInHd5)>3:
            del framesInHd5[0]
            del framesInHd5[-1]
        
        if len(framesInHd5)>6:
            del framesInHd5[0]

        frameSeq=[]

        # build up self.__samplePool
        self.__samplePool = {
            'state':[],
            'action':[],
            'reward':[],
            'next_state':[],
            'done':[],
        }

        itrId=0
        samplePerFrame =0
        lossOfLastPool = 9999999
        loss = 9999999

        while len(frameSeq) >0 or loss > self._maxLossBeforeStepSamples or abs(loss-lossOfLastPool) > (loss * self._maxPctLossDiff/100) :
            lossOfLastPool = loss

            if len(frameSeq) <=0:
                a = copy.copy(framesInHd5)
                random.shuffle(a)
                frameSeq +=a
            
            startPoolSize = len(self.__samplePool['state'])
            cEvicted =0
            if startPoolSize >= max(samplePerFrame, self._trainSize *2):
                # randomly evict half of the poolSize
                sampleIdxs = [a for a in range(min(samplePerFrame, int(startPoolSize/2)))]
                random.shuffle(sampleIdxs)
                for i in sampleIdxs:
                    cEvicted +=1
                    for col in self.__samplePool.keys() :
                        del self.__samplePool[col][i]

            cAppend =0
            strFrames=''
            while len(frameSeq) >0 and len(self.__samplePool['state']) <max(samplePerFrame, self._trainSize *2) :
                strFrames += '%s,' % frameSeq[0]
                frame = self._h5file[frameSeq[0]]
                del frameSeq[0]

                for col in self.__samplePool.keys() :
                    incrematal = list(frame[col].value)
                    samplePerFrame = len(incrematal)
                    self.__samplePool[col] += incrematal
                cAppend += samplePerFrame

            poolSize = len(self.__samplePool['state'])
            self.info('sample pool refreshed: size[%s->%s] by evicting %s and refilling %s samples from %s %d frames await' % (startPoolSize, poolSize, cEvicted, cAppend, strFrames, len(frameSeq)))

            # iterations = self._poolReuses if self._poolReuses >0 else int(round(poolSize / self._trainSize, 0))
            lossOfThisPool = 9999999
            loss = lossOfThisPool/2
            itsInPoll = int((poolSize +self._trainSize -1)/ self._trainSize)
            while itsInPoll>0 or loss > self._maxLossBeforeStepSamples:

                if loss <0.001: loss =0.001 # to avoid divid by zero
                rDiff = abs(loss-lossOfThisPool)*100 / loss
                if itsInPoll<0 and ((rDiff < self._maxPctLossDiff *2) or (loss<0.1 and rDiff < self._maxPctLossDiff *5)):
                    break

                itsInPoll -=1
                lossOfThisPool = loss

                # random sample a dataset with size=self._trainSize from self.__samplePool
                samples ={}
                sampleIdxs = [a for a in range(poolSize)]
                random.shuffle(sampleIdxs)
                del sampleIdxs[self._trainSize :]

                for col in self.__samplePool.keys() :
                    a = [self.__samplePool[col][i] for i in sampleIdxs]
                    samples[col] = np.array(a)

                # call trainMethod to perform tranning
                itrId +=1
                # loss = 9999999
                # while loss > 100000:
                result = trainMethod(samples)
                loss = result.history["loss"][-1]
                self.info('train[%s] done, sampled %d from poolsize[%s], loss[%s]' % (str(itrId).zfill(6), self._trainSize, poolSize, loss))
                yield result # this is a step

            fn_save = '/tmp/DQN_Cnn1Dx4.1548_3.h5'
            self._brain.save(fn_save)
            self.info('saved weights to %s with loss[%s]' %(fn_save, loss))

    def __train_DQN(self, samples):
        # perform DQN training
        Q_next = self._brain.predict(samples['next_state'])
        Q_next_max= np.amax(Q_next, axis=1) # arrary(sampleLen, 1)
        done = np.array(samples['done'] !=0)
        rewards = samples['reward'] + (self._gamma * np.logical_not(done) * Q_next_max) # arrary(sampleLen, 1)
        action_link = np.where(samples['action'] == 1) # array(sizeToBatch, self._actionSize)=>array(2, sizeToBatch)

        Q_target = self._brain.predict(samples['state'])
        Q_target[action_link[0], action_link[1]] = rewards # action_link =arrary(2,sampleLen)

        return self._brain.fit(x=samples['state'], y=Q_target, epochs=self._epochsPerFit, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)

    def __train_DDQN(self, samples):
        if not self._theOther and self._brain :
            model_json = self._brain.to_json()
            self._theOther = model_from_json(model_json)
            self._theOther.set_weights(self._brain.get_weights()) 
            self._theOther.compile(loss='mse', optimizer=Adam(lr=self._learningRate), metrics=['accuracy'])

        if np.random.rand() < 0.5:
            brainPred  = self._brain
            brainTrain = self._theOther
        else:
            brainPred  = self._theOther
            brainTrain = self._brain
        
        Q_next = brainPred.predict(samples['next_state']) # arrary(sampleLen, actionSize)
        Q_next_max= np.amax(Q_next, axis=1) # arrary(sampleLen, 1)
        done = np.array(samples['done'] !=0)
        rewards = samples['reward'] + (self._gamma * np.logical_not(done) * Q_next_max) # arrary(sampleLen, 1)
        action_link = np.where(samples['action'] == 1) # array(sizeToBatch, self._actionSize)=>array(2, sizeToBatch)

        Q_target = self._brain.predict(samples['state'])
        Q_target[action_link[0], action_link[1]] = rewards # action_link =arrary(2,sampleLen)

        return brainTrain.fit(x=samples['state'], y=Q_target, epochs=self._epochsPerFit, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)

########################################################################
class MarketDirClassifier(BaseApplication):

    def __init__(self, program, h5filepath, model_json=None, initWeights= None, recorder =None, **kwargs):
        super(MarketDirClassifier, self).__init__(program, **kwargs)

        self._model_json =model_json
        self._h5filepath =h5filepath
        if not self._h5filepath : 
            h5filepath = os.path.join(self.dataRoot, 'RFrames.h5')
            self._h5filepath = self.getConfig('RFSamples_file', h5filepath)

        self._batchSize           = self.getConfig('batchSize', 128)
        self._trainSize           = self.getConfig('batchesPerTrain', 8) * self._batchSize
        self._poolReuses          = self.getConfig('poolReuses', -1)
        self._epochsPerFit        = self.getConfig('epochsPerFit', 2)
        self._gamma               = self.getConfig('gamma', 0.01)
        self._learningRate        = self.getConfig('learningRate', 0.02)
        self._maxLossBeforeStepSamples  = self.getConfig('maxLossBeforeStepSamples', 1000)
        self._maxPctLossDiff      = self.getConfig('maxPctLossDiff', 2)

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]

        self._brain = None
        self._theOther = None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev): pass

    def doAppInit(self): # return True if succ
        if not super(MarketDirClassifier, self).doAppInit() :
            return False

        self.debug('loading saved ReplaySamples from %s' % self._h5filepath)
        self._h5file = h5py.File(self._h5filepath, 'r')

        if not self._h5file:
            return False

        self._framesInHd5 = []
        for name in self._h5file.keys():
            if 'ReplayFrame:' == name[:len('ReplayFrame:')] :
                self._framesInHd5.append(name)

        # I'd like to skip frame-0 as it most-likly includes many zero-samples
        if len(self._framesInHd5)>3:
            del self._framesInHd5[0]
            del self._framesInHd5[-1]
        
        if len(self._framesInHd5)>6:
            del self._framesInHd5[0]
        if len(self._framesInHd5) <=0:
            return False

        self._stateSize = self._h5file[self._framesInHd5[0]]['state'].shape[1]
        self._actionSize = self._h5file[self._framesInHd5[0]]['action'].shape[1]
        
        if self._model_json:
            self._brain = model_from_json(self._model_json)
            if not self._brain:
                self.error('model_from_json failed')
                return False
        else:
            self._brain = self.__createModel_VGG16_1d()

        #checkpoint = ModelCheckpoint('./weights.best.h5', verbose=0, monitor='loss', mode='min', save_best_only=True)
        #self._fitCallbacks =[checkpoint]
        cbTensorBoard = TensorBoard(log_dir='./out/tb', histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
        self._fitCallbacks =[cbTensorBoard]

        self.__gen = self.__generator()
        return True

    def doAppStep(self):
        if not self.__gen:
            self.stop()
        else:
            try:
                next(self.__gen)
            except Exception as ex:
                self.stop()
                self.logexception(ex)
                raise StopIteration
        
        return super(MarketDirClassifier, self).doAppStep()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __createModel_XXX(self):
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
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered
        model.compile(loss='mse', optimizer=Adam(lr=self._learningRate))
        return model

    def __createModel_VGG16_1d(self):
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        weight_decay = 0.0005

        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        model.add(Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        
        #进行一次归一化
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        #layer2 32*32*64
        model.add(Conv1D(64, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
        #padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
        model.add(MaxPooling1D(2))

        #layer3 16*16*64
        model.add(Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer4 16*16*128
        model.add(Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        
        #layer5 8*8*128
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer6 8*8*256
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer7 8*8*256
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        #layer8 4*4*256
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer9 4*4*512
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer10 4*4*512
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        
        #layer11 2*2*512
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer12 2*2*512
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer13 2*2*512
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        #layer14 1*1*512
        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #layer15 512
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #layer16 512
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered

        # 10
        # model.summary()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model


    def __generator(self):

        frameSeq=[]

        # build up self.__samplePool
        self.__samplePool = {
            'state':[],
            'action':[],
        }

        itrId=0
        samplePerFrame =0
        lossOfLastPool = 9999999
        loss = 9999999

        while len(frameSeq) >0 or loss > self._maxLossBeforeStepSamples or abs(loss-lossOfLastPool) > (loss * self._maxPctLossDiff/100) :
            lossOfLastPool = loss

            if len(frameSeq) <=0:
                a = copy.copy(self._framesInHd5)
                random.shuffle(a)
                frameSeq +=a
            
            startPoolSize = len(self.__samplePool['state'])
            cEvicted =0
            if startPoolSize >= max(samplePerFrame, self._trainSize *2):
                # randomly evict half of the poolSize
                sampleIdxs = [a for a in range(min(samplePerFrame, int(startPoolSize/2)))]
                random.shuffle(sampleIdxs)
                for i in sampleIdxs:
                    cEvicted +=1
                    for col in self.__samplePool.keys() :
                        del self.__samplePool[col][i]

            cAppend =0
            strFrames=''
            while len(frameSeq) >0 and len(self.__samplePool['state']) <max(samplePerFrame, self._trainSize *2) :
                strFrames += frameSeq[0]
                frame = self._h5file[frameSeq[0]]
                del frameSeq[0]

                for col in self.__samplePool.keys() :
                    incrematal = list(frame[col].value)
                    samplePerFrame = len(incrematal)
                    self.__samplePool[col] += incrematal

                if loss <10:
                    state_set = self.__samplePool['state'][poolSize+cAppend:]
                    action_set = self.__samplePool['action'][poolSize+cAppend:]
                    strFrames += '/loss[%s]' %  self._brain.evaluate(x=np.array(state_set), y=np.array(action_set), batch_size=self._batchSize, verbose=1) #, callbacks=self._fitCallbacks)

                strFrames += ','
                cAppend += samplePerFrame

            poolSize = len(self.__samplePool['state'])
            self.info('sample pool refreshed: size[%s->%s] by evicting %s and refilling %s samples from %s %d frames await' % (startPoolSize, poolSize, cEvicted, cAppend, strFrames, len(frameSeq)))

            self._trainSize = poolSize #??????
            lossOfThisPool = 9999999
            loss = lossOfThisPool/2
            itsInPoll = int((poolSize +self._trainSize -1)/ self._trainSize)
            while itsInPoll>0 or loss > self._maxLossBeforeStepSamples:

                if loss <0.001: loss =0.001 # to avoid divid by zero
                rDiff = abs(loss-lossOfThisPool)*100 / loss
                if itsInPoll<0 and ((rDiff < self._maxPctLossDiff *2) or (loss<0.1 and rDiff < self._maxPctLossDiff *5)):
                    break

                itsInPoll -=1
                lossOfThisPool = loss

                # random sample a dataset with size=self._trainSize from self.__samplePool
                sampleIdxs = [a for a in range(poolSize)]
                random.shuffle(sampleIdxs)
                del sampleIdxs[self._trainSize :]

                state_set = [self.__samplePool['state'][i] for i in sampleIdxs]
                action_set = [self.__samplePool['action'][i] for i in sampleIdxs]

                # call trainMethod to perform tranning
                itrId +=1
                result = self._brain.fit(x=np.array(state_set), y=np.array(action_set), epochs=self._epochsPerFit, batch_size=self._batchSize, verbose=1, callbacks=self._fitCallbacks) # ,metrics=['accuracy']) #metrics=['accuracy'],

                loss = result.history["loss"][-1]
                self.info('train[%s] done, sampled %d from poolsize[%s], loss[%s]' % (str(itrId).zfill(6), self._trainSize, poolSize, loss))
                yield result # this is a step

            fn_save = '/tmp/DQN_Cnn1Dx4.1548_3.h5'
            self._brain.save(fn_save)
            self.info('saved weights to %s with loss[%s]' %(fn_save, loss))

########################################################################
if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Gym_AShare.json']

    p = Program()
    p._heartbeatInterval =-1

    SYMBOL = '000001' # '000540' '000001'
    sourceCsvDir = None
    try:
        jsetting = p.jsettings('DQNTrainer/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('DQNTrainer/objectives')
        if not jsetting is None:
            symbol = jsetting([SYMBOL])[0]
    except Exception as ex:
        symbol = SYMBOL
    SYMBOL = symbol

    if not sourceCsvDir or len(sourceCsvDir) <=0:
        for d in ['e:/AShareSample/ETF', '/mnt/e/AShareSample/ETF', '/mnt/m/AShareSample']:
            try :
                if  os.stat(d):
                    sourceCsvDir = d
                    break
            except :
                pass

    if 'Windows' in platform.platform() and '/mnt/' == sourceCsvDir[:5] and '/' == sourceCsvDir[6]:
        drive = '%s:' % sourceCsvDir[5]
        sourceCsvDir = sourceCsvDir.replace(sourceCsvDir[:6], drive)

    p.info('all objects registered piror to DQNTrainer: %s' % p.listByType())
    
    # trainer = p.createApp(DQNTrainer, configNode ='DQNTrainer', h5filepath=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))
    trainer = p.createApp(MarketDirClassifier, configNode ='DQNTrainer', h5filepath=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))

    p.start()
    p.loop()
    p.stop()
