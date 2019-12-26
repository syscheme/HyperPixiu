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

DUMMY_BIG_VAL = 999999
GPUs = backend.tensorflow_backend._get_available_gpus()

########################################################################
class MarketDirClassifier(BaseApplication):

    DEFAULT_MODEL = 'Cnn1Dx4'
    
    def __init__(self, program, h5filepath=None, model_json=None, initWeights= None, recorder =None, **kwargs):
        super(MarketDirClassifier, self).__init__(program, **kwargs)

        self._wkModelId           = self.getConfig('modelId', MarketDirClassifier.DEFAULT_MODEL)

        self._model_json =model_json
        self._h5filepath =h5filepath

        if not self._h5filepath : 
            h5filepath = os.path.join(self.dataRoot, 'RFrames.h5')
            self._h5filepath = self.getConfig('RFSamples_file', h5filepath)
            self._h5filepath = Program.fixupPath(self._h5filepath)

        self._batchSize           = self.getConfig('batchSize', 128)
        self._trainSize           = self.getConfig('batchesPerTrain', 8) * self._batchSize
        self._poolReuses          = self.getConfig('poolReuses', 0)
        self._epochsPerFit        = self.getConfig('epochsPerFit', 2)
        self._lossStop            = self.getConfig('lossStop', 0.1)
        self._lossPctStop         = self.getConfig('lossPctStop', 2)
        self._startLR             = self.getConfig('startLR', 0.01)
        # self._poolEvictRate       = self.getConfig('poolEvictRate', 0.5)
        # if self._poolEvictRate>1 or self._poolEvictRate<=0:
        #     self._poolEvictRate =1

        if len(GPUs) > 0 : # adjust some configurations if currently running on GPUs
            self._batchSize       = self.getConfig('GPU/batchSize',    self._batchSize)
            self._trainSize       = self.getConfig('GPU/batchesPerTrain', 64) * self._batchSize # usually 64 is good for a bottom-line model of GTX1050oc/2G
            self._epochsPerFit    = self.getConfig('GPU/epochsPerFit', self._epochsPerFit)
            self._poolReuses      = self.getConfig('GPU/poolReuses',   self._poolReuses)
            self._startLR         = self.getConfig('GPU/startLR',      self._startLR)

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]

        self._brain = None
        self._outDir = os.path.join(self.dataRoot, self._program.progId)

        self.__knownModels = {
            'VGG16d1'    : self.__createModel_VGG16d1,
            'Cnn1Dx4'    : self.__createModel_Cnn1Dx4,
            'Cnn1Dx4R1'  : self.__createModel_Cnn1Dx4R1,
            }


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
        elif self._wkModelId and len(self._wkModelId) >0:
            wkModelId = '%s.S%sI%sA%s' % (self._wkModelId, self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
            inDir = os.path.join(self.dataRoot, wkModelId)
            try : 
                self.debug('loading saved model from %s' % inDir)
                with open(os.path.join(inDir, 'model.json'), 'r') as mjson:
                    model_json = mjson.read()
                    self._brain = model_from_json(model_json)

                sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
                self._brain.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

                self._wkModelId = wkModelId

                fn_weights = os.path.join(inDir, 'weights.h5')
                self.debug('loading saved weights from %s' %fn_weights)
                self._brain.load_weights(fn_weights)
                self.info('loaded model and weights from %s' %inDir)

            except Exception as ex:
                self.logexception(ex)

            if not self._brain and self._wkModelId in self.__knownModels.keys():
                self._brain = self.__knownModels[self._wkModelId]()

        if not self._brain:
            self._brain = self.__knownModels[MarketDirClassifier.DEFAULT_MODEL]()

        try :
            os.makedirs(self._outDir)
            fn_model =os.path.join(self._outDir, '%s.model.json' %self._wkModelId) 
            with open(fn_model, 'w') as mjson:
                model_json = self._brain.to_json()
                mjson.write(model_json)
                self.info('saved model as %s' %fn_model)
        except :
            pass

        #checkpoint = ModelCheckpoint(os.path.join(self._outDir, '%s.best.h5' %self._wkModelId ), verbose=0, monitor='loss', mode='min', save_best_only=True)
        #self._fitCallbacks =[checkpoint]
        cbTensorBoard = TensorBoard(log_dir=os.path.join(self._outDir, 'tb'), histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

        self._fitCallbacks =[cbTensorBoard]

        self._gen = self.__generator()
        return True

    def doAppStep(self):
        if not self._gen:
            self.stop()
        else:
            try:
                next(self._gen)
            except Exception as ex:
                self.stop()
                self.logexception(ex)
                raise StopIteration
        
        return super(MarketDirClassifier, self).doAppStep()

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __generator(self):

        frameSeq=[]

        # build up self.__samplePool
        self.__samplePool = {
            'state':[],
            'action':[],
        }

        itrId=0
        samplePerFrame =0

        loss = DUMMY_BIG_VAL
        lossMax = loss
        while len(frameSeq) >0 or lossMax > self._lossStop or abs(loss-lossMax) > (lossMax * self._lossPctStop/100) :
            if len(frameSeq) <=0:
                a = copy.copy(self._framesInHd5)
                random.shuffle(a)
                frameSeq +=a
            
            startPoolSize = len(self.__samplePool['state'])
            cEvicted =0
            if startPoolSize >= max(samplePerFrame, self._trainSize *2):
                # # randomly evict half of the poolSize
                # sampleIdxs = [a for a in range(startPoolSize)]
                # random.shuffle(sampleIdxs)
                # nToEvict = int(startPoolSize * self._poolEvictRate)
                # for i in sampleIdxs:
                #     if i >= (startPoolSize - cEvicted): continue

                #     for col in self.__samplePool.keys() :
                #         del self.__samplePool[col][i]
                    
                #     cEvicted +=1
                #     if cEvicted >= nToEvict: break
                cEvicted = startPoolSize # = samplePerFrame
                for col in self.__samplePool.keys() :
                    del self.__samplePool[col][:cEvicted]

            poolSize = len(self.__samplePool['state'])

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

                if loss <10 and len(self.__samplePool['state']) > (poolSize+cAppend + self._batchSize):
                    try :
                        state_set = self.__samplePool['state'][poolSize+cAppend: ]
                        action_set = self.__samplePool['action'][poolSize+cAppend: ]
                        strFrames += '/eval:%s' %  self._brain.evaluate(x=np.array(state_set), y=np.array(action_set), batch_size=self._batchSize, verbose=1) #, callbacks=self._fitCallbacks)
                    except Exception as ex:
                        self.logexception(ex)

                strFrames += ','
                cAppend += samplePerFrame

            poolSize = len(self.__samplePool['state'])
            self.info('sample pool refreshed: size[%s->%s] by evicting %s and refilling %s samples from %s %d frames await' % (startPoolSize, poolSize, cEvicted, cAppend, strFrames, len(frameSeq)))

            # random sample a dataset with size=self._trainSize from self.__samplePool
            sampleSeq = [a for a in range(poolSize)]
            random.shuffle(sampleSeq)
            if self._poolReuses >0:
                tmpseq = copy.copy(sampleSeq)
                for i in range(self._poolReuses) :
                    random.shuffle(tmpseq)
                    sampleSeq += tmpseq

            if len(sampleSeq) >= self._batchSize:
                lossMax = loss if loss < DUMMY_BIG_VAL-1 else 0.0

            while len(sampleSeq) >= self._batchSize:

                if len(sampleSeq) > self._trainSize:
                    sampleIdxs = sampleSeq[:self._trainSize]
                    del sampleSeq[:self._trainSize]
                else :
                    sampleIdxs = sampleSeq
                    sampleSeq = []

                state_set = [self.__samplePool['state'][i] for i in sampleIdxs]
                action_set = [self.__samplePool['action'][i] for i in sampleIdxs]

                # call trainMethod to perform tranning
                itrId +=1
                try :
                    result = self._brain.fit(x=np.array(state_set), y=np.array(action_set), epochs=self._epochsPerFit, batch_size=self._batchSize, verbose=1, callbacks=self._fitCallbacks)

                    loss = result.history["loss"][-1]
                    self.info('train[%s] done, sampled %d from poolsize[%s], loss[%s/%s]' % (str(itrId).zfill(6), self._trainSize, poolSize, loss, lossMax))
                    yield result # this is a step
                except Exception as ex:
                    self.logexception(ex)

                if lossMax < loss:
                    lossMax = loss

            fn_weights = os.path.join(self._outDir, '%s.weights.h5' %self._wkModelId)
            self._brain.save(fn_weights)
            self.info('saved weights to %s with loss[%s]' %(fn_weights, loss))

    #----------------------------------------------------------------------
    # model definitions

    def __createModel_Cnn1Dx4(self):
        '''
        changed input/output dims based on 
        ref: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
        
        when Cnn1Dx4.S1548I4A3, the layers is like the following:
            Layer (type)                 Output Shape              Param #   
            =================================================================
            reshape_2 (Reshape)          (None, 387, 4)            0         
            _________________________________________________________________
            conv1d_9 (Conv1D)            (None, 378, 100)          4100      
            _________________________________________________________________
            conv1d_10 (Conv1D)           (None, 369, 100)          100100    
            _________________________________________________________________
            max_pooling1d_1 (MaxPooling1 (None, 123, 100)          0         
            _________________________________________________________________
            conv1d_11 (Conv1D)           (None, 114, 160)          160160    
            _________________________________________________________________
            conv1d_12 (Conv1D)           (None, 105, 160)          256160    
            _________________________________________________________________
            global_average_pooling1d_1 ( (None, 160)               0         
            _________________________________________________________________
            dropout_1 (Dropout)          (None, 160)               0         
            _________________________________________________________________
            dense_1 (Dense)              (None, 3)                 483       
            =================================================================
            Total params: 521,003
            Trainable params: 521,003
            Non-trainable params: 0
        '''
        self._wkModelId = 'Cnn1Dx4.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
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
        model.compile(loss='mse', optimizer=Adam(lr=self._startLR, decay=1e-5), metrics=['accuracy'])
        return model

    def __createModel_Cnn1Dx4R1(self):
        self._wkModelId = 'Cnn1Dx4R1.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        model.add(Conv1D(128, 3, activation='relu', input_shape=(self._stateSize/EXPORT_FLOATS_DIMS, EXPORT_FLOATS_DIMS)))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(512, 3, activation='relu'))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(100, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.4))
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self._startLR, decay=1e-6), metrics=['accuracy'])
        return model

    def __createModel_VGG16d1(self):
        '''
        changed input/output dims based on 
            Layer (type)                 Output Shape              Param #   
            =================================================================
            reshape_1 (Reshape)          (None, 387, 4)            0         
            _________________________________________________________________
            conv1d_1 (Conv1D)            (None, 387, 64)           832       
            _________________________________________________________________
            activation_1 (Activation)    (None, 387, 64)           0         
            _________________________________________________________________
            batch_normalization_1 (Batch (None, 387, 64)           256       
            _________________________________________________________________
            dropout_1 (Dropout)          (None, 387, 64)           0         
            _________________________________________________________________
            conv1d_2 (Conv1D)            (None, 387, 64)           12352     
            _________________________________________________________________
            activation_2 (Activation)    (None, 387, 64)           0         
            _________________________________________________________________
            batch_normalization_2 (Batch (None, 387, 64)           256       
            _________________________________________________________________
            max_pooling1d_1 (MaxPooling1 (None, 193, 64)           0         
            _________________________________________________________________
            conv1d_3 (Conv1D)            (None, 193, 128)          24704     
            _________________________________________________________________
            activation_3 (Activation)    (None, 193, 128)          0         
            _________________________________________________________________
            batch_normalization_3 (Batch (None, 193, 128)          512       
            _________________________________________________________________
            dropout_2 (Dropout)          (None, 193, 128)          0         
            _________________________________________________________________
            conv1d_4 (Conv1D)            (None, 193, 128)          49280     
            _________________________________________________________________
            activation_4 (Activation)    (None, 193, 128)          0         
            _________________________________________________________________
            batch_normalization_4 (Batch (None, 193, 128)          512       
            _________________________________________________________________
            max_pooling1d_2 (MaxPooling1 (None, 96, 128)           0         
            _________________________________________________________________
            conv1d_5 (Conv1D)            (None, 96, 256)           98560     
            _________________________________________________________________
            activation_5 (Activation)    (None, 96, 256)           0         
            _________________________________________________________________
            batch_normalization_5 (Batch (None, 96, 256)           1024      
            _________________________________________________________________
            dropout_3 (Dropout)          (None, 96, 256)           0         
            _________________________________________________________________
            conv1d_6 (Conv1D)            (None, 96, 256)           196864    
            _________________________________________________________________
            activation_6 (Activation)    (None, 96, 256)           0         
            _________________________________________________________________
            batch_normalization_6 (Batch (None, 96, 256)           1024      
            _________________________________________________________________
            dropout_4 (Dropout)          (None, 96, 256)           0         
            _________________________________________________________________
            conv1d_7 (Conv1D)            (None, 96, 256)           196864    
            _________________________________________________________________
            activation_7 (Activation)    (None, 96, 256)           0         
            _________________________________________________________________
            batch_normalization_7 (Batch (None, 96, 256)           1024      
            _________________________________________________________________
            max_pooling1d_3 (MaxPooling1 (None, 48, 256)           0         
            _________________________________________________________________
            conv1d_8 (Conv1D)            (None, 48, 512)           393728    
            _________________________________________________________________
            activation_8 (Activation)    (None, 48, 512)           0         
            _________________________________________________________________
            batch_normalization_8 (Batch (None, 48, 512)           2048      
            _________________________________________________________________
            dropout_5 (Dropout)          (None, 48, 512)           0         
            _________________________________________________________________
            conv1d_9 (Conv1D)            (None, 48, 512)           786944    
            _________________________________________________________________
            activation_9 (Activation)    (None, 48, 512)           0         
            _________________________________________________________________
            batch_normalization_9 (Batch (None, 48, 512)           2048      
            _________________________________________________________________
            dropout_6 (Dropout)          (None, 48, 512)           0         
            _________________________________________________________________
            conv1d_10 (Conv1D)           (None, 48, 512)           786944    
            _________________________________________________________________
            activation_10 (Activation)   (None, 48, 512)           0         
            _________________________________________________________________
            batch_normalization_10 (Batc (None, 48, 512)           2048      
            _________________________________________________________________
            max_pooling1d_4 (MaxPooling1 (None, 24, 512)           0         
            _________________________________________________________________
            conv1d_11 (Conv1D)           (None, 24, 512)           786944    
            _________________________________________________________________
            activation_11 (Activation)   (None, 24, 512)           0         
            _________________________________________________________________
            batch_normalization_11 (Batc (None, 24, 512)           2048      
            _________________________________________________________________
            dropout_7 (Dropout)          (None, 24, 512)           0         
            _________________________________________________________________
            conv1d_12 (Conv1D)           (None, 24, 512)           786944    
            _________________________________________________________________
            activation_12 (Activation)   (None, 24, 512)           0         
            _________________________________________________________________
            batch_normalization_12 (Batc (None, 24, 512)           2048      
            _________________________________________________________________
            dropout_8 (Dropout)          (None, 24, 512)           0         
            _________________________________________________________________
            conv1d_13 (Conv1D)           (None, 24, 512)           786944    
            _________________________________________________________________
            activation_13 (Activation)   (None, 24, 512)           0         
            _________________________________________________________________
            batch_normalization_13 (Batc (None, 24, 512)           2048      
            _________________________________________________________________
            max_pooling1d_5 (MaxPooling1 (None, 12, 512)           0         
            _________________________________________________________________
            dropout_9 (Dropout)          (None, 12, 512)           0         
            _________________________________________________________________
            flatten_1 (Flatten)          (None, 6144)              0         
            _________________________________________________________________
            dense_1 (Dense)              (None, 512)               3146240   
            _________________________________________________________________
            activation_14 (Activation)   (None, 512)               0         
            _________________________________________________________________
            batch_normalization_14 (Batc (None, 512)               2048      
            _________________________________________________________________
            dense_2 (Dense)              (None, 512)               262656    
            _________________________________________________________________
            activation_15 (Activation)   (None, 512)               0         
            _________________________________________________________________
            batch_normalization_15 (Batc (None, 512)               2048      
            _________________________________________________________________
            dropout_10 (Dropout)         (None, 512)               0         
            _________________________________________________________________
            dense_3 (Dense)              (None, 10)                5130      
            _________________________________________________________________
            dense_4 (Dense)              (None, 3)                 33        
            =================================================================
            Total params: 8,342,955
            Trainable params: 8,332,459
            Non-trainable params: 10,496
        '''
        self._wkModelId = 'VGG16d1.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        weight_decay = 0.0005

        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # model.add(Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        
        #进行一次归一化
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        #layer2 32*32*64
        # model.add(Conv1D(64, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(64, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
        #padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
        model.add(MaxPooling1D(2))

        #layer3 16*16*64
        # model.add(Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer4 16*16*128
        # model.add(Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        
        #layer5 8*8*128
        # model.add(Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer6 8*8*256
        # model.add(Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer7 8*8*256
        # model.add(Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        #layer8 4*4*256
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer9 4*4*512
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        #layer10 4*4*512
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        
        #layer11 2*2*512
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer12 2*2*512
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #layer13 2*2*512
        # model.add(Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

        #layer14 1*1*512
        model.add(Flatten())
        # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #layer15 512
        # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        #layer16 512
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered

        # 10
        # model.summary()
        sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model

########################################################################
# to work as a generator for Keras fit_generator() by reading replay-buffers from HDF5 file
# sample the data as training data
class DQNTrainer(MarketDirClassifier):

    def __init__(self, program, **kwargs):
        super(DQNTrainer, self).__init__(program, **kwargs)
        self._theOther = None

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication

    def doAppInit(self): # return True if succ
        if not super(DQNTrainer, self).doAppInit() :
            return False

        # overwrite MarketDirClassifier's self._gen
        self._gen = self.__generator(self.__train_DDQN)
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __generator(self, trainMethod):

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

        loss = DUMMY_BIG_VAL
        lossMax = loss
        while len(frameSeq) >0 or lossMax > self._lossStop or abs(loss-lossMax) > (lossMax * self._lossPctStop/100) :
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

            # random sample a dataset with size=self._trainSize from self.__samplePool
            sampleSeq = [a for a in range(poolSize)]
            random.shuffle(sampleSeq)
            if self._poolReuses >0:
                tmpseq = copy.copy(sampleSeq)
                for i in range(self._poolReuses) :
                    random.shuffle(tmpseq)
                    sampleSeq += tmpseq

            if len(sampleSeq) >= self._batchSize:
                lossMax = loss if loss < DUMMY_BIG_VAL-1 else 0.0

            while len(sampleSeq) >= self._batchSize:

                if len(sampleSeq) > self._trainSize:
                    sampleIdxs = sampleSeq[:self._trainSize]
                    del sampleSeq[:self._trainSize]
                else :
                    sampleIdxs = sampleSeq
                    sampleSeq = []

                samples = {}
                for col in self.__samplePool.keys() :
                    samples[col] = [self.__samplePool[col][i] for i in sampleIdxs]

                # call trainMethod to perform tranning
                itrId +=1
                result = trainMethod(samples)
                loss = result.history["loss"][-1]
                self.info('train[%s] done, sampled %d from poolsize[%s], loss[%s]' % (str(itrId).zfill(6), self._trainSize, poolSize, loss))
                yield result # this is a step

                if lossMax < loss:
                    lossMax = loss

            fn_weights = os.path.join(self._outDir, '%s.weights.h5' %self._wkModelId)
            self._brain.save(fn_weights)
            self.info('saved weights to %s with loss[%s]' %(fn_weights, loss))

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
            self._theOther.compile(loss='mse', optimizer=Adam(lr=self._startLR), metrics=['accuracy'])

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
if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Gym_AShare.json'] # 'DQNTrainer_VGG16d1.json' 'Gym_AShare.json'

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

    p.info('all objects registered piror to DQNTrainer: %s' % p.listByType())
    
    # trainer = p.createApp(DQNTrainer, configNode ='DQNTrainer', h5filepath=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))
    trainer = p.createApp(MarketDirClassifier, configNode ='DQNTrainer')

    p.start()
    p.loop()
    p.stop()
