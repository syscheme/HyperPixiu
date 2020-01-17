# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''

from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
import HistoryData as hist
from MarketData import EXPORT_FLOATS_DIMS

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

import tensorflow as tf

import sys, os, platform, random, copy, threading
from datetime import datetime
from time import sleep

import h5py, tarfile
import numpy as np

DUMMY_BIG_VAL = 999999
NN_FLOAT = 'float32'
RFGROUP_PREFIX = 'ReplayFrame:'
# GPUs = backend.tensorflow_backend._get_available_gpus()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()

if len(GPUs) >1:
    from keras.utils.training_utils import multi_gpu_model

class Hd5DataGenerator(Sequence):
    def __init__(self, trainer, batch_size):
        self.trainer = trainer
        self.batch_size = batch_size

    def __len__(self):
        return self.trainer.chunksInPool

    def __getitem__(self, index):
        batch = self.trainer.readDataChunk(index)
        return batch['state'], batch['action']

    def __iter__(self):
        for i in range(self.__len__()) :
            batch = self.trainer.readDataChunk(i)
            yield batch['state'], batch['action']

########################################################################
class MarketDirClassifier(BaseApplication):

    DEFAULT_MODEL = 'Cnn1Dx4'
    COMPILE_ARGS ={
    'loss':'categorical_crossentropy', 
    # 'optimizer': sgd,
    'metrics':['accuracy']
    }
    
    def __init__(self, program, replayFrameFiles=None, model_json=None, initWeights= None, recorder =None, **kwargs):
        super(MarketDirClassifier, self).__init__(program, **kwargs)

        self._wkModelId           = self.getConfig('modelId', MarketDirClassifier.DEFAULT_MODEL)

        self._model_json =model_json
        self._replayFrameFiles =replayFrameFiles

        if not self._replayFrameFiles or len(self._replayFrameFiles) <=0: 
            self._replayFrameFiles = self.getConfig('replayFrameFiles', [])
            self._replayFrameFiles = [ Program.fixupPath(f) for f in self._replayFrameFiles ]

        if not self._replayFrameFiles or len(self._replayFrameFiles) <=0: 
            self._replayFrameFiles =[]
            replayFrameDir = self.getConfig('replayFrameDir', None)
            if replayFrameDir:
                replayFrameDir = Program.fixupPath(replayFrameDir)
                try :
                    for _, subdirs, files in os.walk(replayFrameDir, topdown=False):
                        for name in files:
                            if '.h5' != name[-3:] : continue
                            self._replayFrameFiles.append(os.path.join(replayFrameDir, name))
                except:
                    pass

        self._stepMethod          = self.getConfig('stepMethod', None)
        self._repeatsInFile       = self.getConfig('repeatsInFile', 0)
        self._exportTB            = self.getConfig('tensorBoard', 'no').lower() in BOOL_STRVAL_TRUE
        self._batchSize           = self.getConfig('batchSize', 128)
        self._batchesPerTrain     = self.getConfig('batchesPerTrain', 8)
        self._recycleSize         = self.getConfig('recycles', 1)
        self._initEpochs          = self.getConfig('initEpochs', 2)
        self._lossStop            = self.getConfig('lossStop', 0.24) # 0.24 according to average loss value by： grep 'from eval' /mnt/d/tmp/DQNTrainer_14276_0106.log |sed 's/.*loss\[\([^]]*\)\].*/\1/g' | awk '{ total += $1; count++ } END { print total/count }'
        self._lossPctStop         = self.getConfig('lossPctStop', 5)
        self._startLR             = self.getConfig('startLR', 0.01)
        # self._poolEvictRate       = self.getConfig('poolEvictRate', 0.5)
        # if self._poolEvictRate>1 or self._poolEvictRate<=0:
        #     self._poolEvictRate =1

        if len(GPUs) > 0 : # adjust some configurations if currently running on GPUs
            self._stepMethod      = self.getConfig('GPU/stepMethod', self._stepMethod)
            self._exportTB        = self.getConfig('GPU/tensorBoard', 'no').lower() in BOOL_STRVAL_TRUE
            self._batchSize       = self.getConfig('GPU/batchSize',    self._batchSize)
            self._batchesPerTrain = self.getConfig('GPU/batchesPerTrain', 64)  # usually 64 is good for a bottom-line model of GTX1050oc/2G
            self._initEpochs      = self.getConfig('GPU/initEpochs', self._initEpochs)
            self._recycleSize     = self.getConfig('GPU/recycles',   self._recycleSize)
            self._startLR         = self.getConfig('GPU/startLR',      self._startLR)

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]
        self._frameSeq =[]

        self._stateSize, self._actionSize, self._frameSize = None, None, 0
        self._brain = None
        self._outDir = os.path.join(self.dataRoot, self._program.progId)
        self.__lock = threading.Lock()
        self.__thrdsReadAhead = []
        self.__chunksReadAhead = []
        self.__newChunks =[]
        self.__recycledChunks =[]
        self.__convertFrame = self.__frameToBatchs

        self.__latestBthNo=0

        self.__knownModels = {
            'VGG16d1'    : self.__createModel_VGG16d1,
            'Cnn1Dx4'    : self.__createModel_Cnn1Dx4,
            'Cnn1Dx4R1'  : self.__createModel_Cnn1Dx4R1,
            'Cnn1Dx4R1a'  : self.__createModel_Cnn1Dx4R1a,
            'Cnn1Dx4R2'  : self.__createModel_Cnn1Dx4R2,
            }

        STEPMETHODS = {
            'LocalGenerator'   : self.doAppStep_local_generator,
            'DatesetGenerator' : self.doAppStep_keras_dsGenerator,
            'BatchGenerator'   : self.doAppStep_keras_batchGenerator,
            'SliceToDataset'   : self.doAppStep_keras_slice2dataset,
            'DatasetPool'      : self.doAppStep_keras_datasetPool,
        }

        if not self._stepMethod or not self._stepMethod in STEPMETHODS.keys():
            self._stepMethod = 'LocalGenerator'
        
        self.info('taking method[%s]' % (self._stepMethod))
        self._stepMethod = STEPMETHODS[self._stepMethod]

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev): pass

    def doAppInit(self): # return True if succ
        if not super(MarketDirClassifier, self).doAppInit() :
            return False

        if not self._replayFrameFiles or len(self._replayFrameFiles) <=0:
            self.error('no input ReplayFrame files specified')
            return False

        self._replayFrameFiles.sort();
        self.info('ReplayFrame files: %s' % self._replayFrameFiles)

        self.__nextFrameName(False) # probe the dims of state/action from the h5 file
        self.__maxChunks = max(int(self._frameSize/self._batchesPerTrain /self._batchSize), 1) # minimal 8K samples to at least cover a frame

        if self._model_json:
            if len(GPUs) <= 1:
                self._brain = model_from_json(self._model_json)
            else:
                # we'll store a copy of the model on *every* GPU and then combine the results from the gradient updates on the CPU
                with tf.device("/cpu:0"):
                    self._brain = model_from_json(self._model_json)

            if not self._brain:
                self.error('model_from_json failed')
                return False
        
        if not self._brain and self._wkModelId and len(self._wkModelId) >0:
            wkModelId = '%s.S%sI%sA%s' % (self._wkModelId, self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
            inDir = os.path.join(self.dataRoot, wkModelId)
            try : 
                self.debug('loading saved model from %s' % inDir)
                with open(os.path.join(inDir, 'model.json'), 'r') as mjson:
                    model_json = mjson.read()
                    if len(GPUs) <= 1:
                        self._brain = model_from_json(model_json)
                    else:
                        with tf.device("/cpu:0"):
                            self._brain = model_from_json(model_json)

                sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
                self._brain.compile(optimizer=sgd, **MarketDirClassifier.COMPILE_ARGS)

                self._wkModelId = wkModelId

                fn_weights = os.path.join(inDir, 'weights.h5')
                self.debug('loading saved weights from %s' %fn_weights)
                self._brain.load_weights(fn_weights)
                self.info('loaded model and weights from %s' %inDir)

            except Exception as ex:
                self.logexception(ex)

            if not self._brain and not self._wkModelId in self.__knownModels.keys():
                self.warn('unknown modelId[%s], taking % instead' % (self._wkModelId, MarketDirClassifier.DEFAULT_MODEL))
                self._wkModelId = MarketDirClassifier.DEFAULT_MODEL

        if not self._brain:
            if len(GPUs) <= 1:
                self._brain = self.__knownModels[self._wkModelId]()
            else:
                with tf.device("/cpu:0"):
                    self._brain = self.__knownModels[self._wkModelId]()

        try :
            os.makedirs(self._outDir)
            fn_model =os.path.join(self._outDir, '%s.model.json' %self._wkModelId) 
            with open(fn_model, 'w') as mjson:
                model_json = self._brain.to_json()
                mjson.write(model_json)
                self.info('saved model as %s' %fn_model)
        except :
            pass

        if len(GPUs) > 1: # make the model parallel
            self.info('training with m-GPU: %s' % GPUs)
            self._brain = multi_gpu_model(self._brain, gpus=len(GPUs))

        checkpoint = ModelCheckpoint(os.path.join(self._outDir, '%s.best.h5' %self._wkModelId ), verbose=0, monitor='loss', mode='min', save_best_only=True)
        self._fitCallbacks = [checkpoint]
        if self._exportTB :
            cbTensorBoard = TensorBoard(log_dir=os.path.join(self._outDir, 'tb'), histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                    write_graph=True,  # 是否存储网络结构图
                    write_grads=True, # 是否可视化梯度直方图
                    write_images=True) # ,# 是否可视化参数
                    # embeddings_freq=0,
                    # embeddings_layer_names=None, 
                    # embeddings_metadata=None)

            self._fitCallbacks.append(cbTensorBoard)

        self._gen = self.__generator_local()

        return True

    def doAppStep(self):
        if not self._stepMethod:
            self.stop()
            return

        self._stepMethod()
        return super(MarketDirClassifier, self).doAppStep()

    def doAppStep_local_generator(self):
        if not self._gen:
            self.stop()
            return

        try:
            next(self._gen)
        except Exception as ex:
            self.stop()
            self.logexception(ex)
            raise StopIteration

    def doAppStep_keras_batchGenerator(self):
        # frameSeq= [i for i in range(len(self._framesInHd5))]
        # random.shuffle(frameSeq)
        # result = self._brain.fit_generator(generator=self.__gen_readBatchFromFrameEx(frameSeq), workers=2, use_multiprocessing=True, epochs=self._initEpochs, steps_per_epoch=1000, verbose=1, callbacks=self._fitCallbacks)

        result, histEpochs = None, []
        self.refreshPool()
        use_multiprocessing = not 'windows' in self._program.ostype

        try:
            result = self._brain.fit_generator(generator=Hd5DataGenerator(self, self._batchSize), workers=8, use_multiprocessing=use_multiprocessing, epochs=self._initEpochs, steps_per_epoch=1000, verbose=1, callbacks=self._fitCallbacks)
            histEpochs += self.__resultToStepHist(result)
            self.__logAndSaveResult(histEpochs[-1], 'doAppStep_keras_batchGenerator')
        except Exception as ex: self.logexception(ex)

    def doAppStep_keras_dsGenerator(self):
        # ref: https://pastebin.com/kRLLmdxN
        # training_set = tfdata_generator(x_train, y_train, is_training=True, batch_size=_BATCH_SIZE)
        # result = self._brain.fit(training_set.make_one_shot_iterator(), epochs=self._initEpochs, batch_size=self._batchSize, verbose=1, callbacks=self._fitCallbacks)
        # model.fit(training_set.make_one_shot_iterator(), steps_per_epoch=len(x_train) // _BATCH_SIZE
        #     epochs=_EPOCHS, validation_data=testing_set.make_one_shot_iterator(), validation_steps=len(x_test) // _BATCH_SIZE,
        #     verbose=1)

        result, histEpochs = None, []
        self.refreshPool()
        dataset = tf.data.Dataset.from_generator(generator =self.__gen_readDataFromFrame,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=((self._stateSize,), (self._actionSize,)))

        dataset = dataset.batch(self._batchSize).shuffle(100)
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.repeat()

        try :
            result = self._brain.fit(dataset.make_one_shot_iterator(), epochs=self._initEpochs, steps_per_epoch=self.chunksInPool, verbose=1, callbacks=self._fitCallbacks)
            histEpochs += self.__resultToStepHist(result)
            self.__logAndSaveResult(histEpochs[-1], 'doAppStep_keras_dsGenerator')
        except Exception as ex: self.logexception(ex)

    def doAppStep_keras_slice2dataset(self):

        self.__convertFrame = self.__frameToSlices
        result, histEpochs = None, []
        self.refreshPool()

        for i in range(self.chunksInPool) :
            slice = self.readDataChunk(i)
            length = len(slice[0])

            dataset = tf.data.Dataset.from_tensor_slices(slice)
            slice = None # free the memory
            dataset = dataset.batch(self._batchSize)
            if self._initEpochs >1:
                dataset = dataset.repeat() #.shuffle(self._batchSize*2)

            # dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            if 0 ==i: self.info('doAppStep_keras_slice2dataset() starts fitting slice %sx %s' % (length, str(dataset.output_shapes)))
            try :
                result = self._brain.fit(dataset, epochs=self._initEpochs, steps_per_epoch=self._batchesPerTrain, verbose=1, callbacks=self._fitCallbacks)
                histEpochs += self.__resultToStepHist(result)
            except Exception as ex: self.logexception(ex)

        self.__logAndSaveResult(histEpochs[-1], 'doAppStep_keras_slice2dataset')

    def doAppStep_keras_datasetPool(self):

        self.__convertFrame = self.__frameToDatasets
        result, histEpochs = None, []
        self.refreshPool()

        for i in range(self.chunksInPool) :
            dataset = self.readDataChunk(i)
            if self._initEpochs >1:
                dataset = dataset.repeat() # .shuffle(self._batchSize*2)
            # dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            if 0 ==i: self.info('doAppStep_keras_datasetPool() starts fitting ds %s' % str(dataset.output_shapes))
            try :
                result = self._brain.fit(dataset, epochs=self._initEpochs, steps_per_epoch=self._batchesPerTrain, verbose=1, callbacks=self._fitCallbacks)
                histEpochs += self.__resultToStepHist(result)
            except Exception as ex: self.logexception(ex)

        self.__logAndSaveResult(histEpochs[-1], 'doAppStep_keras_datasetPool')

    def __resultToStepHist(self, result):
        if not result: return []

        losshist, accuhist = result.history["loss"], result.history["acc"] if 'acc' in result.history.keys() else [ -1.0 ]
        if len(accuhist) <=1 and 'accuracy' in result.history.keys():
            accuhist = result.history["accuracy"]

        #losshist.reverse()
        #accuhist.reverse()
        #loss, accu, stephist = losshist[0], accuhist[0], []
        
        if len(losshist) == len(accuhist) :
            stephist = ['%.2f%%^%.3f' % (accuhist[i]*100, losshist[i]) for i in range(len(losshist))]
        else:
            stephist = ['%.2f' % (losshist[i]) for i in range(len(losshist))]

        return stephist

    def __logAndSaveResult(self, resFinal, methodName, notes=''):
        if not notes or len(notes) <0: notes=''

        fn_weights = os.path.join(self._outDir, '%s.weights.h5' %self._wkModelId)
        self._brain.save(fn_weights)

        self.info('%s() saved weights %s, result[%s] %s' % (methodName, fn_weights, resFinal, notes))

    # end of BaseApplication routine
    #----------------------------------------------------------------------
    def __gen_readBatchFromFrame(self) :
        frameSeq= []
        while True:
            if len(frameSeq) <=0:
                frameSeq= [i for i in range(len(self._framesInHd5))]
                random.shuffle(frameSeq)
            
            try :
                return self.__gen_readBatchFromFrameEx(frameSeq)
            except StopIteration:
                frameSeq= []

    def __gen_readBatchFromFrameEx(self, frameSeq) :
        while len(frameSeq)>0:
            frameName = self._framesInHd5[frameSeq[0]]
            frame = self._h5file[frameName]
            for i in range(int(8192/self._batchSize)) :
                offset = self._batchSize*i
                yield np.array(list(frame['state'].value)[offset: offset+self._batchSize]), np.array(list(frame['action'].value[offset: offset+self._batchSize]))

            del frameSeq[0]
        raise StopIteration

    def __gen_readDataFromFrame(self) :
        for bth in range(self.chunksInPool) :
            batch = self.readDataChunk(bth)
            for i in range(len(batch['state'])) :
                yield batch['state'][i], batch['action'][i]

    def __fit_gen(self):

        frameSeq= copy.copy(self._framesInHd5)
        random.shuffle(frameSeq)

        dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(len(self._framesInHd5))]))
        dataset = dataset.map(lambda x: self.__readFrame(x)) # list(self._h5file[int(x)]['state'].value), list(self._h5file[int(x)]['action'].value)) # (self.__readFrame)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(self.readFrame, batch_size,
        #     num_parallel_batches=4, # cpu cores
        #     drop_remainder=True if is_training else False))

        dataset = dataset.shuffle(1000 + 3 * self._batchSize)
        dataset = dataset.batch(self._batchSize)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()

        with K.get_session().as_default() as sess:
            while True:
                *inputs, labels = sess.run(next_batch)
                yield inputs, labels
    
    @property
    def chunksInPool(self):
        return len(self.__newChunks)

    def refreshPool(self):
        # build up self.__samplePool

        thrdBusy = True # dummy init value
        while not thrdBusy is None:
            thrdBusy =None
            with self.__lock:
                for th in self.__thrdsReadAhead:
                    thrdBusy = th
                    if thrdBusy: break
        
            if thrdBusy:
                self.warn('refreshPool() readAhead thread is still running, waiting for its completion')
                thrdBusy.join()

        cChunks=0
        with self.__lock:
            if self._frameSize >0:
                cChunks = ((self.__maxChunks * self._batchesPerTrain * self._batchSize) + self._frameSize -1) // self._frameSize
            if cChunks<=0: cChunks =1
            cChunks =int(cChunks)

            self.__thrdsReadAhead = [None] * cChunks

        if not self.__chunksReadAhead or len(self.__chunksReadAhead) <=0:
            self.warn('refreshPool() no readAhead ready, force to read sync-ly')
            # # Approach 1. multiple readAhead threads to read one frame each
            # for i in range(cChunks) :
            #     self.__readAhead(thrdSeqId=i)
            # Approach 2. the readAhead thread that read a list of frames
            self.__readAheadChunks(thrdSeqId=-1, cChunks=cChunks)

        with self.__lock:
            self.__newChunks, self.__samplesFrom = self.__chunksReadAhead, self.__framesReadAhead
            self.__chunksReadAhead, self.__framesReadAhead = [] , []
            self.debug('refreshPool() pool refreshed from readAhead: %s x(%s bth/c, %s samples/bth), reset readAhead to %d and kicking off new round of read-ahead' % (len(self.__newChunks), self._batchesPerTrain, self._batchSize, len(self.__chunksReadAhead)))

            # # Approach 1. kickoff multiple readAhead threads to read one frame each
            # for i in range(cChunks) :
            #     thrd = threading.Thread(target=self.__readAhead, kwargs={'thrdSeqId': i} )
            #     self.__thrdsReadAhead[i] =thrd
            #     thrd.start()

            # Approach 2. kickoff a readAhead thread to read a list of frames
            thrd = threading.Thread(target=self.__readAheadChunks, kwargs={'thrdSeqId': 0, 'cChunks': cChunks } )
            self.__thrdsReadAhead[0] =thrd
            thrd.start()

        newsize = self.chunksInPool
        self.info('refreshPool() pool refreshed from readAhead: %s x(%s bth/c, %s samples/bth) from %s; %s readahead started' % (newsize, self._batchesPerTrain, self._batchSize, ','.join(self.__samplesFrom), cChunks))
        return newsize

    def readDataChunk(self, chunkNo):
        return self.__newChunks[chunkNo]

    def nextDataChunk(self):
        '''
        @return chunk, bRecycledData   - bRecycledData=True if it is from the recycled data
        '''
        ret = None
        with self.__lock:
            if self.__newChunks and len(self.__newChunks) >0:
                ret = self.__newChunks[0]
                del self.__newChunks[0]
                self.__recycledChunks.append(ret)
                if len(self.__recycledChunks) >= ((1+self._recycleSize) *self._batchesPerTrain):
                    random.shuffle(self.__recycledChunks)
                    del self.__recycledChunks[(self._recycleSize *self._batchesPerTrain):]

                return ret, False

            if self._recycleSize>0 and len(self.__recycledChunks) >0:
                ret = self.__recycledChunks[0]
                del self.__recycledChunks[0]
                self.__recycledChunks.append(ret)
        
        bRecycled = True
        # no chunk addressed if reach here, copy the original impl of refreshPool()
        thrdBusy = True # dummy init value
        while not thrdBusy is None:
            thrdBusy =None
            with self.__lock:
                for th in self.__thrdsReadAhead:
                    thrdBusy = th
                    if thrdBusy: break
        
            if ret and thrdBusy: # there is already a background read-ahead thread, so return the result instantly
                return ret, bRecycled

            if thrdBusy:
                self.warn('nextDataChunk() readAhead thread is still running, waiting for its completion')
                thrdBusy.join()

        cChunks=0
        with self.__lock:
            if self._frameSize >0:
                cChunks = ((self.__maxChunks * self._batchesPerTrain * self._batchSize) + self._frameSize -1) // self._frameSize
            if cChunks<=0: cChunks =1
            cChunks =int(cChunks)

            self.__thrdsReadAhead = [None] * cChunks

        if not ret and not self.__chunksReadAhead or len(self.__chunksReadAhead) <=0:
            self.warn('nextDataChunk() no readAhead ready, force to read sync-ly')
            self.__readAheadChunks(thrdSeqId=-1, cChunks=cChunks)

        szRecycled = 0
        with self.__lock:
            self.__newChunks, self.__samplesFrom = self.__chunksReadAhead, self.__framesReadAhead
            self.__chunksReadAhead, self.__framesReadAhead = [], []
            newsize = len(self.__newChunks)
            self.debug('nextDataChunk() pool refreshed from readAhead: %s x(%s bth/c, %s samples/bth), reset readAhead to %d and kicking off new round of read-ahead' % (newsize, self._batchesPerTrain, self._batchSize, len(self.__chunksReadAhead)))
            
            if not ret and self.__newChunks and newsize >0:
                ret = self.__newChunks[0]
                del self.__newChunks[0]
                self.__recycledChunks.append(ret)
                bRecycled = False
            szRecycled = len(self.__recycledChunks)

            thrd = threading.Thread(target=self.__readAheadChunks, kwargs={'thrdSeqId': 0, 'cChunks': cChunks } )
            self.__thrdsReadAhead[0] =thrd
            thrd.start()

        self.info('nextDataChunk() pool refreshed: %s x(%s samples/bth) from %s; %s readahead started, recycled-size:%s' % (newsize, self._batchSize, ','.join(self.__samplesFrom), cChunks, szRecycled))
        return ret, bRecycled

    def __frameToSlices(self, frameDict):
        framelen = 1
        for k,v in frameDict.items():
            framelen = len(v)
            if framelen>= self._batchSize: break

        samplesPerChunk = self._batchesPerTrain * self._batchSize
        cChunks = int(framelen // samplesPerChunk)
        if cChunks <=0 :
            cChunks, samplesPerChunk = 1, framelen

        slices = []
        for i in range(cChunks) :
            bthState  = np.array(frameDict['state'][i*samplesPerChunk: (i+1)*samplesPerChunk])
            bthAction = np.array(frameDict['action'][i*samplesPerChunk: (i+1)*samplesPerChunk])
            slices.append((bthState, bthAction))

        return slices

    def __frameToDatasets(self, frameDict):
        framelen = 1
        for k,v in frameDict.items():
            framelen = len(v)
            if framelen>= self._batchSize: break

        samplesPerChunk = self._batchesPerTrain * self._batchSize
        cChunks = int(framelen // samplesPerChunk)
        if cChunks <=0 :
            cChunks, samplesPerChunk = 1, framelen

        datasets = []
        for i in range(cChunks) :
            bthState  = np.array(frameDict['state'][i*samplesPerChunk: (i+1)*samplesPerChunk])
            bthAction = np.array(frameDict['action'][i*samplesPerChunk: (i+1)*samplesPerChunk])
            dataset = tf.data.Dataset.from_tensor_slices((bthState, bthAction))
            dataset = dataset.batch(self._batchSize)
            datasets.append(dataset)

        return datasets

    def __frameToBatchs(self, frameDict):
        COLS = ['state','action']
        framelen = len(frameDict[COLS[0]])
        bths = []
        cBth = framelen // self._batchSize
        for i in range(cBth):
            batch = {}
            for col in COLS :
                batch[col] = np.array(frameDict[col][self._batchSize*i: self._batchSize*(i+1)]).astype(NN_FLOAT)
            
            bths.append(batch)

        return bths

    def __nextFrameName(self, bPop1stFrameName=False):
        '''
        get the next (h5fileName, frameName, framesAwait) to read from the H5 file
        '''
        h5fileName, nextFrameName, awaitSize = None, None, 0

        with self.__lock:
            if not self._frameSeq or len(self._frameSeq) <=0:
                self._frameSeq =[]

                fileList = copy.copy(self._replayFrameFiles)
                for h5fileName in fileList :
                    framesInHd5 = []
                    try:
                        self.debug('loading ReplayFrame file %s' % h5fileName)
                        with h5py.File(h5fileName, 'r') as h5f:
                            framesInHd5 = []
                            for name in h5f.keys() :
                                if RFGROUP_PREFIX == name[:len(RFGROUP_PREFIX)] :
                                    framesInHd5.append(name[len(RFGROUP_PREFIX):])

                            # I'd like to skip frame-0 as it most-likly includes many zero-samples
                            if len(framesInHd5)>3:
                                del framesInHd5[0:3] # 3frames is about 4mon
                                # del framesInHd5[-1]
                            
                            if len(framesInHd5)>6:
                                del framesInHd5[0]

                            if len(framesInHd5) <=1:
                                self._replayFrameFiles.remove(h5fileName)
                                self.error('file %s elimited as too few ReplayFrames in it' % (h5fileName) )
                                continue

                            f1st = RFGROUP_PREFIX + framesInHd5[0]
                            frameSize  = h5f[f1st]['state'].shape[0]
                            stateSize  = h5f[f1st]['state'].shape[1]
                            actionSize = h5f[f1st]['action'].shape[1]

                            if self._stateSize and self._stateSize != stateSize or self._actionSize and self._actionSize != actionSize:
                                self._replayFrameFiles.remove(h5fileName)
                                self.error('file %s elimited as its dims: %s/state %s/action mismatch working dims %s/state %s/action' % (h5fileName, stateSize, actionSize, self._stateSize, self._actionSize) )
                                continue

                            if self._frameSize < frameSize:
                                self._frameSize = frameSize
                            self._stateSize = stateSize
                            self._actionSize = actionSize

                            self.info('%d ReplayFrames found in %s with dims: %s/state, %s/action' % (len(framesInHd5), h5fileName, self._stateSize, self._actionSize) )

                    except Exception as ex:
                        self._replayFrameFiles.remove(h5fileName)
                        self.error('file %s elimited per IO exception: %s' % (h5fileName, str(ex)) )
                        continue

                    for i in range(max(1, 1+self._repeatsInFile)):
                        seq = [(h5fileName, frmName) for frmName in framesInHd5]
                        # random.shuffle(seq)
                        self._frameSeq += seq

                random.shuffle(self._frameSeq)
                self.info('frame sequence rebuilt: %s frames from %s replay files' % (len(self._frameSeq), len(self._replayFrameFiles)) )

            h5fileName, nextFrameName = self._frameSeq[0]

            if bPop1stFrameName: del self._frameSeq[0]
            awaitSize = len(self._frameSeq)

        return h5fileName, nextFrameName, awaitSize

    def readFrame(self, h5fileName, frameName):
        '''
        read a frame from H5 file
        '''
        COLS = ['state','action']
        frameDict ={}
        try :
            # reading the frame from the h5
            self.debug('readAhead() reading %s of %s' % (frameName, h5fileName))
            with h5py.File(h5fileName, 'r') as h5f:
                frame = h5f[RFGROUP_PREFIX + frameName]

                for col in COLS :
                    if col in frameDict.keys():
                        frameDict[col] += list(frame[col])
                    else : frameDict[col] = list(frame[col])
        except Exception as ex:
            self.logexception(ex)

        return frameDict

    def __readAhead(self, thrdSeqId=0):
        '''
        the background thread to read A frame from H5 file
        reading H5 only works on CPU and is quite slow, so take a seperate thread to read-ahead
        '''
        stampStart = datetime.now()
        h5fileName, nextFrameName, awaitSize = self.__nextFrameName(True)

        frameDict = self.readFrame(h5fileName, nextFrameName)
        lenFrame= 0
        for v in frameDict.values() :
            lenFrame = len(v)
            break

        self.debug('readAhead(%s) read %s samples from %s@%s' % (thrdSeqId, lenFrame, nextFrameName, h5fileName) )
        cvnted = frameDict
        try :
            if self.__convertFrame :
                cvnted = self.__convertFrame(frameDict)
                self.debug('readAhead(%s) converted %s samples of %s@%s into %s chunks' % (thrdSeqId, lenFrame, nextFrameName, h5fileName, len(cvnted)) )
        except Exception as ex:
            self.logexception(ex)

        addSize, raSize=0, 0
        with self.__lock:
            self.__thrdsReadAhead[thrdSeqId] = None

            if isinstance(cvnted, list) :
                self.__chunksReadAhead += cvnted
                addSize, raSize = len(cvnted), len(self.__chunksReadAhead)
            else:
                self.__chunksReadAhead.append(cvnted)
                addSize, raSize = 1, len(self.__chunksReadAhead)

        frameDict, cvnted = None, None
        self.info('readAhead(%s) prepared %s->%s x%s s/bth from %s took %s, %d frames await' % 
            (thrdSeqId, addSize, raSize, self._batchSize, nextFrameName, str(datetime.now() - stampStart), awaitSize))

    def __readAheadChunks(self, thrdSeqId=0, cChunks=1):
        '''
        the background thread to read a number of frames from H5 files
        reading H5 only works on CPU and is quite slow, so take a seperate thread to read-ahead
        '''
        stampStart = datetime.now()
        strFrames =[]
        awaitSize =-1
        addSize, raSize=0, 0

        while cChunks >0 :

            h5fileName, nextFrameName, awaitSize = self.__nextFrameName(True)
            frameDict = self.readFrame(h5fileName, nextFrameName)
            lenFrame= 0
            for v in frameDict.values() :
                lenFrame = len(v)
                break

            self.debug('readAheadChunks(%s) read %s samples from %s@%s' % (thrdSeqId, lenFrame, nextFrameName, h5fileName) )
            strFrames.append('%s@%s' % (nextFrameName, os.path.basename(h5fileName)))
            cvnted = frameDict
            try :
                if self.__convertFrame :
                    cvnted = self.__convertFrame(frameDict)
                    self.debug('readAheadChunks(%s) converted %s samples into %s chunks' % (thrdSeqId, lenFrame, len(cvnted)) )
            except Exception as ex:
                self.logexception(ex)

            with self.__lock:
                size =1
                if isinstance(cvnted, list) :
                    self.__chunksReadAhead += cvnted
                    size = len(cvnted)
                else:
                    self.__chunksReadAhead.append(cvnted)

                cChunks -= 1
                addSize += size

            frameDict, cvnted = None, None

        with self.__lock:
            random.shuffle(self.__chunksReadAhead)
            raSize = len(self.__chunksReadAhead)
            self.__framesReadAhead = strFrames

            if thrdSeqId>=0 and thrdSeqId < len(self.__thrdsReadAhead) :
                self.__thrdsReadAhead[thrdSeqId] = None


        self.info('readAheadChunks(%s) took %s to prepare %s->%s x%s s/bth from %dframes:%s; %d frames await' % 
            (thrdSeqId, str(datetime.now() - stampStart), addSize, raSize, self._batchSize, len(strFrames), ','.join(strFrames), awaitSize))

    def __generator_local(self):

        self.__convertFrame = self.__frameToBatchs

        # build up self.__samplePool
        self.__samplePool = {
            'state':[],
            'action':[],
        }

        itrId=0
        samplePerFrame =0
        trainSize = self._batchesPerTrain*self._batchSize

        loss = DUMMY_BIG_VAL
        lossMax = loss
        idxBatchInPool =int(DUMMY_BIG_VAL)
        while True : #TODO temporarily loop for ever: lossMax > self._lossStop or abs(loss-lossMax) > (lossMax * self._lossPctStop/100) :

            statebths, actionbths =[], []
            cFresh, cRecycled = 0, 0
            while len(statebths) < self._batchesPerTrain :
                bth, recycled = self.nextDataChunk() #= self.readDataChunk(idxBatchInPool)
                if recycled:
                    cRecycled += 1
                else:
                    cFresh += 1

                statebths.append(bth['state'])
                actionbths.append(bth['action'])

            #----------------------------------------------------------
            # continue # if only test read-ahead and pool making-up   #
            #----------------------------------------------------------

            cBths = len(statebths)
            if cBths < self._batchesPerTrain:
                continue

            statechunk = np.concatenate(tuple(statebths))
            actionchunk = np.concatenate(tuple(actionbths))
            statebths, actionbths =[], []

            trainSize = statechunk.shape[0]
            
            stampStart = datetime.now()
            result, lstEpochs, histEpochs = None, [], []
            strEval =''
            loss = max(11, loss)
            sampledAhead = cFresh>(cRecycled*4)
            epochs = self._initEpochs if sampledAhead else 2
            while epochs > 0:
                if len(strEval) <=0 and sampledAhead:
                    try :
                        resEval =  self._brain.evaluate(x=statechunk, y=actionchunk, batch_size=self._batchSize, verbose=1) #, callbacks=self._fitCallbacks)
                        strEval += 'eval[%.2f%%^%.3f]/%s' % (resEval[1]*100, resEval[0], datetime.now() -stampStart)
                    except Exception as ex:
                        self.logexception(ex)

                # call trainMethod to perform tranning
                itrId +=1
                try :
                    epochs2run = epochs
                    epochs =0
                    result = self._brain.fit(x=statechunk, y=actionchunk, epochs=epochs2run, shuffle=True, batch_size=self._batchSize, verbose=1, callbacks=self._fitCallbacks)
                    lstEpochs.append(epochs2run)
                    loss = result.history["loss"][-1]
                    lossImprove =0.0
                    if len(result.history["loss"]) >1 :
                        lossImprove = result.history["loss"][-2] - loss

                    if sampledAhead and loss > self._lossStop and lossImprove > (loss * self._lossPctStop/100) :
                        epochs = epochs2run
                        if lossImprove > (loss * self._lossPctStop *2 /100) :
                            epochs += int(epochs2run/2)

                    if lossMax>=DUMMY_BIG_VAL-1 or lossMax < loss: lossMax = loss
                    histEpochs += self.__resultToStepHist(result)

                    yield result # this is a step

                except Exception as ex:
                    self.logexception(ex)

            strEpochs = '+'.join([str(i) for i in lstEpochs])
            if sampledAhead :
                self.__logAndSaveResult(histEpochs[-1], 'doAppStep_local_generator', 'from %s, %s/%s steps x %s epochs on %dN+%dR samples took %s, hist: %s' % (strEval, trainSize, self._batchSize, strEpochs, cFresh, cRecycled, (datetime.now() -stampStart), ', '.join(histEpochs)) )
            else :
                self.info('doAppStep_local_generator() %s epochs on recycled %dN+%dR samples took %s, hist: %s' % (strEpochs, cFresh, cRecycled, (datetime.now() -stampStart), ', '.join(histEpochs)) )

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
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-5), **MarketDirClassifier.COMPILE_ARGS)

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
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **MarketDirClassifier.COMPILE_ARGS)

        return model

    def __createModel_Cnn1Dx4R1a(self):
        self._wkModelId = 'Cnn1Dx4R1a.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        model.add(Conv1D(128, 3, activation='relu', input_shape=(self._stateSize/EXPORT_FLOATS_DIMS, EXPORT_FLOATS_DIMS)))
        model.add(BatchNormalization())
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
        model.add(Dense(216, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **MarketDirClassifier.COMPILE_ARGS)

        return model

    def __createModel_Cnn1Dx4R2(self):
        self._wkModelId = 'Cnn1Dx4R2.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        model = Sequential()
        model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        model.add(Conv1D(128, 3, activation='relu', input_shape=(self._stateSize/EXPORT_FLOATS_DIMS, EXPORT_FLOATS_DIMS)))
        model.add(BatchNormalization())
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(512, 3, activation='relu'))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Conv1D(100, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(20, activation='relu'))
        
        model.add(Dense(self._actionSize, activation='softmax')) # this is not Q func, softmax is prefered
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **MarketDirClassifier.COMPILE_ARGS)

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
        model.compile(optimizer=sgd, **MarketDirClassifier.COMPILE_ARGS)

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
        self._gen = self.__generator_local(self.__train_DDQN)
        return True

    # end of BaseApplication routine
    #----------------------------------------------------------------------

    def __generator_local(self, trainMethod):

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
        trainSize = self._batchesPerTrain*self._batchSize

        loss = DUMMY_BIG_VAL
        lossMax = loss
        while len(frameSeq) >0 or lossMax > self._lossStop or abs(loss-lossMax) > (lossMax * self._lossPctStop/100) :
            if len(frameSeq) <=0:
                a = copy.copy(self._framesInHd5)
                random.shuffle(a)
                frameSeq +=a
            
            startPoolSize = len(self.__samplePool['state'])
            cEvicted =0
            if startPoolSize >= max(samplePerFrame, trainSize *2):
                # randomly evict half of the poolSize
                sampleIdxs = [a for a in range(min(samplePerFrame, int(startPoolSize/2)))]
                random.shuffle(sampleIdxs)
                for i in sampleIdxs:
                    cEvicted +=1
                    for col in self.__samplePool.keys() :
                        del self.__samplePool[col][i]

            cAppend =0
            strFrames=''
            while len(frameSeq) >0 and len(self.__samplePool['state']) <max(samplePerFrame, trainSize *2) :
                strFrames += '%s,' % frameSeq[0]
                frame = h5f[frameSeq[0]]
                del frameSeq[0]

                for col in self.__samplePool.keys() :
                    incrematal = list(frame[col].value)
                    samplePerFrame = len(incrematal)
                    self.__samplePool[col] += incrematal
                cAppend += samplePerFrame

            poolSize = len(self.__samplePool['state'])
            self.info('sample pool refreshed: size[%s->%s] by evicting %s and refilling %s samples from %s %d frames await' % (startPoolSize, poolSize, cEvicted, cAppend, strFrames, len(frameSeq)))

            # random sample a dataset with size=trainSize from self.__samplePool
            sampleSeq = [a for a in range(poolSize)]
            random.shuffle(sampleSeq)
            if self._recycleSize >0:
                tmpseq = copy.copy(sampleSeq)
                for i in range(self._recycleSize) :
                    random.shuffle(tmpseq)
                    sampleSeq += tmpseq

            if len(sampleSeq) >= self._batchSize:
                lossMax = loss if loss < DUMMY_BIG_VAL-1 else 0.0

            while len(sampleSeq) >= self._batchSize:

                if len(sampleSeq) > trainSize:
                    sampleIdxs = sampleSeq[:trainSize]
                    del sampleSeq[:trainSize]
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
                self.info('train[%s] done, sampled %d from poolsize[%s], loss[%s]' % (str(itrId).zfill(6), trainSize, poolSize, loss))
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

        return self._brain.fit(x=samples['state'], y=Q_target, epochs=self._initEpochs, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)

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

        return brainTrain.fit(x=samples['state'], y=Q_target, epochs=self._initEpochs, batch_size=self._batchSize, verbose=0, callbacks=self._fitCallbacks)

########################################################################
if __name__ == '__main__':

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/DQNTrainer_Cnn1D.json'] # 'DQNTrainer_VGG16d1.json' 'Gym_AShare.json'

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
    
    # trainer = p.createApp(DQNTrainer, configNode ='DQNTrainer', replayFrameFiles=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))
    trainer = p.createApp(MarketDirClassifier, configNode ='DQNTrainer')

    p.start()
    p.loop()
    p.info('loop done, all objs: %s' % p.listByType())
    p.stop()
