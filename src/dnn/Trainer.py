# encoding: UTF-8

'''
A DQN Trainer_classify detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''
from __future__ import division
from abc import abstractmethod

from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
from HistoryData  import H5DSET_DEFAULT_ARGS, listAllFiles, classifyGainRates_screeningTplus1
from dnn.BaseModel import BaseModel, BACKEND_FLOAT
from dnn.Makeups  import Model88_sliced2d
from HistoryData  import SAMPLE_FLOAT # note: SAMPLE_FLOAT MUST compatible with but not equal to BaseModel.INPUT_FLOATS

# ----------------------------
# INDEPEND FROM HyperPX core classes: from MarketData import EXPORT_FLOATS_DIMS
EXPORT_FLOATS_DIMS = 4
DUMMY_BIG_VAL = 999999
RFGROUP_PREFIX = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'

# ----------------------------

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# import tensorflow.keras.engine.saving as saving # import keras.engine.saving as saving
import tensorflow as tf
import numpy as np

import sys, os, platform, random, copy, threading
from datetime import datetime
from time import sleep

import tarfile, pickle, fnmatch
import shutil

import h5py
H5_OPEN_ARGS={}
try:
    # https://docs.h5py.org/en/stable/mpi.html
    from mpi4py import MPI # on CentOS: yum install openmpi-devel; export CC=/usr/lib64/openmpi/bin/mpicc; pip3 install mpi4py
    H5_OPEN_ARGS={'driver':'mpio', 'comm': MPI.COMM_WORLD}
except: pass

CPUs, GPUs = BaseModel.list_processors()
if len(GPUs) >1:
    from keras.utils.training_utils import multi_gpu_model

########################################################################
class Trainer_classify(BaseApplication):

    def __init__(self, program, sampleFiles=None, recorder =None, **kwargs):
        super(Trainer_classify, self).__init__(program, **kwargs)

        self.__wkModelId       = None
        self._fnStartModel     = self.getConfig('startModel', None)
        self._baseModelClass   = self.getConfig('baseModel', 'base').lower()

        self._sampleFiles = sampleFiles
        if not self._sampleFiles or len(self._sampleFiles) <=0: 
            self._sampleFiles = self.getConfig('sampleFiles', [])

        self._dirSamples          = self.getConfig('dirSamples', None)
        self._colnameSamples      = self.getConfig('samplesName', 'state')
        self._colnameClasses      = self.getConfig('classesName', 'action')
        self._shapeSamples        = self.getConfig('shapeSamples', None)
        self._sampleClassSize     = self.getConfig('sampleClassSize', None)
        self._evaluateSamples     = self.getConfig('evaluateSamples', 'yes').lower() in BOOL_STRVAL_TRUE
        self._preBalanced         = self.getConfig('preBalanced',  'no').lower() in BOOL_STRVAL_TRUE

        self._trainables          = self.getConfig('trainableLayers', ['*']) # ['F88.Dense*', 'VClz66from512.2of2']
        self._evalAt              = self.getConfig('evalAt', 5) # how often on trains to perform evaluation
        self._stepMethod          = self.getConfig('stepMethod', None)
        self._exportTB            = self.getConfig('tensorBoard', 'no').lower() in BOOL_STRVAL_TRUE
        self._repeatsInFile       = self.getConfig('repeatsInFile', 0)
        self._readAheadThrds      = self.getConfig('readAheads', 3)

        # training config block, default as CPU
        self._batchSize           = 128
        self._batchesPerTrain     = 8
        self._recycleSize         = 1
        self._initEpochs          = 2
        self._lossStop            = 0.24 # 0.24 according to average loss value by： grep 'from eval' /mnt/d/tmp/replayTrain_14276_0106.log |sed 's/.*loss\[\([^]]*\)\].*/\1/g' | awk '{ total += $1; count++ } END { print total/count }'
        self._lossDiffStop        = 5 # 5% difference
        self._startLR             = 0.01
        
        self.__readTraingConfigBlock('CPU')
        processorModel = 'CPU'
        trainParamConfs = ['CPU']

        # TESTCODE: GPUs = [ {'name': '/device:GPU:0', 'detail': 'device: 0, name: GeForce GTX 1650, pci bus id: 0000:01:00.0, compute capability: 7.5'} ]
        if len(GPUs) > 0 : # adjust some configurations if currently running on GPUs
            self.info('GPUs: %s' % GPUs)
            self.__readTraingConfigBlock('GPU')
            trainParamConfs.append('GPU')

            # apply per model configuration
            processorModel = GPUs[0]['detail']
            configedModels = self.getConfig('GPU/models', [])
            matchedModel = ''
            for m in configedModels:
                if isinstance(m, dict) and 'model' in m.keys() and len(m['model']) > len(matchedModel):
                    cfgm = m['model']
                    if cfgm in processorModel:
                        matchedModel = cfgm
                        self.__readTraingConfigBlock(m)
                        trainParamConfs.append(cfgm)
                        
        self.info('loaded params from: %s for %s and processor: %s' % ('-> '.join(trainParamConfs), BACKEND_FLOAT, processorModel))

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]
        self._frameSeq =[]

        self._evalAt =int(self._evalAt)
        self._frameSize = 0
        self._brain = None
        self.__lock = threading.Lock()
        self.__thrdsReadAhead = []
        self.__chunksReadAhead = []
        self.__newChunks =[]
        self.__recycledChunks =[]
        self._funcConvertFrame = self.__frameToBatchs
        self._funcFilterFrame  = None if self._preBalanced else self.__balanceSamples

        self.__latestBthNo=0
        self.__totalAccu, self.__totalEval, self.__totalSamples, self.__stampRound = 0.0, 0, 0, datetime.now()

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

    def __readTraingConfigBlock(self, configNode ='CPU'):

        if isinstance(configNode, str):
            configNode = self.getConfig(configNode, {})
        
        if isinstance(configNode, dict):
            self._batchSize           = configNode.get('batchSize', self._batchSize)
            self._batchesPerTrain     = configNode.get('batchesPerTrain', self._batchesPerTrain)
            self._recycleSize         = configNode.get('recycles', self._recycleSize)
            self._initEpochs          = configNode.get('initEpochs', self._initEpochs)
            self._lossStop            = configNode.get('lossStop', self._lossStop) 
            self._lossDiffStop        = configNode.get('lossDiffStop', self._lossDiffStop)
            self._startLR             = configNode.get('startLR', self._startLR)

    #----------------------------------------------------------------------
    # impl/overwrite of BaseApplication
    def OnEvent(self, ev): pass

    def doAppInit(self): # return True if succ
        if not super(Trainer_classify, self).doAppInit() :
            return False

        self._fnStartModel = Program.fixupPath(self._fnStartModel)
        self._sampleFiles = [ Program.fixupPath(f) for f in self._sampleFiles ]
        if self._dirSamples and len(self._dirSamples) >0:
            self._dirSamples = Program.fixupPath(self._dirSamples)

        fileList = self.__populateSampleFileList()

        if not fileList or len(fileList) <=0:
            self.error('no input sample files specified')
            return False

        self.debug('initial sample files: %s' % fileList)

        h5fn, framen, await_size = self.__nextFrameName(False) # probe the dims of samples/classes from the h5 file
        if None in [h5fn, framen] :
            self.error('doAppInit() quit per nextFrame failed')
            return False

        self.__maxChunks = max(int(self._frameSize/self._batchesPerTrain /self._batchSize), 1) # minimal 8K samples to at least cover a frame

        if len(GPUs) <= 1:
            # CPU mode
            if self._baseModelClass in [ 'model2d_sliced', 'sliced2d' ]: # lower-ed already
                self._brain = Model88_sliced2d.load(self._fnStartModel)
            else :
                self._brain = BaseModel.load(self._fnStartModel)
        else:
            # GPU mode
            # we'll store a copy of the model on *every* GPU and then combine the results from the gradient updates on the CPU
            with tf.device("/cpu:0"):
                if self._baseModelClass in [ 'model2d_sliced', 'sliced2d' ]: # lower-ed already
                    self._brain = Model88_sliced2d.load(self._fnStartModel)
                else :
                    self._brain = BaseModel.load(self._fnStartModel)
        
        if not self._brain:
            self.error('failed to load startModel[%s]' % self._fnStartModel)
            return False
        
        trainableLayers = []
        if not self._trainables or len(self._trainables) <=0:
            trainableLayers += self._brain.enable_trainable('*')
        else:
            if isinstance(self._trainables, str): self._trainables=self._trainables.split(',')
            for pat in self._trainables:
                trainableLayers += self._brain.enable_trainable(pat)

        trainableLayers = list(set(trainableLayers))
        trainableLayers.sort()
        if len(trainableLayers) >0:
            self.info('trainable-layers: %s' % ','.join(trainableLayers))

        sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        self._brain.compile(optimizer=sgd)
        self.__wkModelId = self._brain.modelId

        try :
            os.makedirs(self.outdir)
        except :
            pass

        if len(GPUs) > 1: # make the model parallel
            self.info('training with m-GPU: %s' % GPUs)
            self._brain = multi_gpu_model(self._brain, gpus=len(GPUs))

        checkpoint = ModelCheckpoint(os.path.join(self.outdir, '%s.best.h5' % self.__wkModelId ), verbose=0, monitor='loss', mode='min', save_best_only=True)
        self._fitCallbacks = [checkpoint]
        if self._exportTB :
            cbTensorBoard = TensorBoard(log_dir=os.path.join(self.outdir, 'tb'), histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
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
        return super(Trainer_classify, self).doAppStep()

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
                                                output_shapes=((self._brain.input_shape[1],), self._brain.output_shape))

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

        self._funcConvertFrame = self.__frameToSlices
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

        self._funcConvertFrame = self.__frameToDatasets
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

        try:
            # because the saving steps may throw exceptions and the saved h5 could be damaged/incompleted, so take two steps to keep good one as XXXX_trained-last.h5
            fn_trained = os.path.join(self.outdir, '%s_trained.%d.h5' %(self.__wkModelId, self.program.pid) )
            if self._brain.save(fn_trained) :
                shutil.copyfile(fn_trained, os.path.join(self.outdir, '%s_trained-last.h5' % self.__wkModelId) )
                self.info('%s() saved trained as %s, result[%s] %s' % (methodName, fn_trained, resFinal, notes))
        except Exception as ex:
            self.logexception('%s() save trained', ex)

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
                yield np.array(list(frame[self._colnameSamples].value)[offset: offset+self._batchSize]), np.array(list(frame[self._colnameClasses].value[offset: offset+self._batchSize]))

            del frameSeq[0]
        raise StopIteration

    def __gen_readDataFromFrame(self) :
        for bth in range(self.chunksInPool) :
            batch = self.readDataChunk(bth)
            for i in range(len(batch[self._colnameSamples])) :
                yield batch[self._colnameSamples][i], batch[self._colnameClasses][i]

    def __fit_gen(self):

        frameSeq= copy.copy(self._framesInHd5)
        random.shuffle(frameSeq)

        dataset = tf.data.Dataset.from_tensor_slices(np.array([i for i in range(len(self._framesInHd5))]))
        dataset = dataset.map(lambda x: self.__readFrame(x)) # list(self._h5file[int(x)][self._colnameSamples].value), list(self._h5file[int(x)][self._colnameClasses].value)) # (self.__readFrame)
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

        cFramesToRead, pendingAhead=0, 0
        with self.__lock:
            if self.__chunksReadAhead:
                pendingAhead = len(self.__chunksReadAhead)

            if self._frameSize >0:
                cFramesToRead = ((self.__maxChunks * self._batchesPerTrain * self._batchSize) + self._frameSize -1) // self._frameSize
            if cFramesToRead<=0: cFramesToRead =1
            cFramesToRead =int(cFramesToRead)

            self.__thrdsReadAhead = [None] * cFramesToRead

        if pendingAhead <=0:
            self.warn('refreshPool() no readAhead ready, force to read sync-ly')
            # # Approach 1. multiple readAhead threads to read one frame each
            # !!!!TODO:
            # !!!!TODO: h5py is thread-unsafe for multithread to read a same h5 file, so group frame-to-read into thread
            # !!!!TODO:
            # for i in range(cFramesToRead) :
            #     self.__readAhead(thrdSeqId=i)
            # Approach 2. the readAhead thread that read a list of frames
            self.__readAheadChunks(thrdSeqId=-1, cFramesToRead=cFramesToRead)
            # Approach 3. the readAhead thread that create subthreads to read a sequence of frames
            # self.__readFramesAhead(thrdSeqId=-1, cFramesToRead=cFramesToRead)

        with self.__lock:
            self.__newChunks, self.__samplesFrom = self.__chunksReadAhead, self.__framesReadAhead
            self.__chunksReadAhead, self.__framesReadAhead = [] , []
            self.debug('refreshPool() pool refreshed from readAhead: %s x(%s bth/c, %s samples/bth), reset readAhead to %d and kicking off new round of read-ahead' % (len(self.__newChunks), self._batchesPerTrain, self._batchSize, len(self.__chunksReadAhead)))

            # # Approach 1. kickoff multiple readAhead threads to read one frame each
            # for i in range(cFramesToRead) :
            #     thrd = threading.Thread(target=self.__readAhead, kwargs={'thrdSeqId': i} )
            #     self.__thrdsReadAhead[i] =thrd
            #     thrd.start()

            # Approach 2. kickoff a readAhead thread to read a list of frames
            thrd = threading.Thread(target=self.__readAheadChunks, kwargs={'thrdSeqId': 0, 'cFramesToRead': cFramesToRead } )

            # Approach 3. the readAhead thread that create subthreads to read a sequence of frames
            # thrd = threading.Thread(target=self.__readFramesAhead, kwargs={'thrdSeqId': 0, 'cFramesToRead': cFramesToRead } )
            self.__thrdsReadAhead[0] =thrd
            thrd.start()

        newsize = self.chunksInPool
        self.info('refreshPool() pool refreshed from readAhead: %s x(%s bth/c, %s samples/bth) from %s; %s readahead started' % (newsize, self._batchesPerTrain, self._batchSize, ','.join(self.__samplesFrom), cFramesToRead))
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

        cFrames=0
        with self.__lock:
            if self._frameSize >0:
                cFrames = (self.__maxChunks * self._batchesPerTrain * self._batchSize) // self._frameSize
            if cFrames<=0: cFrames =1
            cFrames = int(cFrames)

            self.__thrdsReadAhead = [None] * cFrames

        if not ret and not self.__chunksReadAhead or len(self.__chunksReadAhead) <=0:
            self.warn('nextDataChunk() no readAhead ready, force to read sync-ly')
            self.__readAheadChunks(thrdSeqId=-1, cFramesToRead=self._batchesPerTrain) # cFramesToRead=cFrames)
            # self.__readFramesAhead(thrdSeqId=-1, cFramesToRead=self._batchesPerTrain)

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

            thrd = threading.Thread(target=self.__readAheadChunks, kwargs={'thrdSeqId': 0, 'cFramesToRead': self._batchesPerTrain } ) # kwargs={'thrdSeqId': 0, 'cFramesToRead': cFramesToRead } )
            # thrd = threading.Thread(target=self.__readFramesAhead, kwargs={'thrdSeqId': 0, 'cFramesToRead': self._batchesPerTrain } ) # kwargs={'thrdSeqId': 0, 'cFramesToRead': cFramesToRead } )

            self.__thrdsReadAhead[0] =thrd
            thrd.start()

        self.info('nextDataChunk() pool refreshed: %s x(%s samples/bth) from %s; started reading %s+ chunks ahead, recycled-size:%s' % (newsize, self._batchSize, ','.join(self.__samplesFrom), self._batchesPerTrain, szRecycled))
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
            bths_Samples  = np.array(frameDict[self._colnameSamples][i*samplesPerChunk: (i+1)*samplesPerChunk])
            bths_Classes = np.array(frameDict[self._colnameClasses][i*samplesPerChunk: (i+1)*samplesPerChunk])
            slices.append((bths_Samples, bths_Classes))

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
            bths_Samples  = np.array(frameDict[self._colnameSamples][i*samplesPerChunk: (i+1)*samplesPerChunk])
            bths_Classes = np.array(frameDict[self._colnameClasses][i*samplesPerChunk: (i+1)*samplesPerChunk])
            dataset = tf.data.Dataset.from_tensor_slices((bths_Samples, bths_Classes))
            dataset = dataset.batch(self._batchSize)
            datasets.append(dataset)

        return datasets

    def __frameToBatchs(self, frameDict):
        COLS = [self._colnameSamples, self._colnameClasses]
        framelen = len(frameDict[COLS[0]])
        
        # to shuffle within the frame
        shuffledIndx =[i for i in range(framelen)]
        random.shuffle(shuffledIndx)

        bths = []
        cBth = framelen // self._batchSize
        for i in range(cBth):
            batch = {}
            for col in COLS :
                # batch[col] = np.array(frameDict[col][self._batchSize*i: self._batchSize*(i+1)]).astype(SAMPLE_FLOAT)
                batch[col] = np.array([frameDict[col][j] for j in shuffledIndx[self._batchSize*i: self._batchSize*(i+1)]]).astype(SAMPLE_FLOAT)
            
            bths.append(batch)

        return bths

    def __balanceSamples(self, frameDict) :
        '''
            balance the samples, usually reduce some action=HOLD, which appears too many
        '''
        chunk_Classes = np.array(frameDict[self._colnameClasses])
        # AD = np.where(chunk_Classes >=0.99) # to match 1 because action is float read from RFrames
        # kI = [np.count_nonzero(AD[1] ==i) for i in range(3)] # counts of each actions in frame

        # cRowToKeep = max(kI[1:]) + sum(kI[1:]) # = max(kI[1:]) *3
        # # cRowToKeep = int(sum(kI[1:]) /2 *3 +1)

        # # round up by batchSize
        # if self._batchSize >0:
        #     cRowToKeep = int((cRowToKeep + self._batchSize/2) // self._batchSize) *self._batchSize
            
        # idxHolds = np.where(AD[1] ==0)[0].tolist()
        # cHoldsToDel = len(idxHolds) - (cRowToKeep - sum(kI[1:]))
        # if cHoldsToDel>0 :
        #     random.shuffle(idxHolds)
        #     del idxHolds[cHoldsToDel:]
        #     frameDict[self._colnameClasses] = np.delete(frameDict[self._colnameClasses], idxHolds, axis=0)
        #     frameDict[self._colnameSamples]  = np.delete(frameDict[self._colnameSamples],  idxHolds, axis=0)

        AD = np.where(chunk_Classes >=0.99) # to match 1 because action is float read from RFrames
        kI = [np.count_nonzero(AD[1] ==i) for i in range(self._sampleClassSize)] # counts of each actions in frame
        kImax = max(kI)
        idxMax = kI.index(kImax)
        cToReduce = kImax - int(1.6*(sum(kI) -kImax))
        if cToReduce >0:
            idxItems = np.where(AD[1] ==idxMax)[0].tolist()
            random.shuffle(idxItems)
            del idxItems[cToReduce:]
            idxToDel = [int(i) for i in idxItems]
            frameDict[self._colnameClasses] = np.delete(frameDict[self._colnameClasses], idxToDel, axis=0)
            frameDict[self._colnameSamples] = np.delete(frameDict[self._colnameSamples], idxToDel, axis=0)

        return len(frameDict[self._colnameClasses])

    def __populateSampleFileList(self) :
        fileList = copy.copy(self._sampleFiles)
        if len(fileList) <=0 and self._dirSamples and len(self._dirSamples) >0:
            fileList =[]
            files = listAllFiles(self._dirSamples)
            for name in files:
                if self._preBalanced :
                    if '.h5b' != name[-4:] : continue
                elif '.h5' != name[-3:] : 
                    continue

                fileList.append(name)

        fileList.sort()
        self.info('refreshed sample files: %s' % fileList)
        return fileList

    def __nextFrameName(self, bPop1stFrameName=False):
        '''
        get the next (h5fileName, frameName, framesAwait) to read from the H5 file
        '''
        h5fileName, nextFrameName, awaitSize = None, None, 0

        with self.__lock:
            if not self._frameSeq or len(self._frameSeq) <=0:
                self._frameSeq =[]
                fileList = self.__populateSampleFileList()

                for h5fileName in fileList :
                    framesInHd5 = []
                    try:
                        self.debug('loading sample file %s' % h5fileName)
                        with h5py.File(h5fileName, 'r') as h5f:
                            framesInHd5 = []
                            for name in h5f.keys() :
                                if RFGROUP_PREFIX == name[:len(RFGROUP_PREFIX)] or RFGROUP_PREFIX2 == name[:len(RFGROUP_PREFIX2)] :
                                    framesInHd5.append(name)

                            # I'd like to skip frame-0 as it most-likly includes many zero-samples
                            if not self._preBalanced and len(framesInHd5)>3:
                                del framesInHd5[0:3] # 3frames is about 4mon
                                # del framesInHd5[-1]
                            
                            if len(framesInHd5)>6:
                                del framesInHd5[0]

                            if len(framesInHd5) <=1:
                                fileList.remove(h5fileName)
                                self.error('file %s eliminated as too few ReplayFrames in it' % (h5fileName) )
                                continue

                            f1st = framesInHd5[0]
                            frm  = h5f[f1st]
                            frameSize      = frm[self._colnameClasses].shape[0]
                            shape_samples  = frm[self._colnameSamples].shape[1:]
                            shape_classes  = frm[self._colnameClasses].shape[1:]
                            signature =  frm.attrs['signature'] if 'signature' in frm.attrs.keys() else 'n/a'

                            if not self._shapeSamples:     self._shapeSamples = shape_samples
                            if not self._sampleClassSize:  self._sampleClassSize = shape_classes[0]

                            if self._shapeSamples != shape_samples or self._sampleClassSize != shape_classes[0]:
                                fileList.remove(h5fileName)
                                self.error('file %s eliminated per mismatched shapes: samples%s %s-classes against known samples[%s:%s] classes[%s:%s]' % (h5fileName, shape_samples, shape_classes[0], self._colnameSamples, self._shapeSamples, self._colnameClasses, self._sampleClassSize) )
                                continue

                            if self._frameSize < frameSize:
                                self._frameSize = frameSize

                            self.info('%d ReplayFrames found in %s with signature[%s] shapes: samples[%s:%s] classes[%s:%s]' % (len(framesInHd5), h5fileName, signature, self._colnameSamples, self._shapeSamples, self._colnameClasses, self._sampleClassSize) )

                    except Exception as ex:
                        fileList.remove(h5fileName)
                        self.error('file %s elimited per IO exception: %s' % (h5fileName, str(ex)) )
                        continue

                    for i in range(max(1, 1+self._repeatsInFile)):
                        seq = [(h5fileName, frmName) for frmName in framesInHd5]
                        # random.shuffle(seq)
                        self._frameSeq += seq

                random.shuffle(self._frameSeq)
                self.info('frame sequence rebuilt: %s frames from %s replay files, %.2f%%ov%s took %s/round' % (len(self._frameSeq), len(fileList), self.__totalAccu*100.0/(1+self.__totalEval), self.__totalSamples, str(datetime.now() - self.__stampRound)) )
                self.__totalAccu, self.__totalEval, self.__totalSamples, self.__stampRound = 0.0, 0, 0, datetime.now()

            if len(self._frameSeq) >0:
                h5fileName, nextFrameName = self._frameSeq[0]
                if bPop1stFrameName: del self._frameSeq[0]

            awaitSize = len(self._frameSeq)

        return h5fileName, nextFrameName, awaitSize

    def readFrame(self, h5fileName, frameName):
        '''
        read a frame from H5 file
        '''
        COLS = [self._colnameSamples, self._colnameClasses]
        frameDict ={}
        try :
            # reading the frame from the h5
            self.debug('readAhead() reading %s of %s' % (frameName, h5fileName))
            with h5py.File(h5fileName, 'r', **H5_OPEN_ARGS) as h5f:
                frame = h5f[frameName] # h5f[RFGROUP_PREFIX + frameName]

                for col in COLS :
                    if col in frameDict.keys():
                        frameDict[col] += list(frame[col])
                    else : frameDict[col] = list(frame[col])
        except Exception as ex:
            self.logexception(ex)

        return frameDict

    # def __readAhead(self, thrdSeqId=0):
    #     '''
    #     the background thread to read A frame from H5 file
    #     reading H5 only works on CPU and is quite slow, so take a seperate thread to read-ahead
    #     '''
    #     stampStart = datetime.now()
    #     frameDict = None

    #     while True:
    #         try :
    #             h5fileName, nextFrameName, awaitSize = self.__nextFrameName(True)
    #             if not nextFrameName or len(nextFrameName) <=0:
    #                 self.error('readAhead(%s) failed to get nextFrameName' % (thrdSeqId) )
    #                 return

    #             frameDict = self.readFrame(h5fileName, nextFrameName)
    #             if frameDict and len(frameDict) >0: break
    #         except Exception as ex:
    #             thePROG.logexception(ex, 'ssh failed')
        
    #     lenFrame= 0
    #     for v in frameDict.values() :
    #         lenFrame = len(v)
    #         break

    #     self.debug('readAhead(%s) read %s samples from %s@%s' % (thrdSeqId, lenFrame, nextFrameName, h5fileName) )
    #     cvnted = frameDict
    #     try :
    #         if self._funcConvertFrame :
    #             cvnted = self._funcConvertFrame(frameDict)
    #             self.debug('readAhead(%s) converted %s samples of %s@%s into %s chunks' % (thrdSeqId, lenFrame, nextFrameName, h5fileName, len(cvnted)) )
    #     except Exception as ex:
    #         self.logexception(ex)

    #     addSize, raSize=0, 0
    #     with self.__lock:
    #         self.__thrdsReadAhead[thrdSeqId] = None

    #         if isinstance(cvnted, list) :
    #             self.__chunksReadAhead += cvnted
    #             addSize, raSize = len(cvnted), len(self.__chunksReadAhead)
    #         else:
    #             self.__chunksReadAhead.append(cvnted)
    #             addSize, raSize = 1, len(self.__chunksReadAhead)

    #     frameDict, cvnted = None, None
    #     self.info('readAhead(%s) prepared %s->%s x%s s/bth from %s took %s, %d frames await' % 
    #         (thrdSeqId, addSize, raSize, self._batchSize, nextFrameName, str(datetime.now() - stampStart), awaitSize))

    def __readAheadChunks(self, thrdSeqId=0, cFramesToRead=1):
        '''
        the background thread to read a number of frames from H5 files
        reading H5 only works on CPU and is quite slow, so take a seperate thread to read-ahead
        '''
        stampStart = datetime.now()
        strFrames =[]
        awaitSize =-1
        addSize, raSize=0, 0

        self.debug('readAheadChunks(%s) reading samples for %d frames x %ds/chunk' % (thrdSeqId, cFramesToRead, self._batchSize) )
        
        # because h5py only can read one file at same time, sort the frames-to-read by grouping them by filenames
        groupOfFrames = {} # dict filename to framenames
        for i in range(cFramesToRead):
            h5fileName, nextFrameName, awaitSize = self.__nextFrameName(True)
            if h5fileName not in groupOfFrames:
                groupOfFrames[h5fileName] =[]
            groupOfFrames[h5fileName].append(nextFrameName)

        frameSeqToRead = []
        for k, v in groupOfFrames.items():
            v.sort()
            frameSeqToRead += [(k, i) for i in v]

        for h5fileName, nextFrameName in frameSeqToRead :
            frameDict = self.readFrame(h5fileName, nextFrameName)

            lenFrame= 0
            for v in frameDict.values() :
                lenFrame = len(v)
                break

            stampRead = datetime.now()
            self.debug('readAheadChunks(%s) read %s samples from %s@%s, took %s' % (thrdSeqId, lenFrame, nextFrameName, h5fileName, stampRead-stampStart) )
            strFrames.append('%s@%s' % (nextFrameName, os.path.basename(h5fileName)))
            # unnecssary: the following steps of filtering and converting should go to a separated thread to employ CPU cores
            # because readAheadChunks(-1) took 0:02:00.194678 (0:02:00.178583 +0:00:00.000999 +0:00:00.015096) to prepare 65->65 x400 s/bth from 19 frames: ...
            cvnted = frameDict
            nAfterFilter = lenFrame
            try :
                if self._funcFilterFrame :
                    nAfterFilter = self._funcFilterFrame(frameDict)
            except Exception as ex:
                self.logexception(ex)

            stampFiltered = datetime.now()
            try :
                if self._funcConvertFrame :
                    cvnted = self._funcConvertFrame(frameDict)
            except Exception as ex:
                self.logexception(ex)

            stampProcessed = datetime.now()
            self.debug('readAheadChunks(%s) filtered %s from %s samples and converted into %s chunks, took %s + %s' % (thrdSeqId, nAfterFilter, lenFrame, len(cvnted), stampFiltered -stampRead, stampProcessed -stampFiltered) )

            with self.__lock:
                size =1
                if isinstance(cvnted, list) :
                    self.__chunksReadAhead += cvnted
                    size = len(cvnted)
                    cFramesToRead -= size
                else:
                    self.__chunksReadAhead.append(cvnted)
                    cFramesToRead -= 1

                addSize += size

            frameDict, cvnted = None, None

        # finally random.shuffle all samples read
        with self.__lock:
            raSize = len(self.__chunksReadAhead)
            self.__framesReadAhead = strFrames
            random.shuffle(self.__chunksReadAhead)

            if thrdSeqId>=0 and thrdSeqId < len(self.__thrdsReadAhead) :
                self.__thrdsReadAhead[thrdSeqId] = None

        self.info('readAheadChunks(%s) took %s (%s +%s +%s) to prepare %s->%s x%s s/bth from %d frames:%s; %d frames await' % 
            (thrdSeqId, str(datetime.now() - stampStart), str(stampRead -stampStart), str(stampFiltered -stampRead), str(stampProcessed -stampFiltered),
            addSize, raSize, self._batchSize, len(strFrames), ','.join(strFrames), awaitSize))

    """
    
    def __readFramesAhead(self, thrdSeqId =0, cFramesToRead=1):
        '''
        the background thread to read a number of frames from H5 files
        reading H5 only works on CPU and is quite slow, so take a seperate thread to read-ahead
        '''
        stampStart = datetime.now()
        strFrames =[]
        awaitSize =-1
        addSize, raSize=0, 0

        self.debug('readFramesAhead(%s) reading samples for %d frames x %ds/chunk' % (thrdSeqId, cFramesToRead, self._batchSize) )
        if cFramesToRead <1:
            cFramesToRead =1
        groupOfFrames = {} # dict filename to framenames
        for i in range(cFramesToRead):
            h5fileName, nextFrameName, awaitSize = self.__nextFrameName(True)
            if h5fileName not in groupOfFrames:
                groupOfFrames[h5fileName] =[]
            groupOfFrames[h5fileName].append(nextFrameName)
        
        self._readAheadThrds = 3
        frameSeqToRead = [ [] for i in range( min(len(groupOfFrames.keys()), self._readAheadThrds) ) ] # DO NOT: [ [] ] * min(len(groupOfFrames.keys()), self._readAheadThrds)
        seqIdx = [x for x in range(len(frameSeqToRead))]

        for k, v in groupOfFrames.items():
            top = seqIdx[0]
            frameSeqToRead[top] += [(k, i) for i in v]
            seqIdx.sort(key=lambda x: len(frameSeqToRead[x]))

        while len(frameSeqToRead) >0 and len(frameSeqToRead[0]) <=0:
            del frameSeqToRead[0]

        lkReadResult = threading.Lock()
        readResult = {}
        threads = [ None ] * len(frameSeqToRead)
        # --------------------------------
        # the sub-thread to read a sequnce of frame
        def thrd_readFrameSeq(thrdId, frameSeq):
            frmNames =['%s@%s' %(r, f) for f, r in frameSeq]
            self.debug('readFramesAhead() subthrd[%s] starts reading %d frames: %s' % (thrdId, len(frmNames), ','.join(frmNames)))
            
            frmNames =[]
            for filename, framename in frameSeq:
                frmName = '%s@%s'% (framename, os.path.basename(filename))
                sampleFrame = None
                try :
                    sampleFrame = self.readFrame(filename, framename)
                except Exception as ex:
                    pass

                if not sampleFrame or len(sampleFrame) <=0:
                    continue
                
                with lkReadResult:
                    readResult[frmName] = sampleFrame
                    self.debug('readFramesAhead() subthrd[%s] committed %s into readResult, result-size %d' % (thrdId, frmName, len(readResult)))
                    frmNames.append(frmName)
            
            # finished, then unregister self from threads:
            with lkReadResult:
                if thrdId>=0:
                    threads[thrdId] = None
                self.info('readFramesAhead() subthrd[%s] done, read %d frames: %s' % (thrdId, len(frmNames), ','.join(frmNames)))

        # kick of up to self._readAheadThrds threads, each to read a sequence of frames by affiniting filenames
        for i in range(len(frameSeqToRead)) :
            thrd = threading.Thread(target=thrd_readFrameSeq, kwargs={'thrdId': i, 'frameSeq': frameSeqToRead[i]} )
            threads[i] =thrd
            thrd.start()

        # wait for all sub-threads get completed
        while True:
            thrdBusy = None
            with lkReadResult:
                for i in range(len(threads)):
                    if threads[i] and threads[i].is_alive():
                        thrdBusy = threads[i]
                        break
                    threads[i] =None
        
            if not thrdBusy: break
                
            self.info('readFramesAhead() waiting for busy-thread[%s]' %thrdBusy )
            thrdBusy.join()
        
        # when reach here, all the sub threads of reading have completed, and frames are collected in readResult
        # concate the readResult into a large chunk frameDict
        stampRead = datetime.now()
        self.info('readFramesAhead() all %d/%d subthrd done, collected %d frames, took %s' % (len(frameSeqToRead), self._readAheadThrds, len(readResult), (stampRead - stampStart)))

        frameDict = {}
        for fn, frm in readResult.items():
            strFrames.append(fn)
            for k, v in frm.items() :
                if k not in frameDict :
                    frameDict[k] = v
                else:
                    frameDict[k] = np.append(frameDict[k], v, axis=0)
        readResult = None

        lenFrame = len(frameDict[self._colnameClasses])
        nAfterFilter = lenFrame
        cvnted = frameDict
        try :
            if self._funcFilterFrame :
                nAfterFilter = self._funcFilterFrame(frameDict)
        except Exception as ex:
            self.logexception(ex)

        stampFiltered = datetime.now()
        try :
            if self._funcConvertFrame :
                cvnted = self._funcConvertFrame(frameDict)
        except Exception as ex:
            self.logexception(ex)

        stampProcessed = datetime.now()
        self.debug('readFramesAhead(%s) filtered %s from %s samples and converted into %s chunks, took %s + %s' % (thrdSeqId, nAfterFilter, lenFrame, len(cvnted), stampFiltered -stampRead, stampProcessed -stampFiltered) )

        with self.__lock:
            size =1
            if isinstance(cvnted, list) :
                self.__chunksReadAhead += cvnted
                size = len(cvnted)
                cFramesToRead -= size
            else:
                self.__chunksReadAhead.append(cvnted)
                cFramesToRead -= 1

            addSize += size

        frameDict, cvnted = None, None

        with self.__lock:
            raSize = len(self.__chunksReadAhead)
            self.__framesReadAhead = strFrames
            random.shuffle(self.__chunksReadAhead)

            if thrdSeqId>=0 and thrdSeqId < len(self.__thrdsReadAhead) :
                self.__thrdsReadAhead[thrdSeqId] = None

        self.info('readFramesAhead(%s) took %s (%s +%s +%s) to prepare %s->%s x%s s/bth from %d frames:%s; %d frames await' % 
            (thrdSeqId, str(datetime.now() - stampStart), str(stampRead -stampStart), str(stampFiltered -stampRead), str(stampProcessed -stampFiltered),
            addSize, raSize, self._batchSize, len(strFrames), ','.join(strFrames), awaitSize))

    """

    def __generator_local(self):

        # DONOT self._funcConvertFrame = self.__frameToBatchs

        # build up self.__samplePool
        self.__samplePool = {
            self._colnameSamples: [],
            self._colnameClasses: [],
        }

        trainId, itrId = 0, 0
        samplePerFrame =0
        trainSize = self._batchesPerTrain*self._batchSize

        loss = DUMMY_BIG_VAL
        lossMax = loss
        idxBatchInPool =int(DUMMY_BIG_VAL)
        skippedSaves =0
        while True : #TODO temporarily loop for ever: lossMax > self._lossStop or abs(loss-lossMax) > (lossMax * self._lossDiffStop/100) :

            bths_Samples, bths_Classes =[], []
            cFresh, cRecycled = 0, 0
            while len(bths_Samples) < self._batchesPerTrain :
                bth, recycled = self.nextDataChunk() #= self.readDataChunk(idxBatchInPool)
                if recycled:
                    cRecycled += 1
                else:
                    cFresh += 1

                bths_Samples.append(bth[self._colnameSamples])
                bths_Classes.append(bth[self._colnameClasses])

            #----------------------------------------------------------
            # continue # if only test read-ahead and pool making-up   #
            #----------------------------------------------------------

            cBths = len(bths_Samples)
            if cBths < self._batchesPerTrain:
                continue

            chunk_Samples = np.concatenate(tuple(bths_Samples))
            chunk_Classes = np.concatenate(tuple(bths_Classes))
            bths_Samples, bths_Classes =[], []
            trainId +=1

            trainSize = chunk_Samples.shape[0]
            self.__totalSamples += trainSize
            
            stampStart = datetime.now()
            result, lstEpochs, histEpochs = None, [], []
            strEval =''
            loss = max(11, loss)
            sampledAhead = cFresh >0 and (cFresh > cRecycled/4 or skippedSaves >10)
            epochs = self._initEpochs if sampledAhead else 2
            while epochs > 0:
                if self._evaluateSamples and len(strEval) <=0 and sampledAhead and (self._evalAt <=0 or self.__totalEval<=0 or 1 == (trainId % self._evalAt)):
                    try :
                        # eval.1 eval on the samples
                        resEval =  self._brain.evaluate(x=chunk_Samples, y=chunk_Classes, batch_size=self._batchSize, verbose=1) #, callbacks=self._fitCallbacks)
                        strEval += 'from eval[%.2f%%^%.3f]' % (resEval[1]*100, resEval[0])
                        self.__totalAccu += trainSize * resEval[1]
                        self.__totalEval += trainSize

                        # eval.2 action distrib in samples/prediction
                        AD = np.where(chunk_Classes ==1)[1]
                        kI = ['%.2f' % (np.count_nonzero(AD ==i)*100.0/len(AD)) for i in range(self._sampleClassSize)] # the actions percentage in sample
                        predict = self._brain.predict(x=chunk_Samples)
                        predact = np.zeros(len(predict) * self._sampleClassSize).reshape(len(predict), self._sampleClassSize)
                        for r in range(len(predict)):
                            predact[r][np.argmax(predict[r])] =1
                        AD = np.where(predact ==1)[1]
                        kP = ['%.2f' % (np.count_nonzero(AD ==i)*100.0/len(AD)) for i in range(self._sampleClassSize)] # the actions percentage in predictions
                        strEval += 'A%s%%->Prd%s%%' % ('+'.join(kI), '+'.join(kP))
                        
                        # eval.3 duration taken
                        strEval += '/%s, ' % (datetime.now() -stampStart)
                    except Exception as ex:
                        self.logexception(ex)

                # call trainMethod to perform tranning
                itrId +=1
                try :
                    epochs2run = epochs
                    epochs =0
                    result = self._brain.fit(x=chunk_Samples, y=chunk_Classes, epochs=epochs2run, shuffle=True, batch_size=self._batchSize, verbose=1, callbacks=self._fitCallbacks)
                    lstEpochs.append(epochs2run)
                    loss = result.history["loss"][-1]
                    lossImprove =0.0
                    if len(result.history["loss"]) >1 :
                        lossImprove = result.history["loss"][-2] - loss

                    if sampledAhead and loss > self._lossStop and lossImprove > (loss * self._lossDiffStop/100) :
                        epochs = epochs2run
                        if lossImprove > (loss * self._lossDiffStop *2 /100) :
                            epochs += int(epochs2run/2)

                    if lossMax>=DUMMY_BIG_VAL-1 or lossMax < loss: lossMax = loss
                    histEpochs += self.__resultToStepHist(result)

                    yield result # this is a step

                except Exception as ex:
                    self.logexception(ex)

            if len(histEpochs) <=0:
                continue

            strEpochs = '+'.join([str(i) for i in lstEpochs])
            if sampledAhead:
                self.__logAndSaveResult(histEpochs[-1], 'doAppStep_local_generator', '%s%s/%s steps x%s epochs on %dN+%dR samples %.2f%%ov%s took %s, hist: %s' % (strEval, trainSize, self._batchSize, strEpochs, cFresh, cRecycled, self.__totalAccu*100.0/(1+self.__totalEval), self.__totalSamples, (datetime.now() -stampStart), ', '.join(histEpochs)) )
                skippedSaves =0
            else :
                self.info('doAppStep_local_generator() %s epochs on recycled %dN+%dR samples took %s, hist: %s' % (strEpochs, cFresh, cRecycled, (datetime.now() -stampStart), ', '.join(histEpochs)) )
                skippedSaves +=1
    
########################################################################
class Trainer_GainRates(Trainer_classify) :

    def __init__(self, program, **kwargs):
        super(Trainer_GainRates, self).__init__(program, **kwargs)

        self._colnameSamples      = 'state'
        self._colnameClasses      = 'gain_rates'

        #TODO: self._confXXXX     = self.getConfig('XXXX', None)

        self._funcConvertFrame = self.classifyTplus1GainRateOfFrame
        self._funcFilterFrame  = None

    def classifyTplus1GainRateOfFrame(self, frameDict):

        gainClasses = classifyGainRates_screeningTplus1(frameDict[self._colnameClasses])
        samples = np.array(frameDict[self._colnameSamples]).astype(SAMPLE_FLOAT)
        bths = []
        cBth = gainClasses.shape[0] // self._batchSize
        for i in range(cBth):
            batch = {
                self._colnameSamples : samples[i*self._batchSize : (i+1) *self._batchSize,],
                self._colnameClasses : gainClasses[i*self._batchSize : (i+1) *self._batchSize,]
            }

            bths.append(batch)

        return bths

########################################################################
if __name__ == '__main__':

    exportNonTrainable = False
    # sys.argv.append('-x')
    if '-x' in sys.argv :
        exportNonTrainable = True
        sys.argv.remove('-x')

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Trainer.json']

    SYMBOL = '000001' # '000540' '000001'
    sourceCsvDir = None

    p = Program()
    p._heartbeatInterval =-1

    try:
        jsetting = p.jsettings('train/sourceCsvDir')
        if not jsetting is None:
            sourceCsvDir = jsetting(None)

        jsetting = p.jsettings('train/objectives')
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

    p.info('all objects registered piror to Trainer_classify: %s' % p.listByType())
    
    # trainer = p.createApp(Trainer_classify, configNode ='Trainer_classify', replayFrameFiles=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))
    trainer = p.createApp(Trainer_classify, configNode ='train') # Trainer_GainRates

    p.start()

    if exportNonTrainable :
        trainer.exportLayerWeights()
        quit()

    p.loop()
    p.info('loop done, all objs: %s' % p.listByType())
    p.stop()
