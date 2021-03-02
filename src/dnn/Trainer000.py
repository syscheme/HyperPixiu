########################################################################
class Trainer(BaseApplication):

    DEFAULT_MODEL = 'Cnn1Dx4R2'
    COMPILE_ARGS ={
    'loss':'categorical_crossentropy', 
    # 'optimizer': sgd,
    'metrics':['accuracy']
    }
    
    def __init__(self, program, replayFrameFiles=None, model_json=None, initWeights= None, recorder =None, **kwargs):
        super(Trainer, self).__init__(program, **kwargs)

        self._wkModelId      = self.getConfig('brainId', Trainer.DEFAULT_MODEL)

        self._model_json = model_json
        self._replayFrameFiles =replayFrameFiles

        if not self._replayFrameFiles or len(self._replayFrameFiles) <=0: 
            self._replayFrameFiles = self.getConfig('replayFrameFiles', [])
            self._replayFrameFiles = [ Program.fixupPath(f) for f in self._replayFrameFiles ]

        self._stepMethod          = self.getConfig('stepMethod', None)
        self._repeatsInFile       = self.getConfig('repeatsInFile', 0)
        self._exportTB            = self.getConfig('tensorBoard', 'no').lower() in BOOL_STRVAL_TRUE
        self._batchSize           = self.getConfig('batchSize', 128)
        self._batchesPerTrain     = self.getConfig('batchesPerTrain', 8)
        self._recycleSize         = self.getConfig('recycles', 1)
        self._initEpochs          = self.getConfig('initEpochs', 2)
        self._lossStop            = self.getConfig('lossStop', 0.24) # 0.24 according to average loss value by： grep 'from eval' /mnt/d/tmp/replayTrain_14276_0106.log |sed 's/.*loss\[\([^]]*\)\].*/\1/g' | awk '{ total += $1; count++ } END { print total/count }'
        self._lossPctStop         = self.getConfig('lossPctStop', 5)
        self._startLR             = self.getConfig('startLR', 0.01)
        self._evaluateSamples     = self.getConfig('evaluateSamples', 'yes').lower() in BOOL_STRVAL_TRUE
        self._preBalanced         = self.getConfig('preBalanced',      'no').lower() in BOOL_STRVAL_TRUE
        self._evalAt              = self.getConfig('evalAt', 5) # how often on trains to perform evaluation

        # self._nonTrainables       = self.getConfig('nonTrainables',  ['VClz512to20.1of2', 'VClz512to20.2of2']) # non-trainable layers
        # self._nonTrainables       = [x('') for x in self._nonTrainables] # convert to string list
        self._nonTrainables = ['VClz66from512.1of2', 'VClz66from512.2of2']

        # self._poolEvictRate       = self.getConfig('poolEvictRate', 0.5)
        # if self._poolEvictRate>1 or self._poolEvictRate<=0:
        #     self._poolEvictRate =1

        if len(GPUs) > 0 : # adjust some configurations if currently running on GPUs
            self.info('GPUs: %s' % GPUs)
            self._stepMethod      = self.getConfig('GPU/stepMethod', self._stepMethod)
            self._exportTB        = self.getConfig('GPU/tensorBoard', 'no').lower() in BOOL_STRVAL_TRUE
            self._batchSize       = self.getConfig('GPU/batchSize',    self._batchSize)
            self._batchesPerTrain = self.getConfig('GPU/batchesPerTrain', 64)  # usually 64 is good for a bottom-line model of GTX1050oc/2G
            self._initEpochs      = self.getConfig('GPU/initEpochs', self._initEpochs)
            self._recycleSize     = self.getConfig('GPU/recycles',   self._recycleSize)
            self._startLR         = self.getConfig('GPU/startLR',      self._startLR)

            self._models          = self.getConfig('GPU/models',   [])
            gpuType = GPUs[0]['detail'] # TODO: only take the first at the moment
            for m in self._models:
                if not m or not 'model' in m.keys() or not m['model'] in gpuType: continue
                if 'batchSize' in m.keys(): self._batchSize = m['batchSize']
                if 'batchesPerTrain' in m.keys(): self._batchesPerTrain = m['batchesPerTrain']

        if not self._replayFrameFiles or len(self._replayFrameFiles) <=0: 
            self._replayFrameFiles =[]
            replayFrameDir = self.getConfig('replayFrameDir', None)
            if replayFrameDir:
                replayFrameDir = Program.fixupPath(replayFrameDir)
                try :
                    for rootdir, subdirs, files in os.walk(replayFrameDir, topdown=False):
                        for name in files:
                            if self._preBalanced :
                                if '.h5b' != name[-4:] : continue
                            elif '.h5' != name[-3:] : 
                                continue

                            self._replayFrameFiles.append(os.path.join(rootdir, name))
                except:
                    pass

        self.__samplePool = [] # may consist of a number of replay-frames (n < frames-of-h5) for random sampling
        self._fitCallbacks =[]
        self._frameSeq =[]

        self._evalAt =int(self._evalAt)
        self._stateSize, self._actionSize, self._frameSize = None, None, 0
        self._brain = None
        self._outDir = os.path.join(self.dataRoot, '%s/P%s/' % (self.program.baseName, self.program.pid))
        self.__lock = threading.Lock()
        self.__thrdsReadAhead = []
        self.__chunksReadAhead = []
        self.__newChunks =[]
        self.__recycledChunks =[]
        self.__convertFrame = self.__frameToBatchs
        self.__filterFrame  = None if self._preBalanced else self.__balanceSamples

        self.__latestBthNo=0
        self.__totalAccu, self.__totalEval, self.__totalSamples, self.__stampRound = 0.0, 0, 0, datetime.now()

        self.__knownModels_1D = {
            'VGG16d1'    : self.__createModel_VGG16d1,
            'Cnn1Dx4R2'  : self.__createModel_Cnn1Dx4R2,
            'Cnn1Dx4R3'  : self.__createModel_Cnn1Dx4R3,
            'ResNet18d1' : self.__createModel_ResNet18d1,
            'ResNet2Xd1' : self.__createModel_ResNet2Xd1,
            'ResNet2xR1' : self.__createModel_ResNet2xR1,
            'ResNet21'   : self.__createModel_ResNet21,
            'ResNet21R1' : self.__createModel_ResNet21R1,
            'ResNet34d1' : self.__createModel_ResNet34d1,
            'ResNet50d1' : self.__createModel_ResNet50d1,
            }

        self.__knownModels_2D = {
            'ResNet50d2Ext1' : self.__createModel_ResNet50d2Ext1,
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
        if not super(Trainer, self).doAppInit() :
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
                self._brain.compile(optimizer=sgd, **Trainer.COMPILE_ARGS)

                self._wkModelId = wkModelId

                fn_weights = os.path.join(inDir, 'weights.h5')
                self.debug('loading saved weights from %s' %fn_weights)
                self._brain.load_weights(fn_weights)
                self.info('loaded model and weights from %s' %inDir)

                fn_weights = os.path.join(inDir, 'nonTrainables.h5')
                try :
                    if os.stat(fn_weights):
                        self.debug('importing weights of layers[%s] from file %s' % (','.join(self._nonTrainables), fn_weights))
                        lns = importLayerWeights(self._brain, fn_weights, self._nonTrainables)
                        if len(lns) >0:
                            sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
                            self._brain.compile(optimizer=sgd, **Trainer.COMPILE_ARGS)
                            self.info('imported non-trainable weights of layers[%s] from file %s' % (','.join(lns), fn_weights))
                except Exception as ex:
                    self.logexception(ex)

            except Exception as ex:
                self.logexception(ex)

        #TESTCODE: 
        # self.createModel('ResNet50d2Ext1', knownModels = self.__knownModels_2D)

        if not self._brain:
            self._brain, self._wkModelId = self.createModel(self._wkModelId, knownModels = self.__knownModels_2D) # = self.createModel(self._wkModelId)
            self._wkModelId += '.S%sI%sA%s' % (self._stateSize, EXPORT_FLOATS_DIMS, self._actionSize)

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
        return super(Trainer, self).doAppStep()

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

        cFrames=0
        with self.__lock:
            if self._frameSize >0:
                cFrames = (self.__maxChunks * self._batchesPerTrain * self._batchSize) // self._frameSize
            if cFrames<=0: cFrames =1
            cFrames = int(cFrames)

            self.__thrdsReadAhead = [None] * cFrames

        if not ret and not self.__chunksReadAhead or len(self.__chunksReadAhead) <=0:
            self.warn('nextDataChunk() no readAhead ready, force to read sync-ly')
            self.__readAheadChunks(thrdSeqId=-1, cChunks=self._batchesPerTrain) # cChunks=cFrames)

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

            thrd = threading.Thread(target=self.__readAheadChunks, kwargs={'thrdSeqId': 0, 'cChunks': self._batchesPerTrain } ) # kwargs={'thrdSeqId': 0, 'cChunks': cChunks } )
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
        
        # to shuffle within the frame
        shuffledIndx =[i for i in range(framelen)]
        random.shuffle(shuffledIndx)

        bths = []
        cBth = framelen // self._batchSize
        for i in range(cBth):
            batch = {}
            for col in COLS :
                # batch[col] = np.array(frameDict[col][self._batchSize*i: self._batchSize*(i+1)]).astype(NN_FLOAT)
                batch[col] = np.array([frameDict[col][j] for j in shuffledIndx[self._batchSize*i: self._batchSize*(i+1)]]).astype(NN_FLOAT)
            
            bths.append(batch)

        return bths

    def __balanceSamples(self, frameDict) :
        '''
            balance the samples, usually reduce some action=HOLD, which appears too many
        '''
        actionchunk = np.array(frameDict['action'])
        # AD = np.where(actionchunk >=0.99) # to match 1 because action is float read from RFrames
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
        #     frameDict['action'] = np.delete(frameDict['action'], idxHolds, axis=0)
        #     frameDict['state']  = np.delete(frameDict['state'],  idxHolds, axis=0)

        AD = np.where(actionchunk >=0.99) # to match 1 because action is float read from RFrames
        kI = [np.count_nonzero(AD[1] ==i) for i in range(3)] # counts of each actions in frame
        kImax = max(kI)
        idxMax = kI.index(kImax)
        cToReduce = kImax - int(1.6*(sum(kI) -kImax))
        if cToReduce >0:
            idxItems = np.where(AD[1] ==idxMax)[0].tolist()
            random.shuffle(idxItems)
            del idxItems[cToReduce:]
            idxToDel = [lenBefore +i for i in idxItems]
            frameDict['action'] = np.delete(frameDict['action'], idxToDel, axis=0)
            frameDict['state']  = np.delete(frameDict['state'], idxToDel, axis=0)

        return len(frameDict['action'])

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
                                if RFGROUP_PREFIX == name[:len(RFGROUP_PREFIX)] or RFGROUP_PREFIX2 == name[:len(RFGROUP_PREFIX2)] :
                                    framesInHd5.append(name)

                            # I'd like to skip frame-0 as it most-likly includes many zero-samples
                            if not self._preBalanced and len(framesInHd5)>3:
                                del framesInHd5[0:3] # 3frames is about 4mon
                                # del framesInHd5[-1]
                            
                            if len(framesInHd5)>6:
                                del framesInHd5[0]

                            if len(framesInHd5) <=1:
                                self._replayFrameFiles.remove(h5fileName)
                                self.error('file %s eliminated as too few ReplayFrames in it' % (h5fileName) )
                                continue

                            f1st = framesInHd5[0]
                            frm  = h5f[f1st]
                            frameSize  = frm['state'].shape[0]
                            stateSize  = frm['state'].shape[1]
                            actionSize = frm['action'].shape[1]
                            signature =  frm.attrs['signature'] if 'signature' in frm.attrs.keys() else 'n/a'

                            if self._stateSize and self._stateSize != stateSize or self._actionSize and self._actionSize != actionSize:
                                self._replayFrameFiles.remove(h5fileName)
                                self.error('file %s eliminated as its dims: %s/state %s/action mismatch working dims %s/state %s/action' % (h5fileName, stateSize, actionSize, self._stateSize, self._actionSize) )
                                continue

                            if self._frameSize < frameSize:
                                self._frameSize = frameSize
                            self._stateSize = stateSize
                            self._actionSize = actionSize

                            self.info('%d ReplayFrames found in %s with signature[%s] dims: %s/state, %s/action' % (len(framesInHd5), h5fileName, signature, self._stateSize, self._actionSize) )

                    except Exception as ex:
                        self._replayFrameFiles.remove(h5fileName)
                        self.error('file %s elimited per IO exception: %s' % (h5fileName, str(ex)) )
                        continue

                    for i in range(max(1, 1+self._repeatsInFile)):
                        seq = [(h5fileName, frmName) for frmName in framesInHd5]
                        # random.shuffle(seq)
                        self._frameSeq += seq

                random.shuffle(self._frameSeq)
                self.info('frame sequence rebuilt: %s frames from %s replay files, %.2f%%ov%s took %s/round' % (len(self._frameSeq), len(self._replayFrameFiles), self.__totalAccu*100.0/(1+self.__totalEval), self.__totalSamples, str(datetime.now() - self.__stampRound)) )
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
        COLS = ['state','action']
        frameDict ={}
        try :
            # reading the frame from the h5
            self.debug('readAhead() reading %s of %s' % (frameName, h5fileName))
            with h5py.File(h5fileName, 'r') as h5f:
                frame = h5f[frameName] # h5f[RFGROUP_PREFIX + frameName]

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

        self.debug('readAheadChunks(%s) reading samples for %d chunks x %ds/chunk' % (thrdSeqId, cChunks, self._batchSize) )

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
            nAfterFilter = lenFrame
            try :
                if self.__filterFrame :
                    nAfterFilter = self.__filterFrame(frameDict)
            except Exception as ex:
                self.logexception(ex)

            try :
                if self.__convertFrame :
                    cvnted = self.__convertFrame(frameDict)
            except Exception as ex:
                self.logexception(ex)

            self.debug('readAheadChunks(%s) filtered %s from %s samples and converted into %s chunks' % (thrdSeqId, nAfterFilter, lenFrame, len(cvnted)) )

            with self.__lock:
                size =1
                if isinstance(cvnted, list) :
                    self.__chunksReadAhead += cvnted
                    size = len(cvnted)
                    cChunks -= size
                else:
                    self.__chunksReadAhead.append(cvnted)
                    cChunks -= 1

                addSize += size

            frameDict, cvnted = None, None

        with self.__lock:
            raSize = len(self.__chunksReadAhead)
            self.__framesReadAhead = strFrames
            random.shuffle(self.__chunksReadAhead)

            if thrdSeqId>=0 and thrdSeqId < len(self.__thrdsReadAhead) :
                self.__thrdsReadAhead[thrdSeqId] = None

        self.info('readAheadChunks(%s) took %s to prepare %s->%s x%s s/bth from %d frames:%s; %d frames await' % 
            (thrdSeqId, str(datetime.now() - stampStart), addSize, raSize, self._batchSize, len(strFrames), ','.join(strFrames), awaitSize))

    def __generator_local(self):

        self.__convertFrame = self.__frameToBatchs

        # build up self.__samplePool
        self.__samplePool = {
            'state':[],
            'action':[],
        }

        trainId, itrId = 0, 0
        samplePerFrame =0
        trainSize = self._batchesPerTrain*self._batchSize

        loss = DUMMY_BIG_VAL
        lossMax = loss
        idxBatchInPool =int(DUMMY_BIG_VAL)
        skippedSaves =0
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
            trainId +=1

            trainSize = statechunk.shape[0]
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
                        resEval =  self._brain.evaluate(x=statechunk, y=actionchunk, batch_size=self._batchSize, verbose=1) #, callbacks=self._fitCallbacks)
                        strEval += 'from eval[%.2f%%^%.3f]' % (resEval[1]*100, resEval[0])
                        self.__totalAccu += trainSize * resEval[1]
                        self.__totalEval += trainSize

                        # eval.2 action distrib in samples/prediction
                        AD = np.where(actionchunk ==1)[1]
                        kI = ['%.2f' % (np.count_nonzero(AD ==i)*100.0/len(AD)) for i in range(3)] # the actions percentage in sample
                        predict = self._brain.predict(x=statechunk)
                        predact = np.zeros(len(predict) *3).reshape(len(predict), 3)
                        for r in range(len(predict)):
                            predact[r][np.argmax(predict[r])] =1
                        AD = np.where(predact ==1)[1]
                        kP = ['%.2f' % (np.count_nonzero(AD ==i)*100.0/len(AD)) for i in range(3)] # the actions percentage in predictions
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

            if len(histEpochs) <=0:
                continue

            strEpochs = '+'.join([str(i) for i in lstEpochs])
            if sampledAhead:
                self.__logAndSaveResult(histEpochs[-1], 'doAppStep_local_generator', '%s%s/%s steps x%s epochs on %dN+%dR samples %.2f%%ov%s took %s, hist: %s' % (strEval, trainSize, self._batchSize, strEpochs, cFresh, cRecycled, self.__totalAccu*100.0/(1+self.__totalEval), self.__totalSamples, (datetime.now() -stampStart), ', '.join(histEpochs)) )
                skippedSaves =0
            else :
                self.info('doAppStep_local_generator() %s epochs on recycled %dN+%dR samples took %s, hist: %s' % (strEpochs, cFresh, cRecycled, (datetime.now() -stampStart), ', '.join(histEpochs)) )
                skippedSaves +=1
    
    #----------------------------------------------------------------------
    def createModel(self, modelId, knownModels=None):
        if not knownModels:
            knownModels = self.__knownModels_1D

        if not modelId in knownModels.keys():
            self.warn('unknown modelId[%s], taking % instead' % (modelId, Trainer.DEFAULT_MODEL))
            modelId = Trainer.DEFAULT_MODEL


        if len(GPUs) <= 1:
            return knownModels[modelId](), modelId

        with tf.device("/cpu:0"):
            return knownModels[modelId](), modelId

    def exportLayerWeights(self):
        h5fileName = os.path.join(self._outDir, '%s.nonTrainables.h5'% self._wkModelId)
        self.debug('exporting weights of layers[%s] into file %s' % (','.join(self._nonTrainables), h5fileName))
        lns = exportLayerWeights(self._brain, h5fileName, self._nonTrainables)
        self.info('exported weights of layers[%s] into file %s' % (','.join(lns), h5fileName))

    #----------------------------------------------------------------------
    # pretrained 2D models
    # https://tensorflow.google.cn/api_docs/python/tf/keras/applications/ResNet50?hl=zh-cn
    def __createModel_ResNet50d2Ext1(self):
        
        # pretrained = ResNet50(weights='imagenet', classes=1000)
        # may lead to URL fetch failure on https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5: None -- [Errno 11] Resource temporarily unavailable
        # take the pre-downloaded offline-weights
        pretrained = ResNet50(weights=None, classes=1000, input_shape=(32, 32, 3))
        pretrained.load_weights(Program.fixupPath('/mnt/e/AShareSample/resnet50_weights_tf_dim_ordering_tf_kernels.h5'))

        pretrained.trainable = False # freeze those pretrained weights
        
        tuples = self._stateSize/EXPORT_FLOATS_DIMS
        model = Sequential()
        #TODO model.add(Reshape((int(tuples), EXPORT_FLOATS_DIMS), input_shape=(self._stateSize,)))
        model.add(pretrained)
        model.add(Flatten())
        model.add(Dense(518))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # unified final layers Dense(VClz512to20) then Dense(self._actionSize)
        model.add(Dense(20, name='VClz512to20.1of2', activation='relu'))
        model.add(Dense(self._actionSize, name='VClz512to20.2of2', activation='softmax')) # this is not Q func, softmax is prefered
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **Trainer.COMPILE_ARGS)
        model.summary()
        return model

########################################################################
if __name__ == '__main__':

    exportNonTrainable = False
    # sys.argv.append('-x')
    if '-x' in sys.argv :
        exportNonTrainable = True
        sys.argv.remove('-x')

    if not '-f' in sys.argv :
        sys.argv += ['-f', os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../conf/Train.json']

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

    p.info('all objects registered piror to Trainer: %s' % p.listByType())
    
    # trainer = p.createApp(Trainer, configNode ='Trainer', replayFrameFiles=os.path.join(sourceCsvDir, 'RFrames_SH510050.h5'))
    trainer = p.createApp(Trainer, configNode ='train')

    p.start()

    if exportNonTrainable :
        trainer.exportLayerWeights()
        quit()

    p.loop()
    p.info('loop done, all objs: %s' % p.listByType())
    p.stop()
