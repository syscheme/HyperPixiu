# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''

from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
import HistoryData as hist

from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend

import sys, os, platform, random, copy
import h5py, tarfile
import numpy as np

H5_COLS = 'state,action,reward,next_state,done'
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
                    for col in H5_COLS.split(',') :
                        del self.__samplePool[col][i]

            cAppend =0
            strFrames=''
            while len(frameSeq) >0 and len(self.__samplePool['state']) <max(samplePerFrame, self._trainSize *2) :
                strFrames += '%s,' % frameSeq[0]
                frame = self._h5file[frameSeq[0]]
                del frameSeq[0]

                for col in H5_COLS.split(',') :
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

                for col in H5_COLS.split(',') :
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
        for d in ['e:/AShareSample', '/mnt/e/AShareSample/ETF', '/mnt/m/AShareSample']:
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
    
    # trainer = p.createApp(DQNTrainer, configNode ='DQNTrainer', h5filepath='out/IdealDayTrader/CsvToDQN_24106/RFrames_000001.h5')
    trainer = p.createApp(DQNTrainer, configNode ='DQNTrainer', h5filepath='e:/AShareSample/ETF/RFrames_SH510050.h5')
    # rec = p.createApp(hist.TaggedCsvRecorder, configNode ='recorder', filepath = '/tmp/DQNTrainer.tcsv')
    # trainer.setRecorder(rec)

    p.start()
    p.loop()
    p.stop()
