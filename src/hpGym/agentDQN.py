# encoding: UTF-8

'''
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
'''
from .GymTrader import GymTrader, MetaAgent

import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

from abc import ABCMeta, abstractmethod

########################################################################
class agentDQN(MetaAgent):
    def __init__(self, gymTrader, **kwargs):
        super(agentDQN, self).__init__(gymTrader, **kwargs)

    def buildBrain(self): #TODO param brainId to load json/HD5 from dataRoot/brainId
        '''Build the agent's brain
        '''
        return self.buildBrain_00000()

    def buildBrain_00000(self): #TODO param brainId to load json/HD5 from dataRoot/brainId
        '''Build the agent's brain
        '''
        self._brain = self.loadBrain('DQN00000')
        if not self._brain :
            self._brain = Sequential()
            neurons_per_layer = 24
            activation = "relu"
            self._brain.add(Dense(neurons_per_layer,
                            input_dim=self._stateSize,
                            activation=activation))
            self._brain.add(Dense(neurons_per_layer, activation=activation))
            self._brain.add(Dense(self._actionSize, activation='linear'))

        self.__wkBrainId = 'DQN00000'
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

    def isReady(self) :
        return super(agentDQN, self).isReady()

    def saveBrain(self, brainId=None) :
        ''' save the current brain into the dataRoot
        @param a unique brainId must be given
        '''
        if not self._gymTrader.dataRoot or not self._brain :
            raise ValueError("Null brain or Null trader")

        if not brainId or len(brainId) <=0:
            if self.__wkBrainId and len(brainId) >0:
                brainId = brainId
            else : raise ValueError("empty brainId")

        try :
            brainDir = '%s%s/' % (self._gymTrader.dataRoot, brainId)
            os.makedirs(brainDir)
        except:
            pass

        # step 1. save the model file in json
        model_json = self._brain.to_json()
        with open('%smodel.json' % brainDir, 'w') as mjson:
            mjson.write(model_json)
        
        # step 2. save the weights of the model
        self._brain.save('%smodel.json.h5' % brainDir)
        
    def loadBrain(self, brainId) :
        ''' load the previous saved brain
        @param a unique brainId must be given
        '''
        if not self._gymTrader :
            raise ValueError("Null trader")
        if not brainId or len(brainId) <=0:
            raise ValueError("empty brainId")

        try :
            brainDir = '%s%s/' % (self._gymTrader.dataRoot, brainId)
            os.makedirs(brainDir)
        except:
            pass

        brain = None
        # step 1. read the model file in json
        try :
            with open('%smodel.json' % brainDir, 'r') as mjson:
                model_json = mjson.read()
            brain = model_from_json(model_json)

            # step 2. read the weights of the model
            brain.load_weights('%smodel.json.h5' % brainDir)
        except:
            pass

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
            act_values = self._brain.predict(state)
            action[np.argmax(act_values[0])] = 1

        return action

    def gymObserve(self, state, action, reward, next_state, done, warming_up=False):
        '''Memory Management and training of the agent
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self._idxMem = (self._idxMem + 1) % self._memorySize
        self._memory[self._idxMem] = (state, action, reward, next_state, done)
        if warming_up or 0 != (self._idxMem % self._trainInterval):
            return None

        # TODO: if self._epsilon > self._epsilonMin:
        #     self._epsilon -= self.__epsilonDecrement

        state, action, reward, next_state, done = self.__get_batches()
        reward += (self._gamma
                    * np.logical_not(done)
                    * np.amax(self._brain.predict(next_state),
                                axis=1))

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
        return self._brain.fit(x=state, y=q_target, epochs=1, batch_size=self._batchSize, verbose=0)

    def __get_batches(self):
        '''Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        '''
        batch = np.array(random.sample(self._memory, self._batchSize))
        state_batch = np.concatenate(batch[:, 0]).reshape(self._batchSize, self._stateSize)
        action_batch = np.concatenate(batch[:, 1]).reshape(self._batchSize, self._actionSize)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]).reshape(self._batchSize, self._stateSize)
        done_batch = batch[:, 4]

        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

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