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
from keras.callbacks import ModelCheckPoint

from abc import ABCMeta, abstractmethod

########################################################################
class agentDQN(MetaAgent):
    def __init__(self, gymTrader, **kwargs):
        super(agentDQN, self).__init__(gymTrader, **kwargs)
        self._trader = gymTrader

    def buildBrain(self): #TODO param brainId to load json/HD5 from dataRoot/brainId
        '''Build the agent's brain
        '''
        self._brain = self.loadBrain('DQN00000')
        if not self._brain :
            self._brain = Sequential()
            neurons_per_layer = 24
            activation = "relu"
            self._brain.add(Dense(neurons_per_layer,
                            input_dim=self.state_size,
                            activation=activation))
            self._brain.add(Dense(neurons_per_layer, activation=activation))
            self._brain.add(Dense(self.action_size, activation='linear'))

        self.__wkBrainId = 'DQN00000'
        self._brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # checkpointPath ='best.h5'
        # checkpoint = ModelCheckPoint(filepath=checkpointPath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
        # self._brain.flt(x, Y...., callbacks=[checkpoint])
        # more additiona, to customize the checkpoint other than max(val_acc), such as max(mean(y_pred))
        #    import keras.backend as K
        #    def mean_pred(y_true, y_pred):
        #        return K.mean(y_pred)
        #    model.compile(..., metrics=['accuracy', mean_pred])
        #    ModelCheckpoint(..., monitor='val_mean_pred', mode='max', period=1)

        return self._brain

    def saveBrain(self, brainId=None) :
        ''' save the current brain into the dataRoot
        @param a unique brainId must be given
        '''
        if not self._trader.dataRoot or not self._brain :
            raise ValueError("Null brain or Null trader")
        if not brainId or len(brainId) <=0:
            if self.__wkBrainId and len(brainId) >0:
                brainId = brainId
            else : raise ValueError("empty brainId")

        try :
            brainDir = '%s%s/' % (self._trader.dataRoot, brainId)
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
        if not self._trader.dataRoot 
            raise ValueError("Null trader")
        if not brainId or len(brainId) <=0:
            raise ValueError("empty brainId")

        try :
            brainDir = '%s%s/' % (self._trader.dataRoot, brainId)
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

        retun brain

    def gymAct(self, state):
        '''Acting Policy of the agentDQN
        @return one of self.__gymTrader.ACTIONS
        '''
        action = np.zeros(self.action_size)
        if np.random.rand() <= self._epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self._brain.predict(state)
            action[np.argmax(act_values[0])] = 1

        return action

    def gymObserve(self, state, action, reward, next_state, done, warming_up=False):
        '''Memory Management and training of the agent
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self.__idxMem = (self.__idxMem + 1) % self._memorySize
        self.__memory[self.__idxMem] = (state, action, reward, next_state, done)
        if warming_up:
            return None, None, None, None, None

        if (self.__idxMem % self._trainInterval) == 0:
            if self._epsilon > self._epsilonMin:
                self._epsilon -= self.__epsilonDecrement

            state, action, reward, next_state, done = self.__get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self._brain.predict(next_state),
                                 axis=1))

            q_target = self._brain.predict(state)
            q_target[action[0], action[1]] = reward

            return self._brain.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)

    def __get_batches(self):
        '''Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        '''
        batch = np.array(random.sample(self.__memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]).reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1]).reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]).reshape(self.batch_size, self.state_size)
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