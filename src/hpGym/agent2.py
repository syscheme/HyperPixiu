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
class agentDoubleDQN(agentDQN):
    def __init__(self, gymTrader, **kwargs):
        super(agentDoubleDQN, self).__init__(gymTrader, **kwargs)

        # treat the orginal self._brain as the right hemicerebrum to perform predicting, and 
        # additional self._brainTrain to perform training in an additional thread
        self._brainTrain = None
        if self._brain : 
            model_json = self._brain.to_json()
            self._brainTrain = model_from_json(model_json)
            self._brainTrain.set_weights(self._brain.get_weights()) 
            self._brainTrain.compile(loss='mse', optimizer=Adam(lr=self._learningRate))

    def isReady(self) :
        if not self._brainTrain:
            return False

        if not super(agentDoubleDQN, self).isReady():
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

            y = self._brainTrain.predict(next_state_batch)
            reward_batch += (self._gamma * np.logical_not(done_batch) * np.amax(y, axis=1))

            q_target = self._brainTrain.predict(state_batch)
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
            # loss = self._brainTrain.fit(x=state_batch, y=q_target, epochs=1, batch_size=self._batchSize, verbose=0)
            loss = self._brainTrain.train_on_batch(x=state_batch, y=q_target, callbacks=self.__callbacks)

            #if loss < self._loss:
            with self._lock: # update the trained left hemicerebrum to the right
                self._brain.set_weights(self._brainTrain.get_weights()) 
                self._loss =  loss

    def gymObserve(self, state, action, reward, next_state, done, **feedbacks):
        '''Memory Management and training of the agent
        @return tuple:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        '''
        self._statusAttrs = {**self._statusAttrs, **feedbacks}
        if not self._pushToReplay(state, action, reward, next_state, done) :
            return None

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
                #     this_state[i] = train_batch[i][0]
                #     actions[i] = train_batch[i][1]
                #     rewards[i] = train_batch[i][2]
                #     next_state[i] = train_batch[i][3]
                #     dones[i] = train_batch[i][4]
            
                # q1 = actor_network.model.predict([next_state, np.zeros((32, 1))])
                # q1 = np.argmax(q1[0], axis=3)

                # q2 = target_network.model.predict([next_state, np.zeros((32, 1))])
                # q2 = q2[0].reshape((batch_size, env.actions))

                # end_multiplier = -(dones - 1)

                # double_q = q2[range(32), q1.reshape((32))].reshape((32, 1))

                # target_q = rewards + (gamma*double_q*end_multiplier)

                # print "Target Q Shape: ", target_q.shape
                # q = actor_network.model.predict([this_state, actions])
                # #q_of_actions = q[:, train_batch[:, 1]]
                # print target_q.shape
                # actor_network.model.fit([this_state, actions], [np.zeros((32, 1, 1, 4)) ,target_q])
                # target_network.model.set_weights(actor_network.model.get_weights())


        # this basic DQN also performs training in this step
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._sampleBatches()
        batchsize = len(state_batch)

        with self._lock:
            pred_train   = self._brainTrain.predict(next_state_batch)
            maxAct_train = np.argmax(pred_train, axis=1)
            pred_main    = self._brain.predict(next_state_batch)
            
            # https://github.com/p-Mart/Double-Dueling-DQN-Keras/blob/master/DDDQN.py
            double_q     = pred_main[range(batchsize), maxAct_train.reshape((batchsize))].reshape((batchsize, 1))
            reward_batch += (self._gamma * np.logical_not(done) * double_q)

            q_main  = self._brain.predict(state)
            q_main[action_batch[0], action_batch[1]] = reward_batch

            epochs =1
            if len(GPUs) > 0 and len(state) >self._batchSize:
                epochs = self._epochsPerObservOnGpu

            self._loss = self._brain.fit(x=state_batch, y=q_main, epochs=epochs, batch_size=self._batchSize, verbose=0, callbacks=self.__callbacks)
            self._brainTrain.set_weights(self._brain.get_weights())

        return self._loss

