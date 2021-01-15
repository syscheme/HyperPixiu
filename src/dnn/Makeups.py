# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''
from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
from HistoryData  import H5DSET_DEFAULT_ARGS
from dnn.BaseModel  import BaseModel

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D, GlobalAveragePooling1D, ZeroPadding1D
from tensorflow.keras.layers import BatchNormalization, Flatten, add, GlobalAveragePooling2D, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as backend
from tensorflow.keras.utils import Sequence
# from keras.layers.merge import add
from tensorflow.keras import saving

from tensorflow.keras.applications.resnet50 import ResNet50

import tensorflow as tf

import sys, os, platform, random, copy, threading
from datetime import datetime
from time import sleep

import numpy as np

FN_SUFIX_MODEL_JSON = '_model.json'
FN_SUFIX_WEIGHTS_H5 = '_weights.h5'

########################################################################
class Model88(BaseModel) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, outputClasses =3, program=None):
        super(Model88,self).__init__(program)
        self._classesOut = outputClasses

    def __feature88toOut(self, flatternLayerToAppend) :
        lnTag ='F88'
        # unified final layers Dense(VirtualFeature88) then Dense(self._actionSize)

        x = Dropout(0.3, name='%s:Dropout1' %lnTag)(flatternLayerToAppend) #  x= Dropout(0.5)(x)
        x = Dense(88, name='%s:Dense1' %lnTag)(x)
        x = Dense(self._classesOut,  name='%s:Dense2' %lnTag, activation='softmax')(x)
        return x

    @abstractmethod
    def build(self, input_shape):
        '''
        input_shape: shape tuple otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
        '''
        layerIn = Input(shape=input_shape)
        shared_model, shared_modelId = buildSharedModel(layerIn.shape)
        self.tagSubModel(shared_model, shared_modelId)

        o = shared_model(layerIn)
        model = Model(inputs=layerIn, outputs=o)
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **ReplayTrainer.COMPILE_ARGS)
        model.summary()
        return model

    def tagSubModel(self, shared_model, shared_modelId) :
        # add the prefix tag
        if not shared_modelId or len(shared_modelId) <=0: shared_modelId="shared"
        if ':' != shared_modelId[-1]:
            shared_modelId +=':'

        for layer in shared_model.layers:
            if layer.name[:len(shared_modelId)] != shared_modelId:
                layer.name = shared_modelId + layer.name

    @abstractmethod
    def buildSharedModel(self, slice_shape):
        pass

    def create(self, layerIn):
        layerIn = Input(shapeIn)

        self._dnnModel = Model(inputs=layerIn, outputs=x)
        # sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, **ReplayTrainer.COMPILE_ARGS)

        # TODO apply non-trainable feature88 weights

        return self.model

########################################################################
# the old 1584 models
class Model88_1584(BaseModel) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, program=None):
        super(Model88_1584, self).__init__(3, program)

    def build(self, input_shape=(1584,)):
        new_shape = (int(input_shape[0]/4), 4)
        layerIn = Reshape(new_shape, input_shape=input_shape)
        return super(Model88_Cnn1Dx4R2, self).build(new_shape)(layerIn)

# --------------------------------
class Model88_Cnn1Dx4R2(Model88_1584) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, program=None):
        super(Model88_Cnn1Dx4R2, self).__init__(program)

    def buildSharedModel(self, input_shape):
        layerIn = Input(shape=input_shape)
        x = Conv1D(128, 3, activation='relu', inputs=layerIn)
        x = BatchNormalization()(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(512, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(100, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)

        return Model(layerIn, x, name='Cnn1Dx4R2')

# --------------------------------
class Model88_Cnn1Dx4R3(Model88_1584) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, program=None):
        super(Model88_Cnn1Dx4R3, self).__init__(program)

    def buildSharedModel(self, input_shape):
        layerIn = Input(shape=input_shape)
        x = Conv1D(128, 3, activation='relu', inputs=layerIn)
        x = BatchNormalization()(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(512, 3, activation='relu')(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
        x = Conv1D(256, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(100, 3, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)

        return Model(layerIn, x, name='Cnn1Dx4R3')

# --------------------------------
class Model88_VGG16d1(Model88_1584) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, program=None):
        super(Model88_Cnn1Dx4R3, self).__init__(program)

    def buildSharedModel(self, input_shape):
        weight_decay = 0.0005

        layerIn = Input(shape=input_shape)
        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), inputs=layerIn)
        x = Activation('relu')(x)
        
        #进行一次归一化
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        #layer2 32*32*64
        # x = Conv1D(64, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = Conv1D(64, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        #下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
        #padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
        x = MaxPooling1D(2)(x)

        #layer3 16*16*64
        # x = Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        #layer4 16*16*128
        # x = Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        #layer5 8*8*128
        # x = Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        #layer6 8*8*256
        # x = Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        #layer7 8*8*256
        # x = Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)

        #layer8 4*4*256
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        #layer9 4*4*512
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        #layer10 4*4*512
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        #layer11 2*2*512
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        #layer12 2*2*512
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        #layer13 2*2*512
        # x = Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.5)(x)

        #layer14 1*1*512
        x = Flatten()(x)
        # x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        #layer15 512
        # x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        return Model(layerIn, x, name='VGG16d1')

########################################################################
class Model2D_SlicesByChannel(Model88) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __slice2d(x, idxSlice, channels_per_slice=4): 
        slice = x[:,:, idxSlice : idxSlice+4]
        ch2append = channels_per_slice - slice.shape[2] # [0]s to append to fit channel=4
        if ch2append >0:
            slide0 = np.zeros(tuple(list(slice.shape[:2]) +[ch2append])(x)
            slice = np.concatenate((slice, slice0), axis=2)
        return slice

    def __slice2d_flow(inputs, shared_model, idxSlice, channels_per_slice=4):
        channels = inputs.shape[2]
        slice_shape = tuple(list(inputs.shape[:2]) +[channels_per_slice])
        x = Lambda(__slice2d, output_shape=slice_shape, arguments={'idxSlice':idxSlice, 'channels_per_slice':channels_per_slice})(inputs)

        # sample
        lnTag = 's2dflow%s:' % idxSlice
        x = shared_model(x)
        x =Flatten(name='%sflatten' %lnTag)(x)
        x =Dropout(0.3, name='%sdropout' %lnTag)(x)
        x =Dense(518, name='%sD518' %lnTag)(x)
        return x

    def build(self, input_shape=(32,16,8)) :
        channels = input_shape[2]
        channels_per_slice =4
        slice_shape = tuple(list(input_shape[:2]) +[channels_per_slice])
        slices = int(channels / channels_per_slice)
        if 0 != channels % channels_per_slice: slices +=1

        layerIn = Input(shape=input_shape)

        shared_model, shared_modelId = buildSharedModel(slice_shape)
        self.tagSubModel(shared_model, shared_modelId)

        x = [None] * slices
        for i in range(slices):
            x[i] = __slice2d_flow(layerIn, shared_model, i, channels_per_slice=channels_per_slice)
        
        merge = merge(x, mode='concat') # concatenate([x1,x2,x3])

        o = self.__feature88toOut(merge)
        model = Model(inputs=layerIn, outputs=o)
        model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **ReplayTrainer.COMPILE_ARGS)
        model.summary()
        return model

    @abstractmethod
    def buildSharedModel(self, slice_shape):
        #TODO: need to define a channel=4 module
        # refer to:
        # /usr/local/lib64/python3.6/site-packages/keras_applications/resnet50.py
        # /usr/local/lib/python3.6/site-packages/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
        # def ResNet50(include_top=True, ...
        return ResNet50(weights=None, classes=1000, input_shape=slice_shape) # dummy code


    '''
    refer to 
    https://blog.csdn.net/qq_36113741/article/details/105854544
    a = Input(shape=(32,16,8))
    x1 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':0, 'channels_per_slice'=3})(a)
    x2 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':1, 'channels_per_slice'=3})(a)
    x3 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':2, 'channels_per_slice'=3})(a)

    pretrained = ResNet50(weights=None, classes=1000, input_shape=(32, 16, 3))
    x1 =pretrained(x1)
    x1 =Flatten()(x1)
    x1 =Dense(518)(x1)
    x1 =BatchNormalization()(x1)
    x1 =Dropout(0.3)(x1)
    x1 =Dense(88)(x1)

    x2 =pretrained(x2)
    x2 =Flatten()(x2)
    x2 =Dense(518)(x2)
    x2 =BatchNormalization()(x2)
    x2 =Dropout(0.3)(x2)
    x2 =Dense(88)(x2)

    x3 =pretrained(x3)
    x3 =Flatten()(x3)
    x3 =Dense(518)(x3)
    x3 =BatchNormalization()(x3)
    x3 =Dropout(0.3)(x3)
    x3 =Dense(88)(x3)

    merge = merge([x1,x2],mode='concat') # concatenate([x1,x2,x3])
    o = Dense(20, name='VClz512to20.1of2', activation='relu')(merge)
    o = Dense(3, name='VClz512to20.2of2', activation='softmax')(o)
    model = Model(inputs=a, outputs=o)
    model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **ReplayTrainer.COMPILE_ARGS)
    model.summary()
    return model
    '''

########################################################################
if __name__ == '__main__':
    pass
