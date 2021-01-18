# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''
from __future__ import division
from abc import abstractmethod

from Application  import Program, BaseApplication, MetaObj, BOOL_STRVAL_TRUE
from HistoryData  import H5DSET_DEFAULT_ARGS
from dnn.BaseModel  import BaseModel

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Dense, Conv1D, Activation, Dropout, LSTM, Reshape, MaxPooling1D, GlobalAveragePooling1D, ZeroPadding1D
from tensorflow.keras.layers import BatchNormalization, Flatten, add, GlobalAveragePooling2D, Lambda, Concatenate, ZeroPadding2D
from tensorflow.keras import regularizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs

from tensorflow.keras.applications.resnet50 import ResNet50

import tensorflow as tf
import numpy as np
import math

########################################################################
class Model88(BaseModel) :
    '''
    Model88 has a common 88 features at the end
    '''
    CORE_LAYER_PREFIX = 'core.'

    def __init__(self, outputClasses =3, **kwargs):
        super(Model88,self).__init__(**kwargs)
        self._classesOut = outputClasses
        self._startLR    = kwargs.get('startLR', 0.01)

    def _feature88toOut(self, flattern_inputs) :
        # unified final layers Dense(VirtualFeature88) then Dense(self._actionSize)
        lnTag ='F88.'
        x = self._tagged_chain(lnTag, flattern_inputs, Dropout(0.3, name='%sDropout1' %lnTag)) #  x= Dropout(0.5))
        x = self._tagged_chain(lnTag, x, Dense(88, name='%sDense1' %lnTag))
        x = self._tagged_chain(lnTag, x, Dense(self._classesOut,  name='%sDense2' %lnTag, activation='softmax'))
        return x

    @property
    def modelId(self) :
        return 'M88.%s' % self.coreId

    @property
    def coreId(self) :
        coreId = self.__class__.__name__
        if 'Model88_' in coreId:
            coreId = coreId[coreId.index('Model88_') + len('Model88_') :]
        
        return coreId

    @abstractmethod
    def buildup(self, input_shape):
        '''
        @return self.model
        '''
        raise NotImplementedError

    @abstractmethod
    def _buildup_layers(self, input_shape, input_tensor=None):
        '''
        @return output_tensor NOT the model
        '''
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)
            inputs = input_tensor
        else:
            inputs = get_source_inputs(input_tensor)

        # core_model = self._buildup_core(input_shape, input_tensor) # layerIn.shape)
        # core_mId = core_model.name
        # self.__tagCoreModel(core_model, core_mId)
        # x = core_model(input_tensor)
        # x = self._feature88toOut(core_model)
        # self._dnnModel = Model(inputs=inputs, outputs=x, name=self.modelId)
        # # self._dnnModel.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **BaseModel.COMPILE_ARGS)
        # # self._dnnModel.summary()
        # return self.model

        x = self._buildup_core('%s%s.' % (Model88.CORE_LAYER_PREFIX, self.coreId), input_shape, input_tensor)
        x = self._feature88toOut(x)
        return x

    def _tagLayer(layer, lnTag) :
        if '.' != lnTag[-1]: lnTag +='.'

        if layer and layer.name[:len(lnTag)] != lnTag:
            layer._name = lnTag + layer.name

    def _tagged_chain(self, lnTag, input_tensor, layer) :
        Model88._tagLayer(layer, lnTag)
        return layer(input_tensor)

    @abstractmethod
    def _buildup_core(self, lnTag, input_shape, input_tensor): # TODO: input_shape was supposed to get from input_tensor
        '''
        @return output_tensor
        '''
        raise NotImplementedError

    def create(self, layerIn):
        layerIn = Input(shapeIn)

        self._dnnModel = Model(inputs=layerIn, outputs=x)
        # sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, **BaseModel.COMPILE_ARGS)

        # TODO apply non-trainable feature88 weights

        return self.model

########################################################################
# the old 1548 flat models
class Model88_Flat(Model88) :
    '''
    the basic 1548 flat models
    '''
    def __init__(self, **kwargs):
        super(Model88_Flat, self).__init__(outputClasses =3, **kwargs)

    @property
    def modelId(self) :
        return 'M88F.%s' % self.coreId

    @abstractmethod
    def buildup(self, input_shape=(1548, )) :
        layerIn = Input(shape=input_shape)
        new_shape = (int(input_shape[0]/4), 4)
        x = Reshape(new_shape, input_shape=input_shape)(layerIn)
        # m = super(Model88_Flat, self)._buildup_layers(new_shape, x)
        # x = m(x)
        x = self._buildup_layers(new_shape, x)
        self._dnnModel = Model(inputs=get_source_inputs(layerIn), outputs=x)
        return self.model

# --------------------------------
class Model88_Cnn1Dx4R2(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R2, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_shape, input_tensor):

        # x = Conv1D(128, 3, activation='relu')(input_tensor)
        x = self._tagged_chain(lnTag, input_tensor, Conv1D(128, 3, activation='relu'))

        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Dropout(0.3))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(100, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, Dense(512, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())

        return x  # return Model(input_tensor, x, name='%s%s' % (Model88.CORE_LAYER_PREFIX, self.coreId))

# --------------------------------
class Model88_Cnn1Dx4R3(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R3, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_shape, input_tensor):
        x = self._tagged_chain(lnTag, input_tensor, Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Dropout(0.3))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Conv1D(100, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, Dense(512, activation='relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        
        return x  # return Model(input_tensor, x, name='%s%s' % (Model88.CORE_LAYER_PREFIX, self.coreId))

# --------------------------------
class Model88_VGG16d1(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_VGG16d1, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_shape, input_tensor):
        weight_decay = 0.0005

        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, input_tensor, Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        
        #进行一次归一化
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.3))
        #layer2 32*32*64
        # x = Conv1D(64, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(64, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())

        #下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
        #padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))

        #layer3 16*16*64
        # x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))
        
        #layer4 16*16*128
        # x = self._tagged_chain(lnTag, x, Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        
        #layer5 8*8*128
        # x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))
        
        #layer6 8*8*256
        # x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))
        
        #layer7 8*8*256
        # x = self._tagged_chain(lnTag, x, Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))

        #layer8 4*4*256
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))

        #layer9 4*4*512
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))
        
        #layer10 4*4*512
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        
        #layer11 2*2*512
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))

        #layer12 2*2*512
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, Dropout(0.4))

        #layer13 2*2*512
        # x = self._tagged_chain(lnTag, x, Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, Dropout(0.5))

        #layer14 1*1*512
        x = self._tagged_chain(lnTag, x, Flatten())
        # x = self._tagged_chain(lnTag, x, Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())

        #layer15 512
        # x = self._tagged_chain(lnTag, x, Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, Activation('relu'))
        x = self._tagged_chain(lnTag, x, BatchNormalization())

        return x # return Model(input_tensor, x, name=self.coreId)

# --------------------------------
class Model88_ResNet34d1(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_ResNet34d1, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_shape, input_tensor):

        weight_decay = 0.0005

        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, input_tensor, Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        #conv1
        x = self._resBlk_basic(lnTag, x, nb_filter=64, kernel_size=3, padding='valid')
        x = self._tagged_chain(lnTag, x, MaxPooling1D(2))

        #conv2_x
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[64,64,256], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[64,64,256])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[64,64,256])

        #conv3_x
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[128, 128, 512], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[128, 128, 512])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[128, 128, 512])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[128, 128, 512])

        #conv4_x
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[256, 256, 1024])

        #conv5_x
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[512, 512, 2048], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[512, 512, 2048])
        x = self._resBlk_bottleneck(lnTag, x, nb_filters=[512, 512, 2048])

        x = self._tagged_chain(lnTag, x, GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, Flatten())

        return x # return Model(input_tensor, x, name=self.coreId)

    def _resBlk_basic(self, lnTag, x, nb_filter, kernel_size, padding='same', regularizer=None, name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = self._tagged_chain(lnTag, x, Conv1D(nb_filter, kernel_size, padding=padding, activation='relu', name=conv_name, kernel_regularizer= regularizer))
        x = self._tagged_chain(lnTag, x, BatchNormalization(name=bn_name))
        return x

    def _resBlk_identity(self, lnTag, inpt, nb_filter, kernel_size, with_conv_shortcut=False):
        x = self._resBlk_basic(lnTag, inpt, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(lnTag, inpt, nb_filter=nb_filter, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def _resBlk_bottleneck(self, lnTag, inpt, nb_filters, with_conv_shortcut=False):
        k1,k2,k3=nb_filters
        x = self._resBlk_basic(lnTag, inpt, nb_filter=k1, kernel_size=1, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=k2, kernel_size=3, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(lnTag, inpt, nb_filter=k3, kernel_size=1)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x


########################################################################
class Model2D_Sliced(Model88) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(Model2D_Sliced, self).__init__(outputClasses = outputClasses, **kwargs)
        self.__channels_per_slice =4
        self.__features_per_slice =518
        self.__coreId = "NA"
        self.__subModels = {}

    @property
    def modelId(self) :
        return 'M88S%d.%s' % (self.__channels_per_slice, self.coreId)
    
    @property
    def coreId(self) : return self.__coreId

    def __slice2d(x, idxSlice, channels_per_slice): 
        slice = x[:, :, :, idxSlice*channels_per_slice : (idxSlice+1)*channels_per_slice]
        new_shape = (slice.shape[0], slice.shape[1], slice.shape[2], 1) # tuple(list(slice.shape[:3] +[1]))
        tensor0 = tf.zeros(new_shape)
        slice = tf.cat((slice, tensor0), axis=-1)

        ch2append = channels_per_slice - slice.shape[3] # [0]s to append to fit channel=4

        if ch2append >0:
            slice0s = np.zeros(tuple(list(slice.shape[1:3]) +[ch2append]), dtype='float32') # TODO fix this
            tensor0s = tf.convert_to_tensor(slice0s)
            slice += tensor0s
            # slice = np.concatenate((slice, slice0s), axis=2)
        return slice

    def __slice2d_flow(self, inputs, input_shape, core_model, idxSlice):
        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        # x = Lambda(lambda x: x[:, :, :, idxSlice*self.__channels_per_slice : (idxSlice+1)*self.__channels_per_slice], output_shape=slice_shape)(inputs)
        x = Lambda(Model2D_Sliced.__slice2d, output_shape=slice_shape, arguments={'idxSlice':idxSlice, 'channels_per_slice': self.__channels_per_slice})(inputs)
        ch2append = self.__channels_per_slice - x.shape[3]
        if ch2append >0:
            x = ZeroPadding2D(padding=(0, 0, ch2append))(x)

        # common layers to self.__features_per_slice
        lnTag = 'M88S%dflow%s.' % (self.__channels_per_slice, idxSlice)
        tensor_flowClose = core_model(x)

        flowCloseIn = Input(tuple(tensor_flowClose.shape[1:]))
        x =Flatten(name='%sflatten' %lnTag)(flowCloseIn)
        x =Dropout(0.3, name='%sdropout' %lnTag)(x)
        lf=Dense(self.__features_per_slice, name='%sF%d' % (lnTag, self.__features_per_slice))
        x =lf(x)
        m = Model(inputs=flowCloseIn, outputs=x, name='%sC' %lf.name)
        self.__subModels[m.name] =m

        return m(tensor_flowClose)

    def buildup(self, input_shape=(32,32,3)):
        layerIn = Input(shape=input_shape)

        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        slices = int(channels / self.__channels_per_slice)
        if 0 != channels % self.__channels_per_slice: slices +=1

        core_model = self._buildup_core(slice_shape, layerIn)
        self.__coreId = core_model.name
        tagCore = '%s%s' % (Model88.CORE_LAYER_PREFIX, self.__coreId)
        self.__subModels[tagCore] = core_model
        self.__tagCoreModel(core_model, tagCore)
        # x = self._buildup_core('%s%s.' % (Model88.CORE_LAYER_PREFIX, self.coreId), slice_shape, layerIn)

        sliceflows = [None] * slices
        for i in range(slices):
            sliceflows[i] = self.__slice2d_flow(layerIn, input_shape, core_model, i)
        
        # merge the multiple flow-of-slice into a controllable less than F518*2
        merged_tensor = sliceflows[0] if 1 ==len(sliceflows) else Concatenate(axis=1, name='M88S4ConX%d' % slices)(sliceflows) # merge = merge(sliceflows, mode='concat') # concatenate([x1,x2,x3])
        
        closeIn = Input(tuple(merged_tensor.shape[1:]))
        x = closeIn

        dsize = int(math.sqrt(slices))
        if dsize*dsize < slices: dsize +=1
        seq = list(range(dsize))[1:]
        seq.reverse()

        for i in seq:
            x =Dropout(0.5,  name='M88S4M_dropout%d' % i)(x)
            x =Dense(self.__features_per_slice *i, name='M88S4M_F%dx%d' % (self.__features_per_slice, i))(x)

        x = self._feature88toOut(x)
        m = Model(inputs=closeIn, outputs=x, name='F88.F%dx%dC' %(self.__features_per_slice, slices))
        self.__subModels[m.name] =m

        for k, v in self.__subModels.items():
            v.summary()
            v.save('/tmp/%s.h5' % k)
        
        x = m(merged_tensor)
        self._dnnModel = Model(inputs=layerIn, outputs=x, name='%sx%d' %(self.__features_per_slice, slices))

        # self._dnnModel.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **BaseModel.COMPILE_ARGS)
        # self._dnnModel.summary()
        return self.model

    def __tagCoreModel(self, core_model, core_mId) :
        # add the prefix tag
        lnTag = core_mId
        if not lnTag or len(lnTag) <=0: lnTag= Model88.CORE_LAYER_PREFIX
        if Model88.CORE_LAYER_PREFIX != lnTag[:len(Model88.CORE_LAYER_PREFIX)]:
            lnTag = '%s%s' % (Model88.CORE_LAYER_PREFIX, lnTag)
        
        if '.' != lnTag[-1]: lnTag +='.'

        for layer in core_model.layers:
            Model88._tagLayer(layer, lnTag)

    @abstractmethod
    def _buildup_core(self, slice_shape, input_tensor):
        # # a dummy core
        # x = self._tagged_chain(lnTag, input_tensor, Dense(1000, input_shape =slice_shape))
        # x = self._tagged_chain(lnTag, x, Flatten())
        # x = self._tagged_chain(lnTag, x, Dense(518))
        # # return x

        # # inputs=get_source_inputs(input_tensor)
        # return Model(inputs=[input_tensor], outputs=x, name=lnTag[:-1])

        core = ResNet50(weights=None, classes=1000, input_shape=slice_shape) # , input_tensor=input_tensor) # dummy code
        return core

        #TODO: need to define a channel=4 module
        # refer to:
        # /usr/local/lib64/python3.6/site-packages/keras_applications/resnet50.py
        # /usr/local/lib/python3.6/site-packages/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
        # def ResNet50(include_top=True, ...


    '''
    refer to 
    https://blog.csdn.net/qq_36113741/article/details/105854544
    a = Input(shape=(32,16,8))
    x1 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':0, 'channels_per_slice':3})(a)
    x2 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':1, 'channels_per_slice':3})(a)
    x3 = Lambda(__slice2d, output_shape=(32,16,3), arguments={'idxSlice':2, 'channels_per_slice':3})(a)

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
    model.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **BaseModel.COMPILE_ARGS)
    model.summary()
    return model
    '''

########################################################################
if __name__ == '__main__':
    
    model = Model2D_Sliced() # Model88_ResNet34d1() # Model88_Cnn1Dx4R2() # Model2D_Sliced()
    model.buildup()
    model.compile()
    model.summary()
    model.model.save('/tmp/%s.h5' % model.modelId) # model.save_model('/tmp/%s.h5' % model.modelId)
