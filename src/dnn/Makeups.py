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

from tensorflow.keras.models import model_from_json, Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, LSTM, Reshape, Lambda, Concatenate, BatchNormalization, Flatten, add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, ZeroPadding1D
from tensorflow.keras.layers import ZeroPadding2D, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs

from tensorflow.keras.applications.resnet50 import ResNet50

import tensorflow as tf
import numpy as np
import math
import h5py, fnmatch

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
    def buildup(self, input_tensor):
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

        x = self._buildup_core('%s%s.' % (Model88.CORE_LAYER_PREFIX, self.coreId), input_tensor)
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
    def _buildup_core(self, lnTag, input_tensor): 
        '''
        @return output_tensor
        '''
        input_shape = tuple([ int(x) for x in input_tensor.shape[1:]]) # get rid of the leading dim-batch
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

    def _buildup_core(self, lnTag, input_tensor):

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

    def _buildup_core(self, lnTag, input_tensor):
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

    def _buildup_core(self, lnTag, input_tensor):
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

    def _buildup_core(self, lnTag, input_tensor):

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
        self.__model_core = None
        self.__json_core = None
        self.__model_suplementals = {}

    @property
    def modelId(self) :
        return 'M88S%d.%s' % (self.__channels_per_slice, self.coreId)
    
    @property
    def coreId(self) : return self.__coreId

    def get_weights_core(self) : return self.__model_core.get_weights()

    def __slice2d(x, idxSlice, channels_per_slice): 
        slice = x[:, :, :, idxSlice*channels_per_slice : (idxSlice+1)*channels_per_slice]
        # new_shape = (slice.shape[0], slice.shape[1], slice.shape[2], 1) # tuple(list(slice.shape[:3] +[1]))
        # tensor0 = tf.zeros(new_shape)
        # slice = tf.cat((slice, tensor0), axis=-1)

        ch2append = channels_per_slice - slice.shape[3] # [0]s to append to fit channel=4

        if ch2append >0:
            slice0s = np.zeros(tuple(list(slice.shape[1:3]) +[ch2append]), dtype='float32') # TODO fix this
            tensor0s = tf.convert_to_tensor(slice0s)
            slice += tensor0s
            # slice = np.concatenate((slice, slice0s), axis=2)
        return slice

    def __slice2d_flow(self, input_tensor, core_model, idxSlice):
        input_shape = tuple([ int(x) for x in input_tensor.shape[1:]]) # get rid of the leading dim-batch
        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        # x = Lambda(lambda x: x[:, :, :, idxSlice*self.__channels_per_slice : (idxSlice+1)*self.__channels_per_slice], output_shape=slice_shape)(input_tensor)
        x = Lambda(Model2D_Sliced.__slice2d, output_shape=slice_shape, arguments={'idxSlice':idxSlice, 'channels_per_slice': self.__channels_per_slice})(input_tensor)
        # ch2append = self.__channels_per_slice - x.shape[3]
        # if ch2append >0:
        #  NOT WORK   x = ZeroPadding2D(padding=(0, 0, ch2append))(x)

        # common layers to self.__features_per_slice
        lnTag = 'M88S%dflow%s.' % (self.__channels_per_slice, idxSlice)
        tensor_flowClose = core_model(x)

        flowCloseIn = Input(tuple(tensor_flowClose.shape[1:]))
        x =Flatten(name='%sflatten' %lnTag)(flowCloseIn)
        x =Dropout(0.3, name='%sdropout' %lnTag)(x)
        lf=Dense(self.__features_per_slice, name='%sF%d' % (lnTag, self.__features_per_slice))
        x =lf(x)
        m = Model(inputs=flowCloseIn, outputs=x, name='%sC' %lf.name)
        self.__model_suplementals[m.name] =m

        return m(tensor_flowClose)

    def __slice2d_flow2(self, submod_name, model_json, custom_objects, input_tensor, core_model):
        # common layers to self.__features_per_slice

        lnTag = submod_name + '.'
        tensor_flowClose = core_model(input_tensor)

        # construct the submodel
        m = None
        if model_json :
            m = model_from_json(model_json, custom_objects=custom_objects)
            m._name = submod_name

        if not m:
            flowCloseIn = Input(tuple(tensor_flowClose.shape[1:]))
            x =Flatten(name='%sflatten' %lnTag)(flowCloseIn)
            x =Dropout(0.3, name='%sdropout' %lnTag)(x)
            lf=Dense(self.__features_per_slice, name='%sF%d' % (lnTag, self.__features_per_slice))
            x =lf(x)
            m = Model(inputs=flowCloseIn, outputs=x, name=submod_name)

        self.__model_suplementals[m.name] =m
        return m(tensor_flowClose)

    def buildup(self, input_shape=(32, 32, 8)):
        return self.__buildup(None, None, None, input_shape)

    def __buildup(self, jsonCore, jsonSubs, custom_objects, input_shape):
        layerIn = Input(shape=input_shape)

        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        slice_count = int(channels / self.__channels_per_slice)
        if 0 != channels % self.__channels_per_slice: slice_count +=1

        slices = [None] * slice_count
        for i in range(slice_count) :
            slices[i] = Lambda(Model2D_Sliced.__slice2d, output_shape = slice_shape, arguments={'idxSlice':i, 'channels_per_slice': self.__channels_per_slice})(layerIn)

        if not jsonCore:
            self.__model_core = self._buildup_core(slices[0])
            self.__json_core = self.__model_core.to_json()
        else:
            self.__model_core = model_from_json(jsonCore, custom_objects=custom_objects)
            self.__json_core = jsonCore
        
        if not self.__model_core:
            raise ValueError('failed to create model_core')

        self.__coreId = self.__model_core.name
        # can be called at this moment: self.__model_core.save('/tmp/%s.h5' % self.__coreId)
        self.__model_core.summary()

        # tagCore = '%s%s' % (Model88.CORE_LAYER_PREFIX, self.__coreId)
        # self.__tagCoreModel(self.__model_core, tagCore)

        sliceflows = [None] * slice_count
        for i in range(slice_count):
            submod_name = '2D%dS%dflow' % (self.__channels_per_slice, i)
            model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None

            sliceflows[i] = self.__slice2d_flow2(submod_name, model_json, custom_objects, slices[i], self.__model_core)

            # # common layers to self.__features_per_slice
            # lnTag = 'M88S%dflow%s.' % (self.__channels_per_slice, i)
            # tensor_flowClose = self.__model_core(slices[i])
            # flowCloseIn = Input(tuple(tensor_flowClose.shape[1:]))
            # x =Flatten(name='%sflatten' %lnTag)(tensor_flowClose)
            # x =Dropout(0.3, name='%sdropout' %lnTag)(x)
            # lf=Dense(self.__features_per_slice, name='%sF%d' % (lnTag, self.__features_per_slice))
            # x =lf(x)
            # m = Model(flowCloseIn, outputs=x, name='%sC' %lf.name)
            # self.__model_suplementals[m.name] =m
            # sliceflows[i] =m(tensor_flowClose)

        '''
        for i in range(slice_count):
            # common layers to self.__features_per_slice
            lnTag = 'M88S%dflow%s.' % (self.__channels_per_slice, i)
            tensor_flowClose = self.__model_core(slices[i])
            x =Flatten(name='%sflatten' %lnTag)(tensor_flowClose)
            x =Dropout(0.3, name='%sdropout' %lnTag)(x)
            lf=Dense(self.__features_per_slice, name='%sF%d' % (lnTag, self.__features_per_slice))
            x =lf(x)
            sliceflows[i] = x
        '''
        
        # merge the multiple flow-of-slice into a controllable less than F518*2
        merged_tensor = sliceflows[0] if 1 ==len(sliceflows) else Concatenate(axis=1, name='M88S4ConX%d' % slice_count)(sliceflows) # merge = merge(sliceflows, mode='concat') # concatenate([x1,x2,x3])
        
        m = None
        submod_name = '2D%dS%dF%dX' % (self.__channels_per_slice, self.__features_per_slice, slice_count)
        model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None
        if model_json and len(model_json) >0:
            m = model_from_json(jsonCore, custom_objects=custom_objects)
            m._name = submod_name

        if not m:
            closeIn = Input(tuple(merged_tensor.shape[1:]))
            x = closeIn

            dsize = int(math.sqrt(slice_count))
            if dsize*dsize < slice_count: dsize +=1
            seq = list(range(dsize))[1:]
            seq.reverse()

            for i in seq:
                x =Dropout(0.5, name='%s.dropout%d' % (submod_name, i))(x)
                x =Dense(self.__features_per_slice *i, name='%s.F%dx%d' % (submod_name, self.__features_per_slice, i))(x)

            x = self._feature88toOut(x)
            m = Model(inputs=closeIn, outputs=x, name=submod_name)

        self.__model_suplementals[m.name] =m

        for k, v in self.__model_suplementals.items():
            v.summary()
            # print(v.to_json())
            # v.save('/tmp/%s.h5' % k)
        
        x = m(merged_tensor)
        self.__modelId = 'DNN2D%dS%dF%dX.%s' %(self.__channels_per_slice, self.__features_per_slice, slice_count, self.__coreId)
        self._dnnModel = Model(inputs=layerIn, outputs=x, name=self.__modelId)

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
    def _buildup_core(self, input_tensor):
        '''
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_tensor = Input(tuple(input_tensor.shape[1:])) # create a brand-new input_tensor by getting rid of the leading dim-batch
        
        # a dummy core
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)

        # return Model(inputs=get_source_inputs(input_tensor), outputs=x, name='basesliced')
        return Model(input_tensor, outputs=x, name='basesliced')

    def save(self, filepath, saveWeights=True) :
        
        if not self.model: return False

        with h5py.File(filepath, 'w') as h5f:
            # step 1. save json model in h5f['model_config']
            model_json = self.model.to_json()
            g = h5f.create_group('model_config')
            g['model_clz'] = self.__class__.__name__.encode('utf-8')
            g['model_base'] = 'Model2D_Sliced'.encode('utf-8')
            g['model_core'] = self.__model_core.name.encode('utf-8')
            g['json_core'] = self.__json_core.encode('utf-8')
            input_shape = [int(x) for x in list(self.model.input.shape[1:])]
            g.create_dataset('input_shape', data=np.asarray(input_shape))
            BaseModel.save_attributes_to_hdf5_group(g, 'json_suplementals', [k.encode('utf-8') for k in self.__model_suplementals.keys()])
            for k, v in self.__model_suplementals.items() :
                g['subjson_%s' % k] = v.to_json().encode('utf-8')

            if saveWeights:
                g = h5f.create_group('model_weights')
                wg = g.create_group('weights_core')
                BaseModel.save_weights_to_hdf5_group(wg, self.__model_core.layers)

                wg = g.create_group('weights_suplementals')
                for k, v in self.__model_suplementals.items() :
                    subwg = wg.create_group(k)
                    BaseModel.save_weights_to_hdf5_group(subwg, v.layers)

        return True

    @staticmethod
    def load(filepath, custom_objects=None, withWeights=True):
        if h5py is None:
            raise ImportError('load() requires h5py')

        model = None
        with h5py.File(filepath, 'r') as h5f:
            # step 1. load json model defined in h5f['model_config']
            if 'model_config' in h5f:
                gconf = h5f['model_config']
                if 'json_core' not in gconf:
                    raise ValueError('model of %s has no model_json to init Model2D_Sliced' % filepath)

                model_base = gconf['model_base'][()].decode('utf-8') if 'model_base' in gconf else 'base'
                model_clz  = gconf['model_clz'][()].decode('utf-8') if 'model_clz' in gconf else ''
                if not model_base in ['Model2D_Sliced']:
                    raise ValueError('model[%s] of %s is not suitable to load via Model2D_Sliced' % (model_clz, filepath))

                input_shape = gconf['input_shape'][()]
                input_shape = tuple(list(input_shape))
                json_core = gconf['json_core'][()].decode('utf-8')
                json_sub = {}
                json_subnames = BaseModel.load_attributes_from_hdf5_group(gconf, 'json_suplementals')

                for n in json_subnames:
                    json_sub[n] = gconf['subjson_%s' % n][()].decode('utf-8')

                model = Model2D_Sliced()
                model.__buildup(json_core, json_sub, custom_objects, input_shape)

            if not model:
                return model

            # step 2. load weights in h5f['model_weights']
            if withWeights and 'model_weights' in h5f:
                model_weights_group = h5f['model_weights']
                model.load_weights_from_hdf5_group(model_weights_group)
            
            # step 3. by default, disable trainable
            for layer in model.model.layers:
                layer.trainable = False

        return model

    def load_weights_from_hdf5_group(self, group):
        return BaseModel.load_weights_from_hdf5_group_by_name(group, self.model.layers)


# --------------------------------
class Model2D_ResNet50Pre(Model2D_Sliced) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(Model2D_ResNet50Pre, self).__init__(outputClasses = outputClasses, **kwargs)

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        '''
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_shape = tuple([ int(x) for x in input_tensor.shape[1:]])
        return ResNet50(weights=None, classes=1000, input_shape=input_shape)

# --------------------------------
class Model2D_ResNet50(Model2D_Sliced) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(Model2D_ResNet50, self).__init__(outputClasses = outputClasses, **kwargs)

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        '''
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_tensor = Input(tuple(input_tensor.shape[1:])) # create a brand-new input_tensor by getting rid of the leading dim-batch

        bn_axis = 3
        classes = 1000
        pooling = 'max'

        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = Model2D_ResNet50.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = Model2D_ResNet50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = Model2D_ResNet50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = Model2D_ResNet50.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = Model2D_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = Model2D_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = Model2D_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = Model2D_ResNet50.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = Model2D_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = Model2D_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = Model2D_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = Model2D_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = Model2D_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = Model2D_ResNet50.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = Model2D_ResNet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = Model2D_ResNet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        # Create model.
        model = Model(input_tensor, x, name='resnet50') # model = Model(get_source_inputs(input_tensor), x, name='resnet50')

        '''
        # Load weights.
        if weights == 'imagenet':
            if include_top:
                weights_path = keras_utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = keras_utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path)
            if backend.backend() == 'theano':
                keras_utils.convert_all_kernels_in_model(model)
        elif weights is not None:
            model.load_weights(weights)
        '''
        return model

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        """
        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

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
    
    # # model = BaseModel.load('/tmp/test.h5')
    model = Model2D_Sliced.load('/tmp/2dsliced.h5')
    layer_names = model.enable_trainable("F88.Dense*")
    exit(0)

    model = Model2D_Sliced() # Model2D_ResNet50Pre, Model2D_ResNet50, Model2D_Sliced(), Model88_ResNet34d1(), Model88_Cnn1Dx4R2() Model88_VGG16d1 Model88_Cnn1Dx4R3
    model.buildup()
    model.compile()
    model.summary()
    # model.save('/tmp/test.h5')
    model.save('/tmp/2dsliced.h5')

    # cw = model.get_weights_core()
    # model.model.save('/tmp/%s.h5' % model.modelId) # model.save_model('/tmp/%s.h5' % model.modelId)

'''
with h5py.File('/tmp/M88F.Cnn1Dx4R2.h5', 'r') as h5f:
    h5f.visit(lambda x: print(x))

generated by original API: model.save('/tmp/%s.h5' % model.modelId):
model_weights
model_weights/F88.Dense1
model_weights/F88.Dense1/F88.Dense1
model_weights/F88.Dense1/F88.Dense1/bias:0
model_weights/F88.Dense1/F88.Dense1/kernel:0
model_weights/F88.Dense2
model_weights/F88.Dense2/F88.Dense2
model_weights/F88.Dense2/F88.Dense2/bias:0
model_weights/F88.Dense2/F88.Dense2/kernel:0
model_weights/F88.Dropout1
model_weights/core.Cnn1Dx4R2.batch_normalization
model_weights/core.Cnn1Dx4R2.batch_normalization/core.Cnn1Dx4R2.batch_normalization
model_weights/core.Cnn1Dx4R2.batch_normalization/core.Cnn1Dx4R2.batch_normalization/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization/core.Cnn1Dx4R2.batch_normalization/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization/core.Cnn1Dx4R2.batch_normalization/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization/core.Cnn1Dx4R2.batch_normalization/moving_variance:0
model_weights/core.Cnn1Dx4R2.batch_normalization_1
model_weights/core.Cnn1Dx4R2.batch_normalization_1/core.Cnn1Dx4R2.batch_normalization_1
model_weights/core.Cnn1Dx4R2.batch_normalization_1/core.Cnn1Dx4R2.batch_normalization_1/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization_1/core.Cnn1Dx4R2.batch_normalization_1/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization_1/core.Cnn1Dx4R2.batch_normalization_1/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization_1/core.Cnn1Dx4R2.batch_normalization_1/moving_variance:0
model_weights/core.Cnn1Dx4R2.batch_normalization_2
model_weights/core.Cnn1Dx4R2.batch_normalization_2/core.Cnn1Dx4R2.batch_normalization_2
model_weights/core.Cnn1Dx4R2.batch_normalization_2/core.Cnn1Dx4R2.batch_normalization_2/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization_2/core.Cnn1Dx4R2.batch_normalization_2/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization_2/core.Cnn1Dx4R2.batch_normalization_2/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization_2/core.Cnn1Dx4R2.batch_normalization_2/moving_variance:0
model_weights/core.Cnn1Dx4R2.batch_normalization_3
model_weights/core.Cnn1Dx4R2.batch_normalization_3/core.Cnn1Dx4R2.batch_normalization_3
model_weights/core.Cnn1Dx4R2.batch_normalization_3/core.Cnn1Dx4R2.batch_normalization_3/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization_3/core.Cnn1Dx4R2.batch_normalization_3/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization_3/core.Cnn1Dx4R2.batch_normalization_3/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization_3/core.Cnn1Dx4R2.batch_normalization_3/moving_variance:0
model_weights/core.Cnn1Dx4R2.batch_normalization_4
model_weights/core.Cnn1Dx4R2.batch_normalization_4/core.Cnn1Dx4R2.batch_normalization_4
model_weights/core.Cnn1Dx4R2.batch_normalization_4/core.Cnn1Dx4R2.batch_normalization_4/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization_4/core.Cnn1Dx4R2.batch_normalization_4/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization_4/core.Cnn1Dx4R2.batch_normalization_4/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization_4/core.Cnn1Dx4R2.batch_normalization_4/moving_variance:0
model_weights/core.Cnn1Dx4R2.batch_normalization_5
model_weights/core.Cnn1Dx4R2.batch_normalization_5/core.Cnn1Dx4R2.batch_normalization_5
model_weights/core.Cnn1Dx4R2.batch_normalization_5/core.Cnn1Dx4R2.batch_normalization_5/beta:0
model_weights/core.Cnn1Dx4R2.batch_normalization_5/core.Cnn1Dx4R2.batch_normalization_5/gamma:0
model_weights/core.Cnn1Dx4R2.batch_normalization_5/core.Cnn1Dx4R2.batch_normalization_5/moving_mean:0
model_weights/core.Cnn1Dx4R2.batch_normalization_5/core.Cnn1Dx4R2.batch_normalization_5/moving_variance:0
model_weights/core.Cnn1Dx4R2.conv1d
model_weights/core.Cnn1Dx4R2.conv1d/core.Cnn1Dx4R2.conv1d
model_weights/core.Cnn1Dx4R2.conv1d/core.Cnn1Dx4R2.conv1d/bias:0
model_weights/core.Cnn1Dx4R2.conv1d/core.Cnn1Dx4R2.conv1d/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_1
model_weights/core.Cnn1Dx4R2.conv1d_1/core.Cnn1Dx4R2.conv1d_1
model_weights/core.Cnn1Dx4R2.conv1d_1/core.Cnn1Dx4R2.conv1d_1/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_1/core.Cnn1Dx4R2.conv1d_1/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_2
model_weights/core.Cnn1Dx4R2.conv1d_2/core.Cnn1Dx4R2.conv1d_2
model_weights/core.Cnn1Dx4R2.conv1d_2/core.Cnn1Dx4R2.conv1d_2/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_2/core.Cnn1Dx4R2.conv1d_2/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_3
model_weights/core.Cnn1Dx4R2.conv1d_3/core.Cnn1Dx4R2.conv1d_3
model_weights/core.Cnn1Dx4R2.conv1d_3/core.Cnn1Dx4R2.conv1d_3/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_3/core.Cnn1Dx4R2.conv1d_3/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_4
model_weights/core.Cnn1Dx4R2.conv1d_4/core.Cnn1Dx4R2.conv1d_4
model_weights/core.Cnn1Dx4R2.conv1d_4/core.Cnn1Dx4R2.conv1d_4/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_4/core.Cnn1Dx4R2.conv1d_4/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_5
model_weights/core.Cnn1Dx4R2.conv1d_5/core.Cnn1Dx4R2.conv1d_5
model_weights/core.Cnn1Dx4R2.conv1d_5/core.Cnn1Dx4R2.conv1d_5/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_5/core.Cnn1Dx4R2.conv1d_5/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_6
model_weights/core.Cnn1Dx4R2.conv1d_6/core.Cnn1Dx4R2.conv1d_6
model_weights/core.Cnn1Dx4R2.conv1d_6/core.Cnn1Dx4R2.conv1d_6/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_6/core.Cnn1Dx4R2.conv1d_6/kernel:0
model_weights/core.Cnn1Dx4R2.conv1d_7
model_weights/core.Cnn1Dx4R2.conv1d_7/core.Cnn1Dx4R2.conv1d_7
model_weights/core.Cnn1Dx4R2.conv1d_7/core.Cnn1Dx4R2.conv1d_7/bias:0
model_weights/core.Cnn1Dx4R2.conv1d_7/core.Cnn1Dx4R2.conv1d_7/kernel:0
model_weights/core.Cnn1Dx4R2.dense
model_weights/core.Cnn1Dx4R2.dense/core.Cnn1Dx4R2.dense
model_weights/core.Cnn1Dx4R2.dense/core.Cnn1Dx4R2.dense/bias:0
model_weights/core.Cnn1Dx4R2.dense/core.Cnn1Dx4R2.dense/kernel:0
model_weights/core.Cnn1Dx4R2.dropout
model_weights/core.Cnn1Dx4R2.global_average_pooling1d
model_weights/core.Cnn1Dx4R2.max_pooling1d
model_weights/core.Cnn1Dx4R2.max_pooling1d_1
model_weights/core.Cnn1Dx4R2.max_pooling1d_2
model_weights/core.Cnn1Dx4R2.max_pooling1d_3
model_weights/core.Cnn1Dx4R2.max_pooling1d_4
model_weights/input_1
model_weights/reshape
'''


'''
model_config
model_config/input_shape
model_config/json_core
model_config/model_base
model_config/model_clz
model_config/model_core
model_config/subjson_2D4S0flow
model_config/subjson_2D4S1flow
model_config/subjson_2D4S518F2X
model_weights
model_weights/weights_core
model_weights/weights_core/bn_conv1
model_weights/weights_core/bn_conv1/pickled_weights
model_weights/weights_core/conv1
model_weights/weights_core/conv1/pickled_weights
model_weights/weights_suplementals
model_weights/weights_suplementals/2D4S0flow
model_weights/weights_suplementals/2D4S0flow/2D4S0flow.F518
model_weights/weights_suplementals/2D4S0flow/2D4S0flow.F518/pickled_weights
model_weights/weights_suplementals/2D4S1flow
model_weights/weights_suplementals/2D4S1flow/2D4S1flow.F518
model_weights/weights_suplementals/2D4S1flow/2D4S1flow.F518/pickled_weights
model_weights/weights_suplementals/2D4S518F2X
model_weights/weights_suplementals/2D4S518F2X/2D4S518F2X.F518x1
model_weights/weights_suplementals/2D4S518F2X/2D4S518F2X.F518x1/pickled_weights
model_weights/weights_suplementals/2D4S518F2X/F88.Dense1
model_weights/weights_suplementals/2D4S518F2X/F88.Dense1/pickled_weights
model_weights/weights_suplementals/2D4S518F2X/F88.Dense2
model_weights/weights_suplementals/2D4S518F2X/F88.Dense2/pickled_weights
'''