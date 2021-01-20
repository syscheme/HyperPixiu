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
from dnn.Makeups  import BaseModel

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
        self._dnnModel = Model(inputs=get_source_inputs(layerIn), outputs=x, name= self.modelId)
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
class Model88_sliced2d(Model88) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(Model88_sliced2d, self).__init__(outputClasses = outputClasses, **kwargs)
        self.__channels_per_slice =4
        self.__features_per_slice =518
        self.__coreId = "NA"
        self.__dictSubModels = {} # self.__dictSubModels[modelName] = {'model.json': json, 'model': model}
        self.__sizeX = 32

    @property
    def modelId(self) :
        return 'M88S%d.%s' % (self.__channels_per_slice, self.coreId)
    
    @property
    def coreId(self) : return self.__coreId

    # def get_weights_core(self) : return self.__model_core.get_weights()

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

    def __slice2d_flow(self, submod_name, model_json, custom_objects, input_tensor, core_model):
        # common layers to self.__features_per_slice

        lnTag = submod_name + '.'
        tensor_flowClose = core_model(input_tensor)

        # construct the submodel
        m, json_m = None, None
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

        self.__dictSubModels[m.name] = {'model_json': m.to_json(), 'model': m}
        return m(tensor_flowClose)

    def buildup(self, input_shape=(16, 32, 8)):
        return self.__buildup(None, None, None, input_shape)

    def __buildup(self, core_name, jsonSubs, custom_objects, input_shape):
        if self.__sizeX != input_shape[1] or input_shape[0] >32 :
            raise ValueError('shape%s not allowed, must rows<=32 and columns=32' %input_shape)

        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        slice_count = int(channels / self.__channels_per_slice)
        if 0 != channels % self.__channels_per_slice: slice_count +=1
        mNamePrefix = 'S2d%dX%dF%dx%d' %(self.__channels_per_slice, self.__sizeX, self.__features_per_slice, slice_count)

        layerIn = Input(shape=input_shape, name='%s.I%s' % (mNamePrefix, 'x'.join([str(x) for x in input_shape])))
        x = layerIn

        if input_shape[0] <32: # padding Ys at the bottom
            x = ZeroPadding2D(padding=((0, 32 - input_shape[0]), 0), name='%s.padY%d' % (mNamePrefix, input_shape[0]))(x)
            slice_shape = tuple(list(x.shape[1:])[:2] +[self.__channels_per_slice])

        slices = [None] * slice_count
        for i in range(slice_count) :
            slices[i] = Lambda(Model88_sliced2d.__slice2d, arguments={'idxSlice':i, 'channels_per_slice': self.__channels_per_slice},
                                output_shape= slice_shape, name='%s.slice%d' %(mNamePrefix, i),
                                )(x)

        m, json_m = None, None
        if not core_name or core_name not in jsonSubs:
            m = self._buildup_core(slices[0])
            json_m = m.to_json()
        else:
            m = model_from_json(jsonSubs[core_name], custom_objects=custom_objects)
            json_m = jsonSubs[core_name]
        
        if not m :
            raise ValueError('failed to create model_core')

        self.__coreId = m.name
        self.__dictSubModels[self.__coreId] = {'model_json': json_m, 'model': m}
        # can be called at this moment: self.__model_core.save('/tmp/%s.h5' % self.__coreId)
        # m.summary()

        core_m =m
        # tagCore = '%s%s' % (Model88.CORE_LAYER_PREFIX, self.__coreId)
        # self.__tagCoreModel(self.__model_core, tagCore)

        sliceflows = [None] * slice_count
        for i in range(slice_count):
            submod_name = '%sf%d' %(mNamePrefix, i)
            model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None

            sliceflows[i] = self.__slice2d_flow(submod_name, model_json, custom_objects, slices[i], core_m)

        # merge the multiple flow-of-slice into a controllable less than F518*2
        merged_tensor = sliceflows[0] if 1 ==len(sliceflows) else Concatenate(axis=1, name='%s.concat' %mNamePrefix)(sliceflows) # merge = merge(sliceflows, mode='concat') # concatenate([x1,x2,x3])
        
        m, json_m = None, None
        submod_name = '%sC88' % mNamePrefix
        model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None
        if model_json and len(model_json) >0:
            m = model_from_json(model_json, custom_objects=custom_objects)
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

        self.__dictSubModels[m.name] = {'model_json': m.to_json(), 'model': m}

        # for k, v in self.__dictSubModels.items():
        #     v['model'].summary()
        #     print(v.to_json())
        #     # v.save('/tmp/%s.h5' % k)
        
        x = m(merged_tensor)
        self.__modelId = '%s.%s' %(mNamePrefix, self.__coreId)
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
            g['model_base'] = 'Model88_sliced2d'.encode('utf-8')
            input_shape = [int(x) for x in list(self.model.input.shape[1:])]
            g.create_dataset('input_shape', data=np.asarray(input_shape))

            g['model_core'] = self.__coreId.encode('utf-8')
            BaseModel.save_attributes_to_hdf5_group(g, 'json_suplementals', [k.encode('utf-8') for k in self.__dictSubModels.keys()])
            for k, v in self.__dictSubModels.items() :
                g['subjson_%s' % k] = v['model_json'].encode('utf-8')

            if saveWeights:
                g = h5f.create_group('model_weights')
                g = g.create_group('sub_models')
                for k, v in self.__dictSubModels.items() :
                    if 'model' not in v or not v['model']:
                        continue

                    subwg = g.create_group(k)
                    BaseModel.save_weights_to_hdf5_group(subwg, v['model'].layers)

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
                if 'model_core' not in gconf:
                    raise ValueError('model of %s has no model_core to init Model88_sliced2d' % filepath)

                model_base = gconf['model_base'][()].decode('utf-8') if 'model_base' in gconf else 'base'
                model_clz  = gconf['model_clz'][()].decode('utf-8') if 'model_clz' in gconf else ''
                if not model_base in ['Model88_sliced2d']:
                    raise ValueError('model[%s] of %s is not suitable to load via Model88_sliced2d' % (model_clz, filepath))

                input_shape = gconf['input_shape'][()]
                input_shape = tuple(list(input_shape))
                core_name  = gconf['model_core'][()].decode('utf-8')

                json_subnames = BaseModel.load_attributes_from_hdf5_group(gconf, 'json_suplementals')
                json_sub = {}

                for n in json_subnames:
                    json_sub[n] = gconf['subjson_%s' % n][()].decode('utf-8')

                model = Model88_sliced2d()
                model.__buildup(core_name, json_sub, custom_objects, input_shape)

            if not model:
                return model

            # step 2. load weights in h5f['model_weights']
            g_subweights = None
            if withWeights and 'model_weights' in h5f:
                g_subweights = h5f['model_weights']
                g_subweights = g_subweights['sub_models'] if 'sub_models' in g_subweights else None

            if g_subweights:
                laynames = model.load_weights_from_hdf5_group(g_subweights)

        return model

    def load_weights_from_hdf5_group(self, group, trainable=False):
        ret = []
        for k, v in self.__dictSubModels.items() :
            if 'model' not in v or not v['model']: continue
            if k not in group.keys(): continue
            
            m, subwg = v['model'], group[k]
            names = BaseModel.load_weights_from_hdf5_group_by_name(subwg, m.layers)
        
            # step 3. by default, disable trainable
            for layer in m.layers:
                layer.trainable = trainable

            ret += [ '%s/%s' %(k, x) for x in names ]

        return ret

    def enable_trainable(self, layerNamePattern, enable=True) :

        ret = []
        for k, v in self.__dictSubModels.items() :
            if 'model' not in v or not v['model']: continue
            names = BaseModel.enable_trainable_layers(v['model'].layers, layerNamePattern, enable=enable)
            ret += [ '%s/%s' %(k, x) for x in names ]
        return ret

# --------------------------------
class ModelS2d_ResNet50Pre(Model88_sliced2d) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(ModelS2d_ResNet50Pre, self).__init__(outputClasses = outputClasses, **kwargs)

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        '''d
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_shape = tuple([ int(x) for x in input_tensor.shape[1:]])
        return ResNet50(weights=None, classes=1000, input_shape=input_shape)

# --------------------------------
class ModelS2d_ResNet50(Model88_sliced2d) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, outputClasses =3, **kwargs):
        super(ModelS2d_ResNet50, self).__init__(outputClasses = outputClasses, **kwargs)

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

        x = ModelS2d_ResNet50.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = ModelS2d_ResNet50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = ModelS2d_ResNet50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = ModelS2d_ResNet50.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = ModelS2d_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = ModelS2d_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = ModelS2d_ResNet50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = ModelS2d_ResNet50.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = ModelS2d_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = ModelS2d_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = ModelS2d_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = ModelS2d_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = ModelS2d_ResNet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = ModelS2d_ResNet50.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = ModelS2d_ResNet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = ModelS2d_ResNet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        # Create model.
        model = Model(input_tensor, x, name='resnet50') # model = Model(get_source_inputs(input_tensor), x, name='resnet50')

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

########################################################################
if __name__ == '__main__':
    
    # # model = BaseModel.load('/tmp/test.h5')
    # model = Model88_sliced2d.load('/tmp/sliced2d.h5')
    # layer_names = model.enable_trainable("*")
    # exit(0)

    model = ModelS2d_ResNet50Pre() # ModelS2d_ResNet50Pre, ModelS2d_ResNet50, Model88_sliced2d(), Model88_ResNet34d1(), Model88_Cnn1Dx4R2() Model88_VGG16d1 Model88_Cnn1Dx4R3
    model.buildup()
    model.compile()
    model.summary()
    # model.save('/tmp/test.h5')
    model.save('/tmp/sliced2d.h5')

    # cw = model.get_weights_core()
    # model.model.save('/tmp/%s.h5' % model.modelId) # model.save_model('/tmp/%s.h5' % model.modelId)

'''
TO browse the hd5 file:

with h5py.File('/tmp/M88F.Cnn1Dx4R2.h5', 'r') as h5f:
    h5f.visit(lambda x: print(x))

generated by original API: model.save('/tmp/%s.h5' % model.modelId):
model_weights
model_weights/F88.Dense1
...
model_weights/weights_suplementals/2D4S518F2X/F88.Dense2/pickled_weights
'''