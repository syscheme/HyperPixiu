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
from tensorflow.keras.layers import BatchNormalization, Flatten, add, GlobalAveragePooling2D, Lambda, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs

from tensorflow.keras.applications.resnet50 import ResNet50

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
        x = Dropout(0.3, name='%sDropout1' %lnTag)(flattern_inputs) #  x= Dropout(0.5)(x)
        x = Dense(88, name='%sDense1' %lnTag)(x)
        x = Dense(self._classesOut,  name='%sDense2' %lnTag, activation='softmax')(x)
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
    def buildup(self, input_shape, input_tensor=None):
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)
            inputs = input_tensor
        else:
            inputs = get_source_inputs(input_tensor)

        core_model = self.build_core(input_shape, input_tensor) # layerIn.shape)
        core_mId = core_model.name
        self._tagCoreModel(core_model, core_mId)

        x = core_model(input_tensor)
        x = self._feature88toOut(x)
        # x = self._feature88toOut(core_model)

        self._dnnModel = Model(inputs=inputs, outputs=x, name=self.modelId)
        # self._dnnModel.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **BaseModel.COMPILE_ARGS)
        # self._dnnModel.summary()
        return self.model

    def _tagCoreModel(self, core_model, core_mId) :
        # add the prefix tag
        lnTag = core_mId
        if not lnTag or len(lnTag) <=0: lnTag= Model88.CORE_LAYER_PREFIX
        if Model88.CORE_LAYER_PREFIX != lnTag[:len(Model88.CORE_LAYER_PREFIX)]:
            lnTag = '%s%s' % (Model88.CORE_LAYER_PREFIX, lnTag)
        if '.' != lnTag[-1]:
            lnTag +='.'

        for layer in core_model.layers:
            if layer.name[:len(lnTag)] != lnTag:
                layer._name = lnTag + layer.name

    @abstractmethod
    def build_core(self, input_shape, input_tensor): # TODO: input_shape was supposed to get from input_tensor
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

    def buildup(self, input_shape=(1548, ), input_tensor=None):
        layerIn = Input(shape=input_shape)
        new_shape = (int(input_shape[0]/4), 4)
        x = Reshape(new_shape, input_shape=input_shape)(layerIn)
        m = super(Model88_Flat, self).buildup(new_shape, x)
        x = m(x)
        self._dnnModel = Model(inputs=get_source_inputs(layerIn), outputs=x)
        return self.model

# --------------------------------
class Model88_Cnn1Dx4R2(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R2, self).__init__(**kwargs)

    def build_core(self, input_shape, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)
            inputs = input_tensor
        else:
            inputs = get_source_inputs(input_tensor)

        x = Conv1D(128, 3, activation='relu')(input_tensor)
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

        return Model(inputs, x, name=self.coreId)

# --------------------------------
class Model88_Cnn1Dx4R3(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R3, self).__init__(**kwargs)

    def build_core(self, input_shape, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)

        x = Conv1D(128, 3, activation='relu')(input_tensor)
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

        return Model(input_tensor, x, name=self.coreId)

# --------------------------------
class Model88_VGG16d1(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_VGG16d1, self).__init__(**kwargs)

    def build_core(self, input_shape, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)

        weight_decay = 0.0005

        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
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

        return Model(input_tensor, x, name=self.coreId)

# --------------------------------
class Model88_ResNet34d1(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_ResNet34d1, self).__init__(**kwargs)

    def build_core(self, input_shape, input_tensor):
        if input_tensor is None:
            input_tensor = Input(shape=input_shape)

        weight_decay = 0.0005

        #第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
        #对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        # x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
        #conv1
        x= self._resBlk_basic(x, nb_filter=64, kernel_size=3, padding='valid')
        x= MaxPooling1D(2)(x)

        #conv2_x
        x = self._resBlk_bottleneck(x, nb_filters=[64,64,256], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(x, nb_filters=[64,64,256])
        x = self._resBlk_bottleneck(x, nb_filters=[64,64,256])

        #conv3_x
        x = self._resBlk_bottleneck(x, nb_filters=[128, 128, 512], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(x, nb_filters=[128, 128, 512])
        x = self._resBlk_bottleneck(x, nb_filters=[128, 128, 512])
        x = self._resBlk_bottleneck(x, nb_filters=[128, 128, 512])

        #conv4_x
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024])
        x = self._resBlk_bottleneck(x, nb_filters=[256, 256, 1024])

        #conv5_x
        x = self._resBlk_bottleneck(x, nb_filters=[512, 512, 2048], with_conv_shortcut=True)
        x = self._resBlk_bottleneck(x, nb_filters=[512, 512, 2048])
        x = self._resBlk_bottleneck(x, nb_filters=[512, 512, 2048])

        x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)

        return Model(input_tensor, x, name=self.coreId)

    def _resBlk_basic(self, x, nb_filter, kernel_size, padding='same', regularizer=None, name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv1D(nb_filter, kernel_size, padding=padding, activation='relu', name=conv_name, kernel_regularizer= regularizer)(x)
        x = BatchNormalization(name=bn_name)(x)
        return x

    def _resBlk_identity(self, inpt, nb_filter, kernel_size, with_conv_shortcut=False):
        x = self._resBlk_basic(inpt, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        x = self._resBlk_basic(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(inpt, nb_filter=nb_filter, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def _resBlk_bottleneck(self, inpt, nb_filters, with_conv_shortcut=False):
        k1,k2,k3=nb_filters
        x = self._resBlk_basic(inpt, nb_filter=k1, kernel_size=1, padding='same')
        x = self._resBlk_basic(x, nb_filter=k2, kernel_size=3, padding='same')
        x = self._resBlk_basic(x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(inpt, nb_filter=k3, kernel_size=1)
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
        self.__coreId = "NA"

    @property
    def modelId(self) :
        return 'M88S%d.%s' % (self.__channels_per_slice, self.coreId)
    
    @property
    def coreId(self) : return self.__coreId

    def __slice2d(x, idxSlice, channels_per_slice): 
        slice = x[:, :, :, idxSlice*channels_per_slice : (idxSlice+1)*channels_per_slice]
        ch2append = channels_per_slice - slice.shape[3] # [0]s to append to fit channel=4
        if ch2append >0:
            slice0s = np.zeros(tuple(list(slice.shape[:3]) +[ch2append])) # TODO fix this
            slice = np.concatenate((slice, slice0s), axis=2)
        return slice

    def __slice2d_flow(self, inputs, input_shape, core_model, idxSlice):
        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        x = Lambda(Model2D_Sliced.__slice2d, output_shape=slice_shape, arguments={'idxSlice':idxSlice, 'channels_per_slice': self.__channels_per_slice})(inputs)

        # common layers to feature-518
        lnTag = 'M88S%dflow%s.' % (self.__channels_per_slice, idxSlice)
        x = core_model(x)
        x =Flatten(name='%sflatten' %lnTag)(x)
        x =Dropout(0.3, name='%sdropout' %lnTag)(x)
        x =Dense(518, name='%sF518' %lnTag)(x)
        return x

    def buildup(self, input_shape=(32,32,8), input_tensor=None):
        layerIn = Input(shape=input_shape)

        channels = input_shape[2]
        slice_shape = tuple(list(input_shape[:2]) +[self.__channels_per_slice])
        slices = int(channels / self.__channels_per_slice)
        if 0 != channels % self.__channels_per_slice: slices +=1

        core_model = self.build_core(slice_shape, layerIn)
        self.__coreId = core_model.name
        self._tagCoreModel(core_model, self.__coreId)

        sliceflows = [None] * slices
        for i in range(slices):
            sliceflows[i] = self.__slice2d_flow(layerIn, input_shape, core_model, i)
        
        # merge the multiple flow-of-slice into a controllable less than F518*2
        x = Concatenate(axis=1, name='M88S4ConX%d' % slices)(sliceflows) # merge = merge(sliceflows, mode='concat') # concatenate([x1,x2,x3])

        dsize = int(math.sqrt(slices))
        if dsize*dsize < slices: dsize +=1
        seq = list(range(dsize))[1:]
        seq.reverse()

        for i in seq:
            x =Dropout(0.5,  name='M88S4M_dropout%d' % i)(x)
            x =Dense(518 *i, name='M88S4M_F518x%d' % i)(x)

        x = self._feature88toOut(x)
        self._dnnModel = Model(inputs=layerIn, outputs=x, name='%sx%d' %(self.modelId, slices))
        # self._dnnModel.compile(optimizer=Adam(lr=self._startLR, decay=1e-6), **BaseModel.COMPILE_ARGS)
        # self._dnnModel.summary()
        return self.model

    @abstractmethod
    def build_core(self, slice_shape, input_tensor):
        # if input_tensor is None:
        #     input_tensor = Input(shape=input_shape)

        #TODO: need to define a channel=4 module
        # refer to:
        # /usr/local/lib64/python3.6/site-packages/keras_applications/resnet50.py
        # /usr/local/lib/python3.6/site-packages/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
        # def ResNet50(include_top=True, ...
        core = ResNet50(weights=None, classes=1000, input_shape=slice_shape) # , input_tensor=input_tensor) # dummy code
        return core


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
    
    model = Model2D_Sliced() # Model88_Cnn1Dx4R2() # Model2D_Sliced()
    model.buildup()
    model.compile()
    model.summary()
    model.model.save('/tmp/%s.h5' % model.modelId) # model.save_model('/tmp/%s.h5' % model.modelId)
