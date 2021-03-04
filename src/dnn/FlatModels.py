# encoding: UTF-8

from __future__ import division
from abc import abstractmethod

from dnn.BaseModel import *

from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.layers as layers # import layers.Input, layers.Dense, layers.Activation, layers.Dropout, layers.Reshape, layers.Lambda, layers.Concatenate, layers.BatchNormalization, layers.Flatten, add
# from tensorflow.keras.layers import layers.Conv1D, layers.MaxPooling1D, layers.GlobalAveragePooling1D, layers.ZeroPadding1D
# from tensorflow.keras.layers import layers.ZeroPadding2D, layers.GlobalAveragePooling2D, layers.Conv2D, layers.MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs

########################################################################
# the old 1548 flat models
class Model88_Flat(Model88) :
    '''
    the basic 1548 flat models
    '''
    def __init__(self, **kwargs):
        super(Model88_Flat, self).__init__(input_shape=(1548, ), input_name='state1d', output_class_num =3, **kwargs)

    @property
    def modelId(self) :
        return 'state%dF88to%d%s.%s' % (self._input_shape[0], self._output_class_num, self.coreId)

    @abstractmethod
    def buildup(self) :
        tensor_in = layers.Input(shape=self._input_shape)
        new_shape = (int(self._input_shape[0]/4), 4)
        x = layers.Reshape(new_shape, input_shape=self._input_shape)(tensor_in)
        # m = super(Model88_Flat, self)._buildup_layers(new_shape, x)
        # x = m(x)
        x = self._buildup_layers(new_shape, x)
        self._dnnModel = Model(inputs=get_source_inputs(tensor_in), outputs=x, name= self.modelId)
        return self.model

# --------------------------------
class Model88_Cnn1Dx4R2(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R2, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_tensor):

        # x = layers.Conv1D(128, 3, activation='relu')(input_tensor)
        x = self._tagged_chain(lnTag, input_tensor, layers.Conv1D(128, 3, activation='relu'))

        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.3))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(100, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, layers.Dense(512, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

        return x  # return Model(input_tensor, x, name='%s%s' % (Model88.CORE_LAYER_PREFIX, self.coreId))

# --------------------------------
class Model88_Cnn1Dx4R3(Model88_Flat) :
    '''
    Model88 has a common 88 features at the end
    '''
    def __init__(self, **kwargs):
        super(Model88_Cnn1Dx4R3, self).__init__(**kwargs)

    def _buildup_core(self, lnTag, input_tensor):
        x = self._tagged_chain(lnTag, input_tensor, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.3))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(100, 3, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, layers.Dense(512, activation='relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        
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
        # x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, input_tensor, layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        
        #进行一次归一化
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.3))
        #layer2 32*32*64
        # x = layers.Conv1D(64, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(64, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

        #下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
        #padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))

        #layer3 16*16*64
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer4 16*16*128
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(128, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        
        #layer5 8*8*128
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer6 8*8*256
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer7 8*8*256
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(256, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))

        #layer8 4*4*256
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer9 4*4*512
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer10 4*4*512
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        
        #layer11 2*2*512
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer12 2*2*512
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer13 2*2*512
        # x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Conv1D(512, 3, padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.5))

        #layer14 1*1*512
        x = self._tagged_chain(lnTag, x, layers.Flatten())
        # x = self._tagged_chain(lnTag, x, layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

        #layer15 512
        # x = self._tagged_chain(lnTag, x, layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.Activation('relu'))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

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
        # x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, input_tensor, layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        #conv1
        x = self._resBlk_basic(lnTag, x, nb_filter=64, kernel_size=3, padding='valid')
        x = self._tagged_chain(lnTag, x, layers.MaxPooling1D(2))

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

        x = self._tagged_chain(lnTag, x, layers.GlobalAveragePooling1D())
        x = self._tagged_chain(lnTag, x, layers.Flatten())

        return x # return Model(input_tensor, x, name=self.coreId)

    def _resBlk_basic(self, lnTag, x, nb_filter, kernel_size, padding='same', regularizer=None, name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = self._tagged_chain(lnTag, x, layers.Conv1D(nb_filter, kernel_size, padding=padding, activation='relu', name=conv_name, kernel_regularizer= regularizer))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization(name=bn_name))
        return x

    def _resBlk_identity(self, lnTag, inpt, nb_filter, kernel_size, with_conv_shortcut=False):
        x = self._resBlk_basic(lnTag, inpt, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(lnTag, inpt, nb_filter=nb_filter, kernel_size=kernel_size)
            x = layers.add([x, shortcut])
            return x
        else:
            x = layers.add([x, inpt])
            return x

    def _resBlk_bottleneck(self, lnTag, inpt, nb_filters, with_conv_shortcut=False):
        k1,k2,k3=nb_filters
        x = self._resBlk_basic(lnTag, inpt, nb_filter=k1, kernel_size=1, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=k2, kernel_size=3, padding='same')
        x = self._resBlk_basic(lnTag, x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = self._resBlk_basic(lnTag, inpt, nb_filter=k3, kernel_size=1)
            x = layers.add([x, shortcut])
            return x
        else:
            x = layers.add([x, inpt])
            return x

