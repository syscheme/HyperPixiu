# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''
from __future__ import division
from abc import abstractmethod

from dnn.BaseModel import *

from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.layers as layers # import layers.Input, layers.Dense, layers.Activation, layers.Dropout, layers.Reshape, layers.Lambda, layers.Concatenate, layers.BatchNormalization, layers.Flatten, add
# from tensorflow.keras.layers import layers.Conv1D, layers.MaxPooling1D, layers.GlobalAveragePooling1D, layers.ZeroPadding1D
# from tensorflow.keras.layers import layers.ZeroPadding2D, layers.GlobalAveragePooling2D, layers.Conv2D, layers.MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_source_inputs

# from tensorflow.keras.backend import count_params
from tensorflow.keras.applications.resnet50 import ResNet50

########################################################################
class ModelS2d_ResNet50Pre(Model88_sliced) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, **kwargs):
        super(ModelS2d_ResNet50Pre, self).__init__(**kwargs)
        
        # self._maxY = 32
        self._dimMax = tuple([32] + list(self._dimMax[1:]))

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        '''d
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_shape = tuple([ int(x) for x in input_tensor.shape[1:]])
        return ResNet50(weights=None, classes=1000, input_shape=input_shape) #, dtype=INPUT_FLOAT

# --------------------------------
class ModelS2d_ResNet50(Model88_sliced) :
    '''
    2D models with channels expanded by channels=4
    additional autodecoder ref: https://github.com/Alvinhech/resnet-autoencoder/blob/master/autoencoder4.py
    https://blog.csdn.net/qq_42995327/article/details/110219613
    https://blog.csdn.net/nijiayan123/article/details/79416764?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
    '''
    def __init__(self, **kwargs):
        super(ModelS2d_ResNet50, self).__init__(**kwargs)

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        '''
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch

        bn_axis = 3
        classes = 1000
        pooling = 'max'

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

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

        x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        """
        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def deconv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        x = input_tensor
        for i in range(len(filters)):
            x = layers.Conv2DTranspose(filters[i], kernel_shape, activation='relu', padding='same', name='block%s_deconv%d' % (blkId, 1+i))(x)
        
        return x

# --------------------------------
class ModelS2d_VGG16r1(Model88_sliced) :
    '''
    2D models with channels expanded by channels=4
    /usr/local/lib64/python3.6/site-packages/keras_applications/vgg16.py
    additional autodecoder ref: https://github.com/Alvinhech/resnet-autoencoder/blob/master/autoencoder4.py
    '''
    def __init__(self, **kwargs):
        super(ModelS2d_VGG16r1, self).__init__(**kwargs)
        self._dimMax = (20, 32)
        self._encoder, self._decoder = None, None

    # def ResNet50(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    def _buildup_core(self, input_tensor):
        weight_decay = 0.0005
        lnTag = 'vgg16r1'

        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        dimMax = max(list(self._dimMax) + list(input_tensor.shape[1:2]))
        if min(list(input_tensor.shape[1:2])) < dimMax:
            x = self._tagged_chain(lnTag, x, layers.ZeroPadding2D(padding=( (0, dimMax-input_tensor.shape[1]), (0, dimMax-input_tensor.shape[2]) )))

        #layer1 (32,32,64)
        # 对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
        x = self._tagged_chain(lnTag, x, layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.3))

        #layer2 (16,16,64)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling2D(2))

        #layer3 (16,16,128)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer4 (8,8,128)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling2D(2))
        
        #layer5 (8,8,256)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer6 (8,8,256)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer7 (4,4,256)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling2D(2))

        #layer8 (4,4,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer9 (4,4,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))
        
        #layer10 (2,2,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling2D(2))
        
        #layer11 (2,2,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer12 (2,2,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.4))

        #layer13 (1,1,512)
        x = self._tagged_chain(lnTag, x, layers.Conv2D(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())
        x = self._tagged_chain(lnTag, x, layers.MaxPooling2D(2))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.5))

        #layer14 (features_per_slice)
        x = self._tagged_chain(lnTag, x, layers.Flatten())
        x = self._tagged_chain(lnTag, x, layers.Dense(self.features_per_slice, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

        #layer15 (features_per_slice)
        x = self._tagged_chain(lnTag, x, layers.Dense(self.features_per_slice, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(lnTag, x, layers.BatchNormalization())

        # Create model.
        model = Model(input_tensor, x, name=lnTag) 
        return model

    def __prune(x, output_shape=None): # for decoder to make output shape back like input's
        if not output_shape: output_shape = self._dimMax # TODO
        dims = len(output_shape)
        if 1 == dims:
            slice = x[:, :output_shape[0], :]
        elif 2 == dims :
            slice = x[:, :output_shape[0], :output_shape[1] :]

        return slice

    def _buildup_decoder(self, input_tensor):
        weight_decay = 0.0005
        decname = 'devgg16r1'
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor

        #layer14 (self.features_per_slice)
        x = self._tagged_chain(decname, x, layers.Dense(self.features_per_slice,  activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer13 (2,2,512)
        x = self._tagged_chain(decname, x, layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.Reshape((1,1,512))) 
        x = self._tagged_chain(decname, x, layers.UpSampling2D((2,2))) 

        #layer12 (2,2,512)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer11 (2,2,512)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer10 (4,4,512)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.UpSampling2D((2,2))) 

        #layer9 (4,4,512)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer8 (4,4,256)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(512, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer7 (8,8,256)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.UpSampling2D((2,2))) 

        #layer6 (8,8,256)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer5 (8,8,256)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(256, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer4 (16,16,128)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.UpSampling2D((2,2))) 

        #layer3 (16,16,128)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(128, 3, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())

        #layer2 (32,32,64)
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.UpSampling2D((2,2))) 

        #layer1 (32,32,4)
        x = self._tagged_chain(decname, x, layers.BatchNormalization())
        x = self._tagged_chain(decname, x, layers.Conv2DTranspose(self.channels_per_slice, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

        output_shape= tuple(list(self._dimMax) + [ self.channels_per_slice ])
        x = self._tagged_chain(decname, x, layers.Lambda(Model88_sliced._prune, arguments={'output_shape': output_shape}, output_shape= output_shape))
        return Model(input_tensor, x, name=decname) 

    def conv_block(input_tensor, kernel_shape, pool_shape, lst_filters, blkId) :
        """The identity block is the block that has no conv layer at shortcut.
        # Returns
            Output tensor for the block.
        """
        x = input_tensor
        for i in range(len(lst_filters)):
            x = layers.Conv2D(lst_filters[i], kernel_shape, activation='relu', padding='same', name='block%s_conv%d' % (blkId, 1+i))(x)
        
        if max(pool_shape) >1:
            x = layers.MaxPooling2D(pool_shape, strides=pool_shape, name='block%s_pool' % blkId)(x)
        return x

    def deconv_block(input_tensor, kernel_shape, pool_shape, lst_filters, blkId) :
        """The identity block is the block that has no conv layer at shortcut.
        # Returns
            Output tensor for the block.
        """
        x = input_tensor
        x = layers.UpSampling2D(pool_shape, name='block%s_depool' % blkId)(x)
        for i in range(len(lst_filters)):
            x = layers.Conv2DTranspose(lst_filters[i], kernel_shape, activation='relu', padding='same', name='block%s_deconv%d' % (blkId, 1+i))(x)
        
        return x

########################################################################
from dnn.ModelS1d import *
if __name__ == '__main__':
    
    model, fn_template, fn_weightsFrom = None, None, None
    # fn_template = '/tmp/test.h5'
    # model = BaseModel.load(fn_template)
    # fn_template = '/tmp/state18x32x4Y4F518x1To3action.resnet50r1.B32I32_init.h5' # '/tmp/sliced2d.h5'
    # fn_template = '/tmp/foo1d_autoenc_defoo1d.B32I32.h5'
    # fn_weightsFrom = '/mnt/e/AShareSample/state18x32x4Y4F518x1To3action.resnet50_trained-gpu1.20210208.h5'
    # fn_weightsFrom = '/mnt/d/wkspaces/HyperPixiu/out/Trainer/basic1d_autoenc_debasic1d_trained-last.h5'
    # fn_weightsFrom = '/mnt/d/wkspaces/HyperPixiu/out/Trainer/vgg16r1_autoenc_devgg16r1_trained-last.h5'
    
    
    if fn_template and len(fn_template) >0:
        # model = BaseModel.load(fn_template)
        model = Model88_sliced.load(fn_template)

    if not model:
        # model = ModelS2d_ResNet50Pre, ModelS2d_ResNet50, Model88_sliced2d(), Model88_ResNet34d1(), Model88_Cnn1Dx4R2() Model88_VGG16d1 Model88_Cnn1Dx4R3
        # model = ModelS2d_ResNet50(input_shape=(18, 32, 8), output_class_num=3, output_name='action') # forget ModelS2d_ResNet50r1
        model = ModelS2d_VGG16r1(input_shape=(18, 32, 4), output_class_num=3, output_name='a3', output_as_attr=True)
        
        # model = ModelS2d_ResNet50(input_shape=(18, 32, 8), output_class_num=8, output_name='gr8attr', output_as_attr=True)

        # model = ModelS1d_Basic(input_shape=(518, 8), output_class_num=8, output_name='gr8attr', output_as_attr=True)
        # model = ModelS1d_Basic(input_shape=(518, 4), output_class_num=3, output_name='a3')
        
        model.buildup_autoenc() 
        # model.buildup()

    if model and fn_weightsFrom and len(fn_weightsFrom) >0:
        trainables = model.enable_trainable("*")
        trainables = list(set(trainables))
        trainables.sort()
        print('enabled trainable on %d layers: %s' % (len(trainables), '\n'.join(trainables)))

        applied = model.load_weights(fn_weightsFrom, submodel_remap={'resnet50r1':'resnet50',})
        print('applied weights of %s onto %d layers: %s' % (fn_weightsFrom, len(trainables), '\n'.join(applied)))

    model.enable_trainable("state18x32x4Y4F518x1f0.*")
    model.enable_trainable("state18x32x4Y4F518x1C88*")
    model.compile()
    model.summary()

    # trainable_count = count_params(model.trainable_weights)
    # tw = tf.trainable_weights()
    # print('trainable %d vars: %s' % (len(tv), tw))
    # tw = model.model.trainable_weights
    # print('trainable %d weights: %s' % (count_params(tw), tw))

    # cw = model.get_weights_core()
    fn_save='/tmp/%s.B%sI%s.h5' % (model.modelId, BACKEND_FLOAT[5:], INPUT_FLOAT[5:])
    model.save(fn_save)
    print('saved model: %s' % fn_save)

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