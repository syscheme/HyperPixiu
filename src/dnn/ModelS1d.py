from __future__ import division
from abc import abstractmethod

from dnn.BaseModel import *

import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers

########################################################################
class ModelS1d_Basic(Model88_sliced) :
    '''
    '''
    def __init__(self, **kwargs):
        super(ModelS1d_Basic, self).__init__(**kwargs)
        self._dimMax = tuple([518])

    def _buildup_core(self, input_tensor):

        corename = 'basic1d'
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        
        x = self._tagged_chain(corename, x, layers.Dense(self.channels_per_slice, activation='relu')) # (518,4)
        x = self._tagged_chain(corename, x, layers.Flatten()) # (518*4)
        x = self._tagged_chain(corename, x, layers.Dropout(0.3))
        x = self._tagged_chain(corename, x, layers.Dense(518,  activation='relu')) # (64)
        x = self._tagged_chain(corename, x, layers.Dropout(0.3))
        x = self._tagged_chain(corename, x, layers.Dense(518,  activation='relu')) # (64)
        x = self._tagged_chain(corename, x, layers.Dropout(0.3))

        x = self._tagged_chain(corename, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)

        # Create model.
        model = Model(input_tensor, x, name=corename) 
        return model

    def _buildup_decoder(self, input_tensor):
        decname = 'debasic1d'
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        x = self._tagged_chain(decname, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)
        x = self._tagged_chain(decname, x, layers.Dense(518,  activation='relu')) # (518)
        x = self._tagged_chain(decname, x, layers.Dense(518,  activation='relu')) # (518)
        x = self._tagged_chain(decname, x, layers.Dense(518*4,  activation='relu'))  # (518*4)
        x = self._tagged_chain(decname, x, layers.Reshape((518, 4))) # (518,4)
        x = self._tagged_chain(decname, x, layers.Dense(self.channels_per_slice, activation='relu')) # (518,4)

        return Model(input_tensor, x, name=decname) 

# --------------------------------
class ModelS1d_Dense1(Model88_sliced) :
    '''
    '''
    def __init__(self, **kwargs):
        super(ModelS1d_Dense1, self).__init__(**kwargs)
        self._dimMax = tuple([518])
        self.__coreid_ = 'dense1d'

    def _buildup_core(self, input_tensor):

        mname = self.__coreid_
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        
        x = self._tagged_chain(mname, x, layers.Dense(self.channels_per_slice, activation='relu')) # (518, 4)
        x = self._tagged_chain(mname, x, layers.Dense(16, activation='relu')) # (518, 16)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dense(8, activation='relu')) # (518, 16)
        x = self._tagged_chain(mname, x, layers.Flatten()) # (518*8)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.5))
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))

        x = self._tagged_chain(mname, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)

        # Create model.
        model = Model(input_tensor, x, name=mname) 
        return model

    def _buildup_decoder(self, input_tensor):
        mname = 'de' + self.__coreid_
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        x = self._tagged_chain(mname, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(518*8,  activation='relu'))  # (518*4)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Reshape((518, 8))) # (518,2)
        x = self._tagged_chain(mname, x, layers.Dense(8, activation='relu')) # (518,4)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dense(16, activation='relu')) # (518, 4)
        x = self._tagged_chain(mname, x, layers.Dense(self.channels_per_slice, activation='relu')) # (518,4)

        return Model(input_tensor, x, name=mname) 

# --------------------------------
class ModelS1d_Dense2(Model88_sliced) :
    '''
    '''
    def __init__(self, **kwargs):
        super(ModelS1d_Dense2, self).__init__(**kwargs)
        self._dimMax = tuple([518])
        self.__coreid_ = 'dense1d2'

    def _buildup_core(self, input_tensor):

        mname = self.__coreid_
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        
        x = self._tagged_chain(mname, x, layers.Dense(64, activation='relu')) # (518, 4)
        x = self._tagged_chain(mname, x, layers.Dense(64, activation='relu')) # (518, 16)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dense(16, activation='relu')) # (518, 16)
        x = self._tagged_chain(mname, x, layers.Flatten()) # (518*8)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.5))

        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))

        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))

        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))

        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dropout(0.3))

        x = self._tagged_chain(mname, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)

        # Create model.
        model = Model(input_tensor, x, name=mname) 
        return model

    def _buildup_decoder(self, input_tensor):
        mname = 'de' + self.__coreid_
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        x = self._tagged_chain(mname, x, layers.Dense(self.features_per_slice,  activation='relu')) # (self.features_per_slice)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        # x = self._tagged_chain(mname, x, layers.BatchNormalization())

        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(512,  activation='relu')) # (64)
        # x = self._tagged_chain(mname, x, layers.BatchNormalization())

        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        # x = self._tagged_chain(mname, x, layers.BatchNormalization())

        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.Dense(1024,  activation='relu')) # (518)
        x = self._tagged_chain(mname, x, layers.BatchNormalization())

        x = self._tagged_chain(mname, x, layers.Reshape((518, 16))) # (518,2)
        x = self._tagged_chain(mname, x, layers.Dense(16, activation='relu')) # (518,4)
        x = self._tagged_chain(mname, x, layers.Dense(64, activation='relu')) # (518, 4)
        x = self._tagged_chain(mname, x, layers.Dense(64, activation='relu')) # (518, 16)
        # x = self._tagged_chain(mname, x, layers.BatchNormalization())
        x = self._tagged_chain(mname, x, layers.Dense(self.channels_per_slice, activation='relu')) # (518,4)

        return Model(input_tensor, x, name=mname) 

# --------------------------------
class ModelS1d_Cnn1Dr2(Model88_sliced) :
    '''
    '''
    def __init__(self, **kwargs):
        super(ModelS1d_Cnn1Dr2, self).__init__(**kwargs)
        self._dimMax = tuple([518])

    def _buildup_core(self, input_tensor):

        corename = 'cnn1dr2'
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        
        x = self._tagged_chain(corename, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())
        x = self._tagged_chain(corename, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(corename, x, layers.Conv1D(512, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())
        x = self._tagged_chain(corename, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(corename, x, layers.Dropout(0.3))
        x = self._tagged_chain(corename, x, layers.Conv1D(256, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())
        x = self._tagged_chain(corename, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(corename, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())
        x = self._tagged_chain(corename, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(corename, x, layers.Conv1D(128, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())
        x = self._tagged_chain(corename, x, layers.MaxPooling1D(2))
        x = self._tagged_chain(corename, x, layers.Conv1D(100, 3, activation='relu'))
        x = self._tagged_chain(corename, x, layers.GlobalAveragePooling1D())

        x = self._tagged_chain(corename, x, layers.Dense(518, activation='relu'))
        x = self._tagged_chain(corename, x, layers.BatchNormalization())

        # Create model.
        model = Model(input_tensor, x, name=corename) 
        return model

    def _buildup_decoder(self, input_tensor):
        decname = 'decnn1dr2'
        input_tensor = layers.Input(tuple(input_tensor.shape[1:]), dtype=INPUT_FLOAT) # create a brand-new input_tensor by getting rid of the leading dim-batch
        x = input_tensor
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(100, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.UpSampling1D(2))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(128, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.UpSampling1D(2))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(128, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.UpSampling1D(2))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(256, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.UpSampling1D(2))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(256, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(512, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.UpSampling1D(2))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(256, 3, activation='relu'))
        x = self._tagged_chain(decname, x, layers.Conv1DTranspose(128, 3, activation='relu'))

        return Model(input_tensor, x, name=decname) 
