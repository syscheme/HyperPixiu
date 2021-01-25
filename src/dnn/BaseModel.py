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
# ----------------------------
# INDEPEND FROM HyperPX core classes: from MarketData import EXPORT_FLOATS_DIMS
EXPORT_FLOATS_DIMS = 4
DUMMY_BIG_VAL = 999999
NN_FLOAT = 'float32'
RFGROUP_PREFIX = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'
# ----------------------------

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as backend # from tensorflow.keras import backend
# import tensorflow.keras.engine.saving as saving # import keras.engine.saving as saving
import tensorflow as tf

import sys, os, platform, random, copy, threading
from datetime import datetime
from time import sleep

import h5py, tarfile, pickle, fnmatch
import numpy as np

# GPUs = backend.tensorflow_backend._get_available_gpus()
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [{'name':x.name, 'detail':x.physical_device_desc } for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()

if len(GPUs) >1:
    from keras.utils.training_utils import multi_gpu_model

########################################################################
class BaseModel(object) :

    COMPILE_ARGS ={
    'loss':'categorical_crossentropy', 
    # 'optimizer': sgd,
    'metrics':['accuracy']
    }

    def list_GPUs() :
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [{'name':x.name, 'detail':x.physical_device_desc } for x in local_device_protos if 'GPU' == x.device_type]

    GPUs = list_GPUs()

    def __init__(self, **kwargs):
        self._dnnModel = None
        self._program = kwargs.get('program', None)

    @property
    def program(self) :
        return self._program

    @property
    def modelId(self) :
        return self._dnnModel.name if self._dnnModel else 'NilModel'

    @property
    def model(self) : return self._dnnModel

    @property
    def input_shape(self) :
        return tuple(self._dnnModel.input_shape[1:]) if self._dnnModel else None

    @property
    def output_shape(self) :
        return tuple(self._dnnModel.output_shape[1:]) if self._dnnModel else None

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def fit_generator(self, *args, **kwargs):
        return self.model.fit_generator(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def compile(self, compireArgs=None, optimizer=None):
        if not self.model: return

        if not compireArgs or isinstance(compireArgs, dict) and len(compireArgs) <=0:
            compireArgs = BaseModel.COMPILE_ARGS
        if not optimizer:
            optimizer = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(optimizer=optimizer, **compireArgs)
        return self.model

    def enable_trainable(self, layerNamePattern, enable=True) :
        return BaseModel.enable_trainable_layers(self.model.layers, layerNamePattern, enable=enable)

    def save(self, filepath, saveWeights=True) :
        
        if not self.model: return False

        with h5py.File(filepath, 'w') as h5f:
            # step 1. save json model in h5f['model_config']
            model_json = self.model.to_json()
            g = h5f.create_group('model_config')
            BaseModel.hdf5g_setAttribute(g, 'model_clz', self.__class__.__name__)
            BaseModel.hdf5g_setAttribute(g, 'model_base', 'Model88_sliced2d')
            BaseModel.hdf5g_setAttribute(g, 'model_json', model_json.encode('utf-8'))

            input_shape = [int(x) for x in list(self.model.input.shape[1:])]
            BaseModel.hdf5g_setAttribute(g, 'input_shape_s', '(%s)' % ','.join(['%d'%x for x in input_shape]))
            g.create_dataset('input_shape', data=np.asarray(input_shape))

            if saveWeights:
                g = h5f.create_group('model_weights')
                BaseModel.save_weights_to_hdf5_group(g, self.model.layers)

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
                model_base = BaseModel.hdf5g_getAttribute(gconf, 'model_base', 'base')
                model_clz  = BaseModel.hdf5g_getAttribute(gconf, 'model_clz', '')

                if not model_base in [None, 'base', '', '*']:
                    raise ValueError('model[%s] of %s is not suitable to load via BaseModel' % (model_clz, filepath))

                input_shape = gconf['input_shape'][()]
                input_shape = tuple(list(input_shape))

                model_json  = BaseModel.hdf5g_getAttribute(gconf, 'model_json', '')
                if not model_json or len(model_json) <=0 :
                    raise ValueError('model of %s has no model_json to init BaseModel' % filepath)

                m = model_from_json(model_json, custom_objects=custom_objects)
                
                if not m:
                    raise ValueError('failed to build from model_json of %s' % filepath)

                model = BaseModel()
                model._dnnModel =m

            if not model:
                return model

            # step 2. load weights in h5f['model_weights']
            if withWeights and 'model_weights' in h5f:
                model_weights_group = h5f['model_weights']
                model._load_weights_from_hdf5g(model_weights_group)
            
            # step 3. by default, disable trainable
            for layer in model.model.layers:
                layer.trainable = False

        return model

    def _load_weights_from_hdf5g(self, group, trainable = False):
        ret = BaseModel._load_weights_from_hdf5g_by_name(group, self.model.layers)
        for layer in self.model.layers:
            layer.trainable = False

        return ret

    def enable_trainable_layers(layers, layerNamePattern, enable=True) :
        layer_names = []
        for layer in layers:
            if not fnmatch.fnmatch(layer.name, layerNamePattern) :
                continue

            layer.trainable = enable
            trainable_sz = len(layer.trainable_variables)
            if trainable_sz >1:
                layer_names.append('%s(%d)' % (layer.name, int(layer.trainable_variables[1].shape[0])))
        
        return layer_names

    def hdf5g_setAttribute(group, name, data):
        group.attrs[name] = data.encode('utf-8') if isinstance(data, str) else np.asarray(data)

    def hdf5g_getAttribute(group, name, defaultvalue=[]):
        data = defaultvalue
        if name in group.attrs:
            a = group.attrs[name]
            data = a.decode('utf-8') if isinstance(a, bytes) else [n.decode('utf8') for n in a]

        return data

    def save_weights_to_hdf5_group(group, layers):
        '''
        duplicated but simplized from /usr/local/lib/python3.6/...tensorflow/python/keras/engine/hdf5_format.py
        Saves the weights of a list of layers to a HDF5 group.
        group: HDF5 group.
        layers: List of layer instances.
        '''
        saved_layer_names=[]

        for layer in layers:
            weights = layer.get_weights()
            if not weights or len(weights) <=0:
                continue

            g = group.create_group(layer.name)
            pklweights = pickle.dumps(weights)
            npbytes = np.frombuffer(pklweights, dtype=np.uint8)
            param_dset = g.create_dataset('pickled_weights', data=npbytes) # unnecessary: not much to compress compression='gzip' lzf
            saved_layer_names.append(layer.name)

        BaseModel.hdf5g_setAttribute(group, 'layer_names', [n.encode('utf8') for n in saved_layer_names])
        return saved_layer_names

    def _load_weights_from_hdf5g_by_name(group, layers, name_pattern=None):
        '''
        duplicated but simplized from /usr/local/lib/python3.6/...tensorflow/python/keras/engine/hdf5_format.py
        Saves the weights of a list of layers to a HDF5 group.
        group: HDF5 group.
        layers: List of layer instances.
        '''

        loadeds=[]
        layer_names = BaseModel.hdf5g_getAttribute(group, 'layer_names')

        for layer in layers:
            if not layer.trainable or layer.name not in layer_names:
                continue

            if layer.name not in group.keys() or 'pickled_weights' not in group[layer.name].keys():
                # this is not a good model file
                continue

            pklweights = group[layer.name]['pickled_weights'][()].tobytes()
            weights = pickle.loads(pklweights)
            layer.set_weights(weights)
            loadeds.append('%s(%d)' %(layer.name, int(weights[1].shape[0])))
        
        return loadeds

    #---logging -----------------------
    def log(self, level, msg):
        if not self._program: return
        self._program.log(level, 'APP['+self.ident +'] ' + msg)

    def debug(self, msg):
        if not self._program: return
        self._program.debug('APP['+self.ident +'] ' + msg)
        
    def info(self, msg):
        '''正常输出'''
        if not self._program: return
        self._program.info('APP['+self.ident +'] ' + msg)

    def warn(self, msg):
        '''警告信息'''
        if not self._program: return
        self._program.warn('APP['+self.ident +'] ' + msg)
        
    def error(self, msg):
        '''报错输出'''
        if not self._program: return
        self._program.error('APP['+self.ident +'] ' + msg)

    def logexception(self, ex, msg=''):
        '''报错输出+记录异常信息'''
        self.error('%s %s: %s' % (msg, ex, traceback.format_exc()))

