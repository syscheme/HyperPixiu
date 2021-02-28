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
BACKEND_FLOAT = 'float32' # float32(single-preccision) -3.4e+38 ~ 3.4e+38, float16(half~) 5.96e-8 ~ 6.55e+4, float64(double-preccision)
INPUT_FLOAT = 'float32'
# BACKEND_FLOAT = 'float16'
# INPUT_FLOAT = 'float16'

EXPORT_FLOATS_DIMS = 4
DUMMY_BIG_VAL = 999999
RFGROUP_PREFIX = 'ReplayFrame:'
RFGROUP_PREFIX2 = 'RF'
# ----------------------------

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# import tensorflow.keras.engine.saving as saving # import keras.engine.saving as saving
import tensorflow as tf

import sys, os, platform, random, copy, threading
from datetime import datetime
from time import sleep

import h5py, tarfile, pickle, fnmatch
import numpy as np

# # GPUs = backend.tensorflow_backend._get_available_gpus()
# def get_available_gpus():
#     from tensorflow.python.client import device_lib
#     local_device_protos = device_lib.list_local_devices()
#     return [{'name':x.name, 'detail':x.physical_device_desc } for x in local_device_protos if x.device_type == 'GPU']

# GPUs = get_available_gpus()

# if len(GPUs) >0: # didn't help on RTX 2080 tf 14.0+ cuda10.1
#     from tensorflow.keras import backend as backend # from tensorflow.keras import backend
#     os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
#     config = tf.ConfigProto()
#     # config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.9
#     backend.set_session(tf.Session(config=config))

########################################################################
class BaseModel(object) :

    def list_processors() :
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        CPUs, GPUs = [], []
        for x in local_device_protos :
            if 'GPU' == x.device_type :
                GPUs.append({'name':x.name, 'detail':x.physical_device_desc })
            elif 'CPU' == x.device_type :
                CPUs.append({'name':x.name, 'detail':x.physical_device_desc })

        return CPUs, GPUs

    CPUs, GPUs = list_processors()

    def __init__(self, **kwargs):
        self._dnnModel = None
        self._program = kwargs.get('program', None)
        self._startLR = kwargs.get('startLR', 0.01)

        if len(BaseModel.GPUs) >1:
            from keras.utils.training_utils import multi_gpu_model
            # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        if 'float32' != BACKEND_FLOAT : # and not (len(BaseModel.GPUs) >0 and 'Windows' in platform.platform()): # non-float32 excludes Anaconda-on-Win for GPU
            from tensorflow.keras import backend as backend # from tensorflow.keras import backend
            backend.set_floatx(BACKEND_FLOAT)
            backend.set_epsilon(1e-4)
            if self._program:
                self._program.warn('backend set to %s, epsilon[1e-4]' % (BACKEND_FLOAT))
            else: print('backend set to %s, epsilon[1e-4]' % (BACKEND_FLOAT))

        self._defaultOptimizer = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        self._defaultCompileArgs = {
                                    'loss':'categorical_crossentropy', 
                                    # 'optimizer': sgd,
                                    'metrics':['accuracy']
                                    }

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
            compireArgs = self._defaultCompileArgs
        if not optimizer:
            optimizer = self._defaultOptimizer

        self.model.compile(optimizer=optimizer, **compireArgs)
        return self.model

    def enable_trainable(self, layerNamePattern, enable=True) :
        return BaseModel.enable_trainable_layers(self.model.layers, layerNamePattern, enable=enable)

    def save(self, filepath, saveWeights=True) :
        
        if not self.model: return False

        with h5py.File(tmpfile, 'w') as h5f:
            # step 1. save json model in h5f['model_config']
            model_json = self.model.to_json()
            g = h5f.create_group('model_config')
            BaseModel.hdf5g_setAttribute(g, 'model_clz', self.__class__.__name__)
            BaseModel.hdf5g_setAttribute(g, 'model_base', 'Model88_sliced')
            BaseModel.hdf5g_setAttribute(g, 'model_json', model_json.encode('utf-8'))

            input_shape = [int(x) for x in list(self.model.input.shape[1:])]
            BaseModel.hdf5g_setAttribute(g, 'input_shape_s', '(%s)' % ','.join(['%d'%x for x in input_shape]))
            g.create_dataset('input_shape', data=np.asarray(input_shape))

            output_shape = [int(x) for x in list(self.model.output.shape[1:])]
            BaseModel.hdf5g_setAttribute(g, 'output_shape_s', '(%s)' % ','.join(['%d'%x for x in output_shape]))
            g.create_dataset('output_shape', data=np.asarray(output_shape))

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

                output_shape = gconf['output_shape'][()] if 'output_shape' in gconf else None
                if output_shape : output_shape = tuple(list(output_shape))

                model_json  = BaseModel.hdf5g_getAttribute(gconf, 'model_json', '')
                if not model_json or len(model_json) <=0 :
                    raise ValueError('model of %s has no model_json to init BaseModel' % filepath)

                m = model_from_json(model_json, custom_objects=custom_objects)
                
                if not m:
                    raise ValueError('failed to build from model_json of %s' % filepath)

                model = BaseModel()
                if output_shape : model._classesOut = output_shape[0]
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

    def load_weights(self, filepath, import_weight=1.0):
        
        lynames=[]
        if not self.model: return lynames
        with h5py.File(filepath, 'r') as h5f:
            if 'model_weights' not in h5f: return lynames

            g_subweights = h5f['model_weights']
            lynames = model._load_weights_from_hdf5g(g_subweights, import_weight=import_weight)
        
        return lynames
    
    def _load_weights_from_hdf5g(self, group, trainable = False, import_weight=1.0):
        ret = BaseModel._load_weights_from_hdf5g_by_name(group, self.model.layers, import_weight)
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
            # if isinstance(a, str) : data = a 
            # elif isinstance(a, bytes) : data = a.decode('utf-8')
            # elif isinstance(a, list) :
            #     for n in a:
            #         if isinstance(n, str) : data.append(n)
            #         elif isinstance(n, bytes) : data.append(n.decode('utf8'))

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

    def _load_weights_from_hdf5g_by_name(group, layers, name_pattern=None, import_weight=1.0, prefix_remap=[]):
        '''
        duplicated but simplized from /usr/local/lib/python3.6/...tensorflow/python/keras/engine/hdf5_format.py
        Saves the weights of a list of layers to a HDF5 group.
        group: HDF5 group.
        layers: List of layer instances.
        import_weight: (0, 1.0] how much to apply the weights onto the existing layer, each result weight would be (weight_from_h5 * import_weight + weight_of_current* (1 -import_weight))
        '''
        if import_weight<0.0: import_weight =0.0
        elif import_weight >1.0: import_weight =1.0
        w_keep = 1.0 - import_weight

        loadeds=[]
        layer_names = BaseModel.hdf5g_getAttribute(group, 'layer_names')

        for layer in layers:
            ln2load = layer.name
            for rem in prefix_remap:
                if len(rem) >1 and ln2load[:len(rem[0])] == rem[0] :
                    ln2load = rem[1] + ln2load[len(rem[0]):]

            if not layer.trainable or ln2load not in layer_names:
                continue

            if ln2load not in group.keys() or 'pickled_weights' not in group[ln2load].keys():
                # this is not a good model file
                continue

            pklweights = group[ln2load]['pickled_weights'][()].tobytes()
            weights_to_import = pickle.loads(pklweights)

            weights_to_merge = [layer.get_weights(), weights_to_import]
            if weights_to_merge[0][0].shape != weights_to_merge[1][0].shape:
                continue # shape didn't match although same naming
            weightsResult = []

            for t in zip(*weights_to_merge):
                # weightsResult.append( [numpy.array(item).mean(axis=0) for item in zip(*t)])
                weightsResult.append( np.array([np.array(item[0]*w_keep + item[1]*import_weight) for item in zip(*t)]) )
            layer.set_weights(weightsResult)

            loadeds.append('%s(%d)x%s' %(layer.name, int(weights_to_import[1].shape[0]), import_weight))
        
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

