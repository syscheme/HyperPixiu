# encoding: UTF-8

'''
A DQN Trainer detached from gymAgent to perform 'offline' training
It reads the ReplayBuffers, which was output from agentDQN, to train the model. Such a 'offline' trainning would help the online-agent to improve the loss/accurate of the model,
and can also distribute the training load outside of the online agent
'''
from __future__ import division
from abc import abstractmethod

import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np
import math
import h5py, fnmatch, os

AUTOENC_TAG = '_autoenc_'

# ----------------------------
# INDEPEND FROM HyperPX core classes: from MarketData import EXPORT_FLOATS_DIMS
BACKEND_FLOAT = 'float32' # float32(single-preccision) -3.4e+38 ~ 3.4e+38, float16(half~) 5.96e-8 ~ 6.55e+4, float64(double-preccision)
INPUT_FLOAT = 'float32'
# BACKEND_FLOAT = 'float16'
# INPUT_FLOAT = 'float16'

EXPORT_FLOATS_DIMS = 4
DUMMY_BIG_VAL = 999999
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

########################################################################
class Model88(BaseModel) :
    '''
    Model88 has a common 88 features at the end
    '''
    CORE_LAYER_PREFIX = 'core.'

    def __init__(self, input_shape, input_name='state', output_class_num=3, output_name='action', **kwargs):
        super(Model88, self).__init__(**kwargs)
        self._output_class_num = output_class_num
        self._output_name = output_name
        self._input_shape = input_shape
        self._input_name  = input_name
    
        self._output_as_attr = kwargs.get('output_as_attr', False)

    def _feature88toOut(self, flattern_inputs) :
        # unified final layers layers.Dense(VirtualFeature88) then layers.Dense(self._actionSize)
        lnTag ='F88' 
        if 3 != self._output_class_num:
            lnTag += 'o%d' % self._output_class_num
        if self._output_name:
            lnTag += '%s' % self._output_name

        if '.' != lnTag[-1]: lnTag+= '.'

        x = self._tagged_chain(lnTag, flattern_inputs, layers.Dense(888, name='%sD888_1' %lnTag))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.5, name='%sDropout1' %lnTag)) #  x= layers.Dropout(0.5))
        x = self._tagged_chain(lnTag, x, layers.Dense(888, name='%sD888_2' %lnTag))
        x = self._tagged_chain(lnTag, x, layers.Dropout(0.5, name='%sDropout2' %lnTag)) #  x= layers.Dropout(0.5))
        x = self._tagged_chain(lnTag, x, layers.Dense(88, name='%sD88' %lnTag))
        x = self._tagged_chain(lnTag, x, layers.Dense(self._output_class_num, name='%so%d' % (lnTag, self._output_class_num), activation='relu' if self._output_as_attr else 'softmax')) # classifying must take softmax
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
    def buildup(self):
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
            input_tensor = layers.Input(shape=input_shape, dtype=INPUT_FLOAT)
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
        layerIn = layers.Input(shapeIn)

        self._dnnModel = Model(inputs=layerIn, outputs=x)
        # sgd = SGD(lr=self._startLR, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, **BaseModel.COMPILE_ARGS)

        # TODO apply non-trainable feature88 weights

        return self.model

########################################################################
class Model88_sliced(Model88) :
    '''
    2D models with channels expanded by channels=4
    '''
    def __init__(self, input_shape, input_name='state', output_class_num=3, output_name='action', **kwargs):
        super(Model88_sliced, self).__init__(input_shape, input_name, output_class_num, output_name, **kwargs)
        self.__channels_per_slice =4
        self.__features_per_slice =518
        self.__coreId = "NA"
        self.__modelId = None
        self.__dictSubModels = {} # self.__dictSubModels[modelName] = {'model.json': json, 'model': model}
        self._dimMax = (20, 32)

    @property
    def modelId(self) :
        if self.__modelId: return self.__modelId
        return '%sxNTo%d%s.%s' %(self._fmtNamePrefix, self._output_class_num, self._output_name, self.coreId)
    
    @property
    def coreId(self) : return self.__coreId

    @property
    def channels_per_slice(self) : return self.__channels_per_slice
    @property
    def features_per_slice(self) : return self.__features_per_slice

    # def get_weights_core(self) : return self.__model_core.get_weights()

    def __slice(x, idxSlice, channels_per_slice): 
        dims = len(x.shape) -2
        if 1 == dims:
            slice = x[:, :, idxSlice*channels_per_slice : (idxSlice+1)*channels_per_slice]
        elif 2 == dims :
            slice = x[:, :, :, idxSlice*channels_per_slice : (idxSlice+1)*channels_per_slice]

        ch2append = channels_per_slice - slice.shape[-1] # [0]s to append to fit channel=4
        if ch2append >0:
            if 1 == dims:
                slice = tf.pad(slice, ((0,0),(0,0),(0,ch2append)))
            elif 2 == dims :
                slice = tf.pad(slice, ((0,0),(0,0),(0,0),(0,ch2append)))

        return slice

    def __slice_flow(self, submod_name, model_json, custom_objects, input_tensor, core_model):
        # common layers to self.__features_per_slice

        lnTag = submod_name + '.'
        tensor_flowClose = core_model(input_tensor)

        # construct the submodel
        m, json_m = None, None
        if model_json :
            m = model_from_json(model_json, custom_objects=custom_objects)
            m._name = submod_name

        if not m:
            flowCloseIn = layers.Input(tuple(tensor_flowClose.shape[1:]), dtype=INPUT_FLOAT)
            x =layers.Flatten(name='%sflatten' %lnTag)(flowCloseIn)
            x=layers.Dense(self.__features_per_slice, name='%sF%d_1' % (lnTag, self.__features_per_slice))(x)
            x =layers.Dropout(0.5, name='%sdropout' %lnTag)(x)
            x=layers.Dense(self.__features_per_slice, name='%sF%d_2' % (lnTag, self.__features_per_slice))(x)
            m = Model(inputs=flowCloseIn, outputs=x, name=submod_name)

        self.__dictSubModels[m.name] = {'model_json': m.to_json(), 'model': m}
        return m(tensor_flowClose)

    def buildup(self):
        return self.__buildup(None, None, None)

    @property
    def _fmtNamePrefix(self):
        return '%s%sY%dF%d' %(self._input_name, 'x'.join([str(x) for x in self._input_shape]), self.channels_per_slice, self.__features_per_slice)

    def _slicedInput(self):
        channels = self._input_shape[-1]
        shape_beforeCh = self._input_shape[:-1]
        dims = len(shape_beforeCh)
        strShape = 'x'.join([ str(x) for x in shape_beforeCh])

        if (self._dimMax[0] >0 and self._input_shape[0] > self._dimMax[0]) \
            or (dims >= 2 and self._input_shape[1] > self._dimMax[1] )  : # dimX
            raise ValueError('shape%s not allowed, must <=%s' % (self._input_shape, self._dimMax))

        slice_shape = tuple(list(self._input_shape[:dims]) +[ self.channels_per_slice ])
        slice_count = int(channels / self.channels_per_slice)
        if 0 != channels % self.channels_per_slice: slice_count +=1

        mNamePrefix = '%sx%d' %(self._fmtNamePrefix, slice_count)
        tensor_in = layers.Input(shape=self._input_shape, name='%s.input' % (mNamePrefix), dtype=INPUT_FLOAT)
        x = tensor_in # NEVER put Dropout here instantly after Input: x =layers.Dropout(0.5, name='%s.indropout' % mNamePrefix)(tensor_in)

        if 1 == dims and self._input_shape[0] < self._dimMax[0] : # 1D input
            x = layers.ZeroPadding1D(padding=((0, self._dimMax[0] - self._input_shape[0])), name='%s.%dpad%s' % (mNamePrefix, dims, strShape))(x)
        elif 2 == dims and (self._input_shape[0] < self._dimMax[0] or self._input_shape[1] < self._dimMax[1]): # 2D input
            x = layers.ZeroPadding2D(padding=((0, self._dimMax[0] - self._input_shape[0]), (0, self._dimMax[1] - self._input_shape[1])), name='%s.%dpad%s' % (mNamePrefix, dims, strShape))(x)
        slice_shape = tuple(list(x.shape[1:])[:dims] +[self.channels_per_slice])

        slices = [None] * slice_count
        for i in range(slice_count) :
            slices[i] = layers.Lambda(Model88_sliced.__slice, arguments={'idxSlice':i, 'channels_per_slice': self.channels_per_slice},
                                output_shape= slice_shape, name='%s.slice%d' %(mNamePrefix, i)
                                )(x)

        return tensor_in, slices

    def __buildup(self, core_name, jsonSubs, custom_objects):

        tensor_in, slices = self._slicedInput()

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

        slice_count = len(slices)
        sliceflows = [None] * slice_count
        mNamePrefix = '%sx%d' %(self._fmtNamePrefix, slice_count)

        for i in range(slice_count):
            submod_name = '%sf%d' %(mNamePrefix, i)
            model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None

            sliceflows[i] = self.__slice_flow(submod_name, model_json, custom_objects, slices[i], core_m)

        # merge the multiple flow-of-slice into a controllable less than F518*2
        merged_tensor = sliceflows[0] if 1 ==len(sliceflows) else layers.Concatenate(axis=1, name='%s.concat' %mNamePrefix)(sliceflows) # merge = merge(sliceflows, mode='concat')
        
        m, json_m = None, None
        submod_name = '%sC88' % mNamePrefix
        if 3 != self._output_class_num : submod_name +='o%d' % self._output_class_num
        if self._output_name: submod_name += self._output_name

        model_json = jsonSubs[submod_name] if isinstance(jsonSubs, dict) and submod_name in jsonSubs else None
        if model_json and len(model_json) >0:
            m = model_from_json(model_json, custom_objects=custom_objects)
            m._name = submod_name

        if not m:
            closeIn = layers.Input(tuple(merged_tensor.shape[1:]), dtype=BACKEND_FLOAT)
            x = closeIn

            dsize = int(math.sqrt(slice_count))
            if dsize*dsize < slice_count: dsize +=1
            seq = list(range(dsize))[1:]
            seq.reverse()

            for i in seq:
                x =layers.Dropout(0.5, name='%s.dropout%d' % (submod_name, i))(x)
                x =layers.Dense(self.__features_per_slice *i, name='%s.F%dx%d' % (submod_name, self.__features_per_slice, i))(x)

            x = self._feature88toOut(x)
            m = Model(inputs=closeIn, outputs=x, name=submod_name)

        self.__dictSubModels[m.name] = {'model_json': m.to_json(), 'model': m}

        # for k, v in self.__dictSubModels.items():
        #     v['model'].summary()
        #     print(v.to_json())
        #     # v.save('/tmp/%s.h5' % k)
        
        x = m(merged_tensor)
        self.__modelId = '%sx%dTo%d' %(self._fmtNamePrefix, slice_count, self._output_class_num)
        if self._output_name: self.__modelId += self._output_name
        self.__modelId += '.%s' % self.__coreId
        self._dnnModel = Model(inputs=tensor_in, outputs=x, name=self.__modelId)

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

    # @abstractmethod
    def _buildup_decoder(self, input_tensor) :
        raise NotImplementedError

    @abstractmethod
    def _buildup_core(self, input_tensor):
        '''
        unlike the Model88_Flat._buildup_core() returns the output_tensor, the sliced 2D models returns a submodel as core from _buildup_core()
        '''
        input_tensor = layers.Input(tuple(input_tensor.shape[1:])) # create a brand-new input_tensor by getting rid of the leading dim-batch
        
        # a dummy core
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)

        # return Model(inputs=get_source_inputs(input_tensor), outputs=x, name='basesliced')
        return Model(input_tensor, outputs=x, name='basesliced')

    def save(self, filepath, saveWeights=True) :
        
        if not self.model: return False

        with h5py.File(filepath, 'w') as h5f:
            # step 1. save json model in h5f['model_config']
            # NEVER here: model_json = self.model.to_json()
            g = h5f.create_group('model_config')
            BaseModel.hdf5g_setAttribute(g, 'model_clz', self.__class__.__name__)
            BaseModel.hdf5g_setAttribute(g, 'model_base', 'Model88_sliced')
            BaseModel.hdf5g_setAttribute(g, 'model_name', self.model.name)
            BaseModel.hdf5g_setAttribute(g, 'model_core', self.__coreId)
            BaseModel.hdf5g_setAttribute(g, 'input_name', self._input_name)
            BaseModel.hdf5g_setAttribute(g, 'output_name', self._output_name)

            input_shape = [int(x) for x in list(self.model.input.shape[1:])]
            BaseModel.hdf5g_setAttribute(g, 'input_shape_s', '(%s)' % ','.join(['%d'%x for x in input_shape]))
            g.create_dataset('input_shape', data=np.asarray(input_shape))

            output_shape = [int(x) for x in list(self.model.output.shape[1:])]
            BaseModel.hdf5g_setAttribute(g, 'output_shape_s', '(%s)' % ','.join(['%d'%x for x in output_shape]))
            g.create_dataset('output_shape', data=np.asarray(output_shape))

            g.create_dataset('dims_max', data=np.asarray(self._dimMax))

            BaseModel.hdf5g_setAttribute(g, 'json_suplementals', [k.encode('utf-8') for k in self.__dictSubModels.keys()])
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

                model_name = None
                try:
                    model_name = BaseModel.hdf5g_getAttribute(gconf, 'model_name')
                except:
                    model_name = None

                core_name = BaseModel.hdf5g_getAttribute(gconf, 'model_core')
                if not core_name or len(core_name)<0:
                    raise ValueError('model of %s has no model_core to init Model88_sliced' % filepath)

                model_base = BaseModel.hdf5g_getAttribute(gconf, 'model_base', 'base')
                model_clz  = BaseModel.hdf5g_getAttribute(gconf, 'model_clz', '')
                if not model_base in ['Model88_sliced']:
                    raise ValueError('model[%s] of %s is not suitable to load via Model88_sliced' % (model_clz, filepath))

                input_name  = BaseModel.hdf5g_getAttribute(gconf,  'input_name',  'state')
                output_name  = BaseModel.hdf5g_getAttribute(gconf, 'output_name', 'action')

                input_shape = gconf['input_shape'][()]
                input_shape = tuple(list(input_shape))

                output_shape = gconf['output_shape'][()] if 'output_shape' in gconf else None
                if output_shape is not None : output_shape = tuple(list(output_shape))

                dims_max = gconf['dims_max'][()] if 'dims_max' in gconf else None
                if dims_max is not None : dims_max = tuple(list(dims_max))

                json_subnames = BaseModel.hdf5g_getAttribute(gconf, 'json_suplementals')
                json_sub = {}

                for n in json_subnames:
                    json_sub[n] = gconf['subjson_%s' % n][()].decode('utf-8')

                model = Model88_sliced(input_shape=input_shape, input_name=input_name, output_name=output_name, output_class_num=output_shape[0])
                if dims_max: model._dimMax = dims_max

                if model_name and AUTOENC_TAG in model_name:
                     model.__buildup_autoenc(model_name, json_sub, custom_objects)
                else:
                    model.__buildup(core_name, json_sub, custom_objects)
                # unnecessary: model.__coreId = core_name

            if not model:
                return model

            # step 2. load weights in h5f['model_weights']
            g_subweights = None
            if withWeights and 'model_weights' in h5f:
                g_subweights = h5f['model_weights']
                g_subweights = g_subweights['sub_models'] if 'sub_models' in g_subweights else None

            if g_subweights:
                loadeds = model._load_weights_from_hdf5g(g_subweights)

        return model

    def load_weights(self, filepath, submodel_remap={}):
        
        lynames=[]
        if not self.model: return lynames
        with h5py.File(filepath, 'r') as h5f:
            if 'model_weights' not in h5f: return lynames

            g_subweights = h5f['model_weights']
            if 'sub_models' not in g_subweights : return lynames
            g_subweights = g_subweights['sub_models']

            lynames = model._load_weights_from_hdf5g(g_subweights, submodel_remap=submodel_remap)
        
        return lynames
    
    def _load_weights_from_hdf5g(self, group, trainable=False, import_weight=1.0, submodel_remap={}):
        ret = []
        subs=list(group.keys())
        for k, v in self.__dictSubModels.items() :
            if 'model' not in v or not v['model']: continue
            kn = k
            if kn in submodel_remap.keys(): kn =submodel_remap[kn]
            if kn not in group.keys(): continue
            
            m, subwg = v['model'], group[kn]
            loadeds = BaseModel._load_weights_from_hdf5g_by_name(subwg, m.layers, import_weight, prefix_remap=[('%s.'%k, '%s.'%kn)])
        
            # step 3. by default, disable trainable
            for layer in m.layers:
                layer.trainable = trainable

            ret += [ '%s/%s' %(k, x) for x in loadeds ]

        ret.sort()
        return ret

    def enable_trainable(self, layerNamePattern, enable=True) :

        ret = []
        for k, v in self.__dictSubModels.items() :
            if 'model' not in v or not v['model']: continue
            patt = layerNamePattern
            if fnmatch.fnmatch(k, patt) or fnmatch.fnmatch(k +'.', patt):
                # layerNamePattern matched the sub-model name, the entire sub-model trainable
                patt = '*'

            names = BaseModel.enable_trainable_layers(v['model'].layers, patt, enable=enable)
            ret += [ '%s/%s' %(k, x) for x in names ]
        return ret

    def buildup_autoenc(self):
        return self.__buildup_autoenc(None, None, None)

    def __buildup_autoenc(self, model_name, jsonSubs, custom_objects):
        
        tensor_in, slices = self._slicedInput()
        shape_echo = tensor_in.shape[1:]

        enc_name, dec_name = None, None
        if model_name and AUTOENC_TAG in model_name:
            loc = model_name.index(AUTOENC_TAG)
            enc_name = model_name[:loc]
            dec_name = model_name[loc+len(AUTOENC_TAG):].split('.')[0]

        class AutoEnc(Model) :
            def __init__(self, *args, **kwargs) :
                super(AutoEnc, self).__init__(*args, **kwargs)

            def fit(self, *args, **kwargs):
                # take x as y on the arguments
                if len(args) >=2: args[1] =args[0]
                if 'y' in kwargs :
                    if 'x' in kwargs: kwargs['y'] = kwargs['x']
                    elif len(args) >0: kwargs['y'] = args[0]

                return super(AutoEnc, self).fit(*args, **kwargs)

            def evaluate(self, *args, **kwargs):
                # take x as y on the arguments
                if len(args) >=2: args[1] =args[0]
                if 'y' in kwargs :
                    if 'x' in kwargs: kwargs['y'] = kwargs['x']
                    elif len(args) >0: kwargs['y'] = args[0]

                return super(AutoEnc, self).evaluate(*args, **kwargs)

            def summary(self, *args, **kwargs):
                return super(AutoEnc, self).summary(*args, **kwargs)

            def save(self, *args, **kwargs):
                return super(AutoEnc, self).save(*args, **kwargs)

        encoder, decoder = None, None
        
        json_m = None
        if not enc_name or enc_name not in jsonSubs:
            encoder = self._buildup_core(slices[0])
            json_m = encoder.to_json()
        else:
            encoder = model_from_json(jsonSubs[enc_name], custom_objects=custom_objects)
            json_m = jsonSubs[enc_name]

        if not encoder :
            raise ValueError('failed to create model_core')

        self.__dictSubModels[encoder.name] = {'model_json': json_m, 'model': encoder}
        self.__coreId = encoder.name

        encoded = encoder(slices[0])

        if not dec_name or dec_name not in jsonSubs:
            decoder = self._buildup_decoder(encoded)
            json_m = decoder.to_json()
        else:
            decoder = model_from_json(jsonSubs[dec_name], custom_objects=custom_objects)
            json_m = jsonSubs[dec_name]

        self.__dictSubModels[decoder.name] = {'model_json': json_m, 'model': decoder}

        decoded = decoder(encoded) # this is one slice here
        if len(slices) >1:
            decoded = layers.Concatenate(axis=-1)([decoded] + slices[1:])

        if shape_echo != decoded.shape[1:] :
            shape_echo = tuple(shape_echo)
            decoded = layers.Lambda(Model88_sliced._prune, arguments={'output_shape': shape_echo }, output_shape=shape_echo) (decoded)

        self.__modelId = '%s%s%s.%s' % (encoder.name, AUTOENC_TAG, decoder.name, 'x'.join([str(x) for x in list(tensor_in.shape[1:])]))
        self._dnnModel = AutoEnc(tensor_in, decoded, name=self.__modelId)

        self._defaultCompileArgs = {**self._defaultCompileArgs, 'loss':'mse'}
        self._defaultOptimizer = 'adam'

        return self.model

    def _prune(x, output_shape=None): # for decoder to make output shape back like input's
        if not output_shape: output_shape = list(self._dimMax) + [self.channels_per_slice]
        dims = len(output_shape)
        if 2 == dims:
            y = x[:, :output_shape[0], :output_shape[1]]
        elif 3 == dims :
            y = x[:, :output_shape[0], :output_shape[1], :output_shape[2] ]

        return y

