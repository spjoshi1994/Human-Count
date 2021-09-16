## Any function not related to 8b quant hasn't been tested so far. Also, init and regularizer are for expt purpose
from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    x = (0.5 * x) + 0.50001
    return K.clip(x, 0, 1)

def binary_tanh(x): # activation binarization to 1 and 0
    #return 2 * round_through(_hard_sigmoid(x)) - 1
    return round_through(_hard_sigmoid(x))

def binarize(W): # weight binarization to 1 and -1
    #Wb = binary_tanh(W)
    #return Wb
    return 2 * round_through(_hard_sigmoid(W)) - 1


def lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):
    #min_clip = K.round(min_rng*256/(max_rng-min_rng))
    #max_clip = K.round(max_rng*256/(max_rng-min_rng)-1)
    if min_rng==0.0 and max_rng==2.0:
        min_clip = 0
        max_clip = 255
    else:
        min_clip = -128
        max_clip = 127
    wq = 256.0 * w / (max_rng - min_rng)              # to expand [min, max] to [-128, 128]
    wq = K.round(wq)                                  # integer (quantization)
    wq = K.clip(wq, min_clip, max_clip)     # fit into 256 linear quantization
    wq = wq / 256.0 * (max_rng - min_rng)             # back to quantized real number, not integer
    wclip = K.clip(w, min_rng, max_rng)     # linear value w/ clipping
    return wclip + K.stop_gradient(wq - wclip)



class FixedDropout(tf.keras.layers.Dropout):
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.

    It is not possible to define FixedDropout class as global object,

    because we do not have modules for inheritance at first time.



    Issue:

        https://github.com/tensorflow/tensorflow/issues/30946

    """
    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:

            return self.noise_shape



        symbolic_shape = K.shape(inputs)

        noise_shape = [symbolic_shape[axis] if shape is None else shape

                       for axis, shape in enumerate(self.noise_shape)]

        return tuple(noise_shape)







def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.compat.v1.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.compat.v1.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.compat.v1.add_to_collection('losses', weight_decay)
  return var

def weight_init(weights=None, shape=None, dtype=None):
    return weights

import tensorflow.keras.initializers as wt_in


class MyInitializer(wt_in.Initializer):
    def __init__(self, name1="",  **kwargs):
        super(MyInitializer, self).__init__(**kwargs)
        self.name=name1+"kernels"

    def __call__(self, shape=None, dtype=None, **kwargs):
        kernel_init = tf.keras.initializers.TruncatedNormal(
            stddev=0.0001)
        kernel = _variable_with_weight_decay(
            name=self.name, shape=shape,
            wd=0.0001, initializer=kernel_init, trainable=True)

        kernel2 = lin_8b_quant(kernel)
        return kernel2

    def get_config(self):
        return {"name1":self.name}


class MyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,  **kwargs):
        super(MyRegularizer, self).__init__(**kwargs)

    def __call__(self, weights):
        return lin_8b_quant(weights)
        # min_rng = -0.5
        # max_rng = 0.5
        # min_clip = tf.math.rint(min_rng * 256 / (max_rng - min_rng))
        # max_clip = tf.math.rint(max_rng * 256 / (max_rng - min_rng) - 1)
        #
        # wq = 256.0 * weights / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
        # wq = tf.math.rint(wq)  # integer (quantization)
        # wq = tf.clip_by_value(wq, min_clip, max_clip)  # fit into 256 linear quantization
        # wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        # wclip = tf.clip_by_value(weights, min_rng, max_rng)  # linear value w/ clipping
        # return wclip + tf.stop_gradient(wq - wclip)

    def get_config(self):
        return {}

class MyConstraints(tf.keras.constraints.Constraint):
    def __init__(self,name="",  **kwargs):
        super(MyConstraints, self).__init__(**kwargs)
        self.name=name
    def __call__(self, w):
        with tf.compat.v1.variable_scope(self.name + "_CONSTRIANTS") as scope:
            return lin_8b_quant(w)
            # min_rng = -0.5
            # max_rng = 0.5
            # min_clip = -128
            # max_clip = 127
            # wq = 256.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
            # wq = K.round(wq)  # integer (quantization)
            # wq = K.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
            # wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
            # wclip = K.clip(w, min_rng, max_rng)  # linear value w/ clipping
            # return wclip + K.stop_gradient(wq - wclip)


    def get_config(self):
        return {"name":self.name}

### Added this class on Aug 20th
#@tf.keras.utils.register_keras_serializable(package='Custom')
class My_FC_Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,  **kwargs):
        super(My_FC_Regularizer, self).__init__(**kwargs)

    def __call__(self, w):
        min_clip = -32768.0
        max_clip = 32767.0
        min_rng=-32.0
        max_rng=32.0
        wq = 65536.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
        wq = K.round(wq)  # integer (quantization)
        wq = K.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
        wq = wq / 65536.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        wclip = K.clip(w, min_rng, max_rng)  # linear value w/ clipping
        out= wclip + K.stop_gradient(wq - wclip)
        return 0.01 * tf.math.reduce_sum(tf.math.abs(out))
    def get_config(self):
        return {}

#def cust_loss(y_true, y_pred):
#        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

class Keras_nn_loss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


#@tf.keras.utils.register_keras_serializable()
class CastToFloat32(preprocessing.PreprocessingLayer):
    def get_config(self):
        return super().get_config()

    def call(self, inputs):
        if inputs.dtype == tf.float32:
            return inputs
        if inputs.dtype == tf.string:
            return tf.strings.to_number(inputs, tf.float32)
        return tf.cast(inputs, tf.float32)

class TeraGhata(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
