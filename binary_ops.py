from __future__ import absolute_import

import tensorflow as tf
import tensorflow.keras.backend as K


def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.50001
    return K.clip(x, 0, 1)


def binary_tanh(x):  # activation binarization to 1 and 0
    return round_through(_hard_sigmoid(x))


def binarize(W):  # weight binarization to 1 and -1
    return 2 * round_through(_hard_sigmoid(W)) - 1


def lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):
    if min_rng == 0.0 and max_rng == 2.0:
        min_clip = 0
        max_clip = 255
    else:
        min_clip = -128
        max_clip = 127
    wq = 256.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
    wq = K.round(wq)  # integer (quantization)
    wq = K.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
    wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
    wclip = K.clip(w, min_rng, max_rng)  # linear value w/ clipping
    return wclip + K.stop_gradient(wq - wclip)


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
    def __init__(self, name1="", **kwargs):
        super(MyInitializer, self).__init__(**kwargs)
        self.name = name1 + "kernels"

    def __call__(self, shape=None, dtype=None, **kwargs):
        kernel_init = tf.keras.initializers.TruncatedNormal(
            stddev=0.0001)
        kernel = _variable_with_weight_decay(
            name=self.name, shape=shape,
            wd=0.0001, initializer=kernel_init, trainable=True)
        return lin_8b_quant(kernel)

    def get_config(self):
        return {"name1": self.name}


class MyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, goal_std=0.25, l2_lambda=0.5, **kwargs):
        super(MyRegularizer, self).__init__(**kwargs)
        self.goal_std = goal_std
        self.l2_lambda = l2_lambda

    def __call__(self, weights):
        kstd = tf.math.reduce_std(weights)
        std_goal = tf.constant(self.goal_std)
        reg = tf.math.subtract(kstd, std_goal)
        kstd_loss = tf.nn.l2_loss(reg) * self.l2_lambda
        return kstd_loss

    def get_config(self):
        return {"goal_std": self.goal_std, "l2_lambda": self.l2_lambda}


class MyConstraints(tf.keras.constraints.Constraint):
    def __init__(self, name="", **kwargs):
        super(MyConstraints, self).__init__(**kwargs)
        self.name = name

    def __call__(self, w):
        with tf.compat.v1.variable_scope(self.name + "_CONSTRIANTS") as scope:
            return lin_8b_quant(w)

    def get_config(self):
        return {"name": self.name}


class MyBNConstraints(tf.keras.constraints.Constraint):
    def __init__(self, name="", **kwargs):
        super(MyBNConstraints, self).__init__(**kwargs)
        self.name = name

    def __call__(self, w):
        return K.clip(w, 0.00003, 1.0)

    def get_config(self):
        return {"name": self.name}
