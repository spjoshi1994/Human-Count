import inspect
import os

import numpy as np
import tensorflow as tf
import time
from binary_ops import binary_tanh
from binary_ops import binarize
from binary_ops import lin_8b_quant
from binary_ops import binary_wrapper
import config
from tensorflow.python.training import moving_averages
from tf_common import batch_norm_lat

def _variable_on_device(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(
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
        tf.add_to_collection('losses', weight_decay)
    return var

class MV2:
    def __init__(self, is_training):
        self.is_training = is_training

    def build(self, rgb):
        depth = [64, 64, 128, 256]
        bias_on = False
        ####################################################################
        # Quantization layers
        ####################################################################
        if True: # 16b weight (no quant); 8b activation
            fl_w_bin = 8
            fl_a_bin = 8
            ml_w_bin = 8
            ml_a_bin = 8
            sl_w_bin = 8
            # The last layer's activation (sl_a_bin) is always 16b

            min_rng =  0.0 # range of quanized activation
            max_rng =  2.0

        if False: # Stride 2
            self.conv1 = self._fire_layer('conv1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, stride=1)
        else: # Max pool
            self.conv1 = self._fire_layer('conv1', rgb, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, 
                                    min_rng=min_rng, max_rng=max_rng, stride=1)
        self.conv2 = self._mobile_layerv2('conv2', self.conv1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=True, min_rng=min_rng, max_rng=max_rng, expansion=1, bias_on = bias_on)
        self.conv3 = self._mobile_layerv2('conv3', self.conv2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    min_rng=min_rng, max_rng=max_rng, expansion=1, bias_on = bias_on)
        self.conv4 = self._mobile_layerv2('conv4', self.conv3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    min_rng=min_rng, max_rng=max_rng, expansion = 1, bias_on = bias_on)

    def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, 
                    pool_en=True, min_rng=-0.5, max_rng=0.5, stride=1):
        with tf.variable_scope(layer_name):
            ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=stride,
                padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=False) # <----
            tf.summary.histogram('before_bn', ex3x3)
            ex3x3 = self._batch_norm('bn', ex3x3) # <----
            tf.summary.histogram('before_relu', ex3x3)
            ex3x3 = binary_wrapper(ex3x3, layer_name, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
            tf.summary.histogram('after_relu', ex3x3)
            if pool_en:
                pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
            else:
                pool = ex3x3
                tf.summary.histogram('pool', pool)
            return pool

    def _mobile_layerv2(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, 
                        min_rng=-0.5, max_rng = 0.5, stride = 1, expansion=1, bias_on = True):
    
      with tf.variable_scope(layer_name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv1x1',reuse=tf.AUTO_REUSE):
                #expansion layer
                exp1x1 = self._conv_layer('conv1x1', inputs, filters=int(inputs.get_shape()[3])*expansion, 
                            size=1, stride=1, padding='SAME', stddev=stddev, freeze=freeze, relu=False, 
                            w_bin=w_bin, depthwise=False, bias_on = bias_on)
                tf.summary.histogram('1x1_before_bn', exp1x1)
                exp1x1 = self._batch_norm('1x1_bn', exp1x1)
                tf.summary.histogram('1x1_before_relu', exp1x1)
                exp1x1 = binary_wrapper(exp1x1, 'conv1x1', a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
                tf.summary.histogram('1x1_after_relu', exp1x1)

            with tf.variable_scope('dw_conv3x3',reuse=tf.AUTO_REUSE):
                #Depthwise layer
                ex3x3 = self._conv_layer('dw_conv3x3', exp1x1, filters=int(exp1x1.get_shape()[3]), size=3, 
                        stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, 
                        depthwise=True, bias_on = bias_on)
                tf.summary.histogram('dw_before_bn', ex3x3)
                ex3x3 = self._batch_norm('dw_bn', ex3x3) # <----
                tf.summary.histogram('dw_before_relu', ex3x3)
                ex3x3 = binary_wrapper(ex3x3, 'dw_conv3x3', a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
                tf.summary.histogram('dw_after_relu', ex3x3)

            with tf.variable_scope('proconv1x1',reuse=tf.AUTO_REUSE):
                pro1x1 = self._conv_layer('prooconv1x1', ex3x3, filters=oc, size=1, stride=1,padding='SAME', 
                            stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False, bias_on = bias_on)
                tf.summary.histogram('pro1x1_before_bn', pro1x1)
                pro1x1 = self._batch_norm('pro1x1_bn', pro1x1)
                tf.summary.histogram('pro1x1_before_relu', pro1x1)
                pro1x1 = binary_wrapper(pro1x1, 'prooconv1x1', a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
                tf.summary.histogram('pro1x1_after_relu', pro1x1)
        

            if inputs.shape!=pro1x1.shape:
                with tf.variable_scope('netconv1x1',reuse=tf.AUTO_REUSE):
                    net = self._conv_layer('netconv1x1', inputs, filters=oc, size=1, stride=1, padding='SAME', 
                            stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False, bias_on = bias_on)
                    tf.summary.histogram('net1x1_before_bn', net)
                    net = self._batch_norm('net1x1_bn', net)
                    tf.summary.histogram('net1x1_before_relu', net)
                    net = binary_wrapper(net, 'netconv1x1', a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
                    tf.summary.histogram('net1x1_after_relu', net)
                    inputs = net
            resd = tf.add(pro1x1,inputs)
            tf.summary.histogram('add_before_relu', resd)
            resd = binary_wrapper(resd, layer_name, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)
            tf.summary.histogram('add_after_relu', resd)

            if pool_en:
                pool = self._pooling_layer('pool', resd, size=2, stride=2, padding='SAME')
            else:
                pool = resd
            return pool

    def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, w_bin=16, stddev=0.001, depthwise=False, bias_on=True):
        """Convolutional layer operation constructor.

        Args:
        layer_name: layer name.
        inputs: input tensor
        filters: number of output filters.
        size: kernel size.
        stride: stride
        padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
        freeze: if true, then do not train the parameters in this layer.
        xavier: whether to use xavier weight initializer or not.
        relu: whether to use relu or not.
        stddev: standard deviation used for random weight initializer.
        Returns:
        A convolutional layer operation.
        """
        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3] # # of input channel

            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            if depthwise == False: # normal conv 2D
                kernel = _variable_with_weight_decay(
                    'kernels', shape=[size, size, int(channels), filters],
                    wd=0.0005, initializer=kernel_init, trainable=(not freeze))
                # kernel = tf.get_variable('filter', shape = [size, size, int(channels), filters])
            else: # depthwise conv
                # ignore filters parameter (# of ochannel since it's same to ichannel)
                assert int(channels) == filters, "DW conv's ic should be same to oc: {} vs. {}".format(int(channels), filters)
                kernel = _variable_with_weight_decay(
                    'kernels', shape=[size, size, int(channels), 1],
                    wd=0.0005, initializer=kernel_init, trainable=(not freeze))
                
            if w_bin == 1: # binarized conv
                kernel_bin = binarize(kernel)
                tf.summary.histogram('kernel_bin', kernel_bin)
                if depthwise == False:
                    conv = tf.nn.conv2d(inputs, kernel_bin, [1, stride, stride, 1], padding=padding, name='convolution')
                else: # DW CONV
                    conv = tf.nn.depthwise_conv2d(inputs, kernel_bin, [1, stride, stride, 1], padding=padding, name='convolution')
                conv_bias = conv
            elif w_bin == 8: # 8b quantization
                kernel_quant = lin_8b_quant(kernel)
                tf.summary.histogram('kernel_quant', kernel_quant)
                if depthwise == False:
                    conv = tf.nn.conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution')
                else: # DW CONV
                    conv = tf.nn.depthwise_conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution')
                if bias_on:
                    biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
                    biases_quant = lin_8b_quant(biases)
                    conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')
                    tf.summary.histogram('biases_quant', biases_quant)
                else:
                    conv_bias = conv
            else:
                if depthwise == False:
                    conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
                else: # DW CONV
                    conv = tf.nn.depthwise_conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
                if bias_on:
                    biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
                    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
                else:
                    conv_bias = conv
            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            return out
  
    def _pooling_layer(self, layer_name, inputs, size, stride, padding='SAME'):
            """Pooling layer operation constructor.

            Args:
            layer_name: layer name.
            inputs: input tensor
            size: kernel size.
            stride: stride
            padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
            Returns:
            A pooling layer operation.
            """

            with tf.variable_scope(layer_name) as scope:
                out =  tf.nn.max_pool(inputs, 
                                        ksize=[1, size, size, 1], 
                                        strides=[1, stride, stride, 1],
                                        padding=padding)
                return out

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))
            tf.summary.histogram('bn_gamma', gamma)
            tf.summary.histogram('bn_beta',  beta )

            control_inputs = []

            if self.is_training:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9)
                update_moving_var  = moving_averages.assign_moving_average(moving_variance, variance, 0.9)
                control_inputs = [update_moving_mean, update_moving_var]
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
        
            with tf.control_dependencies(control_inputs):
                y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
                y.set_shape(x.get_shape())
            return y
