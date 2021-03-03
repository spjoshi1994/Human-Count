import tensorflow as tf
from tensorflow.python.training import moving_averages
from binary_ops import binary_tanh
from binary_ops import binarize
from binary_ops import lin_8b_quant
from binary_ops import binary_wrapper

xi = tf.contrib.layers.xavier_initializer
xic = tf.contrib.layers.xavier_initializer_conv2d

def batch_norm_lat(x, is_training):

    params_shape = [x.get_shape()[-1]]

    beta = tf.get_variable('beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

    control_inputs = []
    if is_training:
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


def conv2d(name, x, maps_in, maps_out, size=3, stride=1, act=tf.nn.relu, w_bin=16, a_bin=16, 
            min_rng=-0.5, max_rng=0.5, padding='SAME', is_training=False, max_pool = True):
    with tf.variable_scope(name):
        w = tf.get_variable(name="conv2d", shape=[size, size, maps_in, maps_out], initializer=xic())
        if w_bin == 1: # binarized conv
            kernel_bin = binarize(w)
            conv = tf.nn.conv2d(x, kernel_bin, [1, stride, stride, 1], padding=padding, name='convolution')
        elif w_bin == 8: # 8b quantization
            kernel_quant = lin_8b_quant(w)
            conv = tf.nn.conv2d(x, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution')
        else:
            # Perform convolution.
            conv = tf.nn.conv2d(x, w, strides = [1, stride, stride, 1], padding = padding)
        
        bn = batch_norm_lat(conv, is_training)
        l2 = tf.nn.l2_loss(w)

        if act is not None:
            bw = binary_wrapper(bn, name, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)
        else:
            bw = bn
            
        if max_pool == True:
            return tf.nn.max_pool(bw, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool"), l2  
        else:
            return bw, l2

def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)
