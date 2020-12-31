# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc

    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    depth = [32, 32, 32, 64, 128]
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
        fire1 = self._fire_layer  ('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, stride=1)
    else: # Max pool
        fire1 = self._fire_layer  ('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin,
                                   min_rng=min_rng, max_rng=max_rng, stride=1, bias_on=bias_on)
    fire2 = self._mobile_layerv2('fire2', fire1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                 min_rng=min_rng, max_rng=max_rng, expansion=1, bias_on=bias_on)
    fire3 = self._mobile_layerv2('fire3', fire2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                 min_rng=min_rng, max_rng=max_rng, expansion=1, bias_on=bias_on)
    fire4 = self._mobile_layerv2('fire4', fire3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                 pool_en=False, min_rng=min_rng, max_rng=max_rng, expansion = 1, bias_on=bias_on)
    fire5 = self._mobile_layerv2('fire5', fire4, oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                 min_rng=min_rng, max_rng=max_rng, expansion = 1, bias_on=bias_on)
    fire_o = fire5
    print(fire5.shape)
    ####################################################################

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer('conv12', fire_o, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=sl_w_bin, bias_on=bias_on)
    print('self.preds:', self.preds)

  def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, stride=1, bias_on=True):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=stride,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on) # <----
      tf.summary.histogram('before_bn', ex3x3)
      ex3x3 = self._batch_norm('bn', ex3x3) # <----
      tf.summary.histogram('before_relu', ex3x3)
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      tf.summary.histogram('after_relu', ex3x3)
      if pool_en:
        pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
        pool = ex3x3
      tf.summary.histogram('pool', pool)
      return pool

  def _mobile_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, stride=1):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('dw_conv3x3', inputs, filters=int(inputs.get_shape()[3]), size=3, stride=stride,
        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=True) # <----

      tf.summary.histogram('dw_before_bn', ex3x3)
      ex3x3 = self._batch_norm('dw_bn', ex3x3) # <----
      tf.summary.histogram('dw_before_relu', ex3x3)
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      tf.summary.histogram('dw_after_relu', ex3x3)

      if pool_en:
          pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
          pool = ex3x3
      tf.summary.histogram('dw_pool', pool)

      ex1x1 = self._conv_layer('conv1x1', pool, filters=oc, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False) # <----

      tf.summary.histogram('1x1_before_bn', ex1x1)
      ex1x1 = self._batch_norm('1x1_bn', ex1x1) # <----
      tf.summary.histogram('1x1_before_relu', ex1x1)
      ex1x1 = self.binary_wrapper(ex1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      tf.summary.histogram('1x1_after_relu', ex1x1)
      return ex1x1


  def _mobile_layerv2(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng = 0.5, stride = 1, expansion=1, bias_on=True):

      with tf.variable_scope(layer_name,reuse=tf.AUTO_REUSE):
      	#print(inputs.shape)

      	#expansion layer
      	exp1x1 = self._conv_layer('conv1x1', inputs, filters=int(inputs.get_shape()[3])*expansion, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False, bias_on=bias_on)
      	tf.summary.histogram('1x1_before_bn', exp1x1)
      	exp1x1 = self._batch_norm('1x1_bn', exp1x1)
      	tf.summary.histogram('1x1_before_relu', exp1x1)
      	exp1x1 = self.binary_wrapper(exp1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      	tf.summary.histogram('1x1_after_relu', exp1x1)

      	#Depthwise layer
      	ex3x3 = self._conv_layer('dw_conv3x3', exp1x1, filters=int(exp1x1.get_shape()[3]), size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=True, bias_on=bias_on)
      	tf.summary.histogram('dw_before_bn', ex3x3)
      	ex3x3 = self._batch_norm('dw_bn', ex3x3) # <----
      	tf.summary.histogram('dw_before_relu', ex3x3)
      	ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      	tf.summary.histogram('dw_after_relu', ex3x3)
      	pro1x1 = self._conv_layer('prooconv1x1', ex3x3, filters=oc, size=1, stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False, bias_on=bias_on)
      	tf.summary.histogram('pro1x1_before_bn', pro1x1)
      	pro1x1 = self._batch_norm('pro1x1_bn', pro1x1)
      	tf.summary.histogram('pro1x1_before_relu', pro1x1)
      	pro1x1 = self.binary_wrapper(pro1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      	tf.summary.histogram('pro1x1_after_relu', pro1x1)

      	if inputs.shape!=pro1x1.shape:
      		net = self._conv_layer('netconv1x1', inputs, filters=oc, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, depthwise=False, bias_on=bias_on)
      		tf.summary.histogram('net1x1_before_bn', net)
      		net = self._batch_norm('net1x1_bn', net)
      		tf.summary.histogram('net1x1_before_relu', net)
      		net = self.binary_wrapper(net, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      		tf.summary.histogram('net1x1_after_relu', net)
      		inputs = net
      		###print(net.shape)
      		#return tf.add(pro1x1,net)
      	resd = tf.add(pro1x1,inputs)
      	resd = self.binary_wrapper(resd, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu

      	if pool_en:
      		pool = self._pooling_layer('pool', resd, size=2, stride=2, padding='SAME')
      	else:
      		pool = resd
      	return pool






















