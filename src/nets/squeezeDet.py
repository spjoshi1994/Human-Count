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

    depth = [32, 24, 24, 48, 56, 100, 104]


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
        fire1 = self._fire_layer  ('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, stride=2)
    else: # Max pool
        fire1 = self._fire_layer  ('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, min_rng=min_rng, max_rng=max_rng, stride=1)
    fire2 = self._mobile_layer('fire2', fire1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng)
    if False: # stride 2
        fire3 = self._mobile_layer('fire3', fire2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, stride=2)
    else:    # Max pool
        fire3 = self._mobile_layer('fire3', fire2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, stride=1)
    fire4 = self._mobile_layer('fire4', fire3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng)
    fire5 = self._mobile_layer('fire5', fire4, oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
    fire6 = self._mobile_layer('fire6', fire5, oc=depth[5], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng)
    fire7 = self._mobile_layer('fire7', fire6, oc=depth[6], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
    fire_o = fire7

    ####################################################################

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer('conv12', fire_o, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=sl_w_bin)
    print('self.preds:', self.preds)

  def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, stride=1):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=stride,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin) # <----
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
