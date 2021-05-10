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

class TinyYolov2(ModelSkeleton):
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

    depth = [16, 32, 32, 64, 64,128]


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

    # Max pool
    # 224x224x3
    conv1 = self._Conv_BN_ReLU_Pool('conv1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, min_rng=min_rng, max_rng=max_rng)
    
    conv2 = self._Conv_BN_ReLU_Pool('conv2', conv1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng,pool_en=False) # remove pooling from here
    conv3 = self._Conv_BN_ReLU_Pool('conv3', conv2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng)
    conv4 = self._Conv_BN_ReLU_Pool('conv4', conv3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, min_rng=min_rng, max_rng=max_rng)
    conv5 = self._Conv_BN_ReLU_Pool('conv5', conv4, oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,  min_rng=min_rng, max_rng=max_rng)
    conv6 = self._Conv_BN_ReLU_Pool('conv6', conv5, oc=depth[5], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,  min_rng=min_rng, max_rng=max_rng,pool_en=False)
    #conv7 = self._Conv_BN_ReLU_Pool('conv7', conv6, oc=depth[6], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,  min_rng=min_rng, max_rng=max_rng,conv_size=1,pool_en=False)
    output = conv6

    ####################################################################

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer('conv12', output, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=sl_w_bin)
    print('self.preds:', self.preds)

  def _Conv_BN_ReLU_Pool(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, stride=1,conv_size=3):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=conv_size, stride=stride,
          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin) # <----
      ex3x3 = self._batch_norm('bn', ex3x3) # <----
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      if pool_en:
        pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
        pool = ex3x3
      return pool

  
