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

  def _create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
      '''
      :param name: A string. The name of the new variable
      :param shape: A list of dimensions
      :param initializer: User Xavier as default.
      :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
      layers.
      :return: The created variable
      '''
      
      ## TODO: to allow different weight decay to fully connected layer and conv layer
      regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

      new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                      regularizer=regularizer)
      return new_variables

  def _ResNetBlock(self, layer_name, inputs, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, depth1=-1, depth2=-1):
    with tf.variable_scope(layer_name):
      
      ex3x3 = self._conv_layer('conv3by3_0', inputs, filters=depth1, size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin)
      ex3x3 = self._batch_norm('bn0', ex3x3) # <----
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      ex3x3 = self._conv_layer('conv3by3_1', ex3x3, filters=depth2, size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin)      
      ex3x3 = self._batch_norm('bn1', ex3x3) # <----      
      ex3x3 = self.binary_wrapper(ex3x3+inputs, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      
      if pool_en:
          pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
          pool = ex3x3
      
      return pool

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    bin_k = 1 # K for BNN

    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    #depth = [ 32, 32, 32, 64, 64, 128, 128] VGG org
    ####################################################################
    # Quantization layers
    ####################################################################
    if True: # 16b weight (no quant); 8b activation
        fl_w_bin = 16
        fl_a_bin = 16 #8 org 
        ml_w_bin = 16
        ml_a_bin = 16 #8 org
        sl_w_bin = 16
        # The last layer's activation (sl_a_bin) is always 16b

        min_rng =  0.0 # range of quanized activation
        max_rng =  2.0

    
    if False:
      
      depth = [ 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]
      fire1 = self._fire_layer('fire1', self.image_input,oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin,  min_rng=min_rng, max_rng=max_rng)
      fire2 = self._fire_layer('fire2', fire1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)   
      fire3 = self._fire_layer('fire3', fire2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,                min_rng=min_rng, max_rng=max_rng)
      fire4 = self._fire_layer('fire4', fire3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
      fire5 = self._fire_layer('fire5', fire4, oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
      fire6 = self._fire_layer('fire6', fire5, oc=depth[5], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
      fire7 = self._fire_layer('fire7', fire6, oc=depth[6], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,                min_rng=min_rng, max_rng=max_rng)
      fire8 = self._fire_layer('fire8', fire7, oc=depth[7], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
      fire9 = self._fire_layer('fire9', fire8, oc=depth[8], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng)
      fire10 = self._fire_layer('fire10', fire9, oc=depth[9], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,                min_rng=min_rng, max_rng=max_rng)      
      
      fire_o = fire10

    else:      
      
      #depth = [ 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]
      depth = [ 32, 32, 32, 64, 64, 64]
      fire0 = self._fire_layer('fire0', self.image_input,oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng)       
      ResBlk_2 = self._ResNetBlock('ResBlk_2', fire0, depth1=depth[1], depth2=depth[2], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng )
      fire1 = self._fire_layer('fire1', ResBlk_2,oc=depth[3], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng)
      ResBlk_5 = self._ResNetBlock('ResBlk_5', fire1, depth1=depth[4], depth2=depth[5], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng )
      fire_o = ResBlk_5    
    ####################################################################

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer('conv12', fire_o, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=sl_w_bin)
    print('self.preds:', self.preds)
    
  def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5):
    with tf.variable_scope(layer_name):
      ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin)
      #tf.summary.histogram('before_bn', ex3x3)
      ex3x3 = self._batch_norm('bn', ex3x3) # <----
      #tf.summary.histogram('before_relu', ex3x3)
      ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
      #tf.summary.histogram('after_relu', ex3x3)
      
      if pool_en:
          pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
      else:
          pool = ex3x3
      #tf.summary.histogram('pool', pool)

      return pool
  
