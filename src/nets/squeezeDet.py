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
  def __init__(self, mc, freeze_layers=[], gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)
      #self.zero_amt = tf.constant(20, tf.float32)
      #tf.summary.scalar('zero amt', self.zero_amt)


      self._add_forward_graph(freeze_layers)
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self, freeze_layers):
    """NN architecture."""

    mc = self.mc
    bin_k = 1 # K for BNN

    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    depth = [32, 32, 32, 40, 40, 64,  64]  # 64KB for activation, 128KB for program
    depth = [32, 32, 32, 32, 32, 32,  32]  # 64KB for activation,  64KB for program
    depth = [16, 16, 32, 32, 32, 64,  64]  # 32KB for activation,  96KB for program
    depth = [16, 16, 32, 32, 32, 48,  48]  # 32KB for activation,  96KB for program #2 (n2)
    depth = [16, 16, 32, 32, 32, 44,  48]  # 32KB for activation,  96KB for program #2 (n3)
    depth = [ 8, 16, 16, 24, 24, 32,  32]  # 32KB for activation,  96KB for program (low power)
    depth = [ 8,  8, 16, 16, 24, 24,  32]  # 32KB for activation,  96KB for program (low power #2)
    depth = [16, 16, 32, 32, 64, 64, 128]  # 128KB for activation, Unlimited for program
    depth = [32, 32, 32, 64, 64, 128, 128]

    mul_f = 1
    #depth = [ 8*mul_f,  8*mul_f, 16*mul_f, 16*mul_f, 24*mul_f, 24*mul_f, 32*mul_f]  # 32KB for activation,  96KB for program (low power #2)

    ####################################################################
    # Quantization layers
    ####################################################################
    if True: # 8b weight; 8b activation
        fl_w_bin = 8
        fl_a_bin = 8 
        ml_w_bin = 8
        ml_a_bin = 8
        sl_w_bin = 8
        # The last layer's activation (sl_a_bin) is always 16b
        
        min_rng =  0.0 # range of quantized activation only uint8
        max_rng =  2.0

        bias_on = False # no bias for T+ and Crosslink-NX device
    if not len(freeze_layers):
        freeze_layers = [False, False, False, False, False, False, False , False]
    else:
        freeze_layers = [bool(int(item)) for item in freeze_layers.split(',')]


    fire1 = self._fire_layer_3x3('fire1', self.image_input, oc=32, freeze=freeze_layers[0], w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire2 = self._expand_block('fire2', fire1, oc1x1 = 48,oc1_1x1 = 24,oc2_3x3 = 24, freeze=freeze_layers[1], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire3 = self._expand_block('fire3', fire2,oc1x1 = 64,oc1_1x1 = 32,oc2_3x3 = 32,freeze=freeze_layers[2], w_bin=ml_w_bin, a_bin=ml_a_bin,pool_en=False,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire4 = self._expand_block('fire4', fire3,oc1x1 = 80,oc1_1x1 = 40,oc2_3x3 = 40, freeze=freeze_layers[3], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire5 = self._expand_block('fire5', fire4,oc1x1 = 96,oc1_1x1 = 48,oc2_3x3 = 48, freeze=freeze_layers[4], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire6 = self._expand_block('fire6', fire5,oc1x1 = 112,oc1_1x1 = 56,oc2_3x3 = 56, freeze=freeze_layers[5], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire7 = self._expand_block('fire7', fire6, oc1x1 = 128,oc1_1x1 = 64,oc2_3x3=64, freeze=freeze_layers[6], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    fire_o = fire7

    ####################################################################

    if True: # debugging
        self.fire1 = fire1
        self.fire2 = fire2
        self.fire3 = fire3
        self.fire4 = fire4
        self.fire5 = fire5
        self.fire6 = fire6
        self.fire7 = fire7

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer('conv12', fire_o, filters=num_output, freeze=False, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=sl_w_bin, bias_on=bias_on)
    print('self.preds:', self.preds)

  def _fire_layer_3x3(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, bias_on=True, mul_f=1):
    with tf.variable_scope(layer_name):
        ex3x3 = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on, mul_f=mul_f) # <----

        tf.summary.histogram('before_bn', ex3x3)
        ex3x3 = self._batch_norm('bn', ex3x3) 
        tf.summary.histogram('before_relu', ex3x3)
        ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
        tf.summary.histogram('after_relu', ex3x3)
        if pool_en:
            pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
        else:
            pool = ex3x3
        tf.summary.histogram('pool', pool)

        return pool
  def _fire_layer_1x1(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, bias_on=True, mul_f=1):
    with tf.variable_scope(layer_name):
        ex1x1 = self._conv_layer('conv1x1', inputs, filters=oc, size=1, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on, mul_f=mul_f) # <----

        tf.summary.histogram('before_bn', ex1x1)
        ex1x1 = self._batch_norm('bn', ex1x1) 
        tf.summary.histogram('before_relu', ex1x1)
        ex1x1 = self.binary_wrapper(ex1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
        tf.summary.histogram('after_relu', ex1x1)
        if pool_en:
            pool = self._pooling_layer('pool', ex1x1, size=2, stride=2, padding='SAME')
        else:
            pool = ex1x1
        tf.summary.histogram('pool', pool)

        return pool
    
  def _expand_block(self, layer_name, inputs, oc1x1,oc1_1x1,oc2_3x3,stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, bias_on=True, mul_f=1):
    ex1x1 = self._fire_layer_1x1(layer_name+'_1x1', inputs, oc1x1, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=1)

    ex1x1_1 = self._fire_layer_1x1(layer_name+'_1x1_1', ex1x1, oc1_1x1, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=pool_en, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=1)

    ex3x3_2 = self._fire_layer_3x3(layer_name+'_3x3_2', ex1x1, oc2_3x3, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=pool_en, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=1)

    #concat = tf.concat([ex1x1_1, ex3x3_2], 3, name=layer_name+'_concat')
    #tf.summary.histogram('concat', concat)
    elt 

    return concat
