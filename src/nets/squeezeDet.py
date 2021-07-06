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
    if True: # 16b weight (no quant); 8b activation
        fl_w_bin = 8
        fl_a_bin = 8 
        ml_w_bin = 8
        ml_a_bin = 8
        sl_w_bin = 8
        # The last layer's activation (sl_a_bin) is always 16b

        min_rng =  0.0 # range of quanized activation
        max_rng =  2.0

        bias_on = False # no bias for T+
    if not len(freeze_layers):
        freeze_layers = [False, False, False, False, False, False, False , False]
    else:
        freeze_layers = [bool(int(item)) for item in freeze_layers.split(',')]


    fire1 = self._fire_layer_3x3('fire1', self.image_input, oc=16, freeze=freeze_layers[0], w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire2 = self._inception_block('fire2', fire1,oc1x1=12,oc3x3_1=12,oc3x3_2=16,freeze=freeze_layers[2], w_bin=ml_w_bin, a_bin=ml_a_bin,pool_en=False,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire3 = self._inception_block('fire3', fire2,oc1x1=16,oc3x3_1=16,oc3x3_2=20, freeze=freeze_layers[3], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire4 = self._inception_block('fire4', fire3,oc1x1=20,oc3x3_1=20,oc3x3_2=24, freeze=freeze_layers[4], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire5 = self._inception_block('fire5', fire4,oc1x1=24,oc3x3_1=24,oc3x3_2=28, freeze=freeze_layers[5], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    #fire6 = self._inception_block('fire6', fire5, oc1x1=56,oc3x3_1=56,oc3x3_2=64, freeze=freeze_layers[6], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False,min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    
    fire7 = self._fire_layer_3x3('fire7', fire5, oc=64, freeze=freeze_layers[1], w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
    fire_o = fire7

    ####################################################################

    if True: # debugging
        self.fire1 = fire1
        self.fire2 = fire2
        self.fire3 = fire3
        self.fire4 = fire4
        self.fire5 = fire5
        #self.fire6 = fire6
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

  def _fire_layer_1x1(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, bias_on=True, mul_f=1):
    with tf.variable_scope(layer_name):
        ex1x1 = self._conv_layer('conv1x1', inputs, filters=oc, size=1, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on, mul_f=mul_f) # <----

        tf.summary.histogram('before_bn', ex1x1)
        ex1x1 = self._batch_norm('bn', ex1x1) # <----
        tf.summary.histogram('before_relu', ex1x1)
        ex1x1 = self.binary_wrapper(ex1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # <---- relu
        tf.summary.histogram('after_relu', ex1x1)
        if pool_en:
            pool = self._pooling_layer('pool', ex1x1, size=2, stride=2, padding='SAME')
        else:
            pool = ex1x1
        tf.summary.histogram('pool', pool)

        return pool
    
  def _inception_block(self, layer_name, inputs, oc1x1,oc3x3_1,oc3x3_2,stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, min_rng=-0.5, max_rng=0.5, bias_on=True, mul_f=1):
    ex1x1 = self._fire_layer_1x1(layer_name+'_1x1', inputs, oc1x1, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=pool_en, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=1)

    ex3x3_1 = self._fire_layer_3x3(layer_name+'_3x3_1', inputs, oc3x3_1, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on)

    ex3x3_2 = self._fire_layer_3x3(layer_name+'_3x3_2', ex3x3_1, oc3x3_1, stddev=0.01, freeze=False, w_bin=w_bin, a_bin=a_bin, pool_en=pool_en, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on)
    

    if pool_en:
        pool = self._pooling_layer('pool', inputs, size=2, stride=2, padding='SAME')
        concat = tf.concat([ex1x1, ex3x3_2, pool], 3, name=layer_name+'_concat')
    else:
        concat = tf.concat([ex1x1, ex3x3_2], 3, name=layer_name+'_concat')

    tf.summary.histogram('concat', concat)

    return concat
