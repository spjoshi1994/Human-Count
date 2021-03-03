#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   27.08.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import zipfile
import shutil
import os
import sys
import tensorflow as tf
import numpy as np

from urllib.request import urlretrieve
from tqdm import tqdm
import mv2.ssd_base as mv2
import tf_common as tfc

#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0

    #---------------------------------------------------------------------------
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
def conv_map(x, size, shape, stride, name, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable("filter",
                            shape=[shape, shape, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.relu(x)
        l2 = tf.nn.l2_loss(w)
    return x, l2

#-------------------------------------------------------------------------------
def classifier(x, size, mapsize, name):
    with tf.variable_scope(name):
        w = tf.get_variable("filter",
                            shape=[3, 3, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.reshape(x, [-1, mapsize.w*mapsize.h, size])
        l2 = tf.nn.l2_loss(w)
    return x, l2

def classifier_last(x, size, mapsize, name):
    with tf.variable_scope(name):
        w = tf.get_variable("filter",
                            shape=[1, 1, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.reshape(x, [-1, mapsize.w*mapsize.h, size])
        l2 = tf.nn.l2_loss(w)
    return x, l2

#-------------------------------------------------------------------------------
def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)

#-------------------------------------------------------------------------------
def array2tensor(x, name):
    init = tf.constant_initializer(value=x, dtype=tf.float32)
    tensor = tf.get_variable(name=name, initializer=init, shape=x.shape)
    return tensor

#-------------------------------------------------------------------------------
def l2_normalization(x, initial_scale, channels, name):
    with tf.variable_scope(name):
        scale = array2tensor(initial_scale*np.ones(channels), 'scale')
        x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x

#-------------------------------------------------------------------------------
class SSDMV2:
    #---------------------------------------------------------------------------
    def __init__(self, session, preset, is_training):
        self.preset = preset
        self.session = session
        self.__built = False
        self.__build_names()
        self.is_training = is_training
        
    #---------------------------------------------------------------------------
    def build_from_mv2(self, num_classes, a_trous=False,
                       progress_hook='tqdm'):
        """
        Build the model for training.
        :param num_classes:   number of classes
        """
        self.num_classes = num_classes+1
        self.num_vars = num_classes+5
        self.l2_loss = 0
        self.__load_mv2()
        self.__build_ssd_layers()
        self.__select_feature_maps()
        self.__build_classifiers()
        self.__built = True

    #---------------------------------------------------------------------------
    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        """
        Build the model for inference from a metagraph shapshot and weights
        checkpoint.
        """
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.result      = sess.graph.get_tensor_by_name('result/result:0')

    #---------------------------------------------------------------------------
    def build_optimizer_from_metagraph(self):
        """
        Get the optimizer and the loss from metagraph
        """
        sess = self.session
        self.loss = sess.graph.get_tensor_by_name('total_loss/loss:0')
        self.localization_loss = sess.graph.get_tensor_by_name('localization_loss/localization_loss:0')
        self.confidence_loss = sess.graph.get_tensor_by_name('confidence_loss/confidence_loss:0')
        self.l2_loss = sess.graph.get_tensor_by_name('total_loss/l2_loss:0')
        self.optimizer = sess.graph.get_operation_by_name('optimizer/optimizer')
        self.labels = sess.graph.get_tensor_by_name('labels:0')

        self.losses = {
            'total': self.loss,
            'localization': self.localization_loss,
            'confidence': self.confidence_loss,
            'l2': self.l2_loss
        }


    def __load_mv2(self):
        self.image_input = tf.placeholder(tf.float32,name='image_input',shape= [None, 224,224, 3])
        self.mv2_stdaln = mv2.MV2(self.is_training)
        self.mv2_stdaln.build(self.image_input)

    #---------------------------------------------------------------------------
    def __with_loss(self, x, l2_loss):
        self.l2_loss += l2_loss
        return x

    def __build_ssd_layers(self):
        
        h = [256,512,512,
         256, 256,
         128, 256,
         128, 256]
        
        ml_w_bin = 8
        ml_a_bin = 8

        min_rng =  0.0 # range of quanized activation
        max_rng =  2.0
        max_pool = True

        self.c5, l2 = tfc.conv2d("c5", self.mv2_stdaln.conv4, h[0], h[1], size=3, w_bin=ml_w_bin, a_bin=ml_a_bin,
                            min_rng=min_rng, max_rng=max_rng, is_training=self.is_training, max_pool=False)
        self.ssd_conv5 = self.__with_loss(self.c5, l2)
        self.c6, l2 = tfc.conv2d("c6", self.ssd_conv5, h[1], h[2], size=1, w_bin=ml_w_bin, a_bin=ml_a_bin,
                            min_rng=min_rng, max_rng=max_rng, is_training=self.is_training, max_pool=False)
        self.ssd_conv6 = self.__with_loss(self.c6, l2)

        self.c7,l2 = tfc.conv2d("c7", self.ssd_conv6, h[2], h[3], size=3,stride=1, w_bin=ml_w_bin, a_bin=ml_a_bin,
                            min_rng=min_rng, max_rng=max_rng, is_training=self.is_training, max_pool=max_pool)
        self.ssd_conv7 = self.__with_loss(self.c7, l2)

        self.c8,l2 = tfc.conv2d("c8", self.c7, h[3], h[4], size=3,stride=1, w_bin=ml_w_bin, a_bin=ml_a_bin,
                            min_rng=min_rng, max_rng=max_rng, is_training=self.is_training, max_pool=max_pool)
        self.ssd_conv8 = self.__with_loss(self.c8, l2)
        
        self.c9,l2 = tfc.conv2d("c9", self.c8, h[4], h[5], size=3,stride=1, w_bin=ml_w_bin, a_bin=ml_a_bin,
                            min_rng=min_rng, max_rng=max_rng, is_training=self.is_training, max_pool=max_pool)
        self.ssd_conv9 = self.__with_loss(self.c9, l2)
        
    #---------------------------------------------------------------------------

    def __select_feature_maps(self):
        self.__maps = [
            self.ssd_conv6,
            self.c7,
            self.c8,
            self.c9]

        if len(self.preset.maps) == 7:
            self.__maps.append(self.ssd_conv11)

    #---------------------------------------------------------------------------
    def __build_classifiers(self):
        with tf.variable_scope('classifiers'):
            self.__classifiers = []
            for i in range(len(self.__maps)):
                fmap = self.__maps[i]
                map_size = self.preset.maps[i].size
                for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                    name = 'classifier{}_{}'.format(i, j)
                    if i == 5 :
                        clsfier, l2 = classifier_last(fmap, self.num_vars, map_size, name)
                    else :
                        clsfier, l2 = classifier(fmap, self.num_vars, map_size, name)
                    self.__classifiers.append(self.__with_loss(clsfier, l2))

        with tf.variable_scope('output'):
            output      = tf.concat(self.__classifiers, axis=1, name='output')
            self.logits = output[:,:,:self.num_classes]

        with tf.variable_scope('result'):
            self.classifier = tf.nn.softmax(self.logits)
            self.locator    = output[:,:,self.num_classes:]
            self.result     = tf.concat([self.classifier, self.locator],
                                        axis=-1, name='result')

    #---------------------------------------------------------------------------
    def build_optimizer(self, learning_rate=0.001, weight_decay=0.0005,
                        momentum=0.9, global_step=None):

        self.labels = tf.placeholder(tf.float32, name='labels',
                                    shape=[None, None, self.num_vars])

        with tf.variable_scope('ground_truth'):
            #-------------------------------------------------------------------
            # Split the ground truth tensor
            #-------------------------------------------------------------------
            # Classification ground truth tensor
            # Shape: (batch_size, num_anchors, num_classes)
            gt_cl = self.labels[:,:,:self.num_classes]

            # Localization ground truth tensor
            # Shape: (batch_size, num_anchors, 4)
            gt_loc = self.labels[:,:,self.num_classes:]

            # Batch size
            # Shape: scalar
            batch_size = tf.shape(gt_cl)[0]

        #-----------------------------------------------------------------------
        # Compute match counters
        #-----------------------------------------------------------------------
        with tf.variable_scope('match_counters'):
            # Number of anchors per sample
            # Shape: (batch_size)
            total_num = tf.ones([batch_size], dtype=tf.int64) * \
                        tf.to_int64(self.preset.num_anchors)

            # Number of negative (not-matched) anchors per sample, computed
            # by counting boxes of the background class in each sample.
            # Shape: (batch_size)
            negatives_num = tf.count_nonzero(gt_cl[:,:,-1], axis=1)

            # Number of positive (matched) anchors per sample
            # Shape: (batch_size)
            positives_num = total_num-negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([batch_size])*10e-15,
                                          tf.to_float(positives_num))

        #-----------------------------------------------------------------------
        # Compute masks
        #-----------------------------------------------------------------------
        with tf.variable_scope('match_masks'):
            # Boolean tensor determining whether an anchor is a positive
            # Shape: (batch_size, num_anchors)
            positives_mask = tf.equal(gt_cl[:,:,-1], 0)

            # Boolean tensor determining whether an anchor is a negative
            # Shape: (batch_size, num_anchors)
            negatives_mask = tf.logical_not(positives_mask)

        #-----------------------------------------------------------------------
        # Compute the confidence loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('confidence_loss'):
            # Cross-entropy tensor - all of the values are non-negative
            # Shape: (batch_size, num_anchors)
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cl,
                                                            logits=self.logits)

            #-------------------------------------------------------------------
            # Sum up the loss of all the positive anchors
            #-------------------------------------------------------------------
            # Positives - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positives = tf.where(positives_mask, ce, tf.zeros_like(ce))

            # Total loss of positive anchors
            # Shape: (batch_size)
            positives_sum = tf.reduce_sum(positives, axis=-1)

            #-------------------------------------------------------------------
            # Figure out what the negative anchors with highest confidence loss
            # are
            #-------------------------------------------------------------------
            # Negatives - the loss of positive anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            negatives = tf.where(negatives_mask, ce, tf.zeros_like(ce))

            # Top negatives - sorted confience loss with the highest one first
            # Shape: (batch_size, num_anchors)
            negatives_top = tf.nn.top_k(negatives, self.preset.num_anchors)[0]

            #-------------------------------------------------------------------
            # Fugure out what the number of negatives we want to keep is
            #-------------------------------------------------------------------
            # Maximum number of negatives to keep per sample - we keep at most
            # 3 times as many as we have positive anchors in the sample
            # Shape: (batch_size)
            negatives_num_max = tf.minimum(negatives_num, 3*positives_num)

            #-------------------------------------------------------------------
            # Mask out superfluous negatives and compute the sum of the loss
            #-------------------------------------------------------------------
            # Transposed vector of maximum negatives per sample
            # Shape (batch_size, 1)
            negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)

            # Range tensor: [0, 1, 2, ..., num_anchors-1]
            # Shape: (num_anchors)
            rng = tf.range(0, self.preset.num_anchors, 1)

            # Row range, the same as above, but int64 and a row of a matrix
            # Shape: (1, num_anchors)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))

            # Mask of maximum negatives - first `negative_num_max` elements
            # in corresponding row are `True`, the rest is false
            # Shape: (batch_size, num_anchors)
            negatives_max_mask = tf.less(range_row, negatives_num_max_t)

            # Max negatives - all the positives and superfluous negatives are
            # zeroed out.
            # Shape: (batch_size, num_anchors)
            negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))

            # Sum of max negatives for each sample
            # Shape: (batch_size)
            negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)

            #-------------------------------------------------------------------
            # Compute the confidence loss for each element
            #-------------------------------------------------------------------
            # Total confidence loss for each sample
            # Shape: (batch_size)
            confidence_loss = tf.add(positives_sum, negatives_max_sum)

            # Total confidence loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([batch_size]),
                                       tf.div(confidence_loss,
                                              positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar
            self.confidence_loss = tf.reduce_mean(confidence_loss,
                                                  name='confidence_loss')

        #-----------------------------------------------------------------------
        # Compute the localization loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('localization_loss'):
            # Element-wise difference between the predicted localization loss
            # and the ground truth
            # Shape: (batch_size, num_anchors, 4)
            loc_diff = tf.subtract(self.locator, gt_loc)

            # Smooth L1 loss
            # Shape: (batch_size, num_anchors, 4)
            loc_loss = smooth_l1_loss(loc_diff)

            # Sum of localization losses for each anchor
            # Shape: (batch_size, num_anchors)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)

            # Positive locs - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positive_locs = tf.where(positives_mask, loc_loss_sum,
                                     tf.zeros_like(loc_loss_sum))

            # Total loss of positive anchors
            # Shape: (batch_size)
            localization_loss = tf.reduce_sum(positive_locs, axis=-1)

            # Total localization loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            localization_loss = tf.where(tf.equal(positives_num, 0),
                                         tf.zeros([batch_size]),
                                         tf.div(localization_loss,
                                                positives_num_safe))

            # Mean localization loss for the batch
            # Shape: scalar
            self.localization_loss = tf.reduce_mean(localization_loss,
                                                    name='localization_loss')

        #-----------------------------------------------------------------------
        # Compute total loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('total_loss'):
            # Sum of the localization and confidence loss
            # Shape: (batch_size)
            self.conf_and_loc_loss = tf.add(self.confidence_loss,
                                            self.localization_loss,
                                            name='sum_losses')

            # L2 loss
            # Shape: scalar
            self.l2_loss = tf.multiply(weight_decay, self.l2_loss,
                                       name='l2_loss')

            # Final loss
            # Shape: scalar
            self.loss = tf.add(self.conf_and_loc_loss, self.l2_loss,
                               name='loss')

        #-----------------------------------------------------------------------
        # Build the optimizer
        #-----------------------------------------------------------------------
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            optimizer = optimizer.minimize(self.loss, global_step=global_step,
                                           name='optimizer')

        #-----------------------------------------------------------------------
        # Store the tensors
        #-----------------------------------------------------------------------
        self.optimizer = optimizer
        self.losses = {
            'total': self.loss,
            'localization': self.localization_loss,
            'confidence': self.confidence_loss,
            'l2': self.l2_loss
        }

    #---------------------------------------------------------------------------
    def __build_names(self):
        #-----------------------------------------------------------------------
        # Names of the original and new scopes
        #-----------------------------------------------------------------------
        self.original_scopes = [
            'conv1', 'conv2', 'conv3','conv4'
        ]

        self.new_scopes = [
            
        ]

        if len(self.preset.maps) == 7:
            self.new_scopes += ['conv11_1', 'conv11_2']

        for i in range(len(self.preset.maps)):
            for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                self.new_scopes.append('classifiers/classifier{}_{}'.format(i, j))

    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    def build_summaries(self, restore):
        if restore:
            return self.session.graph.get_tensor_by_name('net_summaries/net_summaries:0')

        #-----------------------------------------------------------------------
        # Build the filter summaries
        #-----------------------------------------------------------------------
        names = self.original_scopes + self.new_scopes
        sess = self.session
        #To view nodes in graph
        '''for v in tf.get_default_graph().as_graph_def().node:
            print (v.name)'''
        summaries = []
        with tf.variable_scope('filter_summaries'):
            summaries = []
            for name in names:
                if name == 'conv1':
                    tensor = sess.graph.get_tensor_by_name(name+'/conv3x3/kernels:0')
                else:
                    try:
                        tensor = sess.graph.get_tensor_by_name(name+'/netconv1x1/netconv1x1/kernels:0')
                    except:
                        tensor = sess.graph.get_tensor_by_name(name+'/filter:0')
                summary = tf.summary.histogram(name, tensor)
                summaries.append(summary)

        return tf.summary.merge(summaries, name='net_summaries')
