# Project: squeezeDetOnKeras
# Filename: squeezeDet
# Author: Christopher Ehmann
# Date: 28.11.17
# Organisation: searchInk
# Email: christopher@searchink.com


import binary_ops as bo
import main.utils.utils as utils
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Reshape, Lambda, ReLU, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.models import Model
import sys


def bin_wrapper_layer(x, a_bin=16, min_rng=0.0, max_rng=2.0):  # activation binarization
    x_quant = bo.lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)
    return x_quant


# class that wraps config and model
class SqueezeDet():
    # initialize model from config file
    def __init__(self, config, early_pooling, DEPTHS, useconv3):
        """Init of SqueezeDet Class
        
        Arguments:
            config {[type]} -- dict containing hyperparameters for network building
        """

        # hyperparameter config file
        self.depth = DEPTHS
        self.useconv3 = useconv3
        depth_length = len(self.depth)
        if len(DEPTHS) < 6 or len(DEPTHS) > 10:
            print("Length of Depths Should be in between 6-10")
            sys.exit()
        if early_pooling:
            if depth_length == 6:
                self.pooling = [True, True, True, False, False, False]
            elif depth_length == 7:
                self.pooling = [True, False, True, True, False, False, False]
            elif depth_length == 8:
                self.pooling = [True, False, True, True, False, False, False, False]
            elif depth_length == 9:
                self.pooling = [True, False, True, True, False, False, False, False, False]
            else:
                self.pooling = [True, False, True, True, False, False, False, False, False, False]
        else:
            if depth_length == 6:
                self.pooling = [True, False, True, False, True, False]
            elif depth_length == 7:
                self.pooling = [True, False, True, False, True, False, False]
            elif depth_length == 8:
                self.pooling = [True, False, True, False, True, False, False, False]
            elif depth_length == 9:
                self.pooling = [True, False, True, False, True, False, False, False, False]
            else:
                self.pooling = [True, False, False, True, False, True, False, False, False, False]

        self.config = config
        # create Keras model
        self.model = self._create_model()

    def bin_wrapper_layer(self, x):  # activation binarization
        x_quant = bo.lin_8b_quant(x, min_rng=self.config.QUANT_RANGE[0], max_rng=self.config.QUANT_RANGE[1])
        return x_quant

    def _create_model(self):
        """
        #builds the Keras model from config
        #return: squeezeDet in Keras
        """
        self.input_layer = Input(shape=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.N_CHANNELS),
                                 name="input")
        prev_layer = self.input_layer
        if self.useconv3:
            self.fire1 = self._fire_layer(name="fire1", input=self.input_layer, out_channels=self.depth[0], stdd=0.012,
                                          wt_quant=True, act_quant=True, bn_eps=1e-3, pool=self.pooling[0])
        else:
            self.fire1 = self._mobile_layer(name="fire1", input=prev_layer, out_channels=self.depth[0], stdd=0.012,
                                            wt_quant=True, act_quant=True, bn_eps=1e-3, pool=self.pooling[0],
                                            pool2=True)
        prev_layer = self.fire1
        for layer_count in range(len(self.depth) - 1):
            if layer_count == 0:
                pool2 = True
            else:
                pool2 = False
            prev_layer = self._mobile_layer(name="fire{}".format(str(layer_count + 2)), input=prev_layer,
                                            out_channels=self.depth[layer_count + 1],
                                            stdd=0.012, wt_quant=True, act_quant=True,
                                            bn_eps=1e-3, pool=self.pooling[layer_count + 1], pool2=pool2)

        num_output = self.config.ANCHOR_PER_GRID * (self.config.CLASSES + 1 + 4)
        self.preds = Conv2D(
            name='conv12', filters=num_output, kernel_size=(3, 3), strides=(1, 1), activation=None, padding="SAME",
            use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.01),
            kernel_constraint=bo.MyConstraints('quant_' + 'conv12'))(prev_layer)

        # reshape
        self.pred_reshaped = Reshape((self.config.ANCHORS, -1))(self.preds)
        # pad for loss function so y_pred and y_true have the same dimensions, wrap in lambda layer
        self.pred_padded = Lambda(self._pad)(self.pred_reshaped)
        model = Model(inputs=self.input_layer, outputs=self.pred_padded)

        return model

    def _fire_layer(self, name, input, out_channels, stdd=0.01, wt_quant=True, act_quant=True, bn_eps=1e-6, pool=True):
        """
            wrapper for fire layer constructions

            :param name: name for layer
            :param input: previous layer
            :param s1x1: number of filters for squeezing
            :param e1x1: number of filter for expand 1x1
            :param e3x3: number of filter for expand 3x3
            :param stdd: standard deviation used for intialization
            :return: a keras fire layer
            """
        with tf.variable_scope(name) as scope:
            ex3x3 = Conv2D(
                name=name + '/expand3x3', filters=out_channels, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd),
                kernel_constraint=bo.MyConstraints("quant_" + name))(input)
            bch1 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch1(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "fl_pool")(ex3x3)

        return ex3x3

    # wrapper for padding, written in tensorflow. If you want to change to theano you need to rewrite this!
    def _pad(self, input):
        """
        pads the network output so y_pred and y_true have the same dimensions
        :param input: previous layer
        :return: layer, last dimensions padded for 4
        """

        padding = np.zeros((3, 2))
        padding[2, 1] = 4
        return tf.pad(input, padding, "CONSTANT")

    def _mobile_layer(self, name, input, out_channels=1, stdd=0.02, wt_quant=True, act_quant=True, bn_eps=1e-6,
                      pool=True, pool2=False):
        with tf.variable_scope(name + "_DW") as scope:
            ex3x3 = DepthwiseConv2D(name=name + "_DW", kernel_size=3, padding="SAME", use_bias=False,
                                    depthwise_initializer=TruncatedNormal(stddev=stdd),
                                    depthwise_constraint=bo.MyConstraints("quant_" + name + "_DW"))(input)

            bch2 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch2(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "ml_pool")(ex3x3)
        with tf.variable_scope(name + "_PW") as scope:
            ex3x3 = Conv2D(
                name=name + '_PW', filters=out_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd),
                kernel_constraint=bo.MyConstraints("quant_" + name + "_PW"))(ex3x3)

            bch3 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch3(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool2:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "ml_pool_2")(ex3x3)
            return ex3x3

    # loss function to optimize
    def loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        # handle for config
        mc = self.config

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs
        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(y_pred, mc)

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        # compute class loss,add a small value into log to prevent blowing up
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        # add above losses
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss

    # the sublosses, to be used as metrics during training

    def bbox_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the bbox loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_delta_input = y_true[:, :, 5:9]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID + num_class_probs

        # slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        # slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        return bbox_loss

    def conf_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the conf loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID + num_class_probs

        # slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        # slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        return conf_loss

    def class_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the class loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # slice pred tensor to extract class pred scores and then normalize them
        pred_class_probs = K.reshape(
            K.softmax(
                K.reshape(
                    y_pred[:, :, :, :num_class_probs],
                    [-1, mc.CLASSES]
                )
            ),
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
        )

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # compute class loss
        class_loss = K.sum((labels * (-K.log(pred_class_probs + mc.EPSILON))
                            + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON)))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        return class_loss

    # loss function again, used for metrics to show loss without regularization cost, just of copy of the original loss
    def loss_without_regularization(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        # handle for config
        mc = self.config

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(y_pred, mc)

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc)

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # compute class loss
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        # add above losses 
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss
