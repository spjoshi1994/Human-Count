import sys
import os
import math

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Lambda, Dropout, Flatten, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal
from typing import Union

from kerastuner.engine import hyperparameters
from tensorflow.keras import applications
from tensorflow.python.util import nest

from autokeras import keras_layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils


from typing import Optional

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import autokeras as ak
import math
from glob import glob 

import binary_ops
from binary_ops import *
import argparse

class MyConstraints(tf.keras.constraints.Constraint): ##Used for 8-bit weight quantization is Keras
    def __init__(self,name="", **kwargs):
        super(MyConstraints, self).__init__(**kwargs)
        self.name=name
    def __call__(self, w):
        with tf.compat.v1.variable_scope(self.name + "_CONSTRIANTS") as scope:
            return self.nested_lin_8b_quant(w)
    @staticmethod
    def nested_lin_8b_quant(w, min_rng=-0.5, max_rng=0.5): ## 8-bit activation quantization in Keras using Lambda layer
        if min_rng==0.0 and max_rng==2.0:
            min_clip = 0
            max_clip = 255
        else:
            min_clip = -128
            max_clip = 127
        wq = 256.0 * w / (max_rng - min_rng) # to expand [min, max] to [-128, 128]
        wq = K.round(wq) # integer (quantization)
        wq = K.clip(wq, min_clip, max_clip) # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng) # back to quantized real number, not integer
        wclip = K.clip(w, min_rng, max_rng) # linear value w/ clipping
        return wclip + K.stop_gradient(wq - wclip)

class LatticeConvBlock(block_module.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int or keras_tuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int or hyperparameters.Choice.
            The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        filters: Int or keras_tuner.engine.hyperparameters.Choice. The number of
            filters in the convolutional layers. If left unspecified, it will
            be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float. Between 0 and 1. The dropout rate for after the
            convolutional layers. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
        num_blocks: Optional[Union[int, hyperparameters.Choice]] = None,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        filters: Optional[Union[int, hyperparameters.Choice]] = None,
        max_pooling: Optional[bool] = None,
        separable: Optional[bool] = None,
        dropout: Optional[float] = None,
        use_batchnorm: Optional[bool] = None,
        quantrelu: Optional[bool] = None,
        kernel_quant: Optional[bool] = None,
        img_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        min_layers = 2*(int(math.log(img_size,2)) - 2)+1
        layer_choice = [i for i in range(min_layers,26)]
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3], default=3),
            int,
        )
        self.num_blocks = utils.get_hyperparameter(
            num_blocks,
            hyperparameters.Choice("num_blocks", [1], default=1),
            int,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers",layer_choice, default=25),
            int,
        )
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [4,8,16, 32, 64], default=32
            ),
            int,
        )
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.quantrelu = quantrelu
        self.kernel_quant = kernel_quant

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": hyperparameters.serialize(self.kernel_size),
                "num_blocks": hyperparameters.serialize(self.num_blocks),
                "num_layers": hyperparameters.serialize(self.num_layers),
                "filters": hyperparameters.serialize(self.filters),
                "separable": self.separable,
                "dropout": self.dropout,
                "use_batchnorm": self.use_batchnorm,
                "quantrelu":self.quantrelu,
                "kernel_quant":self.kernel_quant
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(config["kernel_size"])
        config["num_blocks"] = hyperparameters.deserialize(config["num_blocks"])
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["filters"] = hyperparameters.deserialize(config["filters"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        normal_conv_kernel = kernel_size
        separable = False
        conv = tf.keras.layers.Conv2D

        pool = tf.keras.layers.MaxPool2D

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0], default=0)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=True)
        max_pooling = False
        for i in range(utils.add_to_hp(self.num_blocks, hp)):
            for j in range(utils.add_to_hp(self.num_layers, hp)):
                max_pooling = not max_pooling
                if j==1:
                    separable = self.separable
                    if separable is None: # corner case when user doesn't provide separable value
                        separable = hp.Boolean("separable", default=False)
                    if separable:
                        conv1 = tf.keras.layers.DepthwiseConv2D
                        normal_conv_kernel = 1
                if output_node.shape[1]<kernel_size:
                    break
                if separable :
                    if self.kernel_quant :
                        output_node = conv1(
                            kernel_size=kernel_size,
                            padding="same",depthwise_initializer=TruncatedNormal(stddev=0.01,seed=99),
                            activation=None,use_bias=False,depthwise_constraint=MyConstraints("depthwise"+str(i)+str(j)),name="conv_DW_"+str(i)+"_"+str(j)
                        )(output_node)
                    else:
                        print("no quant in kernel")
                        output_node = conv1(
                        kernel_size=kernel_size,
                        padding="same",depthwise_initializer=TruncatedNormal(stddev=0.01,seed=99),\
                                activation=None,use_bias=False,name="conv_DW_"+str(i)+"_"+str(j)
                    )(output_node)
                    if use_batchnorm:
                        output_node = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.9, epsilon=1e-6, fused=False)(output_node)
                    if self.quantrelu:
                        output_node = Lambda(self._lin_8b_quant)(output_node) ##Activation Quantization
                    output_node = tf.keras.layers.ReLU()(output_node)
                    
                    
                if self.kernel_quant :
                    output_node = conv(
                        utils.add_to_hp(
                            self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                        ),
                        normal_conv_kernel,padding="same",kernel_initializer=TruncatedNormal(stddev=0.01,seed=99),\
                                activation=None,use_bias=False,kernel_constraint=MyConstraints("pointwise"+str(i)+str(j)),name="conv_PW_"+str(i)+"_"+str(j)
                    )(output_node)
                else:
                    print('NO QUANTIZATION IN KERNEL')
                    output_node = conv(
                    utils.add_to_hp(
                        self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                    ),
                    normal_conv_kernel, padding="same",kernel_initializer=TruncatedNormal(stddev=0.01,seed=99),\
                            activation=None,use_bias=False,name="conv_PW_"+str(i)+"_"+str(j)
                )(output_node)
                if use_batchnorm:
                    output_node = tf.keras.layers.BatchNormalization(axis=-1,momentum=0.9, epsilon=1e-6, fused=False)(output_node)
                if self.quantrelu:
                    output_node = Lambda(self._lin_8b_quant)(output_node) ##Activation Quantization
                output_node = tf.keras.layers.ReLU()(output_node)
                if max_pooling:
                    output_node = pool(pool_size=(2, 2), strides=2, padding='valid', data_format=None)(output_node)
            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        return "same"
        if all(kernel_size * 2 <= length for length in output_node.shape[1:-1]):
            return "valid"
    
    @staticmethod
    def _lin_8b_quant(w, min_rng=0.0, max_rng=2.0): ## 8-bit activation quantization in Keras using Lambda layer
        if min_rng==0.0 and max_rng==2.0:
            min_clip = 0
            max_clip = 255
        else:
            min_clip = -128
            max_clip = 127
        wq = 256.0 * w / (max_rng - min_rng) # to expand [min, max] to [-128, 128]
        wq = K.round(wq) # integer (quantization)
        wq = K.clip(wq, min_clip, max_clip) # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng) # back to quantized real number, not integer
        wclip = K.clip(w, min_rng, max_rng) # linear value w/ clipping
        return wclip + K.stop_gradient(wq - wclip)


class LatClassificationHead(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        loss: Optional[types.LossType] = None,
        metrics: Optional[types.MetricsType] = None,
        dropout: Optional[float] = None,
        kernel_quant: Optional[bool] = None,
        **kwargs
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        if metrics is None:
            metrics = ["accuracy"]
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        # Infered from analyser.
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None
        self.kernel_quant = kernel_quant

    def infer_loss(self):
        if not self.num_classes:
            return None
        if self.num_classes == 2 or self.multi_label:
            return losses.BinaryCrossentropy()
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multi_label": self.multi_label,
                "dropout": self.dropout,
                "kernel_quant":self.kernel_quant
            }
        )
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        output_node = tf.keras.layers.Flatten()(output_node)
        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        if self.kernel_quant:
            output_node = layers.Dense(self.shape[-1],kernel_constraint=MyConstraints("Dense"),
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=99),bias_initializer="zeros",use_bias=True)(output_node)
        else:
            print('NO QUANTIZATION IN KERNEL')
            output_node = layers.Dense(self.shape[-1], kernel_initializer=tf.keras.initializers.GlorotNormal(seed=99),bias_initializer="zeros",use_bias=True)(output_node)

        if isinstance(self.loss, tf.keras.losses.BinaryCrossentropy):
            output_node = layers.Activation(activations.sigmoid, name=self.name)(
                output_node
            )
        else:
            output_node = layers.Softmax(name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(
            name=self.name, multi_label=self.multi_label
        )


    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        self.loss = self.infer_loss()
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

     
    
    def get_hyper_preprocessors(self):
        hyper_preprocessors = []

        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )

        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32())
            )

        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString())
            )

        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SigmoidPostprocessor()
                )
            )
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SoftmaxPostprocessor()
                )
            )
        elif self.num_classes == 2:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.LabelEncoder(self._labels)
                )
            )
        else:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.OneHotEncoder(self._labels)
                )
            )
        return hyper_preprocessors


def train() :
    print("Separate train and validation folder is provided. No further split will be performed on train data")

    train_data_1 = ak.image_dataset_from_directory(
                                                 directory=train_dir,
                            # Set seed to ensure the same split when loading testing data.
                                seed=99,
                                color_mode="grayscale",
                                image_size=(32, 32),
                                batch_size=BATCH_SIZE)

    test_data_1 = ak.image_dataset_from_directory(
                            directory=val_dir,
                            seed=99,
                            color_mode="grayscale",
                            image_size=(32,32),
                            batch_size=BATCH_SIZE)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay( boundaries=[lt1,lt2,lt3,lt4], values=[0.1,0.01,0.001,0.0001,0.00001])
    cust_opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    kwargs = {"optimizer":cust_opt,"metrics":["accuracy"]}
    input_node = ak.Input(shape=(IMG_SHAPE,IMG_SHAPE,input_channels), name="img")
    output_node1 = LatticeConvBlock(separable=separable, use_batchnorm=use_batchnorm,quantrelu=quantrelu,kernel_quant=kernel_quant,img_size=IMG_SHAPE)(input_node)#(mid_node)
    output_node2 = LatClassificationHead(kernel_quant=kernel_quant,dropout=dropout)(output_node1)
    auto_model = ak.AutoModel(inputs=input_node, outputs=output_node2, overwrite=True, max_trials=max_trials,tuner=tuner,\
            seed=99, max_model_size=max_model_size,objective="val_loss",**kwargs)
    

    # Search
    auto_model.fit(x=train_data_1,epochs=epochs,verbose=1,validation_data=test_data_1,callbacks=[mckpt1])
    auto_model_test_loss, auto_model_test_acc = auto_model.evaluate(test_data_1,verbose=2)
    print("auto model loss and acc eval : {},{} ".format(auto_model_test_loss, auto_model_test_acc))
    best_model = auto_model.export_model()
    fc_index = 0
    for i in range(-10,0,1):
        #print(i,net_model.get_layer(index=i).name)
        if best_model.get_layer(index=i).name=="dense":
            #print(net_model.get_layer(index=i))
            fc_index = i
            print("Found the dense layer index {}".format(fc_index))
    if fc_index==0:
        print("No dense layer index found. Please check the name of the dense layer in this script")
    intermediate_model = tf.keras.Model(best_model.input,best_model.get_layer(index=fc_index).output)
    intermediate_model.save("./intrm_model_loss_{}_acc_{}.h5".format(auto_model_test_loss, auto_model_test_acc),save_format="h5")
    best_model.save("./best_model.h5",save_format="h5")
    print(best_model.summary())
    print("Please note that with image_dataset_from_directory of AutoKeras, the order of classes will be [1,10,11,2,3,4,5,6,7,8,9] instead of [1,2,3,4,5,6,7,8,9,10,11]")
    return



#---------------------------------------------------------------------------------
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Train auto Keras model.') 
    parser.add_argument("--dataset_path", required=True, type=str, help="classification dataset path")
    parser.add_argument("--logdir", default = './logs/',required=False,type=str, help="Checkpoint save directory")
    parser.add_argument("--epochs",default=400, required=False,type=int, help="Number of epochs")
    parser.add_argument("--is_gray", default=True,required=False,type=bool, help="True if grayscale, False for rgb")
    parser.add_argument("--img_shape",default=32, required=False,type=int, help="Input image shape")
    parser.add_argument("--gpu",default="0", required=False,type=str, help="Input image shape")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_and_val_separate = False  # Make it true if there are 2 separate dataset folders for train and validation
    #handgesture = False  # for landmark model make it False
    #if handgesture:
    # Provide total train images to calculate limits for Optimizer in piecewise constant
    if args.dataset_path[-1]=='/':
        total_train_images = len(glob(args.dataset_path+"**/**"))
    else:
        total_train_images = len(glob(args.dataset_path+"/**/**"))
        
    

    data_dir = args.dataset_path
    train_dir = data_dir
    val_dir=data_dir
    IMG_SHAPE = args.img_shape  # Image HxW
    if args.is_gray:
        input_channels = 1  # 1 or 3
        color_mode = "grayscale"  # "grayscale" or "rgb"
    else:
        input_channels = 3  # 1 or 3
        color_mode = "rgb"  # "grayscale" or "rgb"
    L2_value = 0.001
    CLASS_NAMES = None #class names to be used in this order.
   

    BATCH_SIZE = 32 # batch size of images to be pre-processed in train and/or val dataset. Keep it at a value > #total images
    # The actual batch size will be inside fit method. Defaulted to 32
    epochs = args.epochs
    max_trials = 75 # Max number of architectures to be tried by AK to train the best one
    max_model_size=1600000 # To limit the model size
    dropout = 0.8 # Dropout rate

    separable = False # True for Mobilenet like model
    use_batchnorm = True
    kernel_quant = True # whether to use kernel quantization -0.5<=W<0.5
    quantrelu = True # Whether to provide activation quantization
    
    
    steps_per_epoch = math.ceil(total_train_images/BATCH_SIZE)
    total_steps = math.ceil(steps_per_epoch*epochs)
    lt1 = math.ceil(0.65*total_steps)
    lt2 = math.ceil(0.85*total_steps)
    lt3 = math.ceil(0.95*total_steps)
    lt4 = total_steps
    print("steps per epoch {}".format(steps_per_epoch))
    print("total steps {}".format(total_steps))
    print("four limits for SGD are {} , {} , {} and {}".format(lt1,lt2,lt3,lt4))
    print("four limits for SGD are in epochs {} , {} , {} and {}".format(lt1//steps_per_epoch,lt2//steps_per_epoch,lt3//steps_per_epoch,lt4//steps_per_epoch))


    tuner = "hyperband" # Optimizer used by AK to find the best set of hyperparameters
    model_dir = args.logdir+"/models"+"/" # directory where models will be stored
    model_name = "ak_saved_model" # model saved by autokeras directly
    ckpt_model= model_dir+"weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" # To store best ckpt model
    ckpt_model_1 = model_dir+"ckpt_files/" # Similar to ckpt_model and this will be used as the final model


    #---------------------------------------------------------------------------------
    # This ckpt will show us at what epoch, what val_accuracy the model was saved
    mckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_model, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='min', save_freq='epoch')
    # This ckpt saves the best model but without any epoch or accuracy information
    mckpt1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_model_1, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='min', save_freq='epoch')

    # Train the model
    train()
