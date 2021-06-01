import gc
import os
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from main.config.create_config import load_dict
from main.model.dataGenerator import generator_from_data_path
from main.model.squeezeDet import SqueezeDet
from main.pruning_structure import get_output_layer_index


class Training:
    def __init__(self, epochs, depths, earlypooling, config_file, logdir, init_file, cuda_device, validation_freq,
                 freeze_landmark, usecov3):
        self.epochs = epochs
        self.depths = depths
        self.EarlyPooling = earlypooling
        self.optimizer = 'adam'
        self.init_file = init_file
        self.logdir = logdir
        self.VERBOSE = True
        self.REDUCELRONPLATEAU = True  # True ##False for non-mobilenet
        self.num_gpus = 1
        self.cuda_device = cuda_device
        self.validation_freq = validation_freq
        self.freeze_landmark = freeze_landmark
        self.usecov3 = usecov3

        self.checkpoint_dir = logdir + "/train" + "/checkpoints"
        self.tb_dir = logdir + "/train" + "/tensorboard"

        self.img_file = os.path.join(logdir, 'data', 'img_train.txt')
        self.gt_file = os.path.join(logdir, 'data', 'gt_train.txt')

        self.val_img_file = os.path.join(logdir, 'data', 'img_val.txt')
        self.val_gt_file = os.path.join(logdir, 'data', 'gt_val.txt')

        self.config_file = config_file

        self.create_new_log_dir_structure()
        self.load_training_data()
        self.load_config()
        self.start_training()

    def create_new_log_dir_structure(self):
        # delete old checkpoints and tensorboard stuff
        if tf.gfile.Exists(self.checkpoint_dir):
            print('Target directory already Exists : {}'.format(self.checkpoint_dir))
        else:
            tf.gfile.MakeDirs(self.checkpoint_dir)
        if tf.gfile.Exists(self.tb_dir):
            print('Target directory already Exists : {}'.format(self.tb_dir))
        else:
            tf.gfile.MakeDirs(self.tb_dir)

    def load_training_data(self):
        # open files with images and ground truths files with full path names
        with open(self.img_file) as imgs:
            self.img_names = imgs.read().splitlines()
        imgs.close()
        with open(self.gt_file) as gts:
            self.gt_names = gts.read().splitlines()
        gts.close()

        with open(self.val_img_file) as imgs:
            self.val_img_names = imgs.read().splitlines()
        imgs.close()
        with open(self.val_gt_file) as gts:
            self.val_gt_names = gts.read().splitlines()
        gts.close()

    def load_config(self):
        # create config object
        cfg = load_dict(self.config_file)

        # add stuff for documentation to config
        cfg.img_file = self.img_file
        cfg.gt_file = self.gt_file
        cfg.images = self.img_names
        cfg.gts = self.gt_names
        cfg.init_file = self.init_file
        cfg.EPOCHS = self.epochs
        cfg.optimizer = self.optimizer
        cfg.CUDA_VISIBLE_DEVICES = self.cuda_device
        cfg.GPUS = self.num_gpus
        cfg.REDUCELRONPLATEAU = self.REDUCELRONPLATEAU
        self.cfg = cfg

    @staticmethod
    def ckpt_sorting(s):
        return int(str(s).split('-')[0].split('.')[1])

    def start_training(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)

        # compute number of batches per epoch
        nbatches_train, mod = divmod(len(self.img_names), self.cfg.BATCH_SIZE)
        nbatches_val, mod = divmod(len(self.val_img_file), self.cfg.BATCH_SIZE)
        self.cfg.STEPS = nbatches_train

        # print some run info
        print("Number of images: {}".format(len(self.img_names)))
        print("Number of epochs: {}".format(self.epochs))
        print("Number of batches: {}".format(nbatches_train))
        print("Batch size: {}".format(self.cfg.BATCH_SIZE))

        # tf config and session
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        K.set_session(sess)

        # instantiate model
        squeeze = SqueezeDet(self.cfg, self.EarlyPooling, self.depths, self.usecov3)
        # callbacks
        cb = []

        if self.optimizer == "adam":
            opt = optimizers.Adam(lr=self.cfg.LEARNING_RATE, clipnorm=self.cfg.MAX_GRAD_NORM)
            self.cfg.LR = 0.001 * self.num_gpus
        elif self.optimizer == "rmsprop":
            opt = optimizers.RMSprop(lr=0.001 * self.num_gpus, clipnorm=self.cfg.MAX_GRAD_NORM)
            self.cfg.LR = 0.001 * self.num_gpus
        elif self.optimizer == "adagrad":
            opt = optimizers.Adagrad(lr=1.0 * self.num_gpus, clipnorm=self.cfg.MAX_GRAD_NORM)
            self.cfg.LR = 1 * self.num_gpus
        # use default is nothing is given
        else:
            # create sgd with momentum and gradient clipping
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.cfg.LEARNING_RATE,
                                                                         decay_steps=12000, decay_rate=0.8,
                                                                         staircase=True)
            opt = optimizers.SGD(learning_rate=lr_schedule, momentum=self.cfg.MOMENTUM, nesterov=True)
            self.cfg.LR = self.cfg.LEARNING_RATE * self.num_gpus

        # save config file to log dir
        with open(os.path.join(os.path.dirname(self.config_file), 'config.pkl'), 'wb') as f:
            pickle.dump(self.cfg, f, pickle.HIGHEST_PROTOCOL)

        # add tensorboard callback
        tbCallBack = TensorBoard(log_dir=self.tb_dir, histogram_freq=0, write_graph=True, write_images=True)

        cb.append(tbCallBack)

        # if flag was given, add reducelronplateu callback
        if self.REDUCELRONPLATEAU:
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1,
                                          patience=4, min_lr=0.0)
            cb.append(reduce_lr)

        # print keras model summary
        if self.VERBOSE:
            print(squeeze.model.summary())

        # create train generator
        train_generator = generator_from_data_path(self.img_names, self.gt_names, config=self.cfg)
        val_generator = generator_from_data_path(self.val_img_names, self.val_gt_names, config=self.cfg)

        initial_epoch = 0
        if self.init_file is not None:
            print("Weights initialized by name from {}".format(self.init_file))
            squeeze.model.load_weights(self.init_file)
            # initial_epoch = int(os.path.basename(self.init_file).split('-')[0].split('.')[1])
        elif len(os.listdir(self.checkpoint_dir)) > 0:
            ckpt_list = os.listdir(self.checkpoint_dir)
            sorted_list = sorted(ckpt_list, key=self.ckpt_sorting)
            ckpt_path = os.path.join(self.checkpoint_dir, sorted_list[-1])
            squeeze.model.load_weights(ckpt_path)
            print("Weights initialized by name from {}".format(ckpt_path))
            initial_epoch = self.ckpt_sorting(sorted_list[-1])

        if self.freeze_landmark is not None:
            # Freezing some of the layers as per user input
            model_layer_names = []
            max_matched_index = -1
            for layer in squeeze.model.layers:
                model_layer_names.append(layer.name)
                if self.freeze_landmark in model_layer_names[-1]:
                    layerindex = squeeze.model.layers.index(layer)
                    if layerindex > max_matched_index:
                        max_matched_index = layerindex
            for layer in squeeze.model.layers[:max_matched_index]:
                layer.trainable = False
            print("layers frozen till {}".format(squeeze.model.layers[max_matched_index].name))

        # add a checkpoint saver
        ckp_saver = ModelCheckpoint(self.checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True, mode='auto', period=1)
        cb.append(ckp_saver)

        # compile model from squeeze object, loss is not a function of model directly
        squeeze.model.compile(optimizer=opt,
                              loss=[squeeze.loss],
                              metrics=[squeeze.loss_without_regularization, squeeze.bbox_loss, squeeze.class_loss,
                                       squeeze.conf_loss])

        # actually do the training
        squeeze.model.fit_generator(train_generator, epochs=self.epochs, validation_data=val_generator,
                                    validation_steps=nbatches_val, validation_freq=self.validation_freq,
                                    steps_per_epoch=nbatches_train, callbacks=cb, initial_epoch=initial_epoch)

        output_index = get_output_layer_index(squeeze.model)
        final_squeeze_model = tf.keras.Model(squeeze.model.input, squeeze.model.get_layer(index=output_index).output)
        final_squeeze_model.save_weights(self.logdir + "/train" + "/Humancount_train_model_weights.h5")
        final_squeeze_model.save(self.logdir + "/train" + "/Humancount_train_model.h5")
        gc.collect()
