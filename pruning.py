import gc
import os
import pickle
import re

import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_model_optimization.sparsity import keras as sparsity

from main.config.create_config import load_dict
from main.model.dataGenerator import generator_from_data_path
from main.model.squeezeDet import SqueezeDet
from main.pruning_structure import transfer_weights, get_output_layer_index


class Pruning:
    def __init__(self, epochs, finetune_epochs, depths, sparsity, earlypooling, config_file, logdir, init_file,
                 cuda_device, validation_freq, freeze_landmark, usecov3):
        self.epochs = epochs
        self.fientune_epochs = finetune_epochs
        self.depths = depths
        self.sparsity = sparsity
        self.EarlyPooling = earlypooling
        self.optimizer = 'adam'
        self.init_file = init_file
        self.logdir = logdir
        self.validation_freq = validation_freq
        self.freeze_landmark = freeze_landmark
        self.usecov3 = usecov3

        self.VERBOSE = True
        self.REDUCELRONPLATEAU = True  # True ##False for non-mobilenet
        self.num_gpus = 1
        self.cuda_device = cuda_device
        self.train_checkpoint_dir = logdir + "/train" + "/checkpoints"

        # create subdirs for logging of checkpoints and tensorboard stuff
        self.checkpoint_dir = logdir + "/prune" + "/checkpoints"
        self.tb_dir = logdir + "/prune" + "/tensorboard"
        self.checkpoint_dir_fine = logdir + "/fine" + "/checkpoints"
        self.tb_dir_fine = logdir + "/fine" + "/tensorboard"

        self.img_file = os.path.join(logdir, 'data', 'img_train.txt')
        self.gt_file = os.path.join(logdir, 'data', 'gt_train.txt')
        self.val_img_file = os.path.join(logdir, 'data', 'img_val.txt')
        self.val_gt_file = os.path.join(logdir, 'data', 'gt_val.txt')
        self.config_file = config_file

        self.revise_directory_structure()
        self.load_training_data()
        self.load_config()
        self.prune()

    def revise_directory_structure(self):
        # delete old checkpoints and tensorboard stuff
        if tf.gfile.Exists(self.checkpoint_dir):
            tf.gfile.DeleteRecursively(self.checkpoint_dir)

        if tf.gfile.Exists(self.tb_dir):
            tf.gfile.DeleteRecursively(self.tb_dir)

        tf.gfile.MakeDirs(self.tb_dir)
        tf.gfile.MakeDirs(self.checkpoint_dir)

        # delete old checkpoints and tensorboard stuff
        if tf.gfile.Exists(self.checkpoint_dir_fine):
            tf.gfile.DeleteRecursively(self.checkpoint_dir_fine)

        if tf.gfile.Exists(self.tb_dir_fine):
            tf.gfile.DeleteRecursively(self.tb_dir_fine)

        tf.gfile.MakeDirs(self.tb_dir_fine)
        tf.gfile.MakeDirs(self.checkpoint_dir_fine)

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

    def prune(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
        # compute number of batches per epoch
        nbatches_train, mod = divmod(len(self.img_names), self.cfg.BATCH_SIZE)
        nbatches_val, mod = divmod(len(self.val_img_file), self.cfg.BATCH_SIZE)
        self.cfg.STEPS = nbatches_train

        # tf config and session
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        K.set_session(sess)

        # instantiate model
        squeeze = SqueezeDet(self.cfg, self.EarlyPooling, self.depths, self.usecov3)
        # callbacks
        cb = []
        update_step = sparsity.UpdatePruningStep()

        end_step = np.ceil(1.0 * len(self.img_names) / self.cfg.BATCH_SIZE).astype(np.int32) * self.epochs
        print("Pruning steps: {}".format(end_step))

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
        with open(self.logdir + '/prune' + '/config.pkl', 'wb') as f:
            pickle.dump(self.cfg, f, pickle.HIGHEST_PROTOCOL)

        # add tensorboard callback
        tbCallBack = TensorBoard(log_dir=self.tb_dir, histogram_freq=0,
                                 write_graph=True, write_images=True)

        cb.append(tbCallBack)
        cb.append(update_step)
        # if flag was given, add reducelronplateu callback
        if self.REDUCELRONPLATEAU:
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1,
                                          patience=4, min_lr=0.0)
            cb.append(reduce_lr)

        # print keras model summary
        # if self.VERBOSE:
        #     print(squeeze.model.summary())

        if self.init_file != "none":
            print("Weights initialized by name from {}".format(self.init_file))
            # load_only_possible_weights(squeeze.model, init_file, verbose=VERBOSE)
            squeeze.model.load_weights(self.init_file)

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
            print("layers freezed till {}".format(squeeze.model.layers[max_matched_index].name))

        pruned_model = tf.keras.Sequential()
        pruned_model.run_eagerly = True
        for layer in squeeze.model.layers:
            if 'fire1' not in layer.name and re.match(r"fire(\d+)_PW", layer.name):
                sparsity_value = self.sparsity.pop(0) if len(self.sparsity) > 0 else 0.25
                pruning_schedule = sparsity.PolynomialDecay(
                    initial_sparsity=0.01, final_sparsity=float(sparsity_value),
                    begin_step=0, end_step=end_step, frequency=100)
                pruned_model.add(sparsity.prune_low_magnitude(
                    layer,
                    pruning_schedule,
                    block_size=(1, 1)
                ))
            else:
                pruned_model.add(layer)

        # create train generator
        train_generator = generator_from_data_path(self.img_names, self.gt_names, config=self.cfg)
        val_generator = generator_from_data_path(self.val_img_names, self.val_gt_names, config=self.cfg)
        # add a checkpoint saver
        ckp_saver = ModelCheckpoint(self.checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True, mode='auto', period=1)
        cb.append(ckp_saver)

        if self.REDUCELRONPLATEAU:
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1,
                                          patience=4, min_lr=0.0)
            cb.append(reduce_lr)

        # compile model from squeeze object, loss is not a function of model directly
        pruned_model.compile(optimizer=opt, loss=[squeeze.loss],
                             metrics=[squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss])

        pruned_model.fit_generator(train_generator, epochs=self.epochs, validation_data=val_generator,
                                   validation_steps=nbatches_val, validation_freq=self.validation_freq,
                                   steps_per_epoch=nbatches_train, callbacks=cb)
        gc.collect()

        # Fine Tuning model
        final_model = sparsity.strip_pruning(pruned_model)
        final_model.save(self.logdir + "/prune" + "/humancount_pruning_trained_model.h5")

        non_pruned_layers = {}
        non_pruned_depths = [self.depths[0]]
        for i, w in enumerate(final_model.get_weights()):
            if 'fire1' in final_model.weights[i].name:
                continue
            if re.match(r".*\S(fire(\d+)_PW\Skernel)", final_model.weights[i].name):
                weights = np.transpose(w[0][0])
                non_pruned_channels = []
                for j in range(weights.shape[0]):
                    if np.count_nonzero(weights[j]) != 0:
                        non_pruned_channels.append(j)
                non_pruned_channels_length = len(non_pruned_channels)
                if non_pruned_channels_length % 4 != 0:
                    non_pruned_channels_length += (4 - (non_pruned_channels_length % 4))
                non_pruned_depths.append(non_pruned_channels_length)
                l_name = final_model.weights[i].name.split('/')[0]
                non_pruned_layers[l_name] = non_pruned_channels
        fine_model = transfer_weights(self.cfg, final_model, non_pruned_depths, self.EarlyPooling, self.usecov3)
        cb.remove(update_step)
        cb.remove(ckp_saver)
        cb.remove(tbCallBack)
        tbCallBack = TensorBoard(log_dir=self.tb_dir_fine, histogram_freq=0, write_graph=True, write_images=True)
        cb.append(tbCallBack)
        ckp_saver = ModelCheckpoint(self.checkpoint_dir_fine + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True, mode='auto', period=1)
        cb.append(ckp_saver)
        if self.VERBOSE:
            print(fine_model.model.summary())

        fine_model.model.compile(optimizer=opt, loss=[squeeze.loss],
                                 metrics=[squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss])
        fine_model.model.fit_generator(train_generator, epochs=self.fientune_epochs, validation_data=val_generator,
                                       validation_steps=nbatches_val, validation_freq=self.validation_freq,
                                       steps_per_epoch=nbatches_train, callbacks=cb)
        output_index = get_output_layer_index(fine_model.model)
        final_squeeze_model = tf.keras.Model(fine_model.model.input,
                                             fine_model.model.get_layer(index=output_index).output)
        final_squeeze_model.save(self.logdir + "/fine/final_pruned_freezed_model.h5")
        self.non_pruned_depths = non_pruned_depths
