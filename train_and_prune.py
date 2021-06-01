import argparse
import os
import random
import sys

from main.config.create_config import *
from training import Training


def create_dir(path_list):
    for path in path_list:
        print("Creating Direcory: {}".format(path))
        if not os.path.exists(path):
            os.mkdir(path)


class PreTrainingProcesses:
    def __init__(self, kitti_dataset_path, val_split_ratio, logdir, config_name, gray, usedefaultval):
        input_kitti_path = os.path.abspath(kitti_dataset_path)
        output_data_path = os.path.abspath(logdir)
        images_path = os.path.join(input_kitti_path, 'training', 'images')
        labels_path = os.path.join(input_kitti_path, 'training', 'labels')
        data_logdir = os.path.join(output_data_path, 'data')
        config_logdir = os.path.join(output_data_path, 'config')
        config_file_path = os.path.join(output_data_path, 'config', config_name)
        create_dir([output_data_path, data_logdir, config_logdir])

        self.image_train_path = os.path.join(data_logdir, "img_train.txt")
        self.label_train_path = os.path.join(data_logdir, "gt_train.txt")

        self.image_val_path = os.path.join(data_logdir, "img_val.txt")
        self.label_val_path = os.path.join(data_logdir, "gt_val.txt")

        if usedefaultval:
            val_file_path = os.path.join(input_kitti_path, "ImageSets", "val.txt")
            train_file_path = os.path.join(input_kitti_path, "ImageSets", "train.txt")
            with open(val_file_path) as file:
                validation_list = [x.strip() for x in file.readlines()]
            with open(train_file_path) as file:
                train_list = [x.strip() for x in file.readlines()]
            with open(self.image_train_path, 'w') as img_train:
                with open(self.label_train_path, 'w') as gt_train:
                    for element in train_list:
                        img_path_png = os.path.join(images_path, element + ".png")
                        img_path_jpg = os.path.join(images_path, element + ".jpg")
                        label_path = os.path.join(labels_path, element + ".txt")
                        if os.path.exists(label_path):
                            if os.path.exists(img_path_png):
                                out_image_path = img_path_png
                            elif os.path.exists(img_path_jpg):
                                out_image_path = img_path_jpg
                            else:
                                print("Unsupported image type !!")
                                sys.exit()
                            img_train.write(str(out_image_path) + "\n")
                            gt_train.write(str(label_path) + "\n")
                        else:
                            print("Label doesn't exist {}".format(label_path))
            with open(self.image_val_path, 'w') as img_train:
                with open(self.label_val_path, 'w') as gt_train:
                    for element in validation_list:
                        img_path_png = os.path.join(images_path, element + ".png")
                        img_path_jpg = os.path.join(images_path, element + ".jpg")
                        label_path = os.path.join(labels_path, element + ".txt")
                        if os.path.exists(label_path):
                            if os.path.exists(img_path_png):
                                out_image_path = img_path_png
                            elif os.path.exists(img_path_jpg):
                                out_image_path = img_path_jpg
                            else:
                                print("Unsupported image type !!")
                                sys.exit()
                            img_train.write(str(out_image_path) + "\n")
                            gt_train.write(str(label_path) + "\n")
                        else:
                            print("Label doesn't exist {}".format(label_path))
        else:
            self.train_eval_split(images_path, labels_path, val_split_ratio)

        if os.path.exists(config_file_path):
            print("Config File Already Exist at {}".format(config_file_path))
        else:
            cfg = squeezeDet_config(gray)
            save_dict(cfg, config_file_path)

    def train_eval_split(self, images_path, labels_path, val_split_ratio):
        img_names = []
        gt_names = []
        for image in os.listdir(images_path):
            img_names.append(os.path.join(images_path, image))

        for label in os.listdir(labels_path):
            gt_names.append(os.path.join(labels_path, label))

        img_names.sort()
        gt_names.sort()

        shuffled = list(zip(img_names, gt_names))
        random.shuffle(shuffled)
        img_names, gt_names = zip(*shuffled)

        percent_val = int(val_split_ratio)
        percent_train = 100 - percent_val
        n_train = int(np.floor(len(img_names) * percent_train / 100))

        n_val = int(np.floor(len(img_names) * (percent_train + percent_val) / 100))

        assert len(img_names) == len(gt_names)

        with open(self.image_train_path, 'w') as img_train:
            img_train.write("\n".join(img_names[0:n_train]))

        with open(self.label_train_path, 'w') as gt_train:
            gt_train.write("\n".join(gt_names[0:n_train]))

        with open(self.image_val_path, 'w') as img_val:
            img_val.write("\n".join(img_names[n_train:n_val]))

        with open(self.label_val_path, 'w') as gt_val:
            gt_val.write("\n".join(gt_names[n_train:n_val]))


def main(args):
    PreTrainingProcesses(args.dataset_path, int(args.val_set_size), args.logdir, args.configfile, args.gray,
                         args.usedefaultvalset)
    config_path = os.path.join(args.logdir, 'config', args.configfile)
    logdir = os.path.abspath(args.logdir)
    Training(args.epochs[0], args.filterdepths, args.early_pooling, config_path, logdir, args.init,
             args.gpuid, int(args.validation_freq), args.freeze_landmark, args.usecov3)
    if args.runpruning:
        from pruning import Pruning
        trained_model_path = os.path.join(os.path.abspath(logdir), "train", "Humancount_train_model.h5")
        pruning = Pruning(args.epochs[1], args.epochs[2], args.filterdepths, args.sparsity, args.early_pooling,
                          config_path, logdir, trained_model_path,
                          args.gpuid, int(args.validation_freq), args.freeze_landmark, args.usecov3)
        print("Output model depths {}".format(pruning.non_pruned_depths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train squeezeDet model.')
    parser.add_argument("--dataset_path", required=True, help="kitti dataset path")
    parser.add_argument("--val_set_size", default=5, help="validation set percentage (keep it low)")
    parser.add_argument("--validation_freq", default=10, help="validation frequency in #epochs (keep it high)")
    parser.add_argument("--logdir", default='./log', help="target log directory path")
    parser.add_argument("--gray", action="store_true", help="Add flag to crete model with one channel")
    parser.add_argument("--runpruning", action="store_true", help="Add flag to run pruning after training completed")
    parser.add_argument("--early_pooling", action="store_true", help="Perform early pooling on SqueezeDet Network. "
                                                                     "DEFAULT: False")
    parser.add_argument("--epochs", default='110,10,50', help="Training, Pruning, Finetuning Epochs")
    parser.add_argument("--filterdepths", default='32,32,32,64,64,128,128',
                        help="Mention each layer filter depth")
    parser.add_argument("--configfile", default='squeezedet.config',
                        help="pass only filename of config file if you want to reuse "
                             "already existing file at '<logdir>/config/<configfile>'")
    parser.add_argument("--sparsity", default='0.25,0.25,0.35,0.35,0.45,0.45', help="Mention the sparsity")
    parser.add_argument("--init", default=None, help="weight file in .h5 format to start training from. If argument "
                                                     "is none, training starts from the beginning. DEFAULT: None")
    parser.add_argument("--gpuid", default=0, help="GPU id to run training")
    parser.add_argument("--usedefaultvalset", action="store_true", help="use flag if you want to reuse val.txt from "
                                                                        "kitti dataset")
    parser.add_argument("--freeze_landmark", default=None, help="Name subpart of layer till you want to freeze the"
                                                                "weights")
    parser.add_argument("--usecov3", action="store_true", help="use normal convolution instead of dw as first layer")

    args = parser.parse_args()
    args.epochs = [int(item) for item in args.epochs.split(',')]
    args.sparsity = [float(item) for item in args.sparsity.split(',')]
    args.filterdepths = [int(item) for item in args.filterdepths.split(',')]
    main(args)
