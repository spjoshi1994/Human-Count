# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

import json

import numpy as np
from easydict import EasyDict as edict


def squeezeDet_config(gray):
    """Specify the parameters to tune below."""
    cfg = edict()

    # we only care about these, others are omitted
    cfg.CLASS_NAMES = ['person']

    # number of categories to classify
    cfg.CLASSES = len(cfg.CLASS_NAMES)

    # classes to class index dict
    cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, range(cfg.CLASSES)))

    # Probability to keep a node in dropout
    cfg.KEEP_PROB = 0.5

    # a small value used to prevent numerical instability
    cfg.EPSILON = 1e-7

    # threshold for safe exponential operation
    cfg.EXP_THRESH = 1.0

    # image properties
    cfg.IMAGE_WIDTH = 320
    cfg.IMAGE_HEIGHT = 240
    if gray:
        cfg.N_CHANNELS = 1
    else:
        cfg.N_CHANNELS = 3

    # batch sizes
    cfg.BATCH_SIZE = 20
    cfg.VISUALIZATION_BATCH_SIZE = 20

    # SGD + Momentum parameters
    cfg.WEIGHT_DECAY = 0.0001
    cfg.LEARNING_RATE = 0.005
    cfg.MAX_GRAD_NORM = 1.0
    cfg.MOMENTUM = 0.9

    # coefficients of loss function
    cfg.LOSS_COEF_BBOX = 5.0
    cfg.LOSS_COEF_CONF_POS = 75.0
    cfg.LOSS_COEF_CONF_NEG = 100.0
    cfg.LOSS_COEF_CLASS = 1.0

    # thresholds for evaluation
    cfg.NMS_THRESH = 0.4
    cfg.PROB_THRESH = 0.005
    cfg.TOP_N_DETECTION = 10
    cfg.IOU_THRESHOLD = 0.5
    cfg.FINAL_THRESHOLD = 0.0
    div_scale_h = 2.0 * (224 / cfg.IMAGE_HEIGHT)
    div_scale_w = 2.0 * (224 / cfg.IMAGE_WIDTH)

    cfg.ANCHOR_SEED = np.array([[int(368. / div_scale_w), int(368. / div_scale_h)],
                                [int(276. / div_scale_w), int(276. / div_scale_h)],
                                [int(184. / div_scale_w), int(184. / div_scale_h)],
                                [int(138. / div_scale_w), int(138. / div_scale_h)],
                                [int(92. / div_scale_w), int(92. / div_scale_h)],
                                [int(69. / div_scale_w), int(69. / div_scale_h)],
                                [int(46. / div_scale_w), int(46. / div_scale_h)]])

    cfg.ANCHOR_PER_GRID = len(cfg.ANCHOR_SEED)

    cfg.ANCHORS_HEIGHT = 15
    cfg.ANCHORS_WIDTH = 20

    return cfg


def create_config_from_dict(dictionary={}, name="squeeze.config"):
    """Creates a config and saves it
    
    Keyword Arguments:
        dictionary {dict} -- [description] (default: {{}})
        name {str} -- [description] (default: {"squeeze.config"})
    """

    cfg = squeezeDet_config()

    for key, value in dictionary.items():
        cfg[key] = value

    save_dict(cfg, name)


# save a config files to json
def save_dict(dict, name="squeeze.config"):
    # change np arrays to lists for storing
    for key, val, in dict.items():

        if type(val) is np.ndarray:
            dict[key] = val.tolist()

    with open(name, "w") as f:
        json.dump(dict, f, sort_keys=True, indent=0)  # This saves the array in .json format


def load_dict(path):
    """Loads a dictionary from a given path name
    
    Arguments:
        path {[type]} -- string of path
    
    Returns:
        [type] -- [description]
    """

    with open(path, "r") as f:
        cfg = json.load(f)  # This loads the array from .json format

    # changes lists back
    for key, val, in cfg.items():

        if type(val) is list:
            cfg[key] = np.array(val)

    # cast do easydict
    cfg = edict(cfg)

    # create full anchors from seed
    cfg.ANCHOR_BOX, cfg.N_ANCHORS_HEIGHT, cfg.N_ANCHORS_WIDTH = set_anchors(cfg)
    cfg.ANCHORS = len(cfg.ANCHOR_BOX)

    # if you added a class in the config manually, but were to lazy to update
    cfg.CLASSES = len(cfg.CLASS_NAMES)
    cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, range(cfg.CLASSES)))

    return cfg


# compute the anchors for the grid from the seed
def set_anchors(cfg):
    H, W, B = cfg.ANCHORS_HEIGHT, cfg.ANCHORS_WIDTH, cfg.ANCHOR_PER_GRID

    anchor_shapes = np.reshape(
        [cfg.ANCHOR_SEED] * H * W,
        (H, W, B, 2)
    )
    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W + 1) * float(cfg.IMAGE_WIDTH) / (W + 1)] * H * B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H + 1) * float(cfg.IMAGE_HEIGHT) / (H + 1)] * W * B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )

    return anchors, H, W
