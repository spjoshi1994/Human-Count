import numpy as np
import tensorflow as tf
from main.model.squeezeDet import SqueezeDet


def get_output_layer_index(model):
    layer_count = 0
    output_index = layer_count
    output_layer_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size == (3, 3):
            output_index = layer_count
            output_layer_name = layer.name
        layer_count += 1
    if output_layer_name is not None:
        print("Model Output layer {} on index {}".format(output_layer_name, output_index))
    return output_index


def get_batchnormalization_prunned_weights(weights, a_channels):
    mask = []
    for i in range(weights.shape[0]):
        if i in a_channels:
            mask.append(True)
        else:
            mask.append(False)
    return weights[mask]


def get_bias_prunned_weights(weights, a_channels):
    mask = []
    for i in range(weights.shape[0]):
        if i in a_channels:
            mask.append(True)
        else:
            mask.append(False)
    return weights[mask]


def get_normal_convolution_prunned_weights(weights, a_channels, o_channels):
    mask = []
    for i in range(weights.shape[0]):
        if i in a_channels:
            mask.append(True)
        else:
            mask.append(False)
    if len(a_channels) > 0:
        w = np.transpose(weights, (3, 2, 0, 1))
        wt = np.ndarray((len(a_channels), w.shape[1], w.shape[2], w.shape[3]))
        count = 0
        for i in range(w.shape[0]):
            if i in a_channels:
                wt[count, :, :, :] = w[i, :, :, :]
                count = count + 1
        weights = np.transpose(wt, (2, 3, 1, 0))
    if len(o_channels) > 0:
        w = np.transpose(weights, (2, 3, 0, 1))
        wt = np.ndarray((len(o_channels), w.shape[1], w.shape[2], w.shape[3]))
        count = 0
        for i in range(w.shape[0]):
            if i in o_channels:
                wt[count, :, :, :] = w[i, :, :, :]
                count = count + 1
        weights = np.transpose(wt, (2, 3, 0, 1))
    return weights


def get_depthwise_convolution_prunned_weights(weights, a_channels):
    mask = []
    for i in range(weights.shape[0]):
        if i in a_channels:
            mask.append(True)
        else:
            mask.append(False)
    if len(a_channels) > 0:
        w = np.transpose(weights, (2, 3, 0, 1))
        wt = np.ndarray((len(a_channels), w.shape[1], w.shape[2], w.shape[3]))
        count = 0
        for i in range(w.shape[0]):
            if i in a_channels:
                wt[count, :, :, :] = w[i, :, :, :]
                count = count + 1
        weights = np.transpose(wt, (2, 3, 0, 1))
    return weights


def transfer_weights(cfg, pruned_model_obj, final_depths, pooling_early, useconv3):
    empty_model = SqueezeDet(cfg, pooling_early, final_depths, useconv3)
    print(final_depths)
    num_output = cfg.ANCHOR_PER_GRID * (cfg.CLASSES + 1 + 4)
    last_layer_name = "conv12"
    l_a_channels = []
    for i in range(num_output):
        l_a_channels.append(i)

    if cfg['N_CHANNELS'] == 3:
        o_channels = [0, 1, 2]  # ALWWAYS RGB IMAGE
    else:
        o_channels = [0]
    a_channels = [0]
    for layer, elayer in zip(pruned_model_obj.layers, empty_model.model.layers[1:]):
        if layer.__class__.__name__ in ["Conv2D"]:
            if layer.get_weights()[0].shape[0] == 1 and layer.get_weights()[0].shape[1] == 1 :
                o_channels = a_channels
                weights = layer.get_weights()[0]
                w = np.transpose(weights[0][0])
                non_pruned_channels = []
                pruned_channels = []
                for j in range(w.shape[0]):
                    if np.count_nonzero(w[j]) != 0:
                        non_pruned_channels.append(j)
                    else:
                        pruned_channels.append(j)
                non_length = len(non_pruned_channels)
                add = 4 - (non_length % 4) if non_length % 4 != 0 else 0
                if add != 0:
                    non_pruned_channels += pruned_channels[0:add]
                
                if 'fire1' in layer.name:
                    for val in pruned_channels:
                        if val not in non_pruned_channels:
                            non_pruned_channels.append(val)
                a_channels = non_pruned_channels
                if len(layer.get_weights()) == 1:
                    elayer.set_weights([get_normal_convolution_prunned_weights(weights, a_channels, o_channels)])
                else:
                    elayer.set_weights([get_normal_convolution_prunned_weights(weights, a_channels, o_channels),
                                        get_bias_prunned_weights(layer.get_weights()[1], a_channels)])
            else:
                weights = layer.get_weights()[0]
                if layer.name == last_layer_name:
                    o_channels = a_channels
                    a_channels = l_a_channels
                if len(layer.get_weights()) == 1:
                    elayer.set_weights([get_normal_convolution_prunned_weights(weights, a_channels, o_channels)])
                else:
                    elayer.set_weights([get_normal_convolution_prunned_weights(weights, a_channels, o_channels),
                                        get_bias_prunned_weights(layer.get_weights()[1], a_channels)])

        elif layer.__class__.__name__ in ["BatchNormalization"]:
            final_weights = []
            for i in range(len(layer.get_weights())):
                final_weights.append(get_batchnormalization_prunned_weights(layer.get_weights()[i], a_channels))
            elayer.set_weights(final_weights)
        elif layer.__class__.__name__ in ["DepthwiseConv2D"]:
            weights = layer.get_weights()[0]
            if len(layer.get_weights()) == 1:
                elayer.set_weights([get_depthwise_convolution_prunned_weights(weights, a_channels)])
            else:
                elayer.set_weights([get_depthwise_convolution_prunned_weights(weights, a_channels)],
                                   get_bias_prunned_weights(layer.get_weights()[1], a_channels))
    return empty_model
