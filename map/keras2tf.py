# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:53:02 2018

@author: ytan
"""
import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import *

import binary_ops
from binary_ops import *


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()

        print(input_graph_def.node)
        for n in input_graph_def.node:
            print(n.name)

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def convert(output_model_file, weight_file_path=r'./../../../../networks/Keras_ObjectCount/Keras_ObjectCount.h5'):
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)
    with tf.compat.v1.keras.backend.get_session() as sess:
        net_model = tf.keras.models.load_model(weight_file_path, compile=False,
                                               custom_objects={"bo": binary_ops,
                                                               "binary_ops": binary_ops,
                                                               "lin_8b_quant": lin_8b_quant,
                                                               "FixedDropout": FixedDropout,
                                                               "MyInitializer": MyInitializer,
                                                               "MyRegularizer": MyRegularizer,
                                                               "MyConstraints": MyConstraints})
        dataConfigureStr = 'data_format'
        dataFormatType = 'channels_last'
        if len(net_model.layers) > 0 and dataConfigureStr in net_model.layers[0].get_config():
            dataFormatType = net_model.layers[0].get_config()['data_format']
            tf.keras.backend.set_image_data_format(dataFormatType)
            print('setDataConfigure to be same as model configuration %s' % dataFormatType)
        else:
            tf.keras.backend.set_image_data_format(dataFormatType)
            print('Can not find model data_format configuration,  setDataConfigure to be default %s' % dataFormatType)

        num_output = 1
        output_node_prefix = 'output_node'
        pred = [None] * num_output
        pred_node_names = [None] * num_output
        print("*"*100,net_model.input.name.split(":")[0])
        for i in range(num_output):
            pred_node_names[i] = output_node_prefix + str(i)
            pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                                    pred_node_names)
            try:
                if net_model.input.shape and len(net_model.input.shape) == 3:
                    ensure_graph_is_valid(constant_graph)
                    constant_graph = strip_unused_lib.strip_unused(
                        constant_graph, [net_model.input.name.split(":")[0]], pred_node_names,
                        list([net_model.input._dtype.as_datatype_enum]))
                    constant_graph = graph_util.remove_training_nodes(
                        constant_graph, pred_node_names)
                    constant_graph = fuse_resize_and_conv(constant_graph, pred_node_names)
                    ensure_graph_is_valid(constant_graph)
            except:
                pass

        output_fld = os.path.abspath(os.path.dirname(output_model_file))
        tf.io.write_graph(constant_graph, output_fld, os.path.abspath(output_model_file), as_text=False)
        #return constant_graph

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-k", "--kerasmodel", required=True, help="Input Model Path")
#    args = parser.parse_args()
#    modelAbsPath = os.path.abspath(args.kerasmodel)
#    output_file = modelAbsPath.replace('.h5', '.pb')
#    convert(output_file, modelAbsPath)
