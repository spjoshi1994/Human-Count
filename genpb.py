"""
Created on Thu Sep 6 15:26:30 2018

@author: ytan
""" 
import os
import sys
from enum import Enum 
import tensorflow as tf 
from tensorflow.python.platform import gfile
from google.protobuf import text_format as pbtf  

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph
import argparse
import numpy as np

from utils import Size, Point, Overlap, Score, Box, prop2abs, normalize_box
from collections import namedtuple, defaultdict
from math import sqrt, log, exp
import argparse
import math

import multiprocessing as mp
from average_precision import APCalculator, APs2mAP
from training_data import TrainingData
from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdmv2 import SSDMV2
from utils import *
from tqdm import tqdm
import config
import glob


if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)


SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SSDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])


SSD_PRESETS = {
    'mv2': SSDPreset(name = 'mv2',
                        image_size = Size(224, 224),
                        maps = [
                            SSDMap(Size(14, 14), 0.2,   [2, 3, 0.5, 1./3.]),
                            SSDMap(Size(7, 7), 0.375, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 4, 4), 0.55,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 2,  2), 0.725, [2, 0.5])
                        ],
                        extra_scale = 1.075,
                        num_anchors = 1582),}

preset = SSDPreset(name='mv2', image_size=Size(w=224, h=224), maps=[SSDMap(size=Size(w=14, h=14), scale=0.2, aspect_ratios=[2, 3, 0.5, 0.3333333333333333]), SSDMap(size=Size(w=7, h=7), scale=0.375, aspect_ratios=[2, 3, 0.5, 0.3333333333333333]), SSDMap(size=Size(w=4, h=4), scale=0.55, aspect_ratios=[2, 3, 0.5, 0.3333333333333333]), SSDMap(size=Size(w=2, h=2), scale=0.725, aspect_ratios=[2, 0.5])], extra_scale=1.075, num_anchors=1582)

def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step



def gen_pbtxt(dir,name):
  lr_values = '0.001;0.0001;0.00001'
  lr_boundaries = '320000;400000'
  weight_decay = 0.0005
  momentum = 0.9
  # try:
  #     td = TrainingData('pascal-voc')
  #     print('[i] # training samples:   ', td.num_train)
  #     print('[i] # validation samples: ', td.num_valid)
  #     print('[i] # classes:            ', td.num_classes)
  #     print('[i] Image size:           ', td.preset.image_size)
  #     print(td.preset)
  # except (AttributeError, RuntimeError) as e:
  #     print('[!] Unable to load training data:', str(e))
  #     # return 1
  with tf.Session() as sess:
    global_step = None
    # if start_epoch == 0:
    lr_values = lr_values.split(';')
    try:
        lr_values = [float(x) for x in lr_values]
    except ValueError:
        print('[!] Learning rate values must be floats')
        sys.exit(1)

    lr_boundaries = lr_boundaries.split(';')
    try:
        lr_boundaries = [int(x) for x in lr_boundaries]
    except ValueError:
        print('[!] Learning rate boundaries must be ints')
        sys.exit(1)

    ret = compute_lr(lr_values, lr_boundaries)
    learning_rate, global_step = ret
    is_training = False

    net = SSDMV2(sess, preset, is_training)
    # if start_epoch != 0:
    #     print("hi>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     net.build_from_metagraph(metagraph_file, checkpoint_file)
    #     net.build_optimizer_from_metagraph()
    # else:
    net.build_from_mv2(1)
    net.build_optimizer(learning_rate=learning_rate,
                            global_step=global_step,
                            weight_decay=weight_decay,
                            momentum=momentum)
    # if args.generate_pbtxt:
    new_name = name+".pbtxt"
    tf.train.write_graph(sess.graph_def, dir, new_name, as_text=True)



def get_latest_ckpt(file_dir):
  # import glob
  # temp = []
  # p = os.listdir("./checkpoints/")
  # for i in p:
  #   if ".pbtxt" in i:
  #     temp.append(i)
  #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  file_name =os.path.join(file_dir,"*.meta")
  #print(file_name,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  list_of_files = glob.glob(file_name)
  #list_of_files1 = glob.glob("checkpoints/*.pbtxt")
  #print(list_of_files)
  latest_file = max(list_of_files, key=os.path.getmtime)
  return latest_file.split("/")[-1].split(".")[0] 





def get_platform():
  platforms = {
    'linux1': 'Linux',
    'linux2': 'Linux',
    'linux' : 'Linux',
    'darwin': 'OS X',
    'win32': 'Windows'
  }
  if sys.platform not in platforms:
    return sys.platform
  return platforms[sys.platform]

def find_output_nodes(gdef, currNode):
    outNodes = []
    # print('find_output_nodes currNode ', currNode)

    for node in gdef.node:
        # if(node.op=='Sub'):
        # print('Sub node name  input',node.name,node.input)
        if node.name == currNode.name:
            continue
        if (currNode.op == 'Split'):
            for text in node.input:
                # print(currNode.name,'  output : ',node.name, node.input)
                if currNode.name in text:  # As split can be as input  as split:2,split:1, split
                    outNodes.append(node)
                    print(('Split out', node.name))
                    break

        else:
            if currNode.name in node.input:
                # print(currNode.name,'  output : ',node.name, node.input)
                outNodes.append(node)
    return outNodes
class tfPBfileMode(Enum):
     Binary=0
     Text=1
 
def setNodeAttribute(node, tag ,shapeArray):
    if(shapeArray is not None):
        if(tag=='shapes'): # here we assume  always only get first shape in shape list
             if(len(shapeArray)==4):
                  node.attr[tag].list.shape.extend([tf.TensorShape(shapeArray).as_proto()] )
             elif( len(shapeArray)==3): 
                 node.attr[tag].list.shape[0].dim[0].size =1
                 node.attr[tag].list.shape[0].dim[0].size = shapeArray[0]
                 node.attr[tag].list.shape[0].dim[1].size = shapeArray[1]
                 node.attr[tag].list.shape[0].dim[2].size = shapeArray[2]
                 
        if(tag=='shape'): #TODO  Set shape is not working  
                         
             if(len(shapeArray)==4):
                  node.attr[tag].shape.CopyFrom(tf.TensorShape(shapeArray).as_proto())    
             elif( len(shapeArray)==3): 
                 shapeArray4= [None] *4
                 
                 shapeArray4[0] = 1 
                 shapeArray4[1] = shapeArray[1]
                 shapeArray4[2] = shapeArray[2]
                 shapeArray4[3] = shapeArray[3]
                 node.attr[tag].shape.CopyFrom(tf.TensorShape(shapeArray).as_proto())     
       
             
def getInputShapeForTF(node, tag ,forceNumTobe1=True):
    shapeArray= [None] *4
    if(tag=='shapes'): #TODO here we assume and always only get first shape in shape list
         if(len(node.attr[tag].list.shape)>0):
             if(len( node.attr[tag].list.shape[0].dim)==4): 
                  
                 for i in range(len(node.attr[tag].list.shape[0].dim)):
                     shapeArray[i] = node.attr[tag].list.shape[0].dim[i].size
            
             elif( len(node.attr[tag].list.shape[0].dim)==3):
                  
                 shapeArray[0] = 1 
                 shapeArray[1] = node.attr[tag].list.shape[0].dim[0].size
                 shapeArray[2] = node.attr[tag].list.shape[0].dim[1].size
                 shapeArray[3] = node.attr[tag].list.shape[0].dim[2].size
             
    if(tag=='shape'):  
         
         if(len( node.attr[tag].shape.dim)==4): 
              
             for i in range(len(node.attr[tag].shape.dim)):
                 shapeArray[i] = node.attr[tag].shape.dim[i].size
        
         elif( len(node.attr[tag].shape.dim)==3):
              
             shapeArray[0] = 1 
             shapeArray[1] = node.attr[tag].shape.dim[0].size
             shapeArray[2] = node.attr[tag].shape.dim[1].size
             shapeArray[3] = node.attr[tag].shape.dim[2].size
    
    if(tag=='output_shapes'):  
          if(len(node.attr[tag].list.shape)>0):
             if(len( node.attr[tag].list.shape[0].dim)==4): 
                  
                 for i in range(len(node.attr[tag].list.shape[0].dim)):
                     shapeArray[i] = node.attr[tag].list.shape[0].dim[i].size
            
             elif( len(node.attr[tag].list.shape[0].dim)==3):
                  
                 shapeArray[0] = 1 
                 shapeArray[1] = node.attr[tag].list.shape[0].dim[0].size
                 shapeArray[2] = node.attr[tag].list.shape[0].dim[1].size
                 shapeArray[3] = node.attr[tag].list.shape[0].dim[2].size
    
    if(forceNumTobe1 and shapeArray[0] is not None):
        shapeArray[0]=1 
                   
    return shapeArray
             

def getShapeArrays(node): 
    inputShape= [None] *4 
    inputShape = getInputShapeForTF(node,'shape')  
    print('inputShape shape',inputShape)
    if ( inputShape[0] is None):
        inputShape = getInputShapeForTF(node,'shapes')
        print('inputShape shapes',inputShape)
    if (inputShape[0] is not None): 
        return inputShape
    else:
        inputShape = getInputShapeForTF(node,'output_shapes')
        print('output_shapes of input Node',inputShape)
        if(inputShape[0] is None) :
            msg=' **TensorFlow**: can not locate input shape information at: ' +node.name
            print(msg)
        else:
            return inputShape
       
        #raise Exception(msg) 

        
     
def createTensorboard(modelFullPath,tensorboardPath,runLocalimport_pb_to_tensorboard=True):  
      if not os.path.exists(tensorboardPath):
          os.makedirs(tensorboardPath)
      print('tensorboardPath:',tensorboardPath)
      map( os.unlink, (os.path.join( tensorboardPath,f) for f in os.listdir(tensorboardPath)) )
      pb2TB.import_to_tensorboard(modelFullPath,tensorboardPath)

def latestckpt(name):
	return int(name.split(".")[-2].split("-")[-1])
          
def parseCkptFolder(fullPathOfFolder,shapeInforNodeName,inputNodeName, outputNodeName):
    
    filename_w_ext = os.path.basename(fullPathOfFolder)
    modelName, file_extension = os.path.splitext(filename_w_ext)
    if get_platform() == 'Linux':
        folderDir=os.path.dirname(fullPathOfFolder)+'/'
    else:
        folderDir=os.path.dirname(fullPathOfFolder)+'\\'
    
    files = os.listdir(os.getcwd())
    meta_files = [s for s in files if s.endswith('.meta')]
    #meta_files = sorted(meta_files, key = latestckpt)
    meta_files = sorted(meta_files)
    graph_def = tf.GraphDef() 
    ckptFile = os.path.basename(meta_files[-1])
    ckptWith1ndExtension, ckpt_metansion = os.path.splitext(ckptFile)
    ckptWith2ndExtension, ckptextension = os.path.splitext(ckptWith1ndExtension)
    
    if(file_extension=='.pbtxt'):
        with tf.gfile.FastGFile(fullPathOfFolder, 'r') as f:
            graph_str = f.read()
            pbtf.Parse(graph_str, graph_def)  
            ckptPBMode=tfPBfileMode.Text 
    else:
        with tf.gfile.FastGFile(fullPathOfFolder, 'rb') as f:
            graph_def.ParseFromString(f.read()) 
            ckptPBMode=tfPBfileMode.Binary 
    inputShapeArray=[] 
    graph_nodes=[n for n in graph_def.node]
    for node in graph_nodes: 
        if shapeInforNodeName == node.name: 
           inputShapeArray=getShapeArrays(node)  
    return [ckptextension,ckptPBMode, folderDir,inputShapeArray] 
           
  

def settingsConf(modelInfors ): 
    checkpointExt=modelInfors[0]   
    pbFileType=modelInfors[1]
    checkpointPath= modelInfors[2] 
    folderPath=checkpointPath
    shapeArray= modelInfors[3] 
    
    if(pbFileType==tfPBfileMode.Binary):
        msuffix='.pb'   
        readMode='rb'
        binaryPB=True
    elif(pbFileType==tfPBfileMode.Text):
        binaryPB=False
        msuffix='.pbtxt' 
        readMode='r'  
    
    return msuffix,binaryPB,readMode,folderPath,checkpointExt,checkpointPath,shapeArray

def loadGraph(filePath,binaryPB):
    graph_def = tf.GraphDef()
    if(binaryPB):
        with gfile.FastGFile(filePath,'rb') as f:
            graph_def.ParseFromString(f.read())
             
    else:
        with gfile.FastGFile(filePath,'r') as f:
            graph_str = f.read()
            pbtf.Parse(graph_str, graph_def)   
           
            
     # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")         
    return graph,graph_def




   
def convert(file_path,inputNodeName, outputNodeName,msuffix,binaryPB,readMode,folderPath,checkpointExt,checkpointPath,modelName,shapeArray,modifyshapeAttribue ,fixBatchNormal=True) :
    tf.reset_default_graph()
    config = tf.ConfigProto(
            allow_soft_placement = True,
            device_count={"GPU": 0, "CPU": 1}
            
    )
    runIncommdLine=False
    
    g_in,graph_def =  loadGraph(file_path,binaryPB)  
    
    # fix batch normal node nodes  https://github.com/tensorflow/tensorflow/issues/3628
    if(fixBatchNormal): 
          

        for node in graph_def.node:
          if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in  range(len(node.input)):
              if 'moving_' in node.input[index]:
              #if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
                node.input[index] = node.input[index] + '/read'
          elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
          elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
          #elif node.op == 'Assign':
          #    node.op = 'Identity'
          #    if 'use_locking' in node.attr: del node.attr['use_locking']
          #    if 'validate_shape' in node.attr: del node.attr['validate_shape']
          #    if len(node.input) == 2:
          #        # input0: ref: Should be from a Variable node. May be uninitialized.
          #        # input1: value: The value to be assigned to the variable.
          #        node.input[0] = node.input[1]
          #        del node.input[1]
          if('dilations')    in node.attr: del node.attr['dilations']  
          node.device=""
          
          
        #fixVariables not Working  
        fixVariables  =False
        if (fixVariables and node.op == 'VariableV2' and ('batchnorm/var' in node.name or 'batchnorm/mean' in node.name)):
              outputNodes=find_output_nodes(graph_def,node) 
              for index in  range(len( outputNodes )):
                  if(outputNodes[index].op=='Assign'   ):
                        #node.output[index] = node.output[index] + '/read'
                        #outputNodes[index].op ='Identity'
                        outputNodes[index].name = outputNodes[index].name+ '/read'
                        print('Modified %s '%outputNodes[index].name) 
                       
                 
                
                
   
                    
#################### Step 1 Training to inference simplification  , need checkpoint and  .pbtxt files from training   ######################################################      
               
    graphDef = optimize_for_inference_lib.optimize_for_inference(
                        graph_def,
                        [inputNodeName], # an array of the input node(s)
                        [outputNodeName] if type(outputNodeName)  is str  else [item for item in outputNodeName  ], # an array of output nodes
                        tf.float32.as_datatype_enum)
    
     
    if(modifyshapeAttribue):
        
        for n in  graphDef.node: 
            if((n.op=='Placeholder' or n.op=='Reshape') and n.name==inputNodeName):
                print('node to modify',n)
                setNodeAttribute(n,'shape',shapeArray)
                print("--Name of the node - %s shape set to " %  n.name,inputNodeName,shapeArray)
                print('node after modify',n)
    #graphDef=remove_training_nodes(output_graph_def)
    #graphDef = convert_variables_to_constants(sess, graphDef, [outputNodeName]) 
    
    if(runIncommdLine):
        copyfile(file_path,file_path+trainModelSuffix)  
    outputNameSuffix=  '_frozenforInference.pb'
    inferenceSuffix='.Inference'
    tf.train.write_graph(graphDef,folderPath, checkpointPath+modelName+'.pb'+inferenceSuffix, as_text=False)  
    tf.train.write_graph(graphDef,folderPath, checkpointPath+modelName+'.pbtxt'+inferenceSuffix, as_text=True)
        
    
    
    
    pbfileoutput_path=checkpointPath+modelName+outputNameSuffix
    checkpointfile_path=checkpointPath+modelName+checkpointExt
    #modelName='ssdvk' 
    pbfile_path=checkpointPath+modelName+msuffix+inferenceSuffix 
####################   Step 2                    Frozen Inference mode                      ######################################################                     
    
    freeze_graph.freeze_graph(
                        input_graph=pbfile_path, 
                        input_saver='',
                        input_binary=binaryPB,
                        input_checkpoint=checkpointfile_path, # an array of the input node(s)
                        output_node_names=  outputNodeName  if type(outputNodeName)  is str  else  ",".join( outputNodeName),
                        restore_op_name="save/restore_all", #Unused.
                        filename_tensor_name="save/Const:0",# Unused.
                        output_graph=pbfileoutput_path, # an array of output nodes  
                        clear_devices=True,
                        initializer_nodes=''
                        )
####################   Step 3                    Save in tensor board                     ######################################################                                         
    modelFullPath=checkpointPath+modelName+outputNameSuffix                   

def demoCKPT2PB(dataPara):  
    modelFileName=dataPara[0]    
    inputNodeName=dataPara[1]  
    shapeInforNodeName =dataPara[2]
    #outputNodeName= dataPara[3] 
    outputNodeName= dataPara[3] 
    modifyInputShape= dataPara[4] 
    currentFolder= os.path.dirname(os.path.realpath(__file__)) 
    currentFolder= './'
    file_path=currentFolder+modelFileName
    modelName=modelFileName.replace('.pbtxt','')
    print(modelName)
    msuffix,binaryPB,readMode,folderPath,checkpointExt,checkpointPath,shapeArray=settingsConf(parseCkptFolder(file_path,shapeInforNodeName,inputNodeName,outputNodeName)  )
    if(shapeArray is None or len(shapeArray)<1):     
        shapeArray=dataPara[5]
      
    convert(file_path,inputNodeName, outputNodeName,msuffix,binaryPB,readMode,'',checkpointExt,checkpointPath,modelName,shapeArray,modifyInputShape)              


"""
####################                Steps to process a new  tensorflow checkpoints folder                    ###################################################### 
After install dependence (tensorflow and google.protobuf)   in python environment:  

1 copy or generate .pbtxt file into the checkpoint folder
  
  python trainckpt2inferencepb.py   
  alternatively you can use Spyder, PyCharm or other python IDE to run this script


Refer to Readme.txt for more details.  

please Modify following dataPara line to process new checkpoint folder""" 

def main(ckpt_name):
  indexOfModel=0
  dataParas=[[ckpt_name + ".pbtxt",      'image_input',  'image_input',  ['classifiers/classifier0_0/Conv2D','classifiers/classifier0_1/Conv2D','classifiers/classifier0_2/Conv2D','classifiers/classifier0_3/Conv2D', 'classifiers/classifier0_4/Conv2D','classifiers/classifier0_5/Conv2D', 'classifiers/classifier1_0/Conv2D','classifiers/classifier1_1/Conv2D','classifiers/classifier1_2/Conv2D','classifiers/classifier1_3/Conv2D','classifiers/classifier1_4/Conv2D','classifiers/classifier1_5/Conv2D'],    True,[1,224,224,3]]]
  demoCKPT2PB(dataParas[indexOfModel])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  parser.add_argument("--ckpt_dir", type=str, required=True, default="", help="Checkpoint directory")
  # parser.add_argument("--ckpt_name", type=str, required=True, default="", help="Checkpoint name that needs to be frozen")
  parser.add_argument("--freeze", required = False, default = True, action = "store_true", help="Freeze graph for inference.")
  args = parser.parse_args()

  ckpt_name =  get_latest_ckpt(args.ckpt_dir)
  gen_pbtxt(args.ckpt_dir,ckpt_name)
  os.chdir(os.path.abspath(args.ckpt_dir))
  if args.freeze:
    main(ckpt_name)  


