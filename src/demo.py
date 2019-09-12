# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob
import skvideo.io
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg


import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
#from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")



def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      mc.IS_TRAINING = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      mc.IS_TRAINING = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      for f in glob.iglob(FLAGS.input_path):
        print('file name:'+f)
        im = cv2.imread(f) # <---------------------------- BGR format
        im = im.astype(np.float32, copy=False)

        im = cv2.resize(im, (mc.IMAGE_WIDTH*1, mc.IMAGE_HEIGHT*1))
        orig_h, orig_w, _ = [float(v) for v in im.shape]
        y_start = int(orig_h/2-mc.IMAGE_HEIGHT/2)
        x_start = int(orig_w - mc.IMAGE_WIDTH)
        #im = im[y_start:y_start+mc.IMAGE_HEIGHT, x_start:x_start+mc.IMAGE_WIDTH]


        #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # gray color instead of RGB <----------------
        #im_gray = im_gray - np.array([[[128]]])
        im_gray = im - mc.BGR_MEANS # <---------------------------------------------------------------------!!!!!!
        im_gray = im_gray / 128.0
        #im_gray = im_gray.reshape((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1)) <---------------
        # kishan
        #g_conv12 = sess.graph.get_tensor_by_name('conv12/convolution:0')
        #g_boxes = sess.graph.get_tensor_by_name('bbox/trimming/bbox:0')
        #g_probs = sess.graph.get_tensor_by_name('probability/score:0')
        #g_class = sess.graph.get_tensor_by_name('probability/class_idx:0')
        #g_k = sess.graph.get_tensor_by_name('interpret_output/strided_slice_2:0')
        #dt_boxes, dt_probs, dt_class, dt_conv12, dt_k = sess.run(
        #  [g_boxes, g_probs, g_class, g_conv12, g_k],
        #    feed_dict={model.image_input:[im_gray]})

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_gray]})

        #print(np.shape(dt_conv12))
        #print(np.reshape(dt_conv12, 576))
        
        #print(np.shape(dt_boxes))
        #print(len(dt_boxes[0]))
        #print(dt_boxes)

        #print(np.shape(dt_probs))
        #print(len(dt_probs[0]))
        #print(dt_probs)
        
        #print(np.shape(dt_class))
        #print(len(dt_class[0]))
        #print(dt_class)

        #print(np.shape(dt_k))
        #print(dt_k)
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])
        #print(np.shape(final_boxes))
        #print(np.shape(final_probs))
        #print(np.shape(final_class))
        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        # Draw boxes
        print('# of final boxes=', len(keep_idx))
        _draw_box(f,
            #im_gray, final_boxes,
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+':%.2f'% prob \
                for idx, prob in zip(final_class, final_probs)] #,
            #cdict=cls2clr,
        )

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)

        cv2.imwrite(out_file_name, im) # <----- BGR format
        #cv2.imwrite(out_file_name, im_gray) # <----- BGR format
        print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
