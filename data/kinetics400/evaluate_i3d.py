#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shiv
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import i3d

import constants as C

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def evaluate(num_video_frames, rgb_npy, flow_npy, label, result_log_f):
  tf.reset_default_graph()
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type

  imagenet_pretrained = FLAGS.imagenet_pretrained

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, num_video_frames, C._IMAGE_SIZE, C._IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, num_video_frames, C._IMAGE_SIZE, C._IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = np.load(rgb_npy)
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = np.load(flow_npy)
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    result_log_f.write('Norm of logits: %f\n' % np.linalg.norm(out_logits))
    #result_log_f.write('Top classes and probabilities\n')
    result = 1
    init = 0
    for index in sorted_indices[:C._TOP_NUM_RESULTS]:
      if (init == 0): 
          init = 1
          if (kinetics_classes[index] == label):
              result_log_f.write('SUCCESS\n')
          else:
              result = 0
              result_log_f.write('FAILED\n')
      result_log_f.write('%f : %f [%s]\n' % (out_predictions[index], out_logits[index], kinetics_classes[index]))
    sess.close()
    return result
