# -*- coding: utf-8 -*-
from model_T import T_model
from utils import input_setup_MS
from utils import input_setup_PAN
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 48, "The size of batch images [128]")
flags.DEFINE_integer("image_size_MS", 20, "The size of image to use [33]")
flags.DEFINE_integer("image_size_PAN", 80, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride_MS", 3, "The size of stride to apply input image [14]")
flags.DEFINE_integer("stride_PAN", 12, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("summary_dir", "log", "Name of log directory [log]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)


  with tf.Session() as sess:
    srcnn = T_model(sess, 
                  image_size_MS=FLAGS.image_size_MS, 
                  image_size_PAN=FLAGS.image_size_PAN, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
