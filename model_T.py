# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup_MS, 
  input_setup_PAN, 
  imsave,
  merge,
  sobel_gradient,
  lrelu,
  l2_norm,
  linear_map,
  lpls_gradient,
  lpls_gradient_4,
  sobel_gradient_4
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class T_model(object):

  def __init__(self, 
               sess, 
               image_size_MS=20,
               image_size_PAN=80,
               batch_size=48,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size_MS =  image_size_MS
    self.image_size_PAN =  image_size_PAN
    self.image_size_Label = image_size_PAN
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
  ########## MS  Input ###################### 
    with tf.name_scope('MS1_input'):
        self.images_MS1 = tf.placeholder(tf.float32, [None, self.image_size_MS, self.image_size_MS, self.c_dim], name='images_MS1')
    with tf.name_scope('MS2_input'):
        self.images_MS2 = tf.placeholder(tf.float32, [None, self.image_size_MS, self.image_size_MS, self.c_dim], name='images_MS2')        
    with tf.name_scope('MS3_input'):
        self.images_MS3 = tf.placeholder(tf.float32, [None, self.image_size_MS, self.image_size_MS, self.c_dim], name='images_MS3')        
    with tf.name_scope('MS4_input'):
        self.images_MS4 = tf.placeholder(tf.float32, [None, self.image_size_MS, self.image_size_MS, self.c_dim], name='images_MS4')
        
  ########## MS  Label Input ######################         
    with tf.name_scope('MS1_Label'):
        self.Label_MS1 = tf.placeholder(tf.float32, [None, self.image_size_Label, self.image_size_Label, self.c_dim], name='Label_MS1')
    with tf.name_scope('MS2_Label'):
        self.Label_MS2 = tf.placeholder(tf.float32, [None, self.image_size_Label, self.image_size_Label, self.c_dim], name='Label_MS2')        
    with tf.name_scope('MS3_Label'):
        self.Label_MS3 = tf.placeholder(tf.float32, [None, self.image_size_Label, self.image_size_Label, self.c_dim], name='Label_MS3')        
    with tf.name_scope('MS4_Label'):
        self.Label_MS4 = tf.placeholder(tf.float32, [None, self.image_size_Label, self.image_size_Label, self.c_dim], name='Label_MS4')
        
  ########## PAN  Input ######################                         
    with tf.name_scope('PAN_input'):
        self.images_PAN = tf.placeholder(tf.float32, [None, self.image_size_PAN, self.image_size_PAN, self.c_dim], name='images_PAN')

    with tf.name_scope('input'):
        self.input_image_MS1 = self.images_MS1
        self.input_image_MS2 = self.images_MS2
        self.input_image_MS3 = self.images_MS3
        self.input_image_MS4 = self.images_MS4  
        self.input_image_MS  =tf.concat([self.images_MS1,self.images_MS2,self.images_MS3,self.images_MS4],axis=-1)
                                       
        self.input_image_PAN = self.images_PAN

        self.input_Label_MS1 = self.Label_MS1
        self.input_Label_MS2 = self.Label_MS2
        self.input_Label_MS3 = self.Label_MS3
        self.input_Label_MS4 = self.Label_MS4  
        self.input_Label_MS  =tf.concat([self.input_Label_MS1,self.input_Label_MS2,self.input_Label_MS3,self.input_Label_MS4],axis=-1)  
                
    with tf.name_scope('Gradient'): 
        self.Label_MS_gradient_x,self.Label_MS_gradient_y=sobel_gradient_4(self.input_Label_MS)        
        self.HRPAN_gradient_x,self.HRPAN_gradient_y=sobel_gradient(self.images_PAN)                

    with tf.name_scope('fusion'): 
        self.MS2PAN_gradient_x=self.transfer_model(self.Label_MS_gradient_x,reuse=False)
        self.MS2PAN_gradient_y=self.transfer_model(self.Label_MS_gradient_y,reuse=True,update_collection='NO_OPS')

    with tf.name_scope('t_loss'):
        self.t_loss_MS2PAN_grad_x  = tf.reduce_mean(tf.square(self.MS2PAN_gradient_x - self.HRPAN_gradient_x))
        self.t_loss_MS2PAN_grad_y  = tf.reduce_mean(tf.square(self.MS2PAN_gradient_y - self.HRPAN_gradient_y))
        self.t_loss_total=100*(self.t_loss_MS2PAN_grad_x+self.t_loss_MS2PAN_grad_y)
        tf.summary.scalar('t_loss_MS2PAN_grad_x',self.t_loss_MS2PAN_grad_x)
        tf.summary.scalar('t_loss_MS2PAN_grad_y',self.t_loss_MS2PAN_grad_y)        
        tf.summary.scalar('t_loss_total',self.t_loss_total)
        
    self.saver = tf.train.Saver(max_to_keep=100)
    with tf.name_scope('image'):
        tf.summary.image('input_Label_MS',tf.expand_dims(self.input_Label_MS[1,:,:,:],0))  
        tf.summary.image('input_image_PAN',tf.expand_dims(self.input_image_PAN[1,:,:,:],0))  
        tf.summary.image('MS2PAN_gradient_x',tf.expand_dims(self.MS2PAN_gradient_x[1,:,:,:],0))
        tf.summary.image('MS2PAN_gradient_y',tf.expand_dims(self.MS2PAN_gradient_y[1,:,:,:],0))
        tf.summary.image('HRPAN_gradient_x',tf.expand_dims(self.HRPAN_gradient_x[1,:,:,:],0))
        tf.summary.image('HRPAN_gradient_y',tf.expand_dims(self.HRPAN_gradient_y[1,:,:,:],0))

    
  def train(self, config):
    if config.is_train:
      input_setup_MS(self.sess, config,"data/Train_data/Train_MS1")
      input_setup_MS(self.sess, config,"data/Train_data/Train_MS2")
      input_setup_MS(self.sess, config,"data/Train_data/Train_MS3")
      input_setup_MS(self.sess, config,"data/Train_data/Train_MS4")
      input_setup_PAN(self.sess,config,"data/Train_data/Train_PAN")
      input_setup_PAN(self.sess, config,"data/Train_data/Label_MS1")
      input_setup_PAN(self.sess, config,"data/Train_data/Label_MS2")
      input_setup_PAN(self.sess, config,"data/Train_data/Label_MS3")
      input_setup_PAN(self.sess, config,"data/Train_data/Label_MS4")



    if config.is_train:     
      data_dir_MS1 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_MS1","train.h5")
      data_dir_MS2 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_MS2","train.h5")
      data_dir_MS3 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_MS3","train.h5")
      data_dir_MS4 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_MS4","train.h5")
      data_dir_PAN = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Train_PAN","train.h5")
      data_dir_Label_MS1 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Label_MS1","train.h5")
      data_dir_Label_MS2 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Label_MS2","train.h5")
      data_dir_Label_MS3 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Label_MS3","train.h5")
      data_dir_Label_MS4 = os.path.join('./{}'.format(config.checkpoint_dir), "data/Train_data/Label_MS4","train.h5")

      
    train_data_MS1= read_data(data_dir_MS1)
    train_data_MS2= read_data(data_dir_MS2)
    train_data_MS3= read_data(data_dir_MS3)
    train_data_MS4= read_data(data_dir_MS4)
    train_data_PAN= read_data(data_dir_PAN)
    train_data_Label_MS1= read_data(data_dir_Label_MS1)
    train_data_Label_MS2= read_data(data_dir_Label_MS2)
    train_data_Label_MS3= read_data(data_dir_Label_MS3)
    train_data_Label_MS4= read_data(data_dir_Label_MS4)
    
    t_vars = tf.trainable_variables()
    self.trans_vars = [var for var in t_vars if 'transfer_model' in var.name]
    print(self.trans_vars)
    with tf.name_scope('train_step'):
        self.train_trans_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.t_loss_total,var_list=self.trans_vars)        
    self.summary_op = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)    
    tf.initialize_all_variables().run()    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_PAN) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_MS1 = train_data_MS1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_MS2 = train_data_MS2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_MS3 = train_data_MS3[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_MS4 = train_data_MS4[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_PAN = train_data_PAN[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_Label_MS1 = train_data_Label_MS1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_Label_MS2 = train_data_Label_MS2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_Label_MS3 = train_data_Label_MS3[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_Label_MS4 = train_data_Label_MS4[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1          
          _, err_trans,summary_str= self.sess.run([self.train_trans_op, self.t_loss_total,self.summary_op], feed_dict={self.images_MS1: batch_images_MS1,self.images_MS2: batch_images_MS2,self.images_MS3: batch_images_MS3,self.images_MS4: batch_images_MS4,self.images_PAN: batch_images_PAN,self.Label_MS1: batch_Label_MS1,self.Label_MS2: batch_Label_MS2,self.Label_MS3: batch_Label_MS3,self.Label_MS4: batch_Label_MS4})
          self.train_writer.add_summary(summary_str,counter)
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f],loss_trans:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_trans))
        self.save(config.checkpoint_dir, ep)


  def transfer_model(self,img_MS_grad,reuse,update_collection=None):
    with tf.variable_scope('transfer_model',reuse=reuse):   
######################################################### 
#################### grad Layer 1 #######################
#########################################################       
        with tf.variable_scope('layer1_grad'):
            weights=tf.get_variable("w1_grad",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_grad",[16],initializer=tf.constant_initializer(0.0))
            conv1_grad = tf.nn.conv2d(img_MS_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_grad = lrelu(conv1_grad) 
######################################################### 
#################### grad Layer 2 ###########################
#########################################################       
        with tf.variable_scope('layer2_grad'):
            weights=tf.get_variable("w2_grad",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_grad",[16],initializer=tf.constant_initializer(0.0))
            conv2_grad = tf.nn.conv2d(conv1_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_grad = lrelu(conv2_grad)
######################################################### 
#################### grad Layer 3 ###########################
######################################################### 
        with tf.variable_scope('layer3_grad'):
            weights=tf.get_variable("w3_grad",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_grad",[16],initializer=tf.constant_initializer(0.0))
            conv3_grad = tf.nn.conv2d(conv2_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_grad = lrelu(conv3_grad)
######################################################### 
#################### grad Layer 4 ###########################
######################################################### 
        grad_cat_4=tf.concat([conv1_grad,conv3_grad],axis=-1) 
        with tf.variable_scope('layer4_grad'):
            weights=tf.get_variable("w4_grad",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_grad",[16],initializer=tf.constant_initializer(0.0))
            conv4_grad = tf.nn.conv2d(grad_cat_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_grad = lrelu(conv4_grad)
######################################################### 
#################### grad Layer 5 #######################
######################################################### 
        grad_cat_5=tf.concat([img_MS_grad,conv4_grad],axis=-1) 
        with tf.variable_scope('layer5_grad'):
            weights=tf.get_variable("w5_grad",[3,3,20,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5_grad",[8],initializer=tf.constant_initializer(0.0))
            conv5_grad = tf.nn.conv2d(grad_cat_5, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_grad = lrelu(conv5_grad)
######################################################### 
#################### grad Layer 6 #######################
######################################################### 
        with tf.variable_scope('layer6_grad'):
            weights=tf.get_variable("w6_grad",[3,3,8,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b6_grad",[1],initializer=tf.constant_initializer(0.0))
            conv6_grad = tf.nn.conv2d(conv5_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_grad = tf.nn.tanh(conv6_grad)*2
    return conv6_grad


  def save(self, checkpoint_dir, step):
    model_name = "T_model.model"
    model_dir = "%s" % ("T_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s" % ("T_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
