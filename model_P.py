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

class P_model(object):

  def __init__(self, 
               sess, 
               image_size_MS=20,
               image_size_PAN=80,
               batch_size=64,
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
        self.reader = tf.train.NewCheckpointReader('./checkpoint/T_model/T_model.model-'+ str(99))
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
            
    with tf.name_scope('Up_samp'):
        self.LRMS_upsp=tf.image.resize_images(images=self.input_image_MS, size=[self.image_size_PAN, self.image_size_PAN],method=tf.image.ResizeMethod.BICUBIC)

    with tf.name_scope('fusion'): 
        self.weight_res=self.fusion_model(self.input_image_MS,self.input_image_PAN)
        self.fusion_image=self.LRMS_upsp+self.weight_res
        
    with tf.name_scope('Down_samp'):        
        self.fusion_image_downsp=tf.image.resize_images(images=self.fusion_image, size=[self.image_size_MS, self.image_size_MS],method=tf.image.ResizeMethod.BICUBIC)        

        
    with tf.name_scope('Gradient'): 
        self.fusion_image_gradient_x,self.fusion_image_gradient_y=sobel_gradient_4(self.fusion_image)
        self.LRMS_gradient_x,self.LRMS_gradient_y =sobel_gradient_4(self.input_image_MS)        
        self.HRPAN_gradient_x,self.HRPAN_gradient_y=sobel_gradient(self.images_PAN)                               
        self.HRMS2HRPAN_gradient_x = self.transfer_model(self.fusion_image_gradient_x,reuse=False)
        self.HRMS2HRPAN_gradient_y = self.transfer_model(self.fusion_image_gradient_y,reuse=True,update_collection=None)

    with tf.name_scope('g_loss'):
        self.g_loss_HRMS_int  = tf.reduce_mean(tf.abs(self.fusion_image - self.input_Label_MS))
        self.g_loss_LRMS_int  = tf.reduce_mean(tf.abs(self.fusion_image_downsp - self.input_image_MS))
        self.g_loss_HRMS2HRPAN_grad_x = tf.reduce_mean(tf.abs(self.HRMS2HRPAN_gradient_x - self.HRPAN_gradient_x)) 
        self.g_loss_HRMS2HRPAN_grad_y = tf.reduce_mean(tf.abs(self.HRMS2HRPAN_gradient_y - self.HRPAN_gradient_y))          
        self.g_loss_HRMS2HRPAN_grad  = self.g_loss_HRMS2HRPAN_grad_x+self.g_loss_HRMS2HRPAN_grad_y        
        self.g_loss_total=100*(5*self.g_loss_HRMS_int+3*self.g_loss_LRMS_int+1*self.g_loss_HRMS2HRPAN_grad) 
                       
        tf.summary.scalar('g_loss_HRMS_int',self.g_loss_HRMS_int)
        tf.summary.scalar('g_loss_LRMS_int',self.g_loss_LRMS_int)  
        tf.summary.scalar('g_loss_HRMS2HRPAN_grad_x',self.g_loss_HRMS2HRPAN_grad_x)
        tf.summary.scalar('g_loss_HRMS2HRPAN_grad_y',self.g_loss_HRMS2HRPAN_grad_y)              
        tf.summary.scalar('g_loss_HRMS2HRPAN_grad',self.g_loss_HRMS2HRPAN_grad)                     
        tf.summary.scalar('g_loss_total',self.g_loss_total)   


    self.saver = tf.train.Saver(max_to_keep=700)
    with tf.name_scope('image'):
        tf.summary.image('input_image_MS',tf.expand_dims(self.input_image_MS[1,:,:,:],0))  
        tf.summary.image('input_image_PAN',tf.expand_dims(self.input_image_PAN[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))  
        tf.summary.image('input_Label_MS',tf.expand_dims(self.input_Label_MS[1,:,:,:],0))                         
        tf.summary.image('HRMS2HRPAN_gradient_x',tf.expand_dims(self.HRMS2HRPAN_gradient_x[1,:,:,:],0)) 
        tf.summary.image('HRMS2HRPAN_gradient_y',tf.expand_dims(self.HRMS2HRPAN_gradient_y[1,:,:,:],0)) 
        tf.summary.image('HRPAN_gradient_x',tf.expand_dims(self.HRPAN_gradient_x[1,:,:,:],0))  
        tf.summary.image('HRPAN_gradient_y',tf.expand_dims(self.HRPAN_gradient_y[1,:,:,:],0))        
        tf.summary.image('fusion_image_downsp',tf.expand_dims(self.fusion_image_downsp[1,:,:,:],0))


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
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)      
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
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_MS1: batch_images_MS1,self.images_MS2: batch_images_MS2,self.images_MS3: batch_images_MS3,self.images_MS4: batch_images_MS4,self.images_PAN: batch_images_PAN,self.Label_MS1: batch_Label_MS1,self.Label_MS2: batch_Label_MS2,self.Label_MS3: batch_Label_MS3,self.Label_MS4: batch_Label_MS4})
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g))
        self.save(config.checkpoint_dir, ep)

  def fusion_model(self,img_MS,img_PAN):
    with tf.variable_scope('fusion_model'):
        MS_x2=tf.image.resize_images(images=img_MS, size=[2*20, 2*20],method=tf.image.ResizeMethod.BICUBIC) 
        with tf.variable_scope('layer1_MS_x2'):
            weights=tf.get_variable("w1_MS_x2",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_MS_x2",[16],initializer=tf.constant_initializer(0.0))
            conv1_MS_x2=tf.nn.conv2d(MS_x2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_MS_x2 = lrelu(conv1_MS_x2) 
        MS_x4=tf.image.resize_images(images=conv1_MS_x2, size=[4*20, 4*20],method=tf.image.ResizeMethod.BICUBIC) 
        with tf.variable_scope('layer2_MS_x4'):
            weights=tf.get_variable("w2_MS_x4",[3,3,16,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_MS_x4",[4],initializer=tf.constant_initializer(0.0))
            conv2_MS_x4=tf.nn.conv2d(MS_x4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_MS_x4 = lrelu(conv2_MS_x4)       
######################################################### 
#################### MS Layer 1 ###########################
#########################################################       
        with tf.variable_scope('layer1_MS'):
            weights=tf.get_variable("w1_MS",[3,3,4,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_MS",[16],initializer=tf.constant_initializer(0.0))
            conv1_MS=tf.nn.conv2d(conv2_MS_x4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_MS = lrelu(conv1_MS) 
######################################################### 
#################### PAN Layer 1 ###########################
#########################################################        
        with tf.variable_scope('layer1_PAN'):
            weights=tf.get_variable("w1_PAN",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_PAN",[16],initializer=tf.constant_initializer(0.0))
            conv1_PAN=tf.nn.conv2d(img_PAN, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_PAN = lrelu(conv1_PAN) 

######################################################### 
#################### MS Layer 2 #########################
#########################################################    
        MS_dense_2=tf.concat([conv2_MS_x4,conv1_MS,conv1_PAN],axis=-1)
        with tf.variable_scope('layer2_MS'):
            weights=tf.get_variable("w2_MS",[3,3,36,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_MS",[16],initializer=tf.constant_initializer(0.0))
            conv2_MS=tf.nn.conv2d(MS_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_MS = lrelu(conv2_MS)    
#################  Spatial Attention 2 ####################   
        SAttent_2_max=tf.reduce_max(conv2_MS, axis=3, keepdims=True)
        SAttent_2_mean=tf.reduce_mean(conv2_MS, axis=3, keepdims=True)
        SAttent_2_cat_mean_max=tf.concat([SAttent_2_max,SAttent_2_mean],axis=-1)        
        with tf.variable_scope('layer2_atten_map'):
            weights=tf.get_variable("w2_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv2_atten_map=tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_atten_map = tf.nn.sigmoid(conv2_atten_map)  
        conv2_MS_atten_out= conv2_MS*conv2_atten_map  
         
######################################################### 
#################### PAN Layer 2 ########################
#########################################################
        PAN_dense_2=tf.concat([img_PAN,conv1_PAN],axis=-1)      
        with tf.variable_scope('layer2_PAN'):
            weights=tf.get_variable("w2_PAN",[3,3,17,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_PAN",[16],initializer=tf.constant_initializer(0.0))
            conv2_PAN=tf.nn.conv2d(PAN_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_PAN = lrelu(conv2_PAN) 
            
######################################################### 
#################### MS Layer 3 #######################
#########################################################  
        MS_dense_3=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv2_PAN],axis=-1)               
        with tf.variable_scope('layer3_MS'):
            weights=tf.get_variable("w3_MS",[3,3,52,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_MS",[16],initializer=tf.constant_initializer(0.0))
            conv3_MS=tf.nn.conv2d(MS_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_MS = lrelu(conv3_MS) 
            
#################  Spatial Attention 3 ####################   
        SAttent_3_max=tf.reduce_max(conv3_MS, axis=3, keepdims=True)
        SAttent_3_mean=tf.reduce_mean(conv3_MS, axis=3, keepdims=True)
        SAttent_3_cat_mean_max=tf.concat([SAttent_3_max,SAttent_3_mean],axis=-1)        
        with tf.variable_scope('layer3_atten_map'):
            weights=tf.get_variable("w3_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv3_atten_map=tf.nn.conv2d(SAttent_3_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_atten_map = tf.nn.sigmoid(conv3_atten_map)  
        conv3_MS_atten_out= conv3_MS*conv3_atten_map 
        
######################################################### 
#################### PAN Layer 3 #######################
#########################################################  
        PAN_dense_3=tf.concat([img_PAN,conv1_PAN,conv2_PAN],axis=-1)              
        with tf.variable_scope('layer3_PAN'):
            weights=tf.get_variable("w3_PAN",[3,3,33,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_PAN",[16],initializer=tf.constant_initializer(0.0))
            conv3_PAN=tf.nn.conv2d(PAN_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_PAN = lrelu(conv3_PAN) 
  
######################################################### 
#################### MS Layer 4 #########################
#########################################################   
        MS_dense_4=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv3_MS_atten_out,conv3_PAN],axis=-1)         
        with tf.variable_scope('layer4_MS'):
            weights=tf.get_variable("w4_MS",[3,3,68,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_MS",[16],initializer=tf.constant_initializer(0.0))
            conv4_MS=tf.nn.conv2d(MS_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_MS = lrelu(conv4_MS) 
#################  Spatial Attention 4 ####################   
        SAttent_4_max=tf.reduce_max(conv4_MS, axis=3, keepdims=True)
        SAttent_4_mean=tf.reduce_mean(conv4_MS, axis=3, keepdims=True)
        SAttent_4_cat_mean_max=tf.concat([SAttent_4_max,SAttent_4_mean],axis=-1)        
        with tf.variable_scope('layer4_atten_map'):
            weights=tf.get_variable("w4_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv4_atten_map=tf.nn.conv2d(SAttent_4_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_atten_map = tf.nn.sigmoid(conv4_atten_map)  
        conv4_MS_atten_out= conv4_MS*conv4_atten_map  
######################################################### 
#################### PAN Layer 4 #######################
#########################################################  
        PAN_dense_4=tf.concat([img_PAN,conv1_PAN,conv2_PAN,conv3_PAN],axis=-1)              
        with tf.variable_scope('layer4_PAN'):
            weights=tf.get_variable("w4_PAN",[3,3,49,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_PAN",[16],initializer=tf.constant_initializer(0.0))
            conv4_PAN =tf.nn.conv2d(PAN_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_PAN = lrelu(conv4_PAN)   
                
######################################################### 
#################### MS Layer 4_f #########################
#########################################################   
        MS_dense_4_f=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv3_MS_atten_out,conv4_MS_atten_out,conv4_PAN],axis=-1)        
        with tf.variable_scope('layer4_MS_f'):
            weights=tf.get_variable("w4_MS_f",[3,3,84,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_MS_f",[16],initializer=tf.constant_initializer(0.0))
            conv4_MS_f=tf.nn.conv2d(MS_dense_4_f, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_MS_f = lrelu(conv4_MS_f) 
#################  Spatial Attention 4_f ####################   
        SAttent_4_f_max=tf.reduce_max(conv4_MS_f, axis=3, keepdims=True)
        SAttent_4_f_mean=tf.reduce_mean(conv4_MS_f, axis=3, keepdims=True)
        SAttent_4_f_cat_mean_max=tf.concat([SAttent_4_f_max,SAttent_4_f_mean],axis=-1)        
        with tf.variable_scope('layer4_f_atten_map'):
            weights=tf.get_variable("w4_f_atten_map",[5,5,2,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_f_atten_map",[1],initializer=tf.constant_initializer(0.0))
            conv4_f_atten_map=tf.nn.conv2d(SAttent_4_f_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_f_atten_map = tf.nn.sigmoid(conv4_f_atten_map)  
        conv4_MS_f_atten_out= conv4_MS_f*conv4_f_atten_map          
                                
################ residual: LRMS 2 HRMS  ####################### 
####################  Layer 5 ###########################
######################################################### 
        with tf.variable_scope('layer5_Res'):
            weights=tf.get_variable("w5_Res",[3,3,16,8],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5_Res",[8],initializer=tf.constant_initializer(0.0))
            conv5_Res =tf.nn.conv2d(conv4_MS_f_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_Res =lrelu(conv5_Res)  
######################################################### 
####################  Layer 6 ###########################
######################################################### 
        with tf.variable_scope('layer6_Res'):
            weights=tf.get_variable("w6_Res",[3,3,8,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b6_Res",[4],initializer=tf.constant_initializer(0.0))
            conv6_Res =tf.nn.conv2d(conv5_Res, weights, strides=[1,1,1,1], padding='SAME') + bias            
            conv6_Res =tf.nn.tanh(conv6_Res)                                                           
    return conv6_Res
    

  def transfer_model(self,img_MS_grad,reuse,update_collection=None):
    with tf.variable_scope('transfer_model',reuse=reuse):   
######################################################### 
#################### grad Layer 1 #######################
#########################################################       
        with tf.variable_scope('layer1_grad'):
            weights=tf.get_variable("w1_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer1_grad/w1_grad')))
            bias=tf.get_variable("b1_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer1_grad/b1_grad')))
            conv1_grad = tf.nn.conv2d(img_MS_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_grad = lrelu(conv1_grad) 
######################################################### 
#################### grad Layer 2 ###########################
#########################################################       
        with tf.variable_scope('layer2_grad'):
            weights=tf.get_variable("w2_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer2_grad/w2_grad')))
            bias=tf.get_variable("b2_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer2_grad/b2_grad')))
            conv2_grad = tf.nn.conv2d(conv1_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_grad = lrelu(conv2_grad)
######################################################### 
#################### grad Layer 3 ###########################
######################################################### 
        with tf.variable_scope('layer3_grad'):
            weights=tf.get_variable("w3_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer3_grad/w3_grad')))
            bias=tf.get_variable("b3_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer3_grad/b3_grad')))
            conv3_grad = tf.nn.conv2d(conv2_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_grad = lrelu(conv3_grad)
######################################################### 
#################### grad Layer 4 ###########################
######################################################### 
        grad_cat_4=tf.concat([conv1_grad,conv3_grad],axis=-1) 
        with tf.variable_scope('layer4_grad'):
            weights=tf.get_variable("w4_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer4_grad/w4_grad')))
            bias=tf.get_variable("b4_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer4_grad/b4_grad')))
            conv4_grad = tf.nn.conv2d(grad_cat_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_grad = lrelu(conv4_grad)
######################################################### 
#################### grad Layer 5 #######################
######################################################### 
        grad_cat_5=tf.concat([img_MS_grad,conv4_grad],axis=-1) 
        with tf.variable_scope('layer5_grad'):
            weights=tf.get_variable("w5_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer5_grad/w5_grad')))
            bias=tf.get_variable("b5_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer5_grad/b5_grad')))
            conv5_grad = tf.nn.conv2d(grad_cat_5, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_grad = lrelu(conv5_grad)
######################################################### 
#################### grad Layer 6 #######################
######################################################### 
        with tf.variable_scope('layer6_grad'):
            weights=tf.get_variable("w6_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer6_grad/w6_grad')))
            bias=tf.get_variable("b6_grad",initializer=tf.constant(self.reader.get_tensor('transfer_model/layer6_grad/b6_grad')))
            conv6_grad = tf.nn.conv2d(conv5_grad, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_grad = tf.nn.tanh(conv6_grad)*2
    return conv6_grad


  def save(self, checkpoint_dir, step):
    model_name = "P_model.model"
    model_dir = "%s" % ("P_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s" % ("P_model")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
