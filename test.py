# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import scipy.io as scio


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    #data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_MS,img_PAN):
    with tf.variable_scope('fusion_model'):
        MS_x2=tf.image.resize_images(images=img_MS, size=[2*50, 2*50],method=tf.image.ResizeMethod.BICUBIC) 
        with tf.variable_scope('layer1_MS_x2'):
            weights=tf.get_variable("w1_MS_x2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_MS_x2/w1_MS_x2')))
            bias=tf.get_variable("b1_MS_x2",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_MS_x2/b1_MS_x2')))
            conv1_MS_x2=tf.nn.conv2d(MS_x2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_MS_x2 = lrelu(conv1_MS_x2) 
        MS_x4=tf.image.resize_images(images=conv1_MS_x2, size=[4*50, 4*50],method=tf.image.ResizeMethod.BICUBIC) 
        with tf.variable_scope('layer2_MS_x4'):
            weights=tf.get_variable("w2_MS_x4",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_MS_x4/w2_MS_x4')))
            bias=tf.get_variable("b2_MS_x4",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_MS_x4/b2_MS_x4')))
            conv2_MS_x4=tf.nn.conv2d(MS_x4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_MS_x4 = lrelu(conv2_MS_x4)       
######################################################### 
#################### MS Layer 1 ###########################
#########################################################       
        with tf.variable_scope('layer1_MS'):
            weights=tf.get_variable("w1_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_MS/w1_MS')))
            bias=tf.get_variable("b1_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_MS/b1_MS')))
            conv1_MS=tf.nn.conv2d(conv2_MS_x4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_MS = lrelu(conv1_MS) 
######################################################### 
#################### PAN Layer 1 ###########################
#########################################################        
        with tf.variable_scope('layer1_PAN'):
            weights=tf.get_variable("w1_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_PAN/w1_PAN')))
            bias=tf.get_variable("b1_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_PAN/b1_PAN')))
            conv1_PAN=tf.nn.conv2d(img_PAN, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_PAN = lrelu(conv1_PAN) 

######################################################### 
#################### MS Layer 2 #########################
#########################################################    
        MS_dense_2=tf.concat([conv2_MS_x4,conv1_MS,conv1_PAN],axis=-1)
        with tf.variable_scope('layer2_MS'):
            weights=tf.get_variable("w2_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_MS/w2_MS')))
            bias=tf.get_variable("b2_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_MS/b2_MS')))
            conv2_MS=tf.nn.conv2d(MS_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_MS = lrelu(conv2_MS)    
#################  Spatial Attention 2 ####################   
        SAttent_2_max=tf.reduce_max(conv2_MS, axis=3, keepdims=True)
        SAttent_2_mean=tf.reduce_mean(conv2_MS, axis=3, keepdims=True)
        SAttent_2_cat_mean_max=tf.concat([SAttent_2_max,SAttent_2_mean],axis=-1)        
        with tf.variable_scope('layer2_atten_map'):
            weights=tf.get_variable("w2_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_atten_map/w2_atten_map')))
            bias=tf.get_variable("b2_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_atten_map/b2_atten_map')))
            conv2_atten_map=tf.nn.conv2d(SAttent_2_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_atten_map = tf.nn.sigmoid(conv2_atten_map)  
        conv2_MS_atten_out= conv2_MS*conv2_atten_map  
         
######################################################### 
#################### PAN Layer 2 ########################
#########################################################
        PAN_dense_2=tf.concat([img_PAN,conv1_PAN],axis=-1)      
        with tf.variable_scope('layer2_PAN'):
            weights=tf.get_variable("w2_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_PAN/w2_PAN')))
            bias=tf.get_variable("b2_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_PAN/b2_PAN')))
            conv2_PAN=tf.nn.conv2d(PAN_dense_2, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_PAN = lrelu(conv2_PAN) 
            
######################################################### 
#################### MS Layer 3 #######################
#########################################################  
        MS_dense_3=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv2_PAN],axis=-1)               
        with tf.variable_scope('layer3_MS'):
            weights=tf.get_variable("w3_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_MS/w3_MS')))
            bias=tf.get_variable("b3_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_MS/b3_MS')))
            conv3_MS=tf.nn.conv2d(MS_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_MS = lrelu(conv3_MS) 
            
#################  Spatial Attention 3 ####################   
        SAttent_3_max=tf.reduce_max(conv3_MS, axis=3, keepdims=True)
        SAttent_3_mean=tf.reduce_mean(conv3_MS, axis=3, keepdims=True)
        SAttent_3_cat_mean_max=tf.concat([SAttent_3_max,SAttent_3_mean],axis=-1)        
        with tf.variable_scope('layer3_atten_map'):
            weights=tf.get_variable("w3_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_atten_map/w3_atten_map')))
            bias=tf.get_variable("b3_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_atten_map/b3_atten_map')))
            conv3_atten_map=tf.nn.conv2d(SAttent_3_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_atten_map = tf.nn.sigmoid(conv3_atten_map)  
        conv3_MS_atten_out= conv3_MS*conv3_atten_map 
        
######################################################### 
#################### PAN Layer 3 #######################
#########################################################  
        PAN_dense_3=tf.concat([img_PAN,conv1_PAN,conv2_PAN],axis=-1)              
        with tf.variable_scope('layer3_PAN'):
            weights=tf.get_variable("w3_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_PAN/w3_PAN')))
            bias=tf.get_variable("b3_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_PAN/b3_PAN')))
            conv3_PAN=tf.nn.conv2d(PAN_dense_3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_PAN = lrelu(conv3_PAN) 
  
######################################################### 
#################### MS Layer 4 #########################
#########################################################   
        MS_dense_4=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv3_MS_atten_out,conv3_PAN],axis=-1)         
        with tf.variable_scope('layer4_MS'):
            weights=tf.get_variable("w4_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_MS/w4_MS')))
            bias=tf.get_variable("b4_MS",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_MS/b4_MS')))
            conv4_MS=tf.nn.conv2d(MS_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_MS = lrelu(conv4_MS) 
#################  Spatial Attention 4 ####################   
        SAttent_4_max=tf.reduce_max(conv4_MS, axis=3, keepdims=True)
        SAttent_4_mean=tf.reduce_mean(conv4_MS, axis=3, keepdims=True)
        SAttent_4_cat_mean_max=tf.concat([SAttent_4_max,SAttent_4_mean],axis=-1)        
        with tf.variable_scope('layer4_atten_map'):
            weights=tf.get_variable("w4_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_atten_map/w4_atten_map')))
            bias=tf.get_variable("b4_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_atten_map/b4_atten_map')))
            conv4_atten_map=tf.nn.conv2d(SAttent_4_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_atten_map = tf.nn.sigmoid(conv4_atten_map)  
        conv4_MS_atten_out= conv4_MS*conv4_atten_map  
######################################################### 
#################### PAN Layer 4 #######################
#########################################################  
        PAN_dense_4=tf.concat([img_PAN,conv1_PAN,conv2_PAN,conv3_PAN],axis=-1)              
        with tf.variable_scope('layer4_PAN'):
            weights=tf.get_variable("w4_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_PAN/w4_PAN')))
            bias=tf.get_variable("b4_PAN",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_PAN/b4_PAN')))
            conv4_PAN =tf.nn.conv2d(PAN_dense_4, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_PAN = lrelu(conv4_PAN)   
                
######################################################### 
#################### MS Layer 4_f #########################
#########################################################   
        MS_dense_4_f=tf.concat([conv2_MS_x4,conv1_MS,conv2_MS_atten_out,conv3_MS_atten_out,conv4_MS_atten_out,conv4_PAN],axis=-1)        
        with tf.variable_scope('layer4_MS_f'):
            weights=tf.get_variable("w4_MS_f",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_MS_f/w4_MS_f')))
            bias=tf.get_variable("b4_MS_f",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_MS_f/b4_MS_f')))
            conv4_MS_f=tf.nn.conv2d(MS_dense_4_f, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_MS_f = lrelu(conv4_MS_f) 
#################  Spatial Attention 4_f ####################   
        SAttent_4_f_max=tf.reduce_max(conv4_MS_f, axis=3, keepdims=True)
        SAttent_4_f_mean=tf.reduce_mean(conv4_MS_f, axis=3, keepdims=True)
        SAttent_4_f_cat_mean_max=tf.concat([SAttent_4_f_max,SAttent_4_f_mean],axis=-1)        
        with tf.variable_scope('layer4_f_atten_map'):
            weights=tf.get_variable("w4_f_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_f_atten_map/w4_f_atten_map')))
            bias=tf.get_variable("b4_f_atten_map",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_f_atten_map/b4_f_atten_map')))
            conv4_f_atten_map=tf.nn.conv2d(SAttent_4_f_cat_mean_max, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_f_atten_map = tf.nn.sigmoid(conv4_f_atten_map)  
        conv4_MS_f_atten_out= conv4_MS_f*conv4_f_atten_map          
                                
################ residual: LRMS 2 HRMS  ####################### 
####################  Layer 5 ###########################
######################################################### 
        with tf.variable_scope('layer5_Res'):
            weights=tf.get_variable("w5_Res",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_Res/w5_Res')))
            bias=tf.get_variable("b5_Res",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_Res/b5_Res')))
            conv5_Res =tf.nn.conv2d(conv4_MS_f_atten_out, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_Res =lrelu(conv5_Res)  
######################################################### 
####################  Layer 6 ###########################
######################################################### 
        with tf.variable_scope('layer6_Res'):
            weights=tf.get_variable("w6_Res",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_Res/w6_Res')))
            bias=tf.get_variable("b6_Res",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_Res/b6_Res')))
            conv6_Res =tf.nn.conv2d(conv5_Res, weights, strides=[1,1,1,1], padding='SAME') + bias            
            conv6_Res =tf.nn.tanh(conv6_Res)                                                           
    return conv6_Res


    

def input_setup(index):
    padding=0
    sub_MS1_sequence = []
    sub_MS2_sequence = []
    sub_MS3_sequence = []
    sub_MS4_sequence = []
    sub_PAN_sequence = []
    
    input_MS1=(imread(data_MS1[index])-127.5)/127.5
    input_MS1=np.lib.pad(input_MS1,((padding,padding),(padding,padding)),'edge')
    w,h=input_MS1.shape
    input_MS1=input_MS1.reshape([w,h,1])

    input_MS2=(imread(data_MS2[index])-127.5)/127.5
    input_MS2=np.lib.pad(input_MS2,((padding,padding),(padding,padding)),'edge')
    w,h=input_MS2.shape
    input_MS2=input_MS2.reshape([w,h,1])
    
    input_MS3=(imread(data_MS3[index])-127.5)/127.5
    input_MS3=np.lib.pad(input_MS3,((padding,padding),(padding,padding)),'edge')
    w,h=input_MS3.shape
    input_MS3=input_MS3.reshape([w,h,1])
    
    input_MS4=(imread(data_MS4[index])-127.5)/127.5
    input_MS4=np.lib.pad(input_MS4,((padding,padding),(padding,padding)),'edge')
    w,h=input_MS4.shape
    input_MS4=input_MS4.reshape([w,h,1])

    
    input_PAN=(imread(data_PAN[index])-127.5)/127.5
    input_PAN=np.lib.pad(input_PAN,((padding,padding),(padding,padding)),'edge')
    w,h=input_PAN.shape
    input_PAN=input_PAN.reshape([w,h,1])
    
    sub_MS1_sequence.append(input_MS1)
    sub_MS2_sequence.append(input_MS2)
    sub_MS3_sequence.append(input_MS3)
    sub_MS4_sequence.append(input_MS4)
    
    sub_PAN_sequence.append(input_PAN)
    
    
    
    train_data_MS1= np.asarray(sub_MS1_sequence)
    train_data_MS2= np.asarray(sub_MS2_sequence)
    train_data_MS3= np.asarray(sub_MS3_sequence)
    train_data_MS4= np.asarray(sub_MS4_sequence)
    
    
    train_data_PAN= np.asarray(sub_PAN_sequence)
    return train_data_MS1,train_data_MS2,train_data_MS3,train_data_MS4,train_data_PAN

for idx_num in range(699,700):
  num_epoch=idx_num
  while(num_epoch==idx_num):
  
      reader = tf.train.NewCheckpointReader('./checkpoint/Quickbird/P_model/P_model.model-'+ str(num_epoch)) #best_epoch_Quickbird =699
      #reader = tf.train.NewCheckpointReader('./checkpoint/GF-2/P_model/P_model.model-'+ str(num_epoch))     #best_epoch_GF-2 =125
  
      with tf.name_scope('MS1_input'):
          images_MS1 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MS1')
      with tf.name_scope('MS2_input'):
          images_MS2 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MS2')
      with tf.name_scope('MS3_input'):
          images_MS3 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MS3')
      with tf.name_scope('MS4_input'):
          images_MS4 = tf.placeholder(tf.float32, [1,None,None,None], name='images_MS4')
                                        
      with tf.name_scope('PAN_input'):
          images_PAN= tf.placeholder(tf.float32, [1,None,None,None], name='images_PAN')
          
      with tf.name_scope('input'):
          input_image_MS =tf.concat([images_MS1,images_MS2,images_MS3,images_MS4],axis=-1)
          input_image_PAN =images_PAN
          input_image_ms_upsp=tf.image.resize_images(images=input_image_MS, size=[200, 200],method=tf.image.ResizeMethod.BICUBIC)
      with tf.name_scope('fusion'):
          weight_res=fusion_model(input_image_MS,input_image_PAN)
          fusion_image=input_image_ms_upsp+weight_res
  
      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_MS1=prepare_data('data/Test_data/Test_MS1')
          data_MS2=prepare_data('data/Test_data/Test_MS2')
          data_MS3=prepare_data('data/Test_data/Test_MS3')
          data_MS4=prepare_data('data/Test_data/Test_MS4')
          data_PAN=prepare_data('data/Test_data/Test_PAN')
          for i in range(len(data_MS1)):
              train_data_MS1,train_data_MS2,train_data_MS3,train_data_MS4,train_data_PAN=input_setup(i)
              start=time.time()
              result =sess.run(fusion_image,feed_dict={images_MS1: train_data_MS1,images_MS2: train_data_MS2,images_MS3: train_data_MS3,images_MS4: train_data_MS4,images_PAN: train_data_PAN})
              result=result*127.5+127.5
              result = result.squeeze()
              end=time.time()
              image_path = os.path.join(os.getcwd(), 'result','epoch'+str(num_epoch))
              if not os.path.exists(image_path):
                  os.makedirs(image_path)              
              image_path = os.path.join(image_path,str(i+1)+".mat")
              scio.savemat(image_path, {'I':result})
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1
