# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    return data

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
  """
  image = imread(path, is_grayscale=True)
  # Must be normalized
  image = (image-127.5 )/ 127.5 
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
  return input_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    #将图片按序号排序
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
  #print(data)

  return data

def make_data(sess, data, data_dir):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    #savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
    savepath = os.path.join('.', os.path.join('checkpoint',data_dir,'train.h5'))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint',data_dir))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint',data_dir)))
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup_MS(sess,config,data_dir,index=0):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset=data_dir)

  sub_input_sequence = []
  padding = 0 
  if config.is_train:
    for i in xrange(len(data)):
      input_=(imread(data[i])-127.5)/127.5

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      for x in range(0, h-config.image_size_MS+1, config.stride_MS):
        for y in range(0, w-config.image_size_MS+1, config.stride_MS):
          sub_input = input_[x:x+config.image_size_MS, y:y+config.image_size_MS]    
          if data_dir == "Train":
            sub_input=cv2.resize(sub_input, (config.image_size_MS/4,config.image_size_MS/4),interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size_MS/4, config.image_size_MS/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size_MS, config.image_size_MS, 1])            
          sub_input_sequence.append(sub_input)
  arrdata = np.asarray(sub_input_sequence) 
  print(arrdata.shape)
  make_data(sess, arrdata, data_dir)

def input_setup_PAN(sess,config,data_dir,index=0):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  if config.is_train:
    data = prepare_data(sess, dataset=data_dir)

  sub_input_sequence = []
  padding = 0 
  if config.is_train:
    for i in xrange(len(data)):
      input_=(imread(data[i])-127.5)/127.5

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      for x in range(0, h-config.image_size_PAN+1, config.stride_PAN):
        for y in range(0, w-config.image_size_PAN+1, config.stride_PAN):
          sub_input = input_[x:x+config.image_size_PAN, y:y+config.image_size_PAN]     
          # Make channel value
          if data_dir == "Train":
            sub_input=cv2.resize(sub_input, (config.image_size_PAN/4,config.image_size_PAN/4),interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size_PAN/4, config.image_size_PAN/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size_PAN, config.image_size_PAN, 1])            
          sub_input_sequence.append(sub_input)
  arrdata = np.asarray(sub_input_sequence) 
  print(arrdata.shape)
  make_data(sess, arrdata, data_dir)


  if not config.is_train:
    print(nx,ny)
    print(h_real,w_real)
    return nx, ny,h_real,w_real
    
def imsave(image, path):
  return scipy.misc.imsave(path, image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return (img*127.5+127.5)
  
def sobel_gradient(input):
    filter_x=tf.reshape(tf.constant([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]),[3,3,1,1])
    filter_y=tf.reshape(tf.constant([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]),[3,3,1,1])
    d_x=tf.nn.conv2d(input,filter_x,strides=[1,1,1,1], padding='SAME')
    d_y=tf.nn.conv2d(input,filter_y,strides=[1,1,1,1], padding='SAME')
    return d_x, d_y

def sobel_gradient_4(input):
    filter_x=tf.reshape(tf.constant([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]),[3,3,1,1])
    filter_y=tf.reshape(tf.constant([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]),[3,3,1,1])
    d1_x=tf.nn.conv2d(tf.expand_dims(input[:,:,:,0],3),filter_x,strides=[1,1,1,1], padding='SAME')
    d1_y=tf.nn.conv2d(tf.expand_dims(input[:,:,:,0],3),filter_y,strides=[1,1,1,1], padding='SAME')
    d2_x=tf.nn.conv2d(tf.expand_dims(input[:,:,:,1],3),filter_x,strides=[1,1,1,1], padding='SAME')
    d2_y=tf.nn.conv2d(tf.expand_dims(input[:,:,:,1],3),filter_y,strides=[1,1,1,1], padding='SAME')
    d3_x=tf.nn.conv2d(tf.expand_dims(input[:,:,:,2],3),filter_x,strides=[1,1,1,1], padding='SAME')
    d3_y=tf.nn.conv2d(tf.expand_dims(input[:,:,:,2],3),filter_y,strides=[1,1,1,1], padding='SAME')
    d4_x=tf.nn.conv2d(tf.expand_dims(input[:,:,:,3],3),filter_x,strides=[1,1,1,1], padding='SAME')
    d4_y=tf.nn.conv2d(tf.expand_dims(input[:,:,:,3],3),filter_y,strides=[1,1,1,1], padding='SAME')
    d_x=tf.concat([d1_x,d2_x,d3_x,d4_x],axis=-1) 
    d_y=tf.concat([d1_y,d2_y,d3_y,d4_y],axis=-1) 
    return d_x, d_y

def lpls_gradient(input):
    filter=tf.reshape(tf.constant([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]),[3,3,1,1])
    d=tf.nn.conv2d(input,filter,strides=[1,1,1,1], padding='SAME')
    return d

def lpls_gradient_4(input):
    filter=tf.reshape(tf.constant([[1.,1.,1.],[1.,-8.,1.],[1.,1.,1.]]),[3,3,1,1])
    d1=tf.nn.conv2d(tf.expand_dims(input[:,:,:,0],3),filter,strides=[1,1,1,1], padding='SAME')
    d2=tf.nn.conv2d(tf.expand_dims(input[:,:,:,1],3),filter,strides=[1,1,1,1], padding='SAME')
    d3=tf.nn.conv2d(tf.expand_dims(input[:,:,:,2],3),filter,strides=[1,1,1,1], padding='SAME')
    d4=tf.nn.conv2d(tf.expand_dims(input[:,:,:,3],3),filter,strides=[1,1,1,1], padding='SAME')
    d=tf.concat([d1,d2,d3,d4],axis=-1)      
    return d


def linear_map(input,weight):
    output_1=input*weight
    output=tf.reduce_sum(output_1, axis=3, keepdims=True)
    return output


   
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)
    
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm
