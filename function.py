# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import numpy as np
parser = argparse.ArgumentParser()
# Basic model parameters.
batch_size=16
timestep=10
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of images to process in a batch.')



FLAGS = parser.parse_args()

# Constants describing the training process.
h=144
w=192
channels = 3

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

#need to check




def copy_init_snapshot( image, label_heatmap, label_hwxy, itr ):
    #shape of image and label [FLAGS.batch_size,10,h,w,ch]
    image_out=np.zeros([FLAGS.batch_size,itr+timestep,h,w,3])
    label_out_heatmap=np.zeros([FLAGS.batch_size,itr+timestep,h,w,1]) #train itr times
    label_out_hwxy=np.zeros([FLAGS.batch_size,itr+timestep,5]) 
    #Swap first input
    for k in range(FLAGS.batch_size):
        init_heatmap=label_heatmap[k,0,:,:,:]
        init_hwxy=label_hwxy[k,0,:]
        init_img=image[k,0,:,:,:]
        for j in range(itr+timestep):
            if j<(itr+1):
                label_out_heatmap[k,j,:,:,:]=init_heatmap
                label_out_hwxy[k,j,:]=init_hwxy
                image_out[k,j,:,:,:]=np.multiply(np.concatenate([init_heatmap for i in range(channels)],2),init_img)
            else:
                label_out_heatmap[k,j,:,:,:]=label_heatmap[k,j-itr,:,:,:]
                label_out_hwxy[k,j,:]=label_hwxy[k,j-itr,:]
                image_out[k,j,:,:,:]=image[k,j-itr,:,:,:]
    return image_out, label_out_heatmap, label_out_hwxy





def switch_first_frame(name, test_X, image, timestep, itr):
#    if int(name) > 3 and int(name)%2==0:
#    image=cleaning_boundingbox(image)
    image_out=np.zeros([batch_size,itr+timestep,h,w,3])
    #print(np.shape(test_X),np.shape(image))
    #((16, 10, 144, 192, 3), (144, 192, 1))
    #Swap first input
    for k in range(batch_size):
        for j in range(itr+timestep):
            if j<(itr+1):
                ch1=np.reshape(np.multiply(image[:,:,0],test_X[k,0,:,:,0]),[h,w,1])
                ch2=np.reshape(np.multiply(image[:,:,0],test_X[k,0,:,:,1]),[h,w,1])
                ch3=np.reshape(np.multiply(image[:,:,0],test_X[k,0,:,:,2]),[h,w,1])
                init=np.concatenate([ch1,ch2,ch3],2)
                image_out[k,j,:,:,:]=init
            else:
                image_out[k,j,:,:,:]=test_X[k,j-itr,:,:,:]
    return image_out





def switch_first_frame_hwxy(name, test_X, image, timestep, itr):
    image_out=np.zeros([batch_size,itr+timestep,h,w,3])
    #print(np.shape(test_X),np.shape(image))
    #((16, 10, 144, 192, 3), (4-hwxy))
    #Swap first input
    for k in range(batch_size):
        for j in range(itr+timestep):
            if j<(itr+1):
                heatmap=generate_heatmap_hwxy(image)
                ch1=np.reshape(np.multiply(heatmap[:,:,0],test_X[k,0,:,:,0]),[h,w,1])
                ch2=np.reshape(np.multiply(heatmap[:,:,0],test_X[k,0,:,:,1]),[h,w,1])
                ch3=np.reshape(np.multiply(heatmap[:,:,0],test_X[k,0,:,:,2]),[h,w,1])
                init=np.concatenate([ch1,ch2,ch3],2)
                image_out[k,j,:,:,:]=init
            else:
                image_out[k,j,:,:,:]=test_X[k,j-itr,:,:,:]
    return image_out






def crop_around_lonlat(image,y_lonlat_in):
  """From large image,"image", crop sub region(10x10) centering (lon,lat)

  Args:
    image: X-[bachsize,1,feature_size(h*w)*channels]: (24,1,22188)
    y_lonlat_in: [batchsize,1,2] : (24,1,2)
  Returns:
    cropped_image: (24,10(h),10(w),channels) = (24,10,10,2)
  """
  image=np.reshape(image, [FLAGS.batch_size,1,h,w,channels])  
  cropped_image=[];
  for i in range(int(FLAGS.batch_size)):
      lon,lat=y_lonlat_in[i,0,:]
      lon_index=int(lon*w)
      lat_index=int(lat*h)
      lat_lb=lat_index-5
      lat_up=lat_index+5
      lon_lb=lon_index-5
      lon_up=lon_index+5
      if float(lat_index-5)<0.0 : 
          lat_lb=0 
          lat_up=lat_lb+10;
      if float(lat_index+5)>(h-1):
          lat_up=(h-1)
          lat_lb=(h-1)-10
      if float(lon_index-5)<0.0:
          lon_lb=0
          lon_up=lon_lb+10
      if float(lon_index+5)>(w-1):
          lon_up=w-1
          lon_lb=(w-1)-10
      cropped_image.append([image[i,0,lat_lb:lat_up,lon_lb:lon_up,:]])
  cropped_image=np.asarray(np.concatenate(cropped_image,axis=0))
  return cropped_image





def mask_around_lonlat(image_in,y_lonlat_in):
  """From large image,"image", crop sub region(10x10) centering (lon,lat)

  Args:
    image_in: X-[bachsize,timesteps,feature_size(h*w)*channels]: (batch_size,timesteps,22188)
    y_lonlat_in: [batchsize,timesteps,2] : (batch_size,timesteps,2)
  Returns:
    masked_image: (batch_size,timesteps, 86(h)*129(w)*2(channels)): Mask size is 10x10
  """
  batch_size,timesteps,features=np.shape(image_in)
  mask=[];
  image_input=np.copy(image_in)
  for i in range(int(FLAGS.batch_size)):
      mask_t=[];
      for t in range(int(timesteps)):
          image=image_input[i,t,:]
          image=np.reshape(image, [h,w,channels])
          lon,lat=y_lonlat_in[i,t,:]
          lon_index=int(lon*w)
          lat_index=int(lat*h)
          lat_lb=lat_index-10
          lat_up=lat_index+10
          lon_lb=lon_index-10
          lon_up=lon_index+10
          if float(lat_index-10)<0.0 :
              lat_lb=0
              lat_up=lat_lb+20;
          if float(lat_index+10)>(h-1):
              lat_up=h
              lat_lb=h-20
          if float(lon_index-10)<0.0:
              lon_lb=0
              lon_up=lon_lb+20
          if float(lon_index+10)>(w-1):
              lon_up=w
              lon_lb=w-20
          image[ 0:lat_lb,  :,  :]=0
          image[ lat_up:h, :,  :]=0
          image[   :,0:lon_lb,  :]=0
          image[   :,lon_up:w,:]=0
          mask_t.append(image)
      mask.append([mask_t])
  mask=np.asarray(np.concatenate(mask,axis=0))
  mask=np.reshape(mask,[FLAGS.batch_size,timesteps,h*w*channels])
  return mask



def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var




def Inference(images,timesteps):
  """Build the CNN to embed climate image
  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 2, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    
    output_loc=tf.reshape(softmax_linear,[FLAGS.batch_size,1, -1])
  return output_loc



def inference_1layer(images):
  #conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 2, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool1, [FLAGS.batch_size*timesteps, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    output_loc=tf.reshape(softmax_linear,[FLAGS.batch_size,timesteps, -1])
  return output_loc


def old_embedding_1layer(xx):
        #Make embedding of X using tf.nn.conv1d
        #temp=[];
        #for i in range(timesteps):
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(xx, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv1_flatten=tf.reshape(conv1,[FLAGS.batch_size*timesteps,-1]);
        ##FC
        fc1=tf.layers.dense(conv1_flatten,8192)
        fc1=tf.layers.dropout(fc1,rate=0.8);
        x_em=tf.layers.dense(fc1,4096)
        x_em=tf.reshape(x_em,[FLAGS.batch_size,timesteps,-1])
        print("SIZE "+str(x_em));
        return x_em
