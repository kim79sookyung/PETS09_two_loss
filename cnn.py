from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from rnn import *

def Multi_CNN(prediction):
  # Input Layer
  input_layer = tf.reshape(prediction, [ -1, h, w, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=12,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=24,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


#  # Convolutional Layer #2 and Pooling Layer #2
#  conv3 = tf.layers.conv2d(
#      inputs=pool2,
#      filters=32,
#      kernel_size=[3, 3],
#      padding="same",
#      activation=tf.nn.relu)
#  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  print(pool2.get_shape()) #(?, 36, 48, 24) 
  __,a1,a2,a3=pool2.get_shape()
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, a1*a2*a3])
  output = tf.layers.dense(inputs=pool2_flat, units=5)
#  dropout = tf.layers.dropout(
#      inputs=dense, rate=0.5)

  # output Layer
#  output = tf.layers.dense(inputs=dropout, units=5)
  output = tf.reshape(output, [FLAGS.batch_size, -1, 5])

  return output
