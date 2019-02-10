from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from rnn import *

def Multi_CNN(prediction):
    # Refered from SH's code
    # Input Layer
    outputs = tf.reshape(prediction, [ -1, h, w, 1])

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 1, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(outputs, kernel) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)

    pool1 = pool_max(output_conv1_2)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(pool2, kernel) + biases), name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(output_conv3_1, kernel) + biases), name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(output_conv3_2, kernel) + biases), name=scope)

    pool3 = pool_max(output_conv3_3)

    #fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        kernel = weight_variable([shape, 1024])
        biases = bias_variable([1024])
        pool3_flat = tf.reshape(pool3, [-1, shape])
        output_fc1 = tf.nn.relu(tf.layers.batch_normalization(fc(pool3_flat, kernel, biases)), name=scope)

    #fc2
    with tf.name_scope('fc2') as scope:
        kernel = weight_variable([1024, 5])
        biases = bias_variable([5])
        #output_fc2 = fc(output_fc1, kernel, biases)
        output_fc2 = fc(output_fc1, kernel, biases)

    # print(finaloutput.shape())
    #output_sig = tf.sigmoid(output_fc2, name="prediction")

    prediction = tf.reshape(output_fc2, [FLAGS.batch_size, -1, 5])
    #prediction[:,:,-1].assign(tf.sigmoid(prediction[:,:,-1], name="confidence"))

    return prediction
# Didn't apply drop_out here

def weight_variable(shape, name="weights"):
    initial = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32) # He's initialization
    return tf.Variable(initial(shape), name=name)

def bias_variable(shape, name="biases"):
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(input, w):
    return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

def pool_max(input):
    return tf.nn.max_pool(input,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

def fc(input, w, b):
    return tf.matmul(input, w) + b

