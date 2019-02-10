from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from cnn import *
from rnn import *
import numpy as np
#import skimage.measure
parser = argparse.ArgumentParser()


#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.001
X = tf.placeholder("float", [FLAGS.batch_size, None, h, w, channels]) 
Y_heatmap = tf.placeholder("float", [FLAGS.batch_size, None, h,w,1]) #heatmap
Y_hwxy = tf.placeholder("float", [FLAGS.batch_size, None, 5]) #hwxy+confidence
#hwxy in the scale of (576, 768) : need to divide with [576,768,768,576]
timesteps = tf.shape(X)[1]
h=tf.shape(X)[2] 
w=tf.shape(X)[3] 

prediction_image, last_state = ConvLSTM(X) 
loss_op_heatmap=tf.losses.mean_pairwise_squared_error(Y_heatmap,prediction_image)
prediction = Multi_CNN(prediction_image)
print(prediction.get_shape())
loss_op_hwxy=tf.losses.mean_squared_error(Y_hwxy,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
loss_op = loss_op_hwxy+loss_op_heatmap
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    #test_gt("temp",sess,train_op,X,prediction_image, prediction,last_state)
    start=[0,10,20,30,40,50,60,70,80,90]
    end = [10,20,30,40,50,60,70,80,90,100]
    for ii in range(1000):
        name=str(ii)
        for k in range(len(start)):
            train_X,train_Y_heatmap,train_Y_hwxy,val_X,val_Y_heatmap,val_Y_hwxy=read_input("/export/kim79/convLSTM_for_ETC/Model_ts10/tracking_module_test_2_follow_started_hurricane_moving_mnist_test2/PETS200999_ts5_downsample/soo_ts10_downsample4xr_pet09_output_heatmap/data_ts10_ds4x4/",start[k],end[k])
            train(ii,sess,loss_op,train_op,X,Y_heatmap, Y_hwxy,train_X,train_Y_heatmap,train_Y_hwxy,val_X,val_Y_heatmap,val_Y_hwxy,prediction, last_state,fout_log)
        if ii>0 and ii%10==0:
            test_gt(name,sess,train_op,X,prediction_image, prediction,last_state)
            #test(name,sess,train_op,prediction,last_state) 
            save_path = saver.save(sess, "./model_"+str(ii)+".ckpt")
fout_log.close();

