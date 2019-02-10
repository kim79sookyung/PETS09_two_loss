import tensorflow as tf
from tensorflow.contrib import rnn
from rnn import *
from cnn import *
from function import *
import numpy as np
import random



def train(ii,sess,loss_op,train_op,X,Y_heatmap, Y_hwxy, train_X,train_Y_heatmap,train_Y_hwxy,val_X,val_Y_heatmap,val_Y_hwxy,prediction, last_state,fout_log):
    val_best_loss=10; val_best_step=0;
    alpha=0.9
    count=0
    train_size=len(train_X) 
    val_size=len(val_X)
    saver = tf.train.Saver()
    fetches = {'final_state': last_state,
              'prediction_lonlat': prediction}
    for epoch in range(1):
        print(train_size)
        for step in range(train_size):
            stepp=step
            print("train step "+str(stepp))
            X_in, Y_in_heatmap, Y_in_hwxy = copy_init_snapshot(train_X[step], train_Y_heatmap[step], train_Y_hwxy[step], 10)
            #print(np.shape(X_in), np.shape(Y_in_hwxy), np.shape(Y_in_heatmap))
            train=sess.run(train_op, feed_dict={X:X_in, Y_heatmap:Y_in_heatmap, Y_hwxy:Y_in_hwxy})
            # Calculate batch loss in validation set
            if stepp%10 == 0:
                val_sum=0
             #   for j in range(val_size):
                j=random.randint(0,val_size-1)
                X_val,Y_val_heatmap, Y_val_hwxy=copy_init_snapshot(val_X[j], val_Y_heatmap[j], val_Y_hwxy[j], 10)
                lossv= 10000 * sess.run(loss_op,  feed_dict={X:X_val, Y_heatmap:Y_val_heatmap, Y_hwxy: Y_val_hwxy})
                val_sum=lossv
                loss = val_sum
                #Calculated running average of val_loss
                if stepp < 11:  # loss start from very large val at step=0, so start from 10th step
                    val_loss = loss
                elif stepp >11:
                    val_loss = alpha * val_loss + (1-alpha) * loss
                    #write up
                    fout_log.write(str(ii) +" Step " + str(step) + ", Validation running average= " + \
                          "{:.4f}".format(val_loss ** 0.5) + ",Validation Loss= "+\
                          "{:.4f}".format(loss **0.5) + "\n")
                    print("epoch "+str(ii)+" Step " + str(step) + ", Validation running average= " + \
                          "{:.4f}".format(val_loss ** 0.5) + ",Validatiion Loss= "+\
                          "{:.4f}".format(loss **0.5) + "\n")
                    if stepp > 10 and val_loss < val_best_loss:
                        val_best_loss = val_loss
                        save_path = saver.save(sess, "./model.ckpt")
                        print('found new best validation loss:', val_loss)
                        print("Model saved in path: %s" % save_path)
                        count = 0
                    if (epoch > 5) and (val_loss > val_best_loss):
                        count =  count + 1
                        if (count > 10):
                            print("Iteration "+str(it)+"Training DONE!  \n")
                            break





