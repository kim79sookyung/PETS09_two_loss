import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from rnn import *
import numpy as np

threshold = 0.3

def get_max(my_list):
    m = None
    for item in my_list:
        if isinstance(item, list):
            item = get_max(item)
        if not m or m < item:
            m = item
    return m

def get_min(my_list):
    m = None
    for item in my_list:
        if isinstance(item, list):
            item = get_min(item)
        if not m or m > item:
            m = item
    return m

def find_minimum_continuous_digts_cluster(h):
    if len(h)==0:
        return [0]
    else:
        h_index=[]
        h_sub=[h[0]]
        h_cp=h
        for j in xrange(1,len(h)):
            if h_cp[j]==h_cp[j-1]+1:
                h_sub.append(h_cp[j])
            else:
                h_index.append(h_sub)
                h_sub=[h[j]]
        h_index.append(h_sub)
        length=[]
        for j in range(len(h_index)):
            length.append(len(h_index[j]))
        indx=length.index(max(length))
        h_cluster=h_index[indx]
    return h_cluster




def cleaning_boundingbox(image):
    #np.shape(image): [h,w,channels]
    #in image, find tightest bounding box
    image = np.reshape(image,[288,384,3])
    image_1ch = image[:,:,0]+image[:,:,1]+image[:,:,2]
    #print(np.shape(image),np.shape(image_1ch)) #((1, 288, 384, 3), (1, 288, 3))
    #(1) Thresholding 1-d image
    h_index, w_index = np.where( image_1ch > float(threshold) )
    if len(h_index)==0:
        print("threshold too high")
        return np.zeros(np.shape(image))
    else:
        #(2)align w according to h
        h=[h_index[0]]; w=[]; w_inside=[w_index[0]]
        for i in xrange(1,len(h_index)):
            if h_index[i]!=h_index[i-1]:
                h.append(h_index[i])
                w.append(w_inside)
                w_inside=[w_index[i]]
            else:
                w_inside.append(w_index[i])
        w.append(w_inside)
        #(3)bounding box citeria(1) h should be countiuous
        #(3-1)So, find max continuous digits in h cluster
        h_cluster=find_minimum_continuous_digts_cluster(h)
        #(3-2)Then, align w according to h_cluster index
        w_cluster=[]
        for j in range(len(h_cluster)):
            w_cluster.append(w[h.index(h_cluster[j])] )
        w_index_all=[]
        for j in range(len(h_cluster)):
            w_index_all=w_index_all+w_cluster[j]
        #(4)Pick unique indexs only in w 
        w_index=np.unique(w_index_all)
        w_index_count=[]
        length=len(w_cluster)
        for i in w_index:
            count_val=0
            for j in range(len(w_cluster)):
                if i in w_cluster[j]:
                    count_val=count_val+1
            w_index_count.append(count_val)
        w_index_update=[]
        for i in range(len(w_index)):
            if w_index_count[i] > int(length*0.5):
                w_index_update.append(w_index[i])
        #(5) Find max continuous digits in w cluster
        w_index_update=find_minimum_continuous_digts_cluster(w_index_update)
        w_min=get_min(w_index_update)
        w_max=get_max(w_index_update)
        h_max=get_max(h_cluster)
        h_min=get_min(h_cluster)
        print(h_max,h_min,w_max,w_min)
        image_mask =  np.zeros(np.shape(image))
        for ch in range(3):
            for i in range(288):
                for j in range(384):
                    if i > h_min and i < h_max and j > w_min and j < w_max:
                        image_mask[i,j,ch] = image[i,j,ch]
                    else:
                        image_mask[i,j,ch] = 0.0
    return image_mask



#NEED to CORRECT
def test(name,sess,train_op,prediction,last_state):
    fetches = {'final_state': last_state,
              'prediction_image': prediction}
    for ii in range(35): #total 35 tracks in test data
        image=np.load("/export/kim79/h2/Crowd_PETS09/test/div_image_4x4_"+str(ii)+".npy") #one track
        init_label=np.load("/export/kim79/h2/Crowd_PETS09/test/div_hwxy_"+str(ii)+".npy")[0] #(19, 4)
        d1,d2,d3,d4=np.shape(image) #(100,h,w,3)
        output_image = np.zeros([d1,4])
        timestep=10
        for j in range(int(d1/(timestep-1))+1):
            if j ==  int(d1/(timestep-1)): # append to last block
                end = d1
                start = end - timestep
                print("hit")
            else:
                start = j*(timestep-1)
                end = (j+1)*(timestep-1)+1
                if end > d1:
                    end=d1
                    start =  end - timestep
            #print(ii, int(d1/(timestep-1)),j,start,end,end-start)
            test_x=np.reshape(image[start:end,:,:,:],[1, end-start, h, w, channels])
            test_X=np.concatenate([test_x for i in range(batch_size)],0) # all batch has same vector
            if j == 0:
                X_te=switch_first_frame_hwxy(name,test_X,init_label, timestep, 10) #X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels])
            else: #recursive feed back
                o1,o2,o3,o4,o5=np.shape(output)
                X_te=switch_first_frame_hwxy(name,test_X, output[0,o2-1,:], (end-start), 10) #X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels]) 
            eval_out=sess.run(fetches, feed_dict={X:X_te})
            output=eval_out['prediction_image']
            len_out=len(output)
            #print(output[0,10:10+(end-start),:,:,:])
            output_image[start:end,:] = output[0,10:10+(end-start),:]
        print(ii)
        np.save("test_result_rcfb_"+str(name)+"_track_"+str(ii)+".npy", output_image)


#(51,4)
##hwxy in the scale of (576, 768) : need to divide with [576,768,768,576]
def rescale_test(label_out):
   d1,d2=np.shape(label_out) #[51,4]
   label=np.zeros([d1,d2+1])
   for i in range(d1):
       label[i,0]=float(label_out[i,0])/576.0
       label[i,1]=float(label_out[i,1])/768.0
       label[i,2]=float(label_out[i,2])/768.0
       label[i,3]=float(label_out[i,3])/576.0
       #Add confidence score here
       label[i,4]=float(1.0)
   return label


def test_gt(name,sess,train_op,X,prediction_image, prediction,last_state):
    fetches = {'final_state': last_state,
              'prediction_heatmap': prediction_image,
              'prediction_hwxy': prediction }
    for ii in range(35):
        image=np.load("/export/kim79/h2/Crowd_PETS09/test/div_image_4x4_"+str(ii)+".npy") #one track
        init_label_hwxy=np.load("/export/kim79/h2/Crowd_PETS09/test/div_hwxy_"+str(ii)+".npy")
        init_label_hwxy=rescale_test(init_label_hwxy)
        init_label_heatmap=np.load("/export/kim79/h2/Crowd_PETS09/test/div_label_heatmap_"+str(ii)+".npy")
        d1,d2,d3,d4=np.shape(image) #(100,h,w,3)
        output_heatmap = np.zeros(np.shape(image))
        output_hwxy_all = np.zeros([d1,4])
        timestep=10
        for j in range(int(d1/(timestep-1))+1):
            if j ==  int(d1/(timestep-1)): # append to last block
                end = d1
                start = end - timestep
                print("hit")
            else:
                start = j*(timestep-1)
                end = (j+1)*(timestep-1)+1
                if end > d1:
                    end=d1
                    start =  end - timestep
            test_x=np.reshape(image[start:end,:,:,:],[1, end-start, h, w, channels])
            test_X=np.concatenate([test_x for i in range(batch_size)],0) # all batch has same vector
            X_te = switch_first_frame(name, test_X, init_label_heatmap[start,:,:,:], timestep, 10) 
            eval_out=sess.run(fetches, feed_dict={X:X_te})
            output_image = eval_out['prediction_heatmap']
            output_hwxy  = eval_out['prediction_hwxy']
            len_out=len(output_image)
            assert(len(output_image)==len(output_hwxy))
            output_heatmap[start:end,:,:,:] = output_image[0,10:10+(end-start),:,:,:]
            output_hwxy_all[start:end,:] = output_hwxy[0,10:10+(end-start),:]

        print(ii)
        np.save("test_result_"+str(name)+"_track_heatmap_"+str(ii)+".npy", output_heatmap
)
        np.save("test_result_"+str(name)+"_track_hwxy_"+str(ii)+".npy", output_hwxy_all)



