import tensorflow as tf
from function import *
import numpy as np
import random


def generate_data(path,timesteps,batch_size,startt,endt):
    X=[]
    Y=[]
    Y_bb=[]
    index=[i for i in xrange(startt,endt)] #Memory error 1441)] #0, 1, 2, ... , n-1
    for i in range(len(index)):
        print("reading "+str(i)+"th data")
        image=np.load(path+"div_image_"+str(i)+".npy") #(65, 256, 513, 6)
        image=image[:,:,:,0:3]
        image=np.divide(image,255.0)
        label=np.load(path+"div_label2_"+str(i)+".npy")
        label=np.divide(label,255.0)
        a1,a2,a3,a4=np.shape(label)
        label_bb=np.load(path+"div_hwxy_"+str(i)+".npy")
        b1,b2=np.shape(label_bb)
        print(np.shape(label_bb))
        label=np.reshape(label,[a1,a2,a3,3])
        print(np.shape(image),np.shape(label))
        d1,d2,d3,d4=np.shape(image)
        print(a1,b1)
        assert a1==b1
        if a1 > timesteps :
            for j in range(int(a1/(timesteps*0.5))-1):
                start=j*int(timesteps*0.5)
                end=start+timesteps
                X.append(np.reshape(image[start:end,:,:,0:3],[1,timesteps,d2,d3,3]))
                Y.append(np.reshape(label[start:end,:,:,:],[1,timesteps,d2,d3,3]))
                Y_bb.append(np.reshape(label_bb[start:end,:],[1,timesteps,4]))
        if i%100==0: print(np.shape(X))
    s1=len(Y_bb)
    random_index=[ i for i in range(s1) ]
    random.index=random.shuffle(random_index)
    X_r=[]
    Y_r=[]
    Y_bb_r=[]
    #random shuffle the order of dataset
    for i in range(len(random_index)):
        X_r.append(X[random_index[i]])
        Y_r.append(Y[random_index[i]])
        Y_bb_r.append(Y_bb[random_index[i]])
    X_r=np.concatenate(X_r,0)
    Y_r=np.concatenate(Y_r,0)
    Y_bb_r=np.concatenate(Y_bb_r,0)
    print(np.shape(X_r),np.shape(Y_r),np.shape(Y_bb_r))
    num_of_groups=int(s1/batch_size)
    X_sample=X_r[0:batch_size*num_of_groups,:,:,:,:]
    Y_sample=Y_r[0:batch_size*num_of_groups,:,:,:,:]
    Y_bb_sample = Y_bb_r[0:batch_size*num_of_groups,:,:]
    X_sample_last=X_r[s1-batch_size:s1,:,:,:,:]
    Y_sample_last=Y_r[s1-batch_size:s1,:,:,:,:]
    Y_bb_sample_last=Y_bb_r[s1-batch_size:s1,:,:]
    X_sample=np.concatenate([X_sample, X_sample_last],0)
    Y_sample=np.concatenate([Y_sample, Y_sample_last],0)
    Y_bb_sample=np.concatenate([Y_bb_sample, Y_bb_sample_last],0)
    X_sample=np.reshape(X_sample,[num_of_groups+1,batch_size,timesteps,d2,d3,3])
    Y_sample=np.reshape(Y_sample,[num_of_groups+1,batch_size,timesteps,d2,d3,3])
    Y_bb_sample=np.reshape(Y_bb_sample,[num_of_groups+1,batch_size,timesteps,4])
    np.save("X_PETS_"+str(startt)+"_"+str(endt)+".npy",X_sample)
    np.save("Y_PETS_"+str(startt)+"_"+str(endt)+".npy",Y_sample)
    np.save("Y_PETS_bb_"+str(startt)+"_"+str(endt)+".npy",Y_bb_sample)
    return X_sample,Y_sample,Y_bb_sample




#(34, 16, 10, 144, 192, 1)
##hwxy in the scale of (576, 768) : need to divide with [576,768,768,576]
def rescale(label_out):
   d1,d2,d3,d4=np.shape(label_out)
   label=np.zeros([d1,d2,d3,d4+1])
   for i in range(d1):
       for j in range(d2):
           for k in range(d3):
               label[i,j,k,0]=float(label_out[i,j,k,0])/576.0
               label[i,j,k,1]=float(label_out[i,j,k,1])/768.0
               label[i,j,k,2]=float(label_out[i,j,k,2])/768.0
               label[i,j,k,3]=float(label_out[i,j,k,3])/576.0
               #Add confidence score here
               label[i,j,k,4]=float(1.0)

   return label
                

def read_input(path,start,end):
    image_out=np.load(path+"/X_PETS_"+str(start)+"_"+str(end)+".npy") 
    label_hwxy_out=np.load(path+"/Y_PETS_bb_"+str(start)+"_"+str(end)+".npy")
    label_hwxy_out=rescale(label_hwxy_out)
    label_heatmap_out=np.load(path+"/Y_PETS_"+str(start)+"_"+str(end)+".npy")
    print(np.shape(label_hwxy_out),np.shape(image_out))
    d,__,__,__,__,__ = np.shape(image_out)
    tr_image=np.asarray(image_out[0:d-10])
    tr_hwxy_label=np.asarray(label_hwxy_out[0:d-10])
    tr_heatmap_label=np.asarray(label_heatmap_out[0:d-10])
    va_image=np.asarray(image_out[d-10:d])
    va_hwxy_label=np.asarray(label_hwxy_out[d-10:d])
    va_heatmap_label=np.asarray(label_heatmap_out[d-10:d])
    print(np.shape(tr_image),np.shape(tr_heatmap_label),np.shape(tr_hwxy_label),np.shape(va_image),np.shape(va_heatmap_label),np.shape(va_hwxy_label))
    #((3, 16, 25, 144, 192, 3), (3, 16, 25, 144, 192, 1), (3, 16, 25, 4), (10, 16, 25, 144, 192, 3), (10, 16, 25, 144, 192, 1), (10, 16, 25, 4))
    return tr_image,tr_heatmap_label,tr_hwxy_label,va_image,va_heatmap_label,va_hwxy_label






def data_synthesis_new_event2(time_length,max_number_of_balls,height,width): #(10,5,256,513)
    num=random.randint(1,max_number_of_balls)
    image_all=[]; label_all=[]
    for i in range(num):
        start_lon=random.randint(0,width-1); start_lat=random.randint(0,height-1)
        end_lon=random.randint(0,width-1); end_lat=random.randint(0,height-1)
        variance=random.randint(7,50)
        Sigma = np.array([[ variance , 0], [0., variance]])
        if i==0:
            start_time=0
        else:
            start_time=random.randint(1,time_length-1)
        end_time=random.randint(start_time, time_length-1)
        time_div=(end_time+1)-start_time
        lon=[];lat=[];count=0
        for j in range(time_length):
            if j < start_time or j > end_time:
                lon.append(0); lat.append(0);
            else:
                lon.append(start_lon+count*(end_lon-start_lon)/float(time_div))
                lat.append(start_lat+count*(end_lat-start_lat)/float(time_div))
                count=count+1
        image_list=[];label_list=[];
        #Generate image
        for j in range(len(lon)):
            if lon[j]==0: 
                image_list.append(np.zeros([1,height,width]))
            else:
                mu=np.array([float(lon[j]),float(lat[j])])
                image_list.append(multivariate_gaussian(height,width,mu,Sigma))
        image=np.concatenate(image_list,0) #[10,256,513]
        image_all.append(image)
        #Generate label
        if i==0:
            for j in range(len(lon)):
                if j > start_time or j == start_time:
                    label_list.append(np.reshape(image[j,:,:],[1,height,width]))
                else:
                    label_list.append(np.zeros([1,height,width]))
            label=np.concatenate(label_list,0) #[10,256,513]
            label_all.append(label)
        print(start_lon,start_lat)
        print(end_lon, end_lat) 
        print(lon,lat)
    out_image=np.zeros([time_length,height,width])
    out_label=np.zeros([time_length,height,width])
    for j in range(time_length):
        for i in range(num):
            out_image[j,:,:]=out_image[j,:,:]+image_all[i][j,:,:]
            if i==0:
                out_label[j,:,:]=out_label[j,:,:]+label_all[i][j,:,:]
    return out_image, out_label


#train_X,train_Y,val_X,val_Y=read_input("./data_ts10/",0,0)
