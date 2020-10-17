"""
This script goes along my blog post:
'Keras Cats Dogs Tutorial' (https://jkjung-avt.github.io/keras-tutorial/)
"""


import os
#force cpu processing
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import glob
import argparse
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.backend.tensorflow_backend import set_session
import pandas as pd
import math
import sys
import dill
from time import sleep
sys.path.append('../')
from focallosskeras.losses import *

# import keras_metrics
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))


def parse_args():
    """Parse input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files

def get_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
#     if img is None:
#         continue
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    # plt.imshow(x[0])
    # plt.show()
    pred = net.predict(x)[0]
    return x[0],pred

def dump(x):
    return x.split('/')[-1]

if __name__ == '__main__':
#    print("in sleep")
#    time.sleep(2000)
    print("model at work :P")
    inp_path = '/media/vidooly/myfiles2/ml/projects/indian_channel_scanning/9sept/thumbnails/'
    out_dir = '/media/vidooly/myfiles2/ml/projects/indian_channel_scanning/9sept/kiss_output/'
    cls_list = ['adult', 'normal']

    for i in range(11):
        os.makedirs(out_dir+'safe/'+str(i), exist_ok=True)
        os.makedirs(out_dir+'unsafe/'+str(i), exist_ok=True)
        custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss': categorical_focal_loss,
                 'binary_focal_loss': binary_focal_loss}

    # load the trained model
    lines = os.listdir(inp_path)
    done = pd.read_csv('/media/vidooly/myfiles2/ml/projects/indian_channel_scanning/9sept/kiss_output/kiss_vikas_2.txt',header=None)
    print(done.isnull().sum())
    done.dropna(inplace=True)
    chan_vid = done[0]+','+done[1]
    print(chan_vid.head(1).values)
    vidid = set(chan_vid.apply(dump))
    print(len(lines),len(vidid))
    lines = list(set(lines)-vidid)
    print(len(lines))
    imgs=[]
    files_b=[]
    j = 0
    ind=0
    strt=time.time()
    j = 0
    net = load_model('/home/vidooly/ml/projects/user/vikas/kiss_classifier/model/best_path_merge_foc2.hdf5',custom_objects=custom_object)
    for f in lines:
        file = inp_path+f
        try:
            img = image.load_img(file, target_size=(224,224))
            if img is None:
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            imgs.append(x)
            ind+=1
            files_b.append(file)
            if ind==400:
                imgs1=np.array(imgs)
                imgs=[]
                # print(files_b,len(files_b))
                preds=net.predict_on_batch(imgs1)
                ind=0
                t2=time.time()
                print(t2-strt,j)
                with open(out_dir+'kiss_vikas_2.txt','a') as fb:
                    for i,k in zip(files_b,preds):
                        fb.write("{},{}\n".format(i,str(k[1])))
                print(time.time()-t2,'Time in writing files')
                files_b=[]
                # exit()
            # x = np.expand_dims(x, axis=0)
            # pred_ = net.predict(x)
            # pred = net.predict(x)[0]
            # adult_score = round(float(pred[1]), 3)
            # print(math.floor(adult_score*10))
            # exit()
            # print("x: {} ; pred_: {} ; adult_score: {}".format(x, pred_, adult_score))
            # exit(0)
            # adult_score = 1 - adult_score
            # top_inds = pred.argsort()[::-1][:5]

            # prev_i = 0.
            # print("qwewqe: ", str(int(adult_score*10)))
            # out_fn=str(adult_score)+','+file
            # +str(math.floor(adult_score*10)+'/'
            # shutil.copy(f, out_dir+str(math.floor(adult_score*10))+'/'+out_fn)
        except Exception as e:
            print(e)
            pass

        # for i in np.linspace(0, 1, 11)[1:]:
        #     if adult_score >= prev_i and adult_score < i:
        #         shutil.copy(f, "{}/{}/{}_ad,{}".format(out_dir, int(prev_i*10), adult_score, file))
        #     prev_i = i

        j += 1
        # print("Complete: ", j)
        if j%100 == 0:
            print("Complete: {}\r".format(j), end="")
    if ind  <400:
        imgs1=np.array(imgs)
        imgs=[]
        # print(files_b,len(files_b))
        preds=net.predict_on_batch(imgs1)
        ind=0
        t2=time.time()
        print(t2-strt,j)
        with open(out_dir+'kiss_vikas.txt','a') as fb:
            for i,k in zip(files_b,preds):
                fb.write("{},{}\n".format(i,str(k[1])))
            print(time.time()-t2,'Time in writing files')
            files_b=[]

    print("Complete: ", j)

            # for i in top_inds:
            #     print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
