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
import math
import sys
import dill

sys.path.append('../')
#from focallosskeras.losses import *

import keras_metrics
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


if __name__ == '__main__':
    inp_path = '/home/vidooly/ml/projects/user/vikas/kiss_classifier/data/kiss_data_new_aug/'
    out_dir = '/home/vidooly/ml/projects/user/vikas/kiss_classifier/data/kiss_data_new_aug1/'
    cls_list = ['adult', 'normal']

    for i in range(11):
        os.makedirs(out_dir+'safe/'+str(i), exist_ok=True)
        os.makedirs(out_dir+'unsafe/'+str(i), exist_ok=True)
        custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss': categorical_focal_loss,
                 'binary_focal_loss': binary_focal_loss}

    # load the trained model
    net = load_model('/home/vidooly/ml/projects/user/vikas/kiss_classifier/model/best_path_merge_foc2.hdf5',custom_objects=custom_object)

    j = 0
    for root,dirs,files in os.walk(inp_path):
        for file in files:
            f=os.path.join(root,file)
            sub_fol=f.split('/')[-2]
            # print(f)
            # break
            try:
                img = image.load_img(f, target_size=(224,224))
                if img is None:
                    continue
                x = image.img_to_array(img)
                x = preprocess_input(x)
                x = np.expand_dims(x, axis=0)
                pred = net.predict(x)[0]
                # img,pred=get_image(f)
                adult_score = round(float(pred[1]), 3)
                out_fn=str(adult_score)+','+file
                # img.save('out_dir+sub_fol+'/'+str(math.floor(adult_score*10))+'/'+out_fn')
                shutil.copy(f, out_dir+sub_fol+'/'+str(math.floor(adult_score*10))+'/'+out_fn)
            except Exception as e:
                print(e)
                pass
            j += 1
        # print("Complete: ", j)
            if j%100 == 0:
                print("Complete: {}\r".format(j), end="")
    print("Complete: ", j)