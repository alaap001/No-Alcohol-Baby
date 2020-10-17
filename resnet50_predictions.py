# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:32:31 2019

@author: wwech
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import math
import shutil

model = load_model('./Resnet_alcohol_filter.h5')
model.summary()


img_path = 'Final_data/train/unsafe/'
out_dir = 'model_output/'

for i in range(11):
    os.makedirs(out_dir+'safe/'+str(i), exist_ok=True)
    os.makedirs(out_dir+'unsafe/'+str(i), exist_ok=True)

#os.rename(img_path+i,img_path+i.split('.')[0]+', '+str(unsafe_score)+'.jpg')

j = 0
for root,dirs,files in os.walk(img_path):
    for file in files[10001:]:
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
            pred = model.predict(x)[0]
            # img,pred=get_image(f)
            unsafe_score = round(float(pred[0]), 3)
            out_fn=str(unsafe_score)+','+file
            # img.save('out_dir+sub_fol+'/'+str(math.floor(adult_score*10))+'/'+out_fn')
            shutil.copy(f, out_dir+sub_fol+'/'+str(math.floor(unsafe_score*10))+'/'+out_fn)
        except Exception as e:
            print(e)
            pass
        j += 1
    # print("Complete: ", j)
        if j%100 == 0:
            print("Complete: {}\r".format(j), end="")
print("Complete: ", j)

for i in os.listdir(out_dir+'/unsafe/10'):
    print(i.split(',')[1])
    os.rename(out_dir+'/unsafe/10/'+str(i), out_dir+'/unsafe/10/'+i.split(',')[1])
