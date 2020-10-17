# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:06:31 2019

@author: wwech
"""

import requests
import os
import json
import pandas as pd
import shutil
import requests
import pandas as pd
import threading
import numpy as np
from urllib.request import urlopen, urlretrieve, Request


base_api_url = 'https://www.wikiart.org/en/paintings-by-style/impressionism?select=featured&json=2&layout=new&resultType=masonry&page='

responses = dict()

THREAD_COUNTER = 0
THREAD_MAX     = 5


def requesthandle( link, name ):
    r = requests.get( link, stream=True )
    if r.status_code == 200:
        r.raw.decode_content = True
        f = open('.\\data\\impress\\'+str(name)+'.jpg', "wb" )
        shutil.copyfileobj(r.raw, f)
        f.close()
        print("[*] Downloaded Image: %s" % name)

df = pd.DataFrame(columns = ['image_id','image_url'])

for i in range(1,60):
    url = base_api_url + str(i)
    try:
        r= requests.get(url)
    except Exception as e:
        print(e)
    print(r.status_code==200)
    #responses['result'+str(i)] = r.json()['results']
    result = r.json()['Paintings']
    for li in range(0,len(result)):
        #df = df.append({'image_id':result[li]['id'],'image_url':result[li]['urls']['small'],
        #                'image_desc':result[li]['description'],'img_alt_desc':result[li]['alt_description']}, ignore_index=True)
        df = df.append({'image_id':result[li]['id'],'image_url':result[li]['image']}, ignore_index=True)
        
df.to_csv('wikiart.csv')
df = pd.read_csv('wikiart.csv')
## download images
for name,link in zip(df['image_id'],df['image_url']):
    _t = threading.Thread( target=requesthandle, args=(link, name ))
    _t.daemon = True
    _t.start()
    while THREAD_COUNTER >= THREAD_MAX:
        pass
                
while THREAD_COUNTER > 0:
        pass
    
n_threads = 4
start_idx = 0
stop_idx = 3601

directory = "data/"

def get_images(url_id, df):
    try:
        url = df['image_url'].iloc[url_id]
        urlretrieve(url, directory + df['image_id'].iloc[url_id] + '.jpg')
    except Exception as e:
        print(e)
        print(url_id)
        
for i in range(start_idx, stop_idx, n_threads):
    try:
        t1 = threading.Thread(target = get_images, args = (i,df,))
        t2 = threading.Thread(target = get_images, args = (i+1,df,))
        t3 = threading.Thread(target = get_images, args = (i+2,df,))
        t4 = threading.Thread(target = get_images, args = (i+3,df,))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
    except Exception as e:
        print(e)


for index,a in enumerate(df['image_url']):
    r_ = requests.get(a, stream=True )

    if r_.status_code == 200:
        r_.raw.decode_content = True
        f = open("./data/image_"+str(index)+".jpg", "wb" )
        shutil.copyfileobj(r_.raw, f)
        f.close()
        
        
    
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"knife","limit":500}   #creating list of arguments

"""
rest of the categories

"""



categories = ['Beer','Whiskey','Scotch','Liquor']  ## pick up brand names for more accurate results.

beer_api_url = 'https://unsplash.com/napi/search/photos?query=beer&xp=&per_page=30&page='

beer_df = pd.DataFrame(columns = ['image_id','image_url','image_desc','img_alt_desc'])

for i in range(0,350):
    url = beer_api_url + str(i)
    try:
        r= requests.get(url)
        if r.json().get('error'):
            print('Json Id Errors')
        print(r.status_code==200)
        #responses['result'+str(i)] = r.json()['results']
        result = r.json()['results']
        for li in range(0,len(result)):
            beer_df = beer_df.append({'image_id':result[li]['id'],'image_url':result[li]['urls']['small'],
                            'image_desc':result[li]['description'],'img_alt_desc':result[li]['alt_description']}, ignore_index=True)
    except Exception as e:
        print(e)

beer_df.to_csv('unsplash_beer_vidIds.csv')
beer_vid_ids = list(set(beer_df['image_id']))

imag_url_id = []
for i in df['image_url']:
    imag_url_id.append(i[34:].split('?')[0])
    
len(list(set(imag_url_id)))
te = beer_df['image_id'][:1500]
te = list(te)
a_te = df['image_id'][:10000]
a_te = list(a_te)

c_te= list(set(te) - set(a_te))
new_df = pd.DataFrame(columns = ['image_id','image_url','image_desc','img_alt_desc'])

for ac in c_te:
    new_df = pd.concat([new_df,beer_df[beer_df['image_id']==ac]],ignore_index = True)
    
new_df.to_csv('unique_beer.csv')

### Clairfai API

from clarifai.rest import ClarifaiApp
import pandas as pd
app = ClarifaiApp(api_key='81a00109c0b841d39853f1e2bddc64f5')

model = app.models.get("moderation")
df = pd.DataFrame(columns = ['image_name','value','name'])
# predict with the model

for img in os.listdir('./data/Drugs'):
    out = model.predict_by_filename('./data/Drugs/'+img)
    concepts = out['outputs'][0]['data']['concepts'][0]
    print(img,concepts['value'],concepts['name'])
    #df = df.append({'image_name':img,'value':concepts['value'],'name':concepts['name']},ignore_index = True)
    os.rename('./data/Drugs/'+img , './data/Drugs_dest/'+str(img.split('.')[0])+','+concepts['name']+','+str(concepts['value'])+'.jpg')        