import copy
import itertools
import os
import sys
import time
from collections import defaultdict
import json
import numpy as np
import pycocotools.mask as maskUtils
import pickle
from IPython import embed
from PIL import Image


annotation_file = '/public/bme/home/liuyx7/data/chex_data/train.json'


save_dataset = {'images':{},'annotations':{}}
print('loading annotations into memory...')
tic = time.time()
with open(annotation_file, 'r') as f:
    dataset = json.load(f)[:20]
print('Done (t={:0.2f}s)'.format(time.time()- tic))
for i in range(len(dataset)):
    txt_path=dataset[i]['txt_path']
    img_dir='/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'+(txt_path[6:]).replace('.json', '/')
    files = os.listdir(img_dir)
    files.remove('index.html')
    for file_index in range(len(files)):
        files[file_index] = img_dir + files[file_index]

    save_dataset['images'][i]=({'file_name':files[0], 'annotation_id':i, 'id':i})
    report = dataset[i]['report']
    report = report.replace('Findings:\n', '')
    report = report.replace('Impression:\n', '')
    save_dataset['annotations'][i]=({'id':i, 'annotations': report})
    if (i%100==0):
        print(i)
    
f_save = open('/public/bme/home/liuyx7/data/chex_data/mimic_train_20.pkl', 'wb')
pickle.dump(save_dataset, f_save)
f_save.close()

