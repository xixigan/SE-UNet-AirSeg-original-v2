import os
import numpy as np
import time
import SimpleITK as sitk
import warnings
from glob import glob
import json
import random



inputpath = r'F:\datasets\ATM22\TrainBatch1\after\mask'

flist0=os.listdir(inputpath)
flist0.sort()
random.seed(0)
flist=flist0
###################################### split data
random.shuffle(flist)
train_list=[]
val_list=[]
test_list=[]

trainnum=140
valnum=10
testnum=0
for f in flist[0:trainnum]:
    train_list.append(f.split('mask')[0]+'.mha')

for f in flist[trainnum:trainnum+valnum]:
    val_list.append(f.split('mask')[0]+'.mha')

for f in flist[trainnum+valnum:]:
    test_list.append(f.split('mask')[0]+'.mha')



###############################
article_info = {}
data = json.loads(json.dumps(article_info))
f0= {'train': train_list, 
     'val': val_list
     }
data['0'] = f0
with open('./data/base_dict.json', 'w') as f:
    json.dump(data, f,indent=1)


###############################
article_info = {}
data = json.loads(json.dumps(article_info))
data['test'] = test_list
with open('./data/test.json', 'w') as f:
    json.dump(data, f,indent=1)
