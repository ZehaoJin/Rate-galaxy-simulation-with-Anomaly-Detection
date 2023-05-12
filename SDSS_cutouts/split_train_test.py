import os
import numpy as np
from PIL import Image
import random
import shutil
from tqdm import tqdm

test_ratio = 0.10
Foldername='SDSS'
try:
    os.makedirs(Foldername+'/train/0.normal')
    os.makedirs(Foldername+'/test/0.normal')
    os.makedirs(Foldername+'/test/1.abnormal')
except:
    print("Dirctory already there")


## deal with CALIFA images
FileNames = os.listdir('../../cutouts/')
np.random.shuffle(FileNames)
num=len(FileNames)

print('total number of SDSS images: ',num)
print('test ratio: ',test_ratio)


for i,name in tqdm(enumerate(FileNames),total=len(FileNames)):
    if i<(num*test_ratio):
        shutil.copy('../../cutouts/'+name, Foldername+'/test/0.normal/')
    if i>=(num*test_ratio):
        shutil.copy('../../cutouts/'+name, Foldername+'/train/0.normal/')
