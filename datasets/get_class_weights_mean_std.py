#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:36:44 2019

@author: jiaqiwang0301@win.tu-berlin.de
calculate the class_weight (use median frequency balance) over training set
calculate the mean and std over training set
"""


import os 
from PIL import Image
import numpy as np

base_folder = 'SegNet-Tutorial/CamVid/crop/'
#%%
# calculate class weight
#count_class = np.zeros([2]).astype(int)
#annot_folder_path = os.path.join(base_folder,'crop/' )
#file_list = [filename for filename in os.listdir(annot_folder_path)]
#num_file = len(file_list)

#i = 0
#for label_file in file_list:
#    path = os.path.join(annot_folder_path, label_file)
#    label = np.array(Image.open(path).convert('L')).astype(int)
#    class_name = np.unique(label)
#    for item in class_name:
#        num = len(np.where(label == item)[0])
#        count_class[item] += num
#    i += 1    
#    print(f"{i} / {num_file} finished ")
#    
#    
##count_class[-1] = 0
#count_class_scaled = count_class / count_class.sum()
#count_class_median = np.median(count_class_scaled)
#count_class_scaled2 = count_class_median / count_class_scaled
#print(count_class_scaled2)

# pixel count
#array([2870928009,    8028951,    1148421,    4830311,    4897725,
#         50531383])
# current
#[2.25130619e-03, 8.05004041e-01, 5.62802143e+00, 1.33807906e+00,
#       1.31966127e+00, 1.27907404e-01])

#%%
# class mean and std
imag_folder_path = os.path.join(base_folder,'train/' )
file_list = [filename for filename in os.listdir(imag_folder_path)
               if filename.endswith('png')]
num_file = len(file_list)


big_dataset = []
i = 0
for label_file in file_list:
    path = os.path.join(imag_folder_path, label_file)
    imag = np.array(Image.open(path).convert('L')).astype(int).flatten()
    big_dataset.append(imag/255)
    i += 1
    print(f"{i} / {num_file} finished ")
    
big_dataset = np.array(big_dataset)
mean = big_dataset.mean()
std = np.std(big_dataset)
print(f"mean: {mean}  std: {std} ")

# 
    





