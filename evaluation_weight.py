#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:14:09 2020

@author: jiaqiwang0301@win.tu-berlin.de
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#from models import tiramisu_focal as tiramisu
from models import tiramisu as tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training_crack as train_utils
import pandas as pd
import argparse
import json

import os
pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# !!! check if there is normalization or not !!!
# !!! check in_channels 
"""
python3 weights_evaluate2.py --path_folder .weights/alpha015_save_weights \
--start_epoch 94 \
--end_epoch 96 \
--step 2 \
--predict_list SegNet-Tutorial/CamVid/CFD/list/shallow_crack_no_aug_val.txt \
--filename val.csv
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--n_class", type=int, default=2, help="the number of the class")
#    parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
#    parser.add_argument("--weights_name", type=str, default='weights-100.pth', help="path to save the weights")
    parser.add_argument("--path_folder", type=str, default='.weights/MDT_rutting/test', help="path to dataset")
    parser.add_argument("--gamma", type=float, default=2.0, help="gamma value for focal loss")
#    parser.add_argument("--train_list", type=str, default='SegNet-Tutorial/CamVid/CFD_/_train.txt', help="path to save the weights")
    parser.add_argument("--predict_list", type=str, default='SegNet-Tutorial/CamVid/MDT_rutting/file_list/MDT_no_aug_predict.txt', 
                        help="path to dataset")
    parser.add_argument("--start_epoch", type=int, default=60, help="start epoch")
    parser.add_argument("--end_epoch", type=int, default=181, help="end epoch")
    parser.add_argument("--step", type=int, default=2, help="epoch step")
    parser.add_argument("--filename", type=str, default='predict_set.csv', help="csv name")
    parser.add_argument("--tolerance", type=int, default=5, help="tolerance margin")
    
    
    opt = parser.parse_args()
    print(opt)

    n_classes = opt.n_class
    batch_size = opt.batch_size
    

    gamma  = opt.gamma

    #weights_name = opt.weights_name
    WEIGHTS_PATH =  opt.path_folder
    CAMVID_PATH = Path(opt.path_folder)
    
    mean = [0.50898083, 0.52446532, 0.54404199]   
    std = [0.08326811, 0.07471673, 0.07621879] 
#    mean = [0.50932450,0.50932450,0.50932450]   
#    std = [0.147114998,0.147114998,0.147114998] 
#    mean = [0.50932450]   
#    std = [0.147114998] 
    normalize = transforms.Normalize(mean=mean, std=std)  
    val_joint_transformer = transforms.Compose([
#        joint_transforms.JointRandomCrop(640), # commented for fine-tuning
#        joint_transforms.JointRandomHorizontalFlip()
        ])
    
    val_dset = camvid.CamVid2(
        CAMVID_PATH, path=opt.predict_list, 
        joint_transform=val_joint_transformer,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
        
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=batch_size, shuffle=True)
        
    model = tiramisu.FCDenseNet67(n_classes=n_classes,in_channels=3).cuda()
    
    metric = []
    
    #criterion = train_utils.FocalLoss(gamma)
    criterion = nn.NLLLoss().cuda()
    

    for epoch in range(opt.start_epoch,opt.end_epoch,opt.step):
        weights_name = f"weights-{epoch}.pth"
        train_utils.load_weights(model, os.path.join(WEIGHTS_PATH,weights_name))
        val_loss, result2, result5 = train_utils.test1(model, 
                                             val_loader, 
                                             criterion, 
                                             cls_num=n_classes,
                                             tolerance=opt.tolerance) 
    
        result = result5
        metric.append([epoch,result[4][1],result[5][1],result[6][1]])
        print(epoch,np.round(np.array([result[4][1],result[5][1],result[6][1]]),5))
    # precisioin recall f1
#        print(os.path.join(WEIGHTS_PATH,weights_name))
       
#        break
             
# for continuous epoch:   
      
    metric = np.round(np.array(metric),5)
    file_name = opt.filename
    dict_loss = {'epoch':metric[:,0],
                'precision':metric[:,1],
                 'recall':metric[:,2],
                 'f1':metric[:,3]}
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(WEIGHTS_PATH, file_name))  
    
    
    
    
    
'''
record:
.weights/CFD/whole_image_norm/with_aug/with_norm/focal_train/gamma20/test2_save_weights/weights-100.pth
[0.97633 0.80983 0.88532]

'''
