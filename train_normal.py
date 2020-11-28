#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 07:51:35 2019

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

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
import pandas as pd
import argparse
import json

import os
pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_class", type=int, default=2, help="the number of the class")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--starts_epochs", type=int, default=0, help="if there is pretrained weights")
    parser.add_argument("--weights_folder", type=str, default='.weights/', help="path to save the weights")
    parser.add_argument("--path_folder", type=str, default='SegNet-Tutorial/CamVid/CFD/', help="path to dataset")
    parser.add_argument("--train_list", type=str, default='SegNet-Tutorial/CamVid/CFD_generation/_train.txt', help="path to save the weights")
    parser.add_argument("--valid_list", type=str, default='SegNet-Tutorial/CamVid/CFD_generation/_val.txt', help="path to dataset") 
    opt = parser.parse_args()
    print(opt)


    #CAMVID_PATH = Path('/bigguy/data', 'SegNet-Tutorial/CamVid')
    CAMVID_PATH = Path(opt.path_folder)
    WEIGHTS_PATH = os.path.join(opt.weights_folder)
    os.makedirs(WEIGHTS_PATH,exist_ok=True)
    # write the parameter setting into txt files and save it in the same folder with the weights
    with open(os.path.join(opt.weights_folder,'A_note.txt'),'w') as f:
        json.dump(opt.__dict__,f)

    
    weight_filename = opt.pretrained_weights
    #weight_filename = 'weights-100-0.233-0.056.pth'  # if no pretrained weight, then
    #class_weight = torch.FloatTensor([1, 0.8, 0.8, 0.8, 0.8, 1.1])
    class_weight = torch.FloatTensor([1, 1])
        
    n_classes = opt.n_class
    batch_size = opt.batch_size
    evaluation_interval = opt.evaluation_interval
    checkpoint_interval = opt.checkpoint_interval
    train_list=opt.train_list
    valid_list=opt.valid_list    
    N_EPOCHS = opt.epochs
    file_name_train = f"train_loss.csv"
    file_name_val = f"valid_loss.csv"
    

    random_crop_size = 512
    
    criterion = nn.NLLLoss(class_weight.cuda()).cuda()
    #criterion = nn.CrossEntropyLoss(class_weight.cuda()).cuda()
    
    
    print('data source: ', CAMVID_PATH)
    print('weights saved path: ', WEIGHTS_PATH)
    #print('\n there is random crop with size:', random_crop_size)
    
    
    normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
    train_joint_transformer = transforms.Compose([
    #    joint_transforms.JointRandomCrop(random_crop_size), # commented for fine-tuning
        joint_transforms.JointRandomHorizontalFlip()
        ])
    
    train_dset = camvid.CamVid2(CAMVID_PATH, path=train_list,
    #      joint_transform=train_joint_transformer,
          transform=transforms.Compose([
              transforms.ToTensor(),
              normalize          
        ]))    
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True)
    
    
    val_joint_transformer = transforms.Compose([
        #joint_transforms.JointRandomCrop(800), # commented for fine-tuning    
        ])
    val_dset = camvid.CamVid2(
        CAMVID_PATH, path=valid_list, 
        joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))        
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=1, shuffle=True)
    
    
    print("Train: %d" %len(train_loader.dataset.imgs),"Val: %d" %len(val_loader.dataset.imgs))
    print("Classes: %d" % len(train_loader.dataset.classes))
    
    
    inputs, _,_ = next(iter(train_loader))
    print('training image size: ', inputs.size())
    inputs, _,_ = next(iter(val_loader))
    print('evaluation image size: ', inputs.size())
    #print("Inputs: ", inputs.size())
    #print("Targets: ", targets.size())
    #
    #utils.imgs.view_image(inputs[0])
    #utils.imgs.view_annotated(targets[0])
    
    
    #%%
    LR = 1e-4
    LR_DECAY = 0.995
    DECAY_EVERY_N_EPOCHS = 1
    
    torch.cuda.manual_seed(0)
    
    model = tiramisu.FCDenseNet67(n_classes=n_classes).cuda()
    
    if weight_filename is not None:
        train_utils.load_weights(model, os.path.join(WEIGHTS_PATH,weight_filename))
        print('pretrained weights loaded')
        start_epoch = int(weight_filename.split('-')[1]) + 1
    else:
        model.apply(train_utils.weights_init)
        print('train from beginning')
        start_epoch = 0
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    
    
    
    #criterion = train_utils.FocalLoss2d(gamma)
    # if FocalLoss2d: delete the last layer of the tiramisu
    # print('focal loss with gamma = ',gamma)
    
    total_loss = []
    valid_map = []
    
    
    for epoch in range(start_epoch, start_epoch+N_EPOCHS+1):
        print('epoch:',epoch,'-----------------------------')
        print('train:-----------')
        since = time.time()
        ### Train ###
        trn_loss, trn_err = train_utils.train(
            model, train_loader, optimizer, criterion, epoch)
        
        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
            epoch, trn_loss, 1-trn_err))    
        total_loss.append([epoch,trn_loss.item()]) 
        
        time_elapsed = time.time() - since  
        print('Train Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
    
        
        ### Test ###
        if epoch % evaluation_interval == 0:
            print('evaluation:-----------')
            val_loss, result = train_utils.test1(model, 
                                                 val_loader, 
                                                 criterion, 
                                                 epoch,
                                                 cls_num=n_classes) 
            print(f"evaluation results: \n \
            val - loss: {val_loss }    \n \
            pixel accuracy: {result[0]} \n \
            mean class pixel accuracy: {result[1]} \n \
            mIoU: {result[2]} \n \
            weighted mIoU: {result[3]} \n ")
            
            print(f"evaluation results: \n \
            precision: {result[4]}    \n \
            recall: {result[5]}    \n \
            f1: {result[6]}    \n \  ")
    
    
            valid_map.append([epoch,val_loss,result[4][1],result[5][1],result[6][1]])
    
            time_elapsed = time.time() - since  
            print('Total Time {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
                   
        if epoch % checkpoint_interval == 0:
            ### Checkpoint ###    
            train_utils.save_weights(model, epoch, val_loss, result[3],weights_fpath=WEIGHTS_PATH)
        
            ### Adjust Lr ###
            train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                             epoch, DECAY_EVERY_N_EPOCHS)
        
    
    total_loss = np.round(np.array(total_loss),6)
    valid_map = np.round(np.array(valid_map),5)
    
    total_loss = np.array(total_loss)
    valid_map = np.array(valid_map)
    
    dict_loss = {'epoch':total_loss[:,0],'total_loss':total_loss[:,1]}
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(WEIGHTS_PATH,file_name_train),index=False)
    
    dict_loss = pd.DataFrame({
            'epoch':valid_map[:,0],
            'var loss':valid_map[:,1],
            'precision':valid_map[:,2],
            'recall':valid_map[:,3,],
            'f1':valid_map[:,4],
         
            })
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(WEIGHTS_PATH,file_name_val),index=False)  
 
#dict_loss = pd.DataFrame({
#        'epoch':valid_map[:,0],
#        'var loss':valid_map[:,1],
#        'PA':valid_map[:,2],
#        'class_PA':valid_map[:,3],
#        'mIoU':valid_map[:,4],
#        'w_MIoU':valid_map[:,5] 
#     
#        })
#    
#df = pd.DataFrame(dict_loss)
#df.to_csv(os.path.join(WEIGHTS_PATH,file_name_val),index=False)
