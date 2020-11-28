#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:59:06 2019

@author: jiaqiwang0301@win.tu-berlin.de
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training_crack as train_utils
import pandas as pd
import cv2
import argparse
import json
from tqdm import tqdm
from PIL import Image


import os
pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
green is miss detection
blue is pixel predicted to be crack, but the ground truth is not
'''

'''
an example in command line
python3 predict_argparse.py --CAMVID_PATH 'SegNet-Tutorial/CamVid/resized7' \
  --weights_file_folder '.weights/resized7_no_weight_random_crop/' \
  --weight_filename 'weights-200-0.218-0.054.pth' \
  --result_folder 'output/5_without_class_weight_resized7_delete_later/'  
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CAMVID_PATH", type=str,default='SegNet-Tutorial',help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--weights_file_folder", type=str, default=".weights/CFD/patch_training/patch8080/pn13_t10/", help="path to weight")
    parser.add_argument("--weight_filename", type=str, default='weights-100.pth', help="name of the weight file")
    parser.add_argument("--n_classes", type=int, default=2, help="number of the class")
    parser.add_argument("--result_folder", type=str,default='output/patch8080_pn13_t10/test', help="path to save the results")
#    parser.add_argument("--save_image", type=int,default=1, help="if save image or not,1 mans save,0 means no")
    parser.add_argument("--test_list", type=str, default='SegNet-Tutorial/CamVid/CFD/list/add_aug_test.txt', help="path to dataset")
    opt = parser.parse_args()
    print(opt)
        
    CAMVID_PATH = os.path.join(opt.CAMVID_PATH)
    #WEIGHTS_PATH = os.path.join('.weights/with_class_weight_and_randomcrop224/')
    batch_size = opt.batch_size
    weights_file_folder = os.path.join(opt.weights_file_folder)
    weight_filename = opt.weight_filename
    n_classes = opt.n_classes
    result_folder = os.path.join(opt.result_folder,weight_filename)
#    save_image = opt.save_image

    image_name = 'test_{:04d}.png'

    fpath = os.path.join(weights_file_folder,weight_filename)
    if os.path.isfile(weight_filename):  
        raise Exception('file does not exist')

    
    os.makedirs(result_folder,exist_ok=True)
    '''
        predict_folder = os.path.join(result_folder, 'predict')
        FN_folder = os.path.join(result_folder, 'FN')
        TP_FP_FN_folder = os.path.join(result_folder, 'TP_FP_FN')
        os.makedirs(predict_folder,exist_ok=True)
        os.makedirs(FN_folder,exist_ok=True)
        os.makedirs(TP_FP_FN_folder,exist_ok=True)
    '''
    
    txt_file_list = [filename for filename in os.listdir(result_folder)
                      if filename.split('.')[-1] == 'txt' ]
    if len(txt_file_list) > 0:
        for file in txt_file_list:
            os.remove(os.path.join(result_folder,file))
                  
    with open(os.path.join(result_folder,'basic information.txt'),'w') as f:
        json.dump(opt.__dict__,f,indent=2)
        
    mean = [0.50898083, 0.52446532, 0.54404199]   
    std = [0.08326811, 0.07471673, 0.07621879] 
    normalize = transforms.Normalize(mean=mean, std=std)
    
    test_dset = camvid.CamVid2(
        CAMVID_PATH, path=opt.test_list, 
        joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
#            normalize
        ]))        
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=opt.batch_size, shuffle=False)
    
    
    model = tiramisu.FCDenseNet67(n_classes=n_classes,in_channels=3).cuda()
    inputs, _,_ = next(iter(test_loader))
    print('evaluation image size: ', inputs.size())
    
    #model.apply(train_utils.weights_init)
   
    train_utils.load_weights(model, fpath)
    
    c_matrix = np.zeros([n_classes,n_classes])
    result_each_image = []
    
#    for batch_i in tqdm(range(len(test_loader))):
    cmatrix_each = []
#    for batch_i, (inputs, targets, index) in tqdm(enumerate(test_loader)): 
    prev_time = time.time()   
    for batch_i, (inputs, targets, index) in enumerate(test_loader):         
        
        #import pdb; pdb.set_trace()
        #index = index.numpy()
        #pred_result = train_utils.view_sample_predictions_new(model, imgs, targets, n=4)
        data = Variable(inputs.cuda(), volatile=False)
        #label = Variable(targets.cuda())
        with torch.no_grad():
            output = model(data)
            
        pred = train_utils.get_predictions(output)
        batch_size = inputs.size(0)
    
    
    
        for j in range(batch_size):
            #img_utils.view_image(inputs[i])
#            if n_classes == 2:
#                pred[j] = 5 * pred[j]  
            current_time = time.time()          
            evaluation_results = train_utils.evaluate(targets[j].unsqueeze(0),
                                                      pred[j].unsqueeze(0))
            c_matrix0 = train_utils.calculate_confusion_matrix(targets[j].unsqueeze(0),
                                                               pred[j].unsqueeze(0),
                                                               cls_num=n_classes)

            #current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            c_matrix += c_matrix0
            path = index[j]
            c_matrix0 = list(c_matrix0.reshape(-1))
            c_matrix0.insert(0,path.split('/')[-1].split('.')[0])
            cmatrix_each.append(c_matrix0)
            
            image_original = np.array(Image.open(path).convert('RGB'))
            
            
            name = '_' + str(100 * round(evaluation_results[0][1],2)) + \
                   '_' + str(100 * round(evaluation_results[1][1],2))
            img_name = path.split('/')[-1].split('.')[0] + name

            target = targets[j].cpu().numpy()
            prediction = pred[j].cpu().numpy()

			# save prediction images
            img = np.ones_like(image_original)*255
            img = train_utils.draw_color(img,prediction,[255,0,255])
            image_merged = cv2.addWeighted(image_original/255,0.7,img/255,0.3,0,dtype = cv2.CV_32F)
            img_combine = np.concatenate((image_original/255,image_merged), axis=0)
            plt.imsave(os.path.join(result_folder,img_name+'_a_predict.jpg'),img_combine)

			# generate TP,FP,FN images           
            TP, FP, FN = train_utils.generation_TP_FP_FN(target,prediction,t = 5)
            img = np.ones_like(image_original)*255
            img = train_utils.draw_color(img,FN,[0,255,0])

            # save FN images
            image_merged = cv2.addWeighted(image_original/255,0.7,img/255,0.3,0,dtype = cv2.CV_32F)
            img_combine = np.concatenate((image_original/255,image_merged), axis=0)
            plt.imsave(os.path.join(result_folder,img_name+'_c_FN.jpg'),img_combine)

            img = train_utils.draw_color(img,TP,[255,0,0])            
            img = train_utils.draw_color(img,FP,[0,0,255])
            
            # save TP_FP_FN images
            image_merged = cv2.addWeighted(image_original/255,0.7,img/255,0.3,0,dtype = cv2.CV_32F)
            img_combine = np.concatenate((image_original/255,image_merged), axis=0)
            plt.imsave(os.path.join(result_folder,img_name+'_b_TP_FP_FN.jpg'),img_combine)
 
            
            imag_targets = utils.imgs.view_annotated(targets[j],'targets', plot=False, class_num=6)
            imag_pred = utils.imgs.view_annotated(pred[j],'prediction', plot=False, class_num=6)
            img_combine2 = np.concatenate((image_original/255,imag_targets,imag_pred), axis=0)
            #plt.imsave(os.path.join(result_folder,img_name.replace('merge0','merge1')),img_combine2)
            print(path.split('/')[-1].split('.')[0],' finished')
            

    
    
            results = [evaluation_results[0][1],evaluation_results[1][1],evaluation_results[2][1]]
            results = [round(r,3) for r in results]
            results.insert(0,path.split('/')[-1])
            result_each_image.append(results)

    result_each_image = np.array(result_each_image)
    data = pd.DataFrame({
            'index':result_each_image[:,0],
            'precision':result_each_image[:,1],
            'recall':result_each_image[:,2],
            'f1':result_each_image[:,3],
        
            })
    
    data.to_csv(os.path.join(result_folder,'results.csv'))  

    cmatrix_each = np.array(cmatrix_each)
    data = pd.DataFrame({
            'index':cmatrix_each[:,0],
            'FN':cmatrix_each[:,3],
            'TP':cmatrix_each[:,4],
            'FP':cmatrix_each[:,2],
            'precision':result_each_image[:,1],
            'recall':result_each_image[:,2],
            'f1':result_each_image[:,3],
                    
            })
    
    data.to_csv(os.path.join(result_folder,'results_pixel_number.csv'))  
    
    with np.errstate(divide='ignore',invalid='ignore'):
        precision = np.diag(c_matrix)/c_matrix.sum(axis=0)
        
    with np.errstate(divide='ignore',invalid='ignore'):
        recall = np.diag(c_matrix)/c_matrix.sum(axis=1)
    print(precision[1],recall[1])
    
    name = opt.weight_filename +'_'+ str(int(100 * round(precision[1],2))) + \
           '_' + str(int(100 * round(recall[1],2))) +'.txt'
           
    f = open(os.path.join(opt.result_folder, name),'w')      
    f.close()


                #print(img_name, ' finished')            
#            
#            imag_targets = utils.imgs.view_annotated(targets[j],'targets', plot=False, class_num=6)
#            imag_targets[-4:-1,:,:] = 1
#
#            imag_pred = utils.imgs.view_annotated(pred[j],'prediction', plot=False, class_num=6)
#            import pdb; pdb.set_trace()
#             
#
#            if bool(save_image):
#                
#                #img_name = image_name.format(index[j])
#                image_original = np.array(Image.open(path).convert('RGB'))
#  
#                img_name = path.split('/')[-1].split('.')[0] + '_' + str(int(1000*evaluation_results[-1])) + '.png'
#                img_combine = np.concatenate((image_original/255,imag_targets,imag_pred), axis=0)
#                plt.imsave(os.path.join(result_folder,img_name),img_combine)
#                #print(img_name, ' finished')
        
        
##        if batch_i == 5:
##            break
#     
#    result_each_image = np.array(result_each_image)
#    data = pd.DataFrame({
#            'index':result_each_image[:,0],
#            'PA':result_each_image[:,1],
#            'class_PA':result_each_image[:,2],
#            'mIoU':result_each_image[:,3],
#            'w_MIoU':result_each_image[:,4]        
#            })
#    data.to_csv(os.path.join(result_folder,'results.csv'))
#    result = np.round(train_utils.evaluate_whole_dataset(c_matrix),5)
#    total_iou = train_utils.IoU(c_matrix)
#    c_matrix = np.round(c_matrix / c_matrix.sum(),5)
#    #PA, class_PA, mIoU, w_MIoU]
#    final_result = f"evaluation results of the whole test folder: \n \
#            pixel accuracy: {result[0]} \n \
#            mean class pixel accuracy: {result[1]} \n \
#            mIoU: {result[2]} \n \
#            weighted mIoU: {result[3]} \n \
#            IoU of each class:{total_iou}"
#    print(final_result)
#    
#    f = open(os.path.join(result_folder,'result of all test images.txt'),'w')
#    f.write(final_result)
#    f.close()
#    
#    np.savetxt(os.path.join(result_folder,'c_matrix.txt'),c_matrix, fmt='%s',newline='\n',delimiter=' ')


