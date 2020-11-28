#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:41:11 2020

@author: jiaqiwang0301@win.tu-berlin.de
"""

import numpy as np
import utils.training_crack as train_utils
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def calculate_confusion_matrix(targets,preds,cls_num,t=5):
    '''
    claculate confusion matrix for only two class!!!!
    !!! for batch..it also works..I think
    input:
        targets: tensor  ground truth
        preds: tensor  predicted value
        t: tolerance margin
    '''
    pre = preds
    gt = targets
#    targets = targets.data.cpu().numpy().flatten('C')
#    preds = preds.data.cpu().numpy().flatten('C')
    
    c_matrix = confusion_matrix(gt.flatten('C'), pre.flatten('C'),labels=np.arange(cls_num))
    b,w,h = gt.shape
    r = []
    for k in range(b):
        num = 0
        for i in range(w):
            for j in range(h):
                if  pre[k,i,j] == 1 :
                    c = gt[k,max(0,i-t):min(w,i+t+1),max(0,j-t):min(h,j+t+1)]
                    if c[c==1].sum() > 1:
                        num += 1
        r.append(num)
            
    c_matrix[0,1] = c_matrix[0,1] - (sum(r) - c_matrix[1,1])
    c_matrix[1,1] = sum(r)

    return c_matrix

targets = (np.load('output/ts.npy')[0,:,:])
preds = (np.load('output/pred.npy')[0,:,:])/5


t = 5 # tolerance margin
def generation_TP_FP_FN(targets,preds,t = 5):
    TP = np.zeros_like(targets)
    FP = np.zeros_like(targets)
    FN = np.zeros_like(targets)
    w,h = preds.shape
    for i in range(w):
        for j in range(h):
            if  preds[i,j] == 1 :
    
                c = targets[max(0,i-t):min(w,i+t+1),max(0,j-t):min(h,j+t+1)]
                if c[c==1].sum() > 1:
                    TP[i,j] = 1
                else:
                    FP[i,j] = 1
            else:
                if targets[i,j] == 1:
                    FN[i,j] = 1
    return TP, FP, FN

#cm = calculate_confusion_matrix(targets[None,:,:],preds[None,:,:],2,t=5)
#print(cm)

TP, FP, FN = generation_TP_FP_FN(targets,preds,t = 5)
print(TP.sum(),FP.sum(),FN.sum())
w, h = FN.shape
img = np.ones([w,h,3])*255
def draw_color(img,TP,color):
    ind = TP==1
    img1 = img[:,:,0]
    img2 = img[:,:,1]
    img3 = img[:,:,2]
    img1[ind] = color[0]
    img2[ind] = color[1]
    img3[ind] = color[2]
    return np.stack((img1,img2,img3),axis=-1)

img = draw_color(img,TP,[255,0,0])
img = draw_color(img,FP,[0,255,0])
img = draw_color(img,FN,[0,0,255])

path = 'SegNet-Tutorial/CamVid/CFD/aug/aug2_180/images/112.jpg'
image_original = np.array(Image.open(path)).astype(np.float32)

w_img = 0.8
w_label = 0.2  
img = img.astype(np.float32)        
image_merged = cv2.addWeighted(image_original,w_img,img,w_label,0,dtype = cv2.CV_32F) 

plt.figure()
plt.imshow(img)  
    

    

                


