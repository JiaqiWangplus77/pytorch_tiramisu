import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import numpy as np

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
    
def save_weights(model, epoch, loss, err,weights_fpath= '.weights/'):
    #weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fname = 'weights-%d.pth' % (epoch)
    weights_fpath = os.path.join(weights_fpath, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)  
    # for every pixel, get the largest probablilty and its index
    indices = indices.view(bs,h,w)
    return indices

def precision(confusion_matrix):
    '''
    precision = TP/(TP+FP) (axis=0)
    '''
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.diag(confusion_matrix)/confusion_matrix.sum(axis=0)


def recall(confusion_matrix):
    '''
    precision = TP/(TP+FN) (axis=1)
    '''
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.diag(confusion_matrix)/confusion_matrix.sum(axis=1)

def F1(precision,recall):
    '''
    f1 = 2 * precision * recall / (precision + recall)
    '''
    with np.errstate(divide='ignore',invalid='ignore'):
        f1 = 2 * np.multiply(precision,recall) / (recall + precision) 
    return f1
    

def calculate_confusion_matrix(targets,preds,cls_num,t=5):
    '''
    claculate confusion matrix for only two class!!!!
    !!! for batch..it also works..I think
    input:
        targets: tensor  ground truth
        preds: tensor  predicted value
        t: tolerance margin
    '''
    pre = preds.cpu().numpy()
    gt = targets.cpu().numpy() 
#    targets = targets.data.cpu().numpy().flatten('C')
#    preds = preds.data.cpu().numpy().flatten('C')

    c_matrix = confusion_matrix(gt.flatten('C'), pre.flatten('C'),labels=np.arange(cls_num))
    b,w,h = gt.shape
    r = 0
    for k in range(b):
        r = 0
        for i in range(w):
            for j in range(h):
                if  pre[k,i,j] == 1 :
                    c = gt[k,max(0,i-t):min(w,i+t+1),max(0,j-t):min(h,j+t+1)]
                    if c[c==1].sum() > 1:
                        r += 1
            
        c_matrix[0,1] = c_matrix[0,1] - (r - c_matrix[1,1])
        c_matrix[1,1] = r

    return c_matrix

def calculate_confusion_matrix0(targets,preds,cls_num=6):
    '''
    claculate confusion matrix for each target and its respoinding prediction
    !!! for batch..it also works..I think
    input:
        targets: tensor
        preds: tensor
    '''
    targets = targets.data.cpu().numpy().flatten('C')
    preds = preds.data.cpu().numpy().flatten('C')
    
    c_matrix = confusion_matrix(targets, preds,labels=np.arange(cls_num))
    return c_matrix
    
def pixel_accuracy(confusion_matrix):
    '''
    calculate pixel accuracy based on confusion matrix
    '''    
    return np.diag(confusion_matrix).sum()/confusion_matrix.sum()

def weighted_pixel_accuracy_class(confusion_matrix):
    '''
    calculate class pixel accuracy based on confusion matrix
    
    ''' 
    # set the value to 1 to avoid dividing zero problem,
    # based on the characteristic of confusion value
    freq = confusion_matrix.sum(axis=1)/confusion_matrix.sum()
    num_each_class = confusion_matrix.sum(axis=1)
#    num_each_class[num_each_class==0] = 1
    with np.errstate(divide='ignore',invalid='ignore'):
        cls_PA = np.diag(confusion_matrix)/num_each_class
        w_cls_PA = np.multiply(freq[cls_PA>=0],cls_PA[cls_PA>=0]).sum()
    return w_cls_PA

def IoU(confusion_matrix):
    '''
    calculate the intersection over Union for each class
    '''
    intersection = np.diag(confusion_matrix) 
    union_part = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
#    # set the value to 1 to avoid dividing zero problem,
#    # based on the characteristic of confusion value
#    union_part[union_part==0] = 1
    with np.errstate(divide='ignore',invalid='ignore'):
        IoU = intersection / union_part
    return IoU

def MIoU(IoU):
    '''
    calculate the mean intersection over Union for each class
    '''
    return np.nanmean(IoU)

def weighted_MIoU(confusion_matrix):
    '''
    calculate the weighted mean intersection over Union for each class
    '''
    freq = confusion_matrix.sum(axis=1)/confusion_matrix.sum()
    
    intersection = np.diag(confusion_matrix) 
    union_part = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
#    # set the value to 1 to avoid dividing zero problem,
#    # based on the characteristic of confusion value
#    union_part[union_part==0] = 1 
    with np.errstate(divide='ignore',invalid='ignore'):
        IoU = intersection / union_part
        w_MIoU = np.multiply(freq[IoU>=0],IoU[IoU>=0]).sum()
    return w_MIoU

def evaluate_whole_dataset(confusion_matrix):
    PA = pixel_accuracy(confusion_matrix)
    class_PA = weighted_pixel_accuracy_class(confusion_matrix)
    iou = IoU(confusion_matrix)
    mIoU = MIoU(iou)
    w_MIoU = weighted_MIoU(confusion_matrix)
    pre = precision(confusion_matrix)
    rca = recall(confusion_matrix)
    f1 = F1(pre,rca)

    return [PA, class_PA, mIoU, w_MIoU,pre,rca,f1]



def evaluate(targets,preds):
    confusion_matrix = calculate_confusion_matrix(targets,preds,cls_num)
    PA = pixel_accuracy(confusion_matrix)
    class_PA = weighted_pixel_accuracy_class(confusion_matrix)
    iou = IoU(confusion_matrix)
    mIoU = MIoU(iou)
    w_MIoU = weighted_MIoU(confusion_matrix)
    results = [PA, class_PA, mIoU, w_MIoU]
    return results

       

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
#    for i in range(bs):
    incorrect = preds.ne(targets).cpu().sum()
    # torch.ne(input, other, out=None)
    #computes input != other element-wise
    err = incorrect.item()/n_pixels 
    # original err = incorrect/n_pixels, 
    # by adding .item convert incorrect from tensor to scalar
    # (only one element in incorrect)
    
    return round(err,5)

def error_singel_in_batch(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    error = []
    for i in range(bs):
        incorrect = preds.ne(targets).cpu().sum()
        err = incorrect.item() / n_pixels
        error.append(round(err,5))    
    return error

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        if idx % 100 == 0:
            print(idx)
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data
        # original:trn_loss += loss.data[0], but there is a bug
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    for idx, data in enumerate(test_loader):    
#    for data, target in test_loader:
        if idx % 20 == 0:
            print(idx)
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())
#        data = Variable(data.cuda(), volatile=True)
#        target = Variable(target.cuda())
        with torch.no_grad():
            output = model(inputs)
        test_loss += criterion(output, targets).item() #fix the bug here
        pred = get_predictions(output)
        test_error += error(pred, targets.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def test1(model, test_loader, criterion, epoch=1, cls_num=2):
    model.eval()
    test_loss = 0
    test_error = 0
    c_matrix = np.zeros([cls_num, cls_num])
    for idx, data in enumerate(test_loader):    
#    for data, target in test_loader:
        if idx % 20 == 0:
            print(idx)
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())
#        data = Variable(data.cuda(), volatile=True)
#        target = Variable(target.cuda())
        with torch.no_grad():
            output = model(inputs)
        test_loss += criterion(output, targets).item() #fix the bug here
        pred = get_predictions(output)
        c_matrix += calculate_confusion_matrix0(targets,pred,cls_num)
        test_error += error(pred, targets.data.cpu())
        
    result = evaluate_whole_dataset(c_matrix)  
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, result

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    with torch.no_grad():
        output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])
    return pred[i]

def view_sample_predictions_new(model, inputs, targets, n):
    #inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    #label = Variable(targets.cuda())
    with torch.no_grad():
        output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    err = error(pred, targets.data.cpu())

    for i in range(min(n, batch_size)):
        #img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])
    return pred[i]
