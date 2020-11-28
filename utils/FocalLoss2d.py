#from common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()  # torch.Size([262144, 1])


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
            '''
            logit.permute(0, 2, 3, 1) transpose: torch.Size([1, 512, 512, 6])
            torch.Size([262144, 6])
            .contiguous() Returns a contiguous tensor containing the same data as self tensor. 
            '''
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()  # torch.Size([262144, 6])
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)  # torch.Size([262144, 1])
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class RobustFocalLoss2d(nn.Module):
    #assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
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

        prob  = (prob*select).sum(1).view(-1,1)
        prob  = torch.clamp(prob,1e-8,1-1e-8)

        focus = torch.pow((1-prob), self.gamma)
        #focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus,0,2)


        batch_loss = - class_weight *focus*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

if __name__ == "__main__":
    FL = FocalLoss2d(gamma=0.5)
    CE = nn.CrossEntropyLoss()

    inputs = torch.from_numpy(np.load('output/out.npy'))
    targets = torch.from_numpy(np.load('output/targets.npy'))

    inputs_fl = Variable(inputs.clone(), requires_grad=True).cuda()
    targets_fl = Variable(targets.clone()).cuda()

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
#    print('----inputs----')
#    print(inputs)
#    print('---target-----')
#    print(targets)


    ce_loss = CE(inputs_ce, targets_ce)
    fl_loss = FL(inputs_fl, targets_fl)    
    print('ce = {}, fl ={}'.format(ce_loss.item(), fl_loss.item()))
    fl_loss.backward()
    ce_loss.backward()
    #print(inputs_fl.grad.data)
    #print(inputs_ce.grad.data)