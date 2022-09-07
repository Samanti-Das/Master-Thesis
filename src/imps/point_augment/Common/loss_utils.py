#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: loss_utils.py 
@time: 2019/09/23
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""


import numpy as np
import torch
import torch.nn.functional as F

NUM = 1.2#2.0
W = 0.5 #10.0


def cal_loss_raw(pred, target):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''


    eps = 0.2
    #print("size of pred {}".format(pred.size()))
    
    n_class = pred.size(1)
    #print("n_class is {}".format(n_class))
    print("size of pred is {}".format(pred.size()))
    one_hot = torch.zeros_like(pred)
    one_hot[:,target]=1
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss_raw = -(one_hot * log_prb).sum(dim=1)


    loss = loss_raw.mean(axis=0)

    return loss,loss_raw

def mat_loss(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



def cls_loss(pred, pred_aug, gold, pc_tran, aug_tran, pc_feat, aug_feat, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)

    feat_diff = 10.0*mse_fn(pc_feat,aug_feat)
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(1.0-cls_pc_raw)**2).cuda()
    cls_diff = (torch.abs(cls_pc_raw - cls_aug_raw) * (parameters*2)).mean()
    cls_loss = cls_pc + cls_aug  + feat_diff# + cls_diff

    return cls_loss

def aug_loss(pred, pred_aug, target):
    
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    cls_pc, cls_pc_raw = cal_loss_raw(pred, target)
    print("cross_entropy_loss mean is given by size {}".format(cls_pc.size()))
    print("cross_entropy_loss raw is given by size {}".format(cls_pc_raw.size()))
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, target)
    print("cross_entropy_loss aug mean is given by size {}".format(cls_aug.size()))
    print("cross_entropy_loss aug raw is given by size {}".format(cls_aug_raw.size()))
    pc_con = F.softmax(pred, dim=-1)#.max(dim=1)[0]
    one_hot = torch.zeros_like(pred)
    one_hot[:,target]=1
    pc_con = (pc_con*one_hot).max(dim=1)[0]

    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(pc_con) * NUM).cuda()
    
    # both losses are usable
    aug_diff = W * torch.abs(1.0 - torch.exp(cls_aug_raw - cls_pc_raw * parameters)).mean(axis=0)
    #aug_diff =  W*torch.abs(cls_aug_raw - cls_pc_raw*parameters).mean()
    aug_loss = cls_aug + aug_diff
    aug_loss_ = cls_aug

    return aug_loss, aug_loss_

def cal_loss_raw_(pred, target):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''


    eps = 0.2
    #print("size of pred {}".format(pred.size()))
    
    n_class = pred.size(1)
    #print("number of class given by {}".format(n_class))
    #print("n_class is {}".format(n_class))
    #print("size of pred is {}".format(pred.size()))
    one_hot = torch.zeros_like(pred)
    one_hot[:,target]=1
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    #print("log_prb has size {}".format(log_prb.size()))

    loss_raw = -(one_hot * log_prb).sum(dim=1)

    #print("the loss is given by {}".format(loss_raw))
    loss = loss_raw.mean(axis=0)
    #print("the mean loss is given by {}".format(loss))
    return loss,loss_raw


def aug_loss_(pred, pred_aug, target):
    
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    cls_pc, cls_pc_raw = cal_loss_raw_(pred, target)
    #print("cross_entropy_loss mean is given by size {}".format(cls_pc.size()))
    #print("cross_entropy_loss raw is given by size {}".format(cls_pc_raw.size()))
    cls_aug, cls_aug_raw = cal_loss_raw_(pred_aug, target)
    #print("cross_entropy_loss aug mean is given by size {}".format(cls_aug.size()))
    #print("cross_entropy_loss aug raw is given by size {}".format(cls_aug_raw.size()))
    pc_con = F.softmax(pred, dim=-1)#.max(dim=1)[0]
    #print("dimension of pc_con after applying softmax is {}".format(pc_con.size()))
    one_hot = torch.zeros_like(pred)
    #print("one hot is given by size {}".format(one_hot.size()))
    one_hot[:,target]=1
    #print("one hot is given by  {}".format(one_hot))
    pc_con = (pc_con*one_hot).max(dim=1)[0]
    #print("dimension of pc_con is {}".format(pc_con.size()))

    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(pc_con) * NUM).cuda()
    #print("parameters are given by {}".format(parameters))
    #print("parameters are given by size {}".format(parameters.size()))
    
    # both losses are usable
    #print("cls_aug_raw has dim {}".format(cls_aug_raw.size()))
    aug_diff = W * torch.abs(1.0 - torch.exp(cls_aug_raw - cls_pc_raw * parameters)).mean(axis=0)
    #aug_diff =  W*torch.abs(cls_aug_raw - cls_pc_raw*parameters).mean()
    aug_loss = cls_aug + aug_diff
    aug_loss_ = cls_aug
    #print("aug_loss dimension is {}".format(aug_loss.size()))
    #print("aug_loss is {}".format(aug_loss))
    

    return aug_loss, aug_loss_

def get_randla_loss(logits, labels, class_weights):
    class_weights = torch.from_numpy(class_weights).float().to(logits.device)
    logits = logits.reshape(-1, len(class_weights))
    labels = labels.reshape(-1)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    
   
    n_points = output_loss.shape[0]
        
    output_loss = output_loss.sum() / n_points
    
    return output_loss

#hello
def get_loss(logits, labels, class_weights, mask=None):
    class_weights = torch.from_numpy(class_weights).float().to(logits.device)
    logits = logits.reshape(-1, len(class_weights))
    labels = labels.reshape(-1)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    
    if mask is not None:
        output_loss = output_loss * mask
        n_points = torch.sum(mask)
    else:
        n_points = output_loss.shape[0]
        
    output_loss = output_loss.sum() / n_points
    
    return output_loss





