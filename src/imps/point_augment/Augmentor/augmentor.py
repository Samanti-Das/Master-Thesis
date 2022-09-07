#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: augmentor.py
@time: 2019/09/16
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
from imps.point_augment.Common import loss_utils, point_augment_utils

def batch_quat_to_rotmat(q, out=None):

    B = q.size(0)

    if out is None:
        out = q.new_empty(B, 3, 3)

    # 2 / squared quaternion 2-norm
    len = torch.sum(q.pow(2), 1)
    s = 2/len

    s_ = torch.clamp(len,2.0/3.0,3.0/2.0)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = (1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = (1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = (1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s))#.mul(s_)

    return out, s_
    
class Augmentor_Rotation(nn.Module):
    def __init__(self,dim):
        super(Augmentor_Rotation, self).__init__()
        self.fc1 = nn.Linear(dim + 64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(B, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)
        return x, None


class Augmentor_Displacement(nn.Module):
    def __init__(self, dim):
        super(Augmentor_Displacement, self).__init__()

        self.conv1 = torch.nn.Conv1d(dim+64+64, 64, 1)

        self.conv2 = torch.nn.Conv1d(64, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 16, 1)
        self.conv4 = torch.nn.Conv1d(16, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = self.conv4(x)

        return x


class Augmentor(nn.Module):
    def __init__(self,dim=64,in_dim=3):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.rot = Augmentor_Rotation(self.dim)
        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, pt, global_feature, noise):


        B, C, N = pt.size()
        raw_pt = pt[:,:3,:].contiguous()
        normal = pt[:,3:,:].transpose(1, 2).contiguous() if C > 3 else None


        x = F.relu((self.conv1(raw_pt)))
        x = F.relu((self.conv2(x)))
        pointfeat = x
        #x = F.relu((self.conv3(x)))
        #x = F.relu((self.conv4(x)))
        #x = torch.max(x, 2, keepdim=True)[0]
        
        global_feature = global_feature.unsqueeze(0)

        feat_r = global_feature #x.view(-1, 1024)
        feat_r = torch.cat([feat_r,noise],1)
        rotation, scale = self.rot(feat_r)
        #print("rotation is {}".format(rotation))
        
        feat_d = global_feature.view(-1, 64, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d, noise_d],1)
        displacement = self.dis(feat_d)
        #print("displacement is {}".format(displacement))
        pt = raw_pt.transpose(2, 1).contiguous() 

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0  
        if p1 > possi:
            pt_aug = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt_aug = pt.transpose(1, 2).contiguous()


        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt_aug = pt_aug + displacement
        
        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt_aug = torch.cat([pt_aug,normal],1)
        
        # displacement of instance
        #p3 = random.uniform(0, 1)
        #displacement_factor = random.uniform(-0.5, 0.5)
        #if p3 > possi:
        #    pt_aug = torch.add(pt_aug, displacement_factor)
         
        # elastic distortion of instance
        # p4 = random.uniform(0, 1)
        # if p4 > possi:
        #     for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
        #                 pt_aug = point_augment_utils.elastic_distortion(
        #                     pt_aug, granularity, magnitude
        #                 )
            
        # flip at centre for an instance
        p5 = random.uniform(0,1)

        if p5 > possi:
             pt_aug = point_augment_utils.flip_in_center(pt_aug)
          
        #pt_aug_clamped = pt_aug
        pt_aug_clamped = torch.clamp(pt_aug, min=-0.5, max=0.5)
        
        

                         
        return pt_aug_clamped