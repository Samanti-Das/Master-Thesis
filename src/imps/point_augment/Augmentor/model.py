import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from Augment.pointnet import PointNetCls
from Augment.augmentor import Augmentor
import numpy as np
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics
import Common.data_utils as d_utils
import random
from Common import loss_utils

self_dim = 3

#Segmentor = 
augmentor = Augmentor().cuda()
print('No existing Augment, starting training from scratch...')
start_epoch = 0

# optimizer_c = torch.optim.Adam(
# classifier.parameters(),
# lr=self.opts.learning_rate,
# betas=(0.9, 0.999),
# eps=1e-08,
# weight_decay=self.opts.decay_rate
# )

optimizer_a = torch.optim.Adam(
            augmentor.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001
        )

global_epoch = 0
best_tst_accuracy = 0.0
blue = lambda x: '\033[94m' + x + '\033[0m'

print("Start Training")

PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation

points = PointcloudScaleAndTranslate(points)
points = points.transpose(2, 1).contiguous()

noise = 0.02 * torch.randn(1, 1024).cuda()

#classifier = classifier.train()
augmentor = augmentor.train()
optimizer_a.zero_grad()
#optimizer_c.zero_grad()

aug_pc = augmentor(points, noise)
pred_pc, pc_tran, pc_feat = classifier(points)
pred_aug, aug_tran, aug_feat = classifier(aug_pc)

augLoss  = loss_utils.aug_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, ispn=ispn)
clsLoss = loss_utils.cls_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, pc_feat,
                                aug_feat, ispn=ispn)


augLoss.backward(retain_graph=True)
clsLoss.backward(retain_graph=True)

optimizer_c.step()
optimizer_a.step()

