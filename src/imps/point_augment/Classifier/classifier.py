import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



# Classifier model for classifying features to corresponding labels:
class RandlaClassifier(nn.Module):
    def __init__(self, feat_shape_layer_2, N_CLASS): # valid for 3 encoding layers only
        super(RandlaClassifier, self).__init__()
        self.feat_shape = feat_shape_layer_2
        self.num_class = N_CLASS
        self.fc1 = nn.Linear(self.feat_shape, 45)
        #self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(45, self.num_class )
        self.dropout = nn.Dropout(p=0.3)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)
        #self.bn1 = nn.BatchNorm1d(100)
        #self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

    def forward(self, feat):
        x = F.relu(self.fc1(feat))
        x = self.dp1(x)
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.dropout(self.fc2(x)))
        x = self.dp2(x)       
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=-1)
        return x
    