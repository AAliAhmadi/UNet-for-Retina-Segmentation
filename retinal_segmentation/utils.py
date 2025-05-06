#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import random
import time
import cv2
import imageio
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F

def seeding(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DICEBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DICEBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice
        return Dice_BCE

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=1)  # (512, 512, 3)
    return mask

