#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:08:35 2020

@author: yanis
"""

import torch.optim as optim
from torchsummary import summary
from models import initialize_model
from train import train_model
import datasets 
import argparse
import os
import torch
import matplotlib.pyplot as plt

from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""
bpath = './firstTest'
img_path = '/Users/yanis/GehlerDataset/im'
target_path = '/Users/yanis//GehlerDataset/colorMean.csv'

epochs = 2
batchsize = 8

#model_ft,input_size = initialize_model(model_name = "resnet101", num_classes = 38, feature_extract = True, use_pretrained=True)
#model_ft.train()

#summary(model_ft, input_size=(3, 512, 512))

dataloaders = datasets.get_dataloader(img_path,target_path, fraction=0.7, batch_size=4)

gd = datasets.GehlerDataset(img_path = img_path,
                                target_path = target_path,
                                transform = None,
                                seed = 12)

from torchvision import transforms

data_transform = transforms.Compose([
        datasets.RandomFlip(),
        datasets.RandomRotate(10,0.5),
        datasets.ToTensor()
        ])

for i in range(10):
    min_value = data_transform(gd[i])['image'].max()
    print(f'min_value :{min_value}')



#trained_model = train_model(model_ft, criterion, dataloaders,
                            #optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
#torch.save(model_ft, os.path.join(bpath, 'weights.pt'))
