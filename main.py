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


# Create the dataloader
dataloaders = get_dataloader(img_path,target_path, batch_size=batchsize,fraction = 0.002)
iterator = iter(dataloaders['Train'])
tensor = next(iterator)['image']
showImgFromTensor(tensor)




#trained_model = train_model(model_ft, criterion, dataloaders,
                            #optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
#torch.save(model_ft, os.path.join(bpath, 'weights.pt'))
