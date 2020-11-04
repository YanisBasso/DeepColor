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


# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {}


# Create the dataloader
dataloaders = datasets.get_dataloader(img_path,target_path, batch_size=batchsize)
i = 0
for samples in iter(dataloaders['Train']):
    image = samples['image'][0]
    print('here')
    i+=1
    if i == 3 :
        break
    def showImgFromTensor(tensor):
      img = tensor.numpy().transpose((1, 2, 0))
      plt.figure()
      plt.imshow(img)
    
    showImgFromTensor(image)

#trained_model = train_model(model_ft, criterion, dataloaders,
                            #optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
#torch.save(model_ft, os.path.join(bpath, 'weights.pt'))
