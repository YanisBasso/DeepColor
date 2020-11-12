#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:51:56 2020

@author: yanis
"""

from torch import nn
from torchvision import models

class FinetuneResNet(nn.Module):
    
    def __init__(self,backbone_version,num_classes,feature_extracting,use_pretrained = True):
        super(FinetuneResNet, self).__init__()
        assert backbone_version in ['resnet18','resnet50','resnet101']
        self.backbone_version = backbone_version 
        #Download pre-trained module
        self.model = getattr(models,self.backbone_version)(use_pretrained)
        #Freeze the feature extracter part 
        self._set_parameter_requires_grad(feature_extracting)
        #Create regression layers 
        self.num_classes = num_classes
        #Add regression part 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs,512),
                nn.ReLU(inplace=True),
                nn.LayerNorm(512),
                nn.Linear(512,256),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256),
                nn.Linear(256,self.num_classes)
             )
        
    def _set_parameter_requires_grad(self, feature_extracting):
            if feature_extracting:
                for param in self.model.parameters():
                    param.requires_grad = False
        
        
    def forward(self,x):
        x = self.model(x)
    
    def __str__(self):
        return str(self.model)
        
        
        
        
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs,512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256,num_classes)
         )
        input_size = 224
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs,512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256,num_classes)
        )
        input_size = 224
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs,512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256,num_classes)

        )
        input_size = 224
    elif model_name == 'vgg16':
      model_ft = models.vgg16_bn(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.classifier[0].in_features
      model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs,512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256,num_classes)
        )
    else:
          print("Invalid model name, exiting...")
          exit()

    return model_ft, input_size
'''

    
    
    
    
    