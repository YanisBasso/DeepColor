#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:08:35 2020

@author: yanis
"""

from ConfigParser import ConfigParser
from train import Trainer
import argparse
import torch
import models
import datasets

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(config):
    
    #create a config parser 
    config = ConfigParser.from_args(args)
    
    # setup data_loader instances
    dataloaders = datasets.get_dataloader(img_path = "/Users/yanis/GehlerDataset/im",
                                          target_path = "/Users/yanis/GehlerDataset/colorMean.csv",
                                          fraction=0.7,
                                          batch_size=4)
    
    #setup model 
    model = config.init_obj('arch', models)
    #print(model)
    
    # get function handles of loss and metrics
    criterion = torch.nn.MSELoss(reduction='mean')
    metrics = {}
    
    # build optimizer
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    #print(optimizer)
    
    trainer = Trainer(model = model,
                      dataloaders = dataloaders,
                      optimizer = optimizer,
                      criterion = criterion,
                      metrics = metrics,
                      config = config)
    
    trainer.train()
    
    print(trainer.model)
    
    

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Deep Color')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    #create a config parser 
    config = ConfigParser.from_args(args)
    
    main(config)




#trained_model = train_model(model_ft, criterion, dataloaders,
                            #optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
#torch.save(model_ft, os.path.join(bpath, 'weights.pt'))
