#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:08:35 2020

@author: yanis
"""

from ConfigParser import ConfigParser
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Deep Color')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)




#trained_model = train_model(model_ft, criterion, dataloaders,
                            #optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
#torch.save(model_ft, os.path.join(bpath, 'weights.pt'))
