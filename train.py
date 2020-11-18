#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:54:05 2020

@author: yanis
"""

import csv
import copy
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import os


class Trainer(object):
  """
  Trainer class
  """
  def __init__(self,model,dataloaders,optimizer,criterion,metrics,config):
    self.config = config
    self.device, device_ids = self._prepare_device(config['n_gpu'])
    print(self.device,device_ids)
    self.model = model.to(self.device)
    if len(device_ids) > 1:
        self.model = torch.nn.DataParallel(model, device_ids=device_ids)
    self.criterion = criterion 
    self.metrics = metrics
    self.optimizer = optimizer
    self.num_epochs = config['trainer']['epochs']
    self.start_epoch = 1
    assert type(dataloaders) == dict
    self.dataloaders = dataloaders
    self.fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
                      [f'Train_{m}' for m in metrics.keys()] + \
                      [f'Test_{m}' for m in metrics.keys()]
    #self.best_loss = 1e10
    self.checkpoint_dir = config.save_dir
    self.save_period = config['trainer']['save_period']
    
    self.log_path = self.checkpoint_dir / 'log.csv'
    with open(self.log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
        writer.writeheader()

    if config.resume is not None:
      self._resume_checkpoint(config.resume)

  def train(self):
    """
    Training protocol 
    """
    #best_model_wts = copy.deepcopy(self.model.state_dict())
    since = time.time()
    for epoch in range(self.start_epoch,self.num_epochs+1):

      print('Epoch {}/{}'.format(epoch, self.num_epochs))
      print('-' * 10)
      batchsummary = {a : [0] for a in self.fieldnames}

      for phase in ['Train','Test']:
        if phase == 'Train':
          self.model.train()
        else :
          self.model.eval()
        for sample in tqdm(iter(self.dataloaders[phase])):
          
          inputs = sample['image'].to(self.device)
          targets = sample['target'].to(self.device)
          # zero the parameter gradients
          self.optimizer.zero_grad()
          with torch.set_grad_enabled(phase == 'Train'):
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            
            if self.device == torch.device("cuda:0"):
                y_pred = outputs.data.cpu().numpy().ravel()
                y_true = targets.data.cpu().numpy().ravel()
            else :
                y_pred = outputs.data.numpy().ravel()
                y_true = targets.data.numpy().ravel()
            for name, metric in self.metrics.items():
              batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred))

          # backward + optimize only if in training phase
          if phase == 'Train':
              loss.backward()
              self.optimizer.step()

        batchsummary['epoch'] = epoch
        epoch_loss = loss
        batchsummary[f'{phase}_loss'] = epoch_loss.item()
        print('{} Loss: {:.4f}'.format(phase, loss))
      for field in self.fieldnames[3:]:
        batchsummary[field] = np.mean(batchsummary[field])
      print(batchsummary)

      with open(self.log_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
        writer.writerow(batchsummary)
        
        # deep copy the model
        #if phase == 'Test' and loss < best_loss:
            #best_loss = loss
            #best_model_wts = copy.deepcopy(self.model.state_dict())

      if epoch%self.save_period == 0:
        self._save_checkpoint(epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return self.model

  def _save_checkpoint(self,epoch):
    """
    Saving checkpoints
    :param epoch: current epoch number
    """
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'config': self.config
    }
    filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    torch.save(state, filename)
    print("Saving checkpoint: {} ...".format(filename))

  
  def _resume_checkpoint(self,resume_path):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    print("Loading checkpoint: {}...".format(resume_path))
    checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch']+1
    
    assert checkpoint['config']['arch'] == self.config['arch']
    assert checkpoint['config']['optimizer']['type'] == self.config['optimizer']['type']

    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
  
  def _prepare_device(self, n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
                            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                targets = sample['target'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = targets.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(
                phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model