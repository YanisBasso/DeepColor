#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:28:40 2020

@author: yanis
"""

from __future__ import print_function, division
import os
from pathlib import Path
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from tqdm import tqdm
from utils import vsrgb2linear
import cv2 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ChengDataset(Dataset):
    """Cheng dataset with rg colorchecker patch values as target"""
    
    def __init__(self,img_path,target_path,transform=None,seed=None,fraction=None,subset=None):
        
        self.img_path = img_path
        self.target_path = target_path
        self.targets = pd.read_csv(target_path) 
        self.transform = transform
        self.camera_sensors = next(os.walk(img_path))[1]
        
        first_dir = os.path.join(self.img_path,self.camera_sensors[0])
        self.ids = next(os.walk(first_dir))[2]
        
        print(ids)
        
        if fraction :
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.ids))
                np.random.shuffle(indices)
                self.ids = self.ids[indices]
            if subset == 'Test':
                self.ids = self.ids[:int(
                    np.ceil(len(self.ids)*(1-self.fraction)))]
            else:
                self.ids = self.ids[int(
                    np.ceil(len(self.ids)*(1-self.fraction))):]
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        raise NotImplementedError
        
    def _get_image_ids(self):
        
        first_dir = os.path.join(self.img_path,self.camera_sensors[0])
        ids = next(os.walk(first_dir))[2]
        ids = [index.split('_')[1] for index in ids]
        return ids
        
class GehlerDataset(Dataset):
    """Gheler dataset with rg colorchecker patch values as target"""

    def __init__(self, img_path, target_path, transform=None,seed=None, fraction=None, subset=None):

        self.img_path = img_path
        self.target_path = target_path
        self.targets = pd.read_csv(target_path) 
        self.transform = transform
        self.ids = next(os.walk(img_path))[2]
        if '.DS_Store' in self.ids :
            self.ids.remove('.DS_Store')
            self.ids = np.array(self.ids)
        if fraction :
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.ids))
                np.random.shuffle(indices)
                self.ids = self.ids[indices]
            if subset == 'Test':
                self.ids = self.ids[:int(
                    np.ceil(len(self.ids)*(1-self.fraction)))]
            else:
                self.ids = self.ids[int(
                    np.ceil(len(self.ids)*(1-self.fraction))):]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #get img
        name = self.ids[idx]
        img_name = os.path.join(self.img_path,name)
        image = io.imread(img_name)
        if image.max() > 1:
          image = image/255.0

        #get mask
        target = self.targets.query('name == "{}"'.format(name[:-4]), inplace = False) 
        coeffs = np.array(target.values[0][1:]).astype(np.float32)
        if coeffs.max() > 1:
          coeffs = coeffs/255.0
        sample = {'image': image, 'target': coeffs}
      
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def getName(self,idx) :
        """
        get the name of a given image 
        
        :idx: int - index of image in the dataset 
        :return: string - name of the image without extension
        """
        nameWithExtenstion = Path(self.ids[idx])
        return nameWithExtenstion.stem
    
class Srgb2Linear(object):
    """
    Data transfromation - apply gamma transformation 
    """
    
    def __call__(self, sample):
        """
        :sample: numpy array or dict for training mode
        :return: numpy array or dict according to the input
        """
        if type(sample) == dict:
            image,target = sample['image'], sample['target']
        elif type(sample) == np.ndarray:
            image = sample
        
        shape = image.shape
        image = image.flatten()
        image = vsrgb2linear(image)
        image = image.reshape(shape)
        
        if type(sample) == dict:
            return {'image': image, 'target': target}
        elif type(sample) == np.ndarray:
            return image    

class RemoveShading(object):
    """
    Data transfromation - Normalize each pixel by its brightness value
    """
    def __call__(self, sample):
        """
        :sample: numpy array or dict for training mode
        :return: numpy array or dict according to the input
        """
        if type(sample) == dict:
            image,target = sample['image'], sample['target']
        elif type(sample) == np.ndarray:
            image = sample

        
        lum = np.sum(image,axis=2)
        image[:,:,0] = image[:,:,0]/lum
        image[:,:,1] = image[:,:,1]/lum
        image[:,:,2] = image[:,:,2]/lum
        image = np.nan_to_num(image)
        image.astype(np.float32)
        
        if type(sample) == dict:
            return {'image': image, 'target': target}
        elif type(sample) == np.ndarray:
            return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if type(sample) == dict:
          image, target = sample['image'], sample['target']
        elif type(sample) == np.ndarray:
          image = sample

        image = image.transpose((2, 0, 1))

        if type(sample) == dict:
          return {'image': torch.from_numpy(image).float(),
                  'target': torch.from_numpy(target).float()}
        elif type(sample) == np.ndarray:
          return torch.from_numpy(image).float()

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
      if type(sample) == dict:
        image,target = sample['image'], sample['target']
      elif type(sample) == np.ndarray:
        image = sample
    
      h, w = image.shape[:2]
      if isinstance(self.output_size, int):
          if h > w:
              new_h, new_w = self.output_size * h / w, self.output_size
          else:
              new_h, new_w = self.output_size, self.output_size * w / h
      else:
          new_h, new_w = self.output_size

      new_h, new_w = int(new_h), int(new_w)
      image = cv2.resize(image, (new_w,new_h), interpolation = cv2.INTER_NEAREST)

      if type(sample) == dict:
        return {'image': image, 'target': target}
      elif type(sample) == np.ndarray:

        return image

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def norm(self,tensor):
        tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        return tensor 
    
    def __call__(self,sample):
        if type(sample) == dict:
            image,target = sample['image'], sample['target']
            image = self.norm(image)
            return {'image': image, 'target': target}
            
        elif type(sample) in [np.ndarray,torch.Tensor]:
            image = sample
            image = self.norm(image)
            print(image)
            return image

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
            

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if type(sample) == dict:
          image,target = sample['image'], sample['target']
        elif type(sample) == np.ndarray:
          image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        if type(sample) == dict:
          return {'image': image, 'target': target}
        elif type(sample) == np.ndarray:
          return image


def get_dataloader(img_path,target_path, fraction=0.7, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
    """
    data_transforms = {
        'Train': transforms.Compose([ 
           #Srgb2Linear(),
           #RemoveShading(),
           Rescale(225),
           RandomCrop(224),
           ToTensor(),
           Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])]),
        'Test': transforms.Compose([
           #Srgb2Linear(),
           #RemoveShading(),
           Rescale(230),
           RandomCrop(224),
           ToTensor(),
           Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])])
    }

    image_datasets = {x: GehlerDataset(img_path = img_path,
                                target_path = target_path,
                                transform = data_transforms[x],
                                seed = 12,
                                fraction = fraction,
                                subset = x)
                  for x in ['Train', 'Test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=8)
                   for x in ['Train', 'Test']}
    return dataloaders
    


    
    