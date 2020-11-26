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
from torchvision import transforms
import torchvision.transforms.functional as F
from utils import vsrgb2linear, vlinear2srgb,linear2srgb,srgb2linear
import cv2 
import random 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class GehlerDataset(Dataset):
    """Regression dataset made from Gehler dataset"""
    def __init__(self, dir_path, load_target=False, remove_cc=None, seed=None, 
                 fraction=None, subset=None, transform=None):
        """
        :param dir_path: Name of the path where the images and coordinates 
            are stored.
        :param target_path: Name of the cvs file containing target. If None, 
            the targets are computed from the image.
        :param remove_cc: Boolean which specify if the color checker has to be 
            hide during image loading.
        :param seed: Specify a seed for the train and test split.
        :param fraction: A float value from 0 to 1 which specifies the validation split fraction.

        :param subset: 'Train' or 'Test' to select the appropriate set.
        :param transform: Optional transform to be applied on a sample.
        """
        self.dir_path = Path(dir_path)
        self.img_path = self.dir_path / 'im'
        self.coord_path = self.dir_path / 'coord'
        self.transform = transform
        self.remove_cc = remove_cc
        self.load_target = load_target
        
        #list image names 
        self.ids = next(os.walk(self.img_path))[2]
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
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #get img 
        name  = self.ids[idx]
        img_path = str(self.img_path / name)
        img = io.imread(img_path)
        #img = img/255
        
        if self.load_target :
            target = self._load_target(Path(name).stem)
        else : 
            patches_coord = self._get_patches_coordinates(Path(name).stem)
            target = self._extract_target(patches_coord,img)
        
        if self.remove_cc :
            mire_coord = self._get_mire_coordinates(Path(name).stem)
            mire_coord[:,0] = mire_coord[:,0]*img.shape[1]
            mire_coord[:,1] = mire_coord[:,1]*img.shape[0]
            pts = mire_coord.astype(np.int32)
            mask = np.zeros(img.shape[:2], dtype=img.dtype)
            mask = cv2.fillPoly(mask,[pts],(255,255))
            img[mask == 255] = 0 
        
        sample = {'image': img, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def _get_mire_coordinates(self,name):
        """
        """
        path = self.coord_path / '{}_macbeth.txt'.format(name)
        with open(path) as fp :
            lines = fp.readlines()
            x,y = lines[0].split(' ')
            r_x = np.float32(x)
            r_y = np.float32(y)
            mire_coord  = np.zeros((4,2))
            for i,line in enumerate(lines[1:5]):
                x,y = line.split(' ')
                mire_coord[i%4,0] = np.float32(x)/r_x
                mire_coord[i%4,1] = np.float32(y)/r_y
        mire = mire_coord.copy()
        mire_coord[3],mire_coord[2] = mire[2],mire[3]
        return mire_coord 
    
    def _get_patches_coordinates(self,name):
        path = self.coord_path / '{}_macbeth.txt'.format(name)
        with open(path) as fp :
            lines = fp.readlines()
            x,y = lines[0].split(' ')
            r_x = np.float32(x)
            r_y = np.float32(y)
            patches_coord = np.zeros((24,4,2))
            for i,line in enumerate(lines[5:]):
                x,y = line.split(' ')
                patches_coord[i//4,i%4,0] = np.float32(x)/r_x
                patches_coord[i//4,i%4,1] = np.float32(y)/r_y
        return patches_coord
    
    def _extract_target(self,patches_coord,img):
        patches_coord[:,:,0] = patches_coord[:,:,0]*img.shape[1]
        patches_coord[:,:,1] = patches_coord[:,:,1]*img.shape[0]
        targets = np.zeros((24,3))
        for i,patche in enumerate(patches_coord):
            pts = patche.astype(np.int32)
            mask = np.zeros(img.shape[:2], dtype=img.dtype)
            mask = cv2.fillPoly(mask,[pts],(255,255))
            pixel_info = img[mask == 255]
            targets[i] = np.mean(pixel_info,axis=0)
        return targets
    
    def _load_target(self,name):
        path = self.dir_path / "target" / f"{name}.txt"
        targets = np.zeros((24,3))
        with open(path,"r") as fp :
            lines = fp.readlines()
            for i,line in enumerate(lines) :
                line = line.rstrip()
                r,g,b = line.split(',')
                targets[i,:] = [r,g,b]
        return targets
            
    
    def get_name(self,idx):
        return Path(self.ids[idx]).stem
    

##################################
# Data augmentation 
##################################

class RandomFlip(object):
    """
    Data augmentation - Random horizontal and vertical flip
    """   
    def __init__(self,p_vert=0.5,p_hor=0.5):
        self.p_vert = p_vert
        self.p_hor = p_hor
        
    def __call__(self,sample):
        assert type(sample) == dict 
        image = sample['image']
        #Horizontal flip
        if random.random() < self.p_hor  :
            image = np.flip(image,1).copy()
        #Vertical flip
        if random.random() < self.p_vert  :
            image = np.flip(image,0).copy()
        sample['image'] = image
        return sample

class RandomRotate(object):
    """
    Data augmentation - Random Rotation 
    """
    def __init__(self,angle_max,p):
        self.angle_max = angle_max
        self.p = p
        
    def __call__(self,sample):
        assert type(sample) == dict
        image = sample['image']
        
        if random.random() < self.p:
            angle = (random.random()*2 - 1)*self.angle_max
            image = transform.rotate(image,angle)
        sample['image'] = image
        return sample 

class RandomColorShift(object):
    """
    Data augmentation - Random Rotation 
    """
    def __init__(self,min_value,max_value):
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        
    def __call__(self,sample):
        assert type(sample) == dict
        image,target = sample['image'],sample['target']
        
        #image = srgb2linear(image)
        #target = srgb2linear(target)
        
        r_shift = random.random()*(self.max_value - self.min_value) + self.min_value
        b_shift = random.random()*(self.max_value - self.min_value) + self.min_value

        image[:,:,0] = image[:,:,0]*r_shift
        image[:,:,2] = image[:,:,2]*b_shift
        
        target[:,0] = target[:,0]*r_shift
        target[:,2] = target[:,2]*b_shift
        
        #image = linear2srgb(image)
        #target = linear2srgb(target)

        return {'image':image,'target':target}
        
        
        
        
        
        

##################################
# Data preprocessing 
##################################
        
    
class Srgb2Linear(object):
    """
    Data transfromation - apply inverse gamma transformation 
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

class Linear2srgb(object):
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
        image = vlinear2srgb(image)
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

class RemoveShadingTarget(object):
    
    def __call__(self,sample):
        target = sample['target']
        lum = np.sum(target,axis=1)
        target[:,0] = target[:,0]/lum
        target[:,1] = target[:,1]/lum
        target[:,2] = target[:,2]/lum
        sample['target'] = target[:,:2]
        return sample 

class PrepareTarget(object):
    
    def __call__(self,sample):
        target = sample['target']
        
        #Linearise
        target = srgb2linear(target)
        
        #Remove shading 
        lum = np.sum(target,axis=1)
        target[:,0] = target[:,0]/lum
        target[:,1] = target[:,1]/lum
        target[:,2] = target[:,2]/lum
        target = target[:,:2]
        
        #Create white-patche values 
        target[18] = np.mean(target[18:],axis=0)
        sample['target'] = target[:19].ravel()
        return sample 
    
##################################
# Data preprocessing 
##################################


def get_dataloader(dir_path, target_path=None, fraction=0.7, batch_size=32):
    data_transforms = {
        'Train': transforms.Compose([ 
           Rescale(225),
           RandomCrop(224),
           RandomColorShift(0.6,1.4),
           RandomFlip(),
           RandomRotate(10,0.5),
           PrepareTarget(),
           ToTensor(),
           Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
                     ]),
        'Test': transforms.Compose([
           Rescale(230),
           RandomCrop(224),
           PrepareTarget(),
           ToTensor(),
           Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
                     ])
    }
    
    image_datasets = {x: GehlerDataset(dir_path = dir_path,
                                       target_path = target_path,
                                       transform = data_transforms[x],
                                       remove_cc = True,
                                       seed=12, 
                                       fraction=fraction, 
                                       subset=x)
                      for x in ['Train', 'Test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['Train', 'Test']}
    
    return dataloaders
    

def get_eval_dataset(dir_path, target_path=None, fraction=0.7) :
    
    data_transform = transforms.Compose([
                                     Rescale(230),
                                     RandomCrop(224),
                                     RemoveShadingTarget(),
                                     ToTensor(),
                                     Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
    
    
    test_dataset = GehlerDataset(dir_path = dir_path,
                                 target_path = target_path,
                                 transform = data_transform,
                                 remove_cc = True,
                                 seed=12, 
                                 fraction=fraction, 
                                 subset='Test')
    
    return data_transform,test_dataset


if __name__ == "__main__":
    
    from time import time 
    
    
    data_transforms = {
        'Train': transforms.Compose([ 
            Rescale(225),
            RandomCrop(224),
            RandomColorShift(0.8,1.2),
            RandomFlip(),
            RandomRotate(10,0.5),
            PrepareTarget(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
                      ]),
        'Test': transforms.Compose([
            Rescale(230),
            RandomCrop(224),
            PrepareTarget(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
                      ])
    }
    
    image_datasets = {x: GehlerDataset(dir_path = "/Users/yanis/GehlerDataset",
                                        load_target = True,
                                        transform = data_transforms[x],
                                        remove_cc = False,
                                        seed=12, 
                                        fraction=0.7, 
                                        subset=x)
                      for x in ['Train', 'Test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                  shuffle=True, num_workers=4)
                    for x in ['Train', 'Test']}
    
    

        
    
    iterator = iter(dataloaders['Train'])
    sample = next(iterator)
    print(sample)



    
