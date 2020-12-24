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
    def __init__(self, dir_path, load_target=False, load_weight=True, remove_cc=None, seed=None, 
                 fraction=None, subset=None, transform=None, nb_image = None):
        """
        :param dir_path: Name of the path where the images and coordinates 
                        are stored.
        :param load_target: If True, targets are load from txt files. If False,
                        target are computed from images 
        :param load_weight: If True, weights are load from txt files. If False, 
                        no weighted are loaded
        :param remove_cc: Boolean which specify if the color checker has to be 
                        hide during image loading.
        :param seed: Specify a seed for the train and test split.
        :param fraction: A float value from 0 to 1 which specifies the validation split fraction.
        :param subset: 'Train' or 'Test' to select the appropriate set.
        :param transform: Optional transform to be applied on a sample.
        :param nb_image: Specify the number of image to consider. 
        """
        self.dir_path = Path(dir_path)
        self.img_path = self.dir_path / 'im'
        self.coord_path = self.dir_path / 'coord'
        self.weight_path = self.dir_path / 'weight'
        self.transform = transform
        self.remove_cc = remove_cc
        self.load_target = load_target
        self.load_weight = load_weight
        self.nb_image = nb_image
        
        #list image names 
        self.ids = next(os.walk(self.img_path))[2]
        if '.DS_Store' in self.ids :
            self.ids.remove('.DS_Store')
            
        if nb_image :
            assert nb_image > 0 and nb_image < len(self.ids)
            self.ids= self.ids[:nb_image]
            
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
        img = (img/255).astype(np.float32)
        
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
        
        if self.load_weight : 
            sample['weight'] = self._load_weight(Path(name).stem)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def _get_mire_coordinates(self,name):
        """
        Get color checker coordinates in normalized coordinate 
        
        :param name: image name without extension
        :return: Normalized position of the color checker's corners in the 
            following order : Upper-Left/Upper-Right/Lower-Right/Lower-Left
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
        """
        Get color patches coordinates in normalized coordinate 
        
        :param name: image name without extension
        :return: Normalized positions of the color patches's corners in the 
            following order : Upper-Left/Upper-Right/Lower-Right/Lower-Left. 
        """
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
        """
        Load targets from text file 
        
        :param name: image name without extension
        :return: Array of RGB values - mean color for each patch
        """
        path = self.dir_path / "target" / f"{name}.txt"
        targets = np.zeros((24,3))
        with open(path,"r") as fp :
            lines = fp.readlines()
            for i,line in enumerate(lines) :
                line = line.rstrip()
                r,g,b = line.split(',')
                targets[i,:] = [r,g,b]
        return targets
    
    def _load_weight(self,name):
        """
        Load weights from text file 
        
        :param name: image name without extension
        :return: Array of size (19,) number of pixel for each class
        """
        path = self.weight_path / f"{name}.txt"
        weights = np.zeros((38,))
        with open(path,"r") as fp :
            lines = fp.readlines()
            for i,line in enumerate(lines) :
                weights[i] = np.float32(line)
        return weights


    def get_patch_distribution(self,idx):
        """
        Function that return the color distribution of each patch of an image
        
        :param idx: index of image
        :return: list of list of pixel values 
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #get img 
        name  = self.ids[idx]
        img_path = str(self.img_path / name)
        img = io.imread(img_path)
        img = (img/255).astype(np.float32)
        
        if transform:
            img = self.transform(img)
        
        patches_coord = self._get_patches_coordinates(Path(name).stem)
        patches_coord[:,:,0] = patches_coord[:,:,0]*img.shape[1]
        patches_coord[:,:,1] = patches_coord[:,:,1]*img.shape[0]
        color_distributions = []
        for i,patche in enumerate(patches_coord):
            pts = patche.astype(np.int32)
            mask = np.zeros(img.shape[:2], dtype=img.dtype)
            mask = cv2.fillPoly(mask,[pts],(255,255))
            pixel_info = img[mask == 255]
            color_distributions.append(pixel_info)
        return color_distributions
        
    
    def get_name(self,idx):
        """
        Get image name without extension 
        """
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
        sample['image'] = image
        sample['target'] = target
        
        return sample
        
        

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
        image = image[:,:,:2]
        image = image.astype(np.float32)
        
        if type(sample) == dict:
            return {'image': image, 'target': target}
        elif type(sample) == np.ndarray:
            return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if type(sample) == dict:
          
            # Place Channel dimension in first place
            image = sample['image']
            image = image.transpose((2, 0, 1))
            sample['image'] = image
          
            # Convert numpy arrays into torch.tensor
            for key in sample : 
                tensor = torch.from_numpy(sample[key]).float()
                sample[key] = tensor
                
            return sample
        
        elif type(sample) == np.ndarray:
            image = sample
            image = image.transpose((2, 0, 1))
            return torch.from_numpy(image).float()          

class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    : param output_size (tuple or int): Desired output size. If tuple, output is
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
          sample['image'] = image
          sample['target'] = target
          return sample
      elif type(sample) == np.ndarray:
          return image

class Normalize(object):
    """
    Apply Distribution normalisation on RGB channels
    
    :param mean: list of means values for each channel
    :param std: list of standard deviation for each channel
    :param inplace: boolean. If true normalisation is done in place
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def norm(self,tensor):
        tensor = F.normalize(tensor, self.mean, self.std, self.inplace)
        return tensor 
    
    def __call__(self,sample):
        """
        :param sample: If Numpy array, transformation is applied on the array. 
                If dict, transformation is applied on the value of key 'image'.
        """
        if type(sample) == dict:
            image = sample['image']
            image = self.norm(image)
            sample['image'] = image
            return sample
            
        elif type(sample) in [np.ndarray,torch.Tensor]:
            image = sample
            image = self.norm(image)
            return image

class UnNormalize(object):
    """
    Remove the effect of distribution normalisation on image
    
    :param mean: list of means values for each channel
    :param std: list of standard deviation values for each channel
    :param inplace: boolean. If true normalisation is done in place
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        :param tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
            

class RandomCrop(object):
    """Crop randomly the image in a sample.

    :param output_size (tuple or int): Desired output size. If int, square crop
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
            image = sample['image']
        elif type(sample) == np.ndarray:
            image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        if type(sample) == dict:
            sample['image'] = image
            return sample
        elif type(sample) == np.ndarray:
            return image

class PrepareTarget(object):
    """ 
    Prepare Target to the desire format : inverse gamma function and 
    normalisation by triplet RGB brightness (R+G+B).
    """
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
# Data preprocessing Pipeline 
##################################


def get_dataloader(dir_path, load_target=True, fraction=0.7, batch_size=32):
    """
    Prepare a dataloader with the basline data preprocessing protocol 
    
    :param dir_path: dataset path 
    :param load_target: If True, load target from txt file. If False, compute them 
        from input images
    :param fraction: Percentage of train sample 
    :param batch_size: Size of training batches
    
    :return: Dict of Dataloader. Key 'Train' for training and key 'Test' for 
    validation and test
    """
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
                                       load_target=load_target,
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
    

def get_eval_dataset(dir_path, load_target=True, fraction=0.7) :
    
    data_transform = transforms.Compose([
                                     Rescale(230),
                                     RandomCrop(224),
                                     PrepareTarget(),
                                     ToTensor(),
                                     Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
    
    
    test_dataset = GehlerDataset(dir_path = dir_path,
                                 tload_target = True,
                                 transform = data_transform,
                                 remove_cc = False,
                                 seed=12, 
                                 fraction=fraction, 
                                 subset='Test')
    
    return data_transform,test_dataset


if __name__ == "__main__":

    data_transform = transforms.Compose([
            Rescale(225),
            RandomCrop((224)),
            RandomColorShift(0.8,1.2),
            RandomFlip(),
            RandomRotate(10,0.5),
            PrepareTarget(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
            ])
    

    image_dataset = GehlerDataset(dir_path = "/Users/yanis/GehlerDataset",
                                  load_target = True,
                                  load_weight = True,
                                  transform = data_transform,
                                  remove_cc = False,
                                  seed=12, 
                                  fraction=0.7, 
                                  subset='Train')
    
    sample = image_dataset[1]
    print('image shape :',sample['image'].shape)
    print('keys :',sample.keys())
    print('targets :', sample['target'].shape)
    print('weights :',sample['weight'].shape)
    

    
