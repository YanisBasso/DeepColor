#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:33:14 2020

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

class GehlerDataset2(Dataset):
    """Regression dataset made from Gehler dataset"""
    def __init__(self, dir_path, target_path=None, remove_cc=None, seed=None, 
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
        
        self.target_path = target_path
        if target_path :
            self.targets = pd.read_csv(self.target_path) 
        
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
        img = np.array(cv2.imread(img_path, -1), dtype='uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255
        
        
        if self.target_path :
            target = self.targets.query('name == "{}"'.format(name[:-4]), inplace = False) 
            target = np.array(target.values[0][1:]).astype(np.float32)
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
    
    def get_name(self,idx):
        return Path(self.ids[idx]).stem
    

def verif_coord_mire(img,patches_coord):
    
    img_with_patche = img.copy()
    patches_coord[:,:,0] = patches_coord[:,:,0]*img.shape[1]
    patches_coord[:,:,1] = patches_coord[:,:,1]*img.shape[0]
    for patche in patches_coord :
        for x,y in zip(patche[:,0],patche[:,1]):
            cv2.circle(img_with_patche, (int(x), int(y)), 0, (0, 255, 0), 5) 
    plt.figure(figsize = (10,10))
    plt.imshow(img_with_patche)
    

def create_mire(targets):
    f = 10
    color_checker = np.zeros((4*f,6*f,3))
    for idx,colour in enumerate(targets):
        i = idx//6
        j = idx%6
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 0] = colour[0]
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 1] = colour[1]
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 2] = colour[2]
    if color_checker.max()>1:
        color_checker = color_checker/255
    plt.figure()
    plt.imshow(color_checker)
    plt.show()
        
        

class RemoveShadingTarget(object):
    
    def __call__(self,sample):
        target = sample['target']
        lum = np.sum(target,axis=1)
        target[:,0] = target[:,0]/lum
        target[:,1] = target[:,1]/lum
        target[:,2] = target[:,2]/lum
        sample['target'] = target[:,:2]
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
        
        image = srgb2linear(image)
        target = srgb2linear(target)
        
        r_shift = random.random()*(self.max_value - self.min_value) + self.min_value
        b_shift = random.random()*(self.max_value - self.min_value) + self.min_value
        
        image[:,:,0] = image[:,:,0]*r_shift
        image[:,:,2] = image[:,:,2]*b_shift
        
        target[:,0] = target[:,0]*r_shift
        target[:,2] = target[:,2]*b_shift
        
        image = linear2srgb(image)
        target = linear2srgb(target)

        return {'image':image,'target':target}
        
    
if __name__ == '__main__':
    
    
    gd = GehlerDataset2(dir_path = "/Users/yanis/GehlerDataset",
                       remove_cc = True,
                       seed=12, 
                       fraction=0.7, 
                       subset='Train', 
                       transform=None
                       )
    
    sample = gd[20]
    plt.figure()
    plt.imshow(sample['image'])
    create_mire(sample['target'])
    transform = RandomColorShift(0.6,1.4)
    
    sample = transform(sample)
    plt.figure()
    plt.imshow(sample['image'])
    create_mire(sample['target'])
    
    
    
    
    
    
