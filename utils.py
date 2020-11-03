#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:14:41 2020

@author: yanis
"""
import numpy as np 
import malplotlib.pyplot as plt 

def srgb_to_linear(x):
        if x <= 0.0:
            return 0.0
        elif x>=1: 
            return 1.0
        elif x<0.04045:
            return x / 12.92
        else:
            return ((x + 0.055) / 1.055) ** 2.4


vsrgb_to_linear = np.vectorize(srgb_to_linear)

def showImgFromTensor(tensor):
    """
    Show images from a batch 
    :tensor: Pytorch tensor whose shape (N,C,H,W) with N number of images
    """
    ntensor = tensor.numpy()
    imageCount = ntensor.shape[0]
    fig = plt.figure(figsize = (20,10))
    fig.suptitle('Examples of Pre-processed Images',fontsize=20)
    columns = 4
    if imageCount > 4 : 
        rows = imageCount//columns
    else :
        rows = 1
    for i in range(imageCount):
        img = tensor.numpy()[i].transpose((1, 2, 0))
        i+=1
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

