#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:14:41 2020

@author: yanis
"""
import numpy as np 
import malplotlib.pyplot as plt 
from numpy.linalg import inv 

def px_srgb2linear(x):
        if x <= 0.0:
            return 0.0
        elif x>=1: 
            return 1.0
        elif x<0.04045:
            return x / 12.92
        else:
            return ((x + 0.055) / 1.055) ** 2.4


vsrgb2linear = np.vectorize(px_srgb2linear)

def px_linear2srgb(x):
    if x <= 0.0:
        return 0.0
    elif x >= 1:
        return 1.0
    elif x < 0.0031308:
        return x * 12.92
    else:
        return x ** (1 / 2.4) * 1.055 - 0.055
    
vlinear2srgb = np.vectorize(px_linear2srgb)

def srgb2linear(image):
    
    shape = image.shape
    image = image.flatten()
    image = vsrgb2linear(image)
    image = image.reshape(shape)
    
    return image

def linear2srgb(image):
    
    shape = image.shape
    image = image.flatten()
    image = vlinear2srgb(image)
    image = image.reshape(shape)
    
    return image

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
    
##################################
# Homography correction 
##################################

def als_rg(A,B,max_iter = 50,tol = 10**(-20)):
    """Compute homography correction matrix by ALS 
    
    :A: input matrix of shape (N,3) where N is the number of patch considered
    :B: target matrix 
    :max_iter: maximum iteration number in the optimisation process
    :tol: tolerance to stop the optimisation process 
    
    :return errs: residual error 
    :return D: shading matrice of shape (N,N)
    :return M: homographie matrice 
    """
    n_it = 0
    errs = [np.inf]
    d_err = np.inf 
    
    P = A
    Pk = A
    Q = B
    
    while (n_it < max_iter and d_err > tol):
        n_it += 1
        #1 - update D
        d = np.diagonal(np.dot(Pk,Q.transpose())/ np.dot(Pk,Pk.transpose()))
        D = np.diag(d)
        
        #2 - update M 
        #Moore-Penrose pseudo-inverse of D*P
        P_d = np.dot(D,P)
        N = np.dot(inv(np.dot(P_d.transpose(),P_d)),P_d.transpose())
        
        M = np.dot(N,Q)
        
        #3 - update Pk
        Pk = np.dot(P,M)
        
        #Calculate the mean error 
        Diff = (np.dot(D,Pk) - Q)**2
        err = np.mean(np.mean(Diff))
        d_err = errs[-1] - err 
        errs.append(err)
    
    return errs,D,M

