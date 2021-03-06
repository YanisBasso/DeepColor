#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:14:41 2020

@author: yanis
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import inv 
from skimage import io,color 
from colorCheckerValues import MACBETH_COLOR,MACBETH_COLOR_HEX
import torch 

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


    
##################################
# Color space transformation
##################################
    
def xyz2rgb(xyz):
    M = np.array([[ 3.2404542, -0.9692660,  0.0556434],
                  [-1.5371385,  1.8760108, -0.2040259],
                  [-0.4985314,  0.0415560,  1.0572252]])
    rgb = np.dot(xyz,M)
    return rgb

def rgb2lab(rgb):
    return color.rgb2lab(rgb)

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

def homographyCorrection(image,A,B):
    
    h,w,_ = image.shape
    image = image.ravel()
    image = vsrgb2linear(image)
    image = image.reshape((h*w,3))
    
    _,D,H = als_rg(A,B)
    
    xyzFlat = np.dot(image,H)
    
    rgb_corrected = xyz2rgb(xyzFlat)
    rgb_corrected = vlinear2srgb(rgb_corrected.flatten())
    rgb_corrected = np.reshape(rgb_corrected,(h,w,3))
    
    return rgb_corrected

##################################
# Metrics
##################################


def deltaRG(y_pred,y_true):
  """
  Compute the median distance in the rg domain between predictions and 
  target values
  
  :y_pred: network prediction vector 
  :y_true: target vector computed on color checker 
  :return: deltaRG
  """
  deltaRG = 0
  for j in range(19):
    deltaRG += np.sqrt((y_pred[2*j] - y_true[2*j])**2 + (y_pred[2*j+1] - y_true[2*j+1])**2)
  return deltaRG/19

def deltaRG_per_patch(y_pred,y_true):
    deltaRG = np.zeros((19))
    for i in range(19):
        deltaRG[i] = np.sum((y_pred[2*i:2*(i+1)] - y_true[2*i:2*(i+1)])**2)
        deltaRG[i] = np.sqrt(deltaRG[i])
    return deltaRG
        

def deltaIll(y_pred,y_true):
    
    y_pred = y_pred.reshape((19,2))
    y_true = y_true.reshape((19,2))
    errs = np.zeros((24))
    for i in range(19):
        a = y_pred[i]
        b = y_pred[i]
        c = a.dot(b)/np.sqrt(a.dot(a)*b.dot(b))
        errs[i] = np.arccos(c)
    return np.mean(errs)

def deltaE(I,J):
    """
    Delta E error 
    """
    dE = np.sqrt(np.sum((I - J)**2,axis = 2))
    dE = np.mean(dE)
    return dE 


##################################
# loading function
##################################

def load_macbetch_colorspace_coord(path):
    """Load xyz stand value from MacBeth color checker 
    """
    colorMatrix =  np.zeros((24,3))
    
    with open(path) as fp:
        lines = fp.readlines()
        for i,line in enumerate(lines):
            a,b,c = line.split()
            colorMatrix[i] = np.array([[np.float32(a),np.float32(b),np.float32(c)]])
    colorMatrix = colorMatrix.reshape((6,4,3))
    colorMatrix = colorMatrix.swapaxes(0,1)
    colorMatrix = colorMatrix.reshape((24,3))
    return colorMatrix


##################################
# Visualisation tools 
##################################

def visuPredGap_rgDomain(y_pred,y_truth,ax=None,**kwargs):
  '''
  Plot prediction and target value on rg domain
  '''
  ax = ax or plt.gca()
  scaleRange = np.linspace(0,1,5)
  for j in range(19):
      ax.scatter(y_pred[2*j],y_pred[2*j+1],c=MACBETH_COLOR_HEX[j])
      ax.scatter(y_truth[2*j],y_truth[2*j+1],c=MACBETH_COLOR_HEX[j],marker='x',s = 100)

  ax.axis(xmin=0,xmax=1)
  ax.set_xlabel('r channel',fontsize=20)
  ax.set_ylabel('g channel',fontsize=20)
  ax.set_xticks(scaleRange)
  ax.set_yticks(scaleRange)
  ax.tick_params(axis="x", labelsize=16)
  ax.tick_params(axis="y", labelsize=16)
  ax.set_facecolor('#000000')
  return ax

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
        rows = imageCount//columns + 1
    else :
        rows = 1
    for i in range(imageCount):
        img = tensor.numpy()[i].transpose((1, 2, 0))
        i+=1
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    
def unormalize(tensor,mean,std):
    for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    return tensor

def reverse_transform(tensor):
  tensor = unormalize(tensor,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  image = tensor.numpy()
  image = image.transpose((1, 2, 0))
  return image

def visualizePrediction(model,dataloader):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()

  sample  =  next(iter(dataloader))
  inputs = sample['image'].to(device)
  targets = sample['target'].data
  outputs = model(inputs)

  if device == torch.device("cuda:0"):
    targets = targets.cpu()
    outputs = outputs.cpu().detach()
    inputs = inputs.cpu()
  
  targets.detach().numpy()
  outputs.detach().numpy()
  
  nb_image = min(4,inputs.shape[0])
  for i in range(nb_image):
    image_plot = reverse_transform(inputs[i].cpu())
    y_pred = outputs[i]
    y_true = targets[i]
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ax1.imshow(image_plot)
    ax1.axis('off')
    visuPredGap_rgDomain(y_pred,y_true,ax2)

def showResult(model,dataset,transform,ind,**kwargs):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    sample = dataset[ind]
    initialSample = sample
    sample = transform(sample)
    inputs = sample['image'].to(device)
    inputs = inputs.unsqueeze(0)
    y_true = sample['target'].data.cpu().numpy()
    outputs = model(inputs)
    outputs = outputs.data.cpu().numpy()
    y_pred = outputs[0]
    
    image_sRGB = initialSample['image']
    
    A  = y_pred.reshape((19,2))
    A = np.stack((A[:,0],A[:,1],1-A[:,0]-A[:,1]),axis=1)
    
    B = load_macbetch_colorspace_coord('./MacbethColorSpace/ciexyz_std.txt')
    B = B[:19]
    
    image_sRGB_corrected = homographyCorrection(image_sRGB,A,B)
    
    name = dataset.getName(ind) + '.png'
    image_name = os.path.join('im_corrected',name)
    image_GT = io.imread(image_name)/255.0

    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1,ncols=4,figsize=(25,5),**kwargs)
    ax1.imshow(image_sRGB)
    ax1.set_title('Input Image',**kwargs)
    ax1.axis('off')
    ax2.imshow(image_sRGB_corrected)
    ax2.set_title('Our Corrected Image',**kwargs)
    ax2.axis('off')
    ax3.imshow(image_GT)
    ax3.set_title('CC Corrected Image',**kwargs)
    ax3.axis('off')
    visuPredGap_rgDomain(y_pred,y_true,ax4)
    ax4.set_title('RG Histogramm',**kwargs)
  
def gentab(err,title):
    """
    Generate an error table from an error distribution 
    Compute and print the mean error, the median error, the 95% quantile and 
    the max error encounter during evaluation test. 
    
    :param err: list of errors
    :param title: String. Name of the metric evaluated
    """
    n = len(title)
    if type(err) == np.ndarray:
        err = err.tolist()
    err_mean = np.mean(err)
    err_median = np.quantile(err,0.5)
    err_qu95 = np.quantile(err,0.95)
    err_max = np.max(err)
    print('\n')
    print('_'*(45+n))
    print('   {}       Mean   Median    95pct      max'.format(' '*n))
    print('{4}   :  {0:8.2f} {1:8.2f} {2:8.2f} {3:8.2f}'.format(err_mean,err_median,err_qu95,err_max,title))
    print('_'*(45+n))
    
def add_mire(y_pred, image, f=10, padd= 5, **kwargs):
    h,w,_ = image.shape
    colors = y_pred.reshape((24,3))
    color_checker = np.zeros((4*f,6*f,3))
    for idx,color in enumerate(colors):
        i = idx//6
        j = idx%6
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 0] = color[0]
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 1] = color[1]
        color_checker[i*f:(i+1)*f,j*f:(j+1)*f, 2] = color[2]
    color_checker = color_checker
    image[h-4*f-padd:h-padd,padd:6*f+padd] = color_checker
    return image


def ellipse_parameter(pixel_info):
    """
    Calculates the parameters of the covariance error ellipse associated at 
    each color patche
    
    :param pixel_info: pixels values of a given region of interest 
    :return means: mean color of the ROI 
    :return largest_eigenval: largest eigen value of the covariance matrix
    :return smallest_eigenval: smallest eigen value of the covariance matrix
    :return angle: angle between the x-axis and the direction of the eigen 
        vector associated to the largest eigen value
    """
    #Covariance matrix
    cov = np.cov(pixel_info.transpose())
    #Calculate the eigenvectors and eigenvalues
    a = np.linalg.eig(cov)
    eigenval = a[0]
    eigenvec = a[1]
    #Get the index of the lagest and smallest eigenvalue 
    largest_eigenval_ind = np.argmax(eigenval)
    largest_eigenval = np.max(eigenval)
    smallest_eigenval = np.min(eigenval)
    #Get the corresponding eigenvector 
    largest_eigenvect = eigenvec[:,largest_eigenval_ind]
    #Calculate the angle between x-axis and largest eigenvector
    angle = np.arctan2(largest_eigenvect[1],largest_eigenvect[0])
    if angle < 0:
        angle = angle + 2*np.pi
    
    #get mean
    means = np.mean(pixel_info, axis = 0)
    return means,largest_eigenval,smallest_eigenval,angle

def add_ellipse(mean,lamda1,lambda2,phi,thresh = 5.991,color = '#cccccc',ax=None):
    """
    Add a confidence ellipse from a given 2D gaussian distribution 
    
    :param mean: Mean value 
    :param lambda1: largest eigen value of the covariance matrix
    :param lambda2: smallest eigen value of the covaraince matrix 
    :param phi: angle between the x-axis and the direction of the eigen vector 
        associated to the largest eigen value
        
    :param thresh: ellipse threshold 
    :param color: Plotting color 
    :param ax: matplolib ax where the ellipse will be added
    """
    from matplotlib.patches import Ellipse
    
    ax = ax or plt.gca()
    center = (mean[0],mean[1])
    w = 2*np.sqrt(thresh*lamda1)
    h = 2*np.sqrt(thresh*lambda2)
    phi = phi*180/np.pi
    ellipse = Ellipse(center,
            width=w,
            height=h,
            angle=phi,
            edgecolor='black',
            facecolor=color,
            linewidth = 1.5,
            alpha=0.7)
    ax.add_patch(ellipse)
    

