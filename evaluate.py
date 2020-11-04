#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:26:13 2020

@author: yanis
"""
import utils 

def evaluate(model,test_dataset,transform,bpath):
    
    since = time.time()
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    deltaE_errs = [] #store Dela E lab error. 
    deltatRG_errs = [] #store average distance in lab domain.
    
    for i,sample in tqdm(enumerate(test_dataset)):
        initialSample = sample
        sample = data_transforms_tests(sample)
        inputs = sample['image'].to(device)
        inputs = inputs.unsqueeze(0)
        y_true = sample['target'].data.cpu().numpy()
        outputs = model(inputs)
        outputs = outputs.data.cpu().numpy()
        y_pred = outputs[0]
        deltatRG_errs.append(deltaRG(y_pred,y_true))
        
        image_sRGB = initialSample['image']
        
        A  = y_pred.reshape((19,2))
        A = np.stack((rgb[:,0],rgb[:,1],1-rgb[:,0]-rgb[:,1]),axis=1)
        
        B = load_macbetch_colorspace_coord('./MacbethColorSpace/ciexyz_std.txt')
        B = xyz_std[:19]
        
        image_sRGB_corrected = homographyCorrection(image_sRGB,A,B)
        
        name = test_dataset.getName(i) + '.png'
        image_name = os.path.join('im_corrected',name)
        image_GT = io.imread(image_name)/255.0
    
        h,w,c = image_GT.shape
        image_sRGB_corrected_lab = rgb2lab(image_sRGB_corrected)
        image_GT_lab = rgb2lab(image_GT)
    
        labDiff = deltaE(image_GT_lab)
        deltaE_errs.append(labDiff)
        
        
        
        
        
        
    
    
    
    
    