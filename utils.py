#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:14:41 2020

@author: yanis
"""
import numpy as np 

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