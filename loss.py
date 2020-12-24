#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:02:35 2020

@author: yanis
"""


import torch


def mse_loss(input, target):
    return ((input - target) ** 2).mean()

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()