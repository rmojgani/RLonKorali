#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:56:02 2023

@author: rm99
"""
import numpy as np
from scipy.stats import multivariate_normal

def multivariat_fit(x,y):
    covxy = np.cov(x,y, rowvar=False)
    meanxy=np.mean(x),np.mean(y)
    rv = multivariate_normal(mean=meanxy, cov=covxy, allow_singular=False)
    xv, yv = np.meshgrid(np.linspace(x.min(),x.max(),100),
                         np.linspace(y.min(),y.max(),100), indexing='ij')
    pos = np.dstack((xv, yv))

    return xv, yv, rv, pos, meanxy, covxy