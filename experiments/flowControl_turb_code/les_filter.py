#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:45:12 2023

@author: rm99
https://github.com/omersan/ML-Parameterization/blob/3d3de76699651b0be62ad04bed96341231262ba1/Kraichnan_Turbulence/CNN/run_3/utils.py
"""
import numpy as np
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    **filters** the solution field  === keeping the size of the data same

    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc[:-1,:-1]