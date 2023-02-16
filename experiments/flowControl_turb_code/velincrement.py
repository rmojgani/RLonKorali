#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:49:02 2023

@author: rm99
"""
##%% increment-u δu
def realVelInc_fast(u,ax,r):
    # https://github.com/cselab/MARL_LES/blob/acd73f9c6c6195bda90209f1d7a8441993e547f4/plot_compute_structure.py#L90
    nx, ny, nz = np.shape(u)
    ret = np.zeros((nx,ny,nz,2))
    # Roll array elements along a given axis. Elements that roll
    # beyond the last position are re-introduced at the first.
    ret[:,:,:,0] = np.roll(u,  int(r), axis=ax) - u
    ret[:,:,:,1] = np.roll(u, -int(r), axis=ax) - u
    return ret