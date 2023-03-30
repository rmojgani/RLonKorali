#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updated on March 30, 2023
copied from:
    https://github.com/envfluids/spectra/blob/a3dc4e232bd7e0ac52745d6a82c05bb705390977/util/Python/spectrum_angle_average_fun.py
updated on March 20, 2023
Created on Thu Jan 19 11:41:28 2023
"""
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
#%%
def spectrum_angle_average(w1_hat, NX, kx, Ksq):
    # if not len(kx):
    #     kx       = (2*np.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),
    #                                               np.arange((-NX/2+1),0,dtype=np.float64)))
    # if not len(Kabs):
    #     [Ky,Kx]  = np.meshgrid(kx,kx)
    #     Kx,Ky = np.meshgrid(kx,kx)
    #     Ksq      = (Kx**2 + Ky**2)
    #     Kabs     = np.sqrt(Ksq)


    # Angle averaged energy spectrum
    arr_len = int(0.5*np.sqrt((NX*NX + NX*NX)))
    n = arr_len + 1
    kplot = np.linspace(2,n,n)
    eplot = np.zeros(n)
    
    # Energy spectrum for all wavenumbers
    # es = abs(wor_hat)**2/Kabs

    es = np.power(np.abs(w1_hat),2)/NX/NX/Ksq
    es[0,0]=0;
    
    for k in range(0,n):
        eplot[k] = 0;
        ic = 0
        for j in range(0,NX):
            for i in range(0,NX):
                kk = np.sqrt(kx[i]**2 + kx[j]**2);
                if kk >= (k - 0.5) and kk < (k+0.5):
                    ic = ic+1
                    eplot[k] = eplot[k] + es[i,j]
        eplot[k] = eplot[k] / ic
       
    eplot=eplot/NX**4
    
    return kplot, eplot

#%%
def spectrum_angle_average_vec(es, Kabs, NX):#, kx):#, , invKsq):
    '''
    Angle averaged energy spectrum
    '''
    arr_len = int(0.5*NX)#
    #arr_len = int(np.ceil(0.5*np.sqrt((NX*NX + NX*NX))))
    kplot = np.array(range(arr_len))
    eplot = np.zeros(arr_len)
    
    # spectrum for all wavenumbers
    for k in kplot[1:]:
        unmask = np.logical_and(Kabs>=(k-0.5), Kabs<(k+0.5))
        eplot[k] = k*np.sum(es[unmask]);
     
    eplot[0]=es[0,0]
    
    kplot_str = '\sqrt{\kappa_x^2+\kappa_y^2}'
    return kplot, eplot, kplot_str
#%%
# def energy_es(w1_hat, Kabs, invKsq, NX ):
    
#     # Energy spectrum for all wavenumbers
#     # es = abs(wor_hat)**2/Kabs

#     # # Jan 2023
#     # es = np.power(np.abs(w1_hat),2)/NX/NX/Ksq
#     # es[0,0]=0;
    
#     # March 2023
#     # psi_hat = -w1_hat*(1/Kabs);
#     # psi_hat = -w1_hat/Kabs;
#     psi_hat = -invKsq*w1_hat
#     # es = 0.5*np.abs((np.conj(psi_hat).T)*w1_hat)/(NX**4);#/(NX**4); % MATLAB
#     es = 0.5*np.abs((np.conj(psi_hat))*w1_hat)/(NX**3);#/(NX**4); Question: NX**2?Rambod or NX**4? Yifei
    
#     return es
def energy_es(w1_hat, invKsq, NX , Kabs):
    
    psi_hat = -invKsq*w1_hat
    es = 0.5*np.abs((np.conj(psi_hat))*w1_hat)
    
    return es
#%%
def enstrophy_es(w1_hat, Kabs, NX ):
    # March 2023
    es = 0.5*np.abs((np.conj(w1_hat).T)*w1_hat)
    
    return es
#%%

