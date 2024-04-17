#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:59:42 2023

@author: rm99
"""
# omega 
# psi
import numpy as np
import matplotlib.pyplot as plt
def hat_2_real(u_hat):
    return np.real(np.fft.ifft2(u_hat))
from les_filter import les_filter
#%%
# def Germanodiff(a, b, myfilter, c1=1, c2=1):
#     return c1*myfilter(a*b)-c2*myfilter(a)*myfilter(b)
#%%
def D_dir(u_hat, K_dir):
    Du_Ddir = 1j*K_dir*u_hat
    return Du_Ddir    

#%%
# dx = Lx/NX

# x        = np.linspace(0, Lx-dx, num=NX)
# kx       = (2*np.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),
#                                         np.arange((-NX/2+1),0,dtype=np.float64)
#                                         ))   
# [Y,X]    = np.meshgrid(x,x)
# [Ky,Kx]  = np.meshgrid(kx,kx)
# Ksq      = (Kx**2 + Ky**2)
# Kabs     = np.sqrt(Ksq)
# Ksq[0,0] = 1e12
# invKsq   = 1/Ksq
# Ksq[0,0] = 0
# invKsq[0,0] = 0
# #%%

# 
def smag_dynamic_cs(w1_hat, invKsq, NX, NY, Kx, Ky):
    psi_hat = -w1_hat*invKsq
    u1_hat = D_dir(psi_hat,Ky) # u_hat = (1j*Ky)*psi_hat
    v1_hat = -D_dir(psi_hat,Kx) # v_hat = -(1j*Kx)*psi_hat
    
    dudx_hat = D_dir(u1_hat,Kx)
    dudy_hat = D_dir(u1_hat,Ky)
    
    dvdx_hat = D_dir(v1_hat,Kx)
    dvdy_hat = D_dir(v1_hat,Ky)
    
    u = hat_2_real(u1_hat)
    v = hat_2_real(v1_hat)
    

    
    dudx = hat_2_real(dudx_hat)
    dvdy = hat_2_real(dvdy_hat)
    
    # plt.figure(figsize=(10,6), dpi=400)
    # plt.subplot(2,3,1)
    # plt.pcolor(dudx,cmap='bwr');plt.colorbar()
    # plt.subplot(2,3,2)
    # plt.pcolor(dvdy,cmap='bwr');plt.colorbar()
    # plt.subplot(2,3,3)
    # # plt.title(str(np.round(np.sum(np.sum(dudx+dvdy)),2)))
    
    # plt.title(r'$\epsilon$= {:.2E}'.format(np.sum(np.sum(dudx+dvdy)),2))
    
    # plt.pcolor(dudx+dvdy,cmap='bwr');plt.colorbar()
    # assert dudx==-dvdy
    #%%
    # https://github.com/omersan/ML-Parameterization/blob/3d3de76699651b0be62ad04bed96341231262ba1/Kraichnan_Turbulence/CNN/run_4/utils.py
    # S11 = hat_2_real(dudx_hat)
    # S12 = hat_2_real(0.5*(dudy_hat+dvdx_hat))
    # S22 = hat_2_real(dvdy_hat)
    # Sbar1 = (np.power(S11,2) + 2*np.power(S12,2) + np.power(S22,2)  )**0.5
    
    # # Sbar = hat_2_real(Sbar_hat)
    # plt.figure(figsize=(10,6), dpi=400)
    
    # plt.pcolor(Sbar1,cmap='bwr',vmin=0,vmax=10);plt.colorbar()
    
    
    dudx = hat_2_real(dudx_hat)
    dudy = hat_2_real(dudy_hat)
    dvdx = hat_2_real(dvdx_hat)    
    dvdy = hat_2_real(dvdy_hat)
    
    S11 = dudx
    S12 = 0.5*(dudy+dvdx)
    S22 = dvdy
    
    Sbar = ( S11**2 + 2*(S12**2) + S22**2 )**0.5
    
    # plt.figure(figsize=(10,6), dpi=400)
    # plt.pcolor(Sbar,cmap='bwr',vmin=0,vmax=10);plt.colorbar()
    
    
    grid_filter_width = 2*np.pi/(NX/2) #0.19634954084936207# 4*np.pi/NLES
    test_filter_width = 2*grid_filter_width
    
    NXC,NYC = int(0.5*NX), int(0.5*NY)
    myfilter = lambda u: les_filter(NX, NY, NXC, NYC, u)
    Germanodiff = lambda a,b,c1=1,c2=1: c1*myfilter(a*b)-c2*myfilter(a)*myfilter(b)
    
    
    M11 = Germanodiff(Sbar, S11, 2*(grid_filter_width**2), 2*(test_filter_width**2)) 
    M12 = Germanodiff(Sbar, S12, 2*(grid_filter_width**2), 2*(test_filter_width**2))
    M22 = Germanodiff(Sbar, S22, 2*(grid_filter_width**2), 2*(test_filter_width**2))
    
    L11 = Germanodiff(u, u)
    L12 = Germanodiff(u, v)
    L22 = Germanodiff(v, v)
    
    Lr11 = L11 - 0.5*(L11 + L22);
    Lr12 = L12;
    Lr22 = L22 - 0.5*(L11 + L22);
    
    aa = M11*Lr11+2.0*M12*Lr12+M22*Lr22 # This is equal to: M11*L11+2.0*M12*L12+M22*L22
    bb = (M11**2) + 2*(M12**2) + (M22**2)
    
    cs_Local_SM = aa/bb
    cs_local_SM_withClipping = 0.5*(cs_Local_SM+np.abs(cs_Local_SM))
    cs_Averaged_SM = aa.sum()/bb.sum()
    cs_Averaged_SM_withClipping = 0.5*(cs_Averaged_SM+np.abs(cs_Averaged_SM))
    
    return cs_Local_SM, cs_local_SM_withClipping, cs_Averaged_SM, cs_Averaged_SM_withClipping, aa
