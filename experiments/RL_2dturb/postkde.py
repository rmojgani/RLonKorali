#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:23:31 2022

@author: rm99
"""
from PDE_KDE import myKDE
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
# directory = '/home/rm99/Mount/aa/flowControl_turb/flowControl_turb_1'
directory = '/home/rm99/Mount/aa/flowControl_turb/flowControl_turb'


num_file = 0
omega_M = np.zeros((64, 64))
for filename in os.listdir(directory):
    print(filename)

    if filename.endswith(".mat") and filename.startswith("N64_"):
        print(filename)
        mat_contents = sio.loadmat(filename)
        w1_hat = mat_contents['w_hat']
        omega = np.real(np.fft.ifft2(w1_hat))
        if omega.max() > 10:
            print(omega.max())
        omega_M = np.append(omega_M, omega)
        num_file += 1
    #     try:
    #         mat_contents = sio.loadmat(filename)
    #         w1_hat = mat_contents['w_hat']
    #         omega = np.real(np.fft.ifft2(w1_hat))
    #         omega_M = np.append(omega_M, omega)
    #         num_file += 1
    #         print(max(omega))
    #     except:
    #         print('none')
    # else:
    #     continue


# %%

CL = 0.2346

std_omega_DNS = 1# 6.0705


std_omega = 1#std_omega_DNS#np.std(omega_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M)
plt.semilogy(Vecpoints/std_omega, exp_log_kde, 'k', alpha=0.75, linewidth=2, label=r'Leith ($C_L=$'+str(CL)+r')')

# Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega)
# plt.semilogy(Vecpoints/std_omega, exp_log_kde)

plt.xlabel(r'$\omega / \sigma(\omega)$, $\sigma(\omega)$=' +
           str(np.round(std_omega, 2)))
plt.ylabel('PDF with '+str(num_file)+' samples')
plt.grid(which='major', linestyle='--',
         linewidth='1.0', color='black', alpha=0.25)
plt.grid(which='minor', linestyle='-',
         linewidth='0.5', color='red', alpha=0.25)

plt.ylim([1e-4, 1e-1])
#plt.xlim([-8, 8])

pdf_DNS = np.loadtxt('/home/rm99/Mount/aa/pdf_DNS.dat')
plt.semilogy(pdf_DNS[:,0]/std_omega_DNS, pdf_DNS[:,1], 'k', linewidth=4.0, alpha=0.25, label='DNS')
plt.legend(loc="upper left")
plt.title(r'Comparison of PDF of $\omega$')
