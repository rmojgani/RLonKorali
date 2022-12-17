#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update : Dec 14
"""
from PDE_KDE import myKDE
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
# directory = '/home/rm99/Mount/aa/flowControl_turb/flowControl_turb_1'
directory = '.'

NLES = 128
Fn = 25
CASENO = 4

METHOD = 'Smag' # 'Leith' , 'Smag'
CL = 0.17

num_file = 0
omega_M = []# np.zeros((NLES, NLES))
for filename in os.listdir(directory):
    print(filename)

    if filename.endswith(".mat") and filename.startswith("N"+str(NLES)+"_"):
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
if CASENO == 4:
    CASE = r'Case 4 ($Re = 20x10^3, k_f=25$)'
elif CASENO == 1:
    CASE = r'Case 1 ($Re = 20x10^3, k_f=4$)'


if CASENO == 4:
    std_omega_DNS = 12.85
if CASENO == 1:
    std_omega_DNS = 6.0705


std_omega = std_omega_DNS#np.std(omega_M)

Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M)
plt.semilogy(Vecpoints/std_omega, exp_log_kde, 'k', alpha=0.75, linewidth=2, label=METHOD+r'($C=$'+str(CL)+r')')

# Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega)
# plt.semilogy(Vecpoints/std_omega, exp_log_kde)

plt.xlabel(r'$\omega / \sigma(\omega)$, $\sigma(\omega)$=' +
           str(np.round(std_omega, 2)))
plt.ylabel('PDF with '+str(num_file)+' samples')
plt.grid(which='major', linestyle='--',
         linewidth='1.0', color='black', alpha=0.25)
plt.grid(which='minor', linestyle='-',
         linewidth='0.5', color='red', alpha=0.25)

plt.ylim([1e-6, 1e-1])
#plt.xlim([-8, 8])

pdf_DNS = np.loadtxt('pdf_DNS_Re20kf25.dat')
plt.semilogy(pdf_DNS[:,0]/std_omega_DNS, pdf_DNS[:,1], 'k', linewidth=4.0, alpha=0.25, label='DNS')
plt.legend(loc="upper left")
plt.title(r'Comparison of PDF of $\omega$, '+CASE)


filename_save = '2Dturb_N'+str(NLES)+'_'+METHOD+str(CL)

plt.savefig(filename_save+'_pdf.png', bbox_inches='tight', dpi=450)
#%%
pdf_dns_= np.hstack((pdf_DNS,pdf_DNS[:,0].reshape(-1,1)/std_omega_DNS))
np.savetxt(filename_save+"_pdfdns.txt", pdf_dns_, delimiter='\t')


pdf_les_= np.stack((Vecpoints, exp_log_kde,Vecpoints/std_omega),axis=1)
np.savetxt(filename_save+"_pdf.txt", pdf_les_, delimiter='\t')
#%% Plot enstrophy
col1, col2 = 0, 0
ens_M = np.zeros((int(NLES/2),num_file))
tke_M = np.zeros((int(NLES/2),num_file))

for filename in os.listdir(directory):

    if filename.endswith("ens.out"):
        print(filename)

        ens_i = np.loadtxt(filename)[:-1,1]
        
        if len(filename)<=21:
            ens_dns = ens_i
            Kplot =  np.loadtxt(filename)[:-1,0]
        else:
            ens_M[:,col1] = ens_i
            col1 +=1
    
    if filename.endswith("tke.out"):
        print(filename)

        tke_i = np.loadtxt(filename)[:-1,1]
        
        if len(filename)<=21:
            tke_dns = tke_i
            Kplot =  np.loadtxt(filename)[:-1,0]
        else:
            tke_M[:,col2] = tke_i
            col2 +=1
#%%
ens_ave = np.mean(ens_M,axis=1)
tke_ave = np.mean(tke_M,axis=1)
#%%
kplot_str = '\kappa_{x}'

plt.figure(figsize=(10,14))

# Energy 
plt.subplot(3,2,3)
plt.loglog(Kplot,tke_ave,'k')
plt.plot([Fn,Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot,tke_dns,':k', alpha=0.5, linewidth=4)
plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')
plt.xlim([1,1e3])
plt.ylim([1e-4,1e2])


# Enstrophy
plt.subplot(3,2,4)
plt.loglog(Kplot, ens_ave,'k')
plt.plot([Fn,Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot, ens_dns,':k', alpha=0.5, linewidth=4)
plt.title(rf'$\varepsilon({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')
plt.xlim([1,1e2])
#plt.ylim([1e-5,1e0])
plt.ylim([1e-4,1e-1])

for i in [3,4]:
    plt.subplot(3,2,i)

    plt.grid(which='major', linestyle='--',
             linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
             linewidth='0.5', color='red', alpha=0.25)
    
    
plt.savefig(filename_save+'_spec.png', bbox_inches='tight', dpi=450)
#%%
tke_ave_=np.stack((Kplot,tke_ave)).T
np.savetxt(filename_save+"_tkeave.txt", tke_ave_,delimiter='\t')

ens_ave_=np.stack((Kplot,ens_ave)).T
np.savetxt(filename_save+"_ensave.txt", ens_ave_,delimiter='\t')


ens_ave_=np.stack((Kplot,tke_dns)).T
np.savetxt(filename_save+"tkedns.txt", ens_ave_,delimiter='\t')


ens_ave_=np.stack((Kplot,ens_dns)).T
np.savetxt(filename_save+"_ensdns.txt", ens_ave_,delimiter='\t')

