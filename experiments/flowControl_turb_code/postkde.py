#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update : Dec 20
"""
from PDE_KDE import myKDE
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# directory = '/home/rm99/Mount/aa/flowControl_turb/flowControl_turb_1'
# directory = '.'
NLES = 16

sys.path.append('_result_vracer_C1_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost/')
directory = '_result_vracer_C1_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CS_nAgents16/CSpost/'
# METHOD = 'smagRL' # 'Leith' , 'Smag'
METHOD = 'smagRL'#'smag' # 'Leith' , 'Smag'

# sys.path.append('_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/')
# directory = '_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/'
# METHOD = 'smag0d17' # 'Leith' , 'Smag'

Fn = 4
CASENO = 1

CL = ''

num_file = 0
omega_M = []# np.zeros((NLES, NLES))
for filename in os.listdir(directory):

    if METHOD in filename and str(NLES) in filename  and filename.endswith('.mat'):
        print(filename)
        mat_contents = sio.loadmat(directory+filename)
        w1_hat = mat_contents['w_hat']
        omega = np.real(np.fft.ifft2(w1_hat))
        if omega.max() > 10:
            print(omega.max())
        omega_M = np.append(omega_M, omega)
        omega_M = np.append(omega_M, -omega)

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

Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M,BANDWIDTH=1.0)
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

# pdf_DNS = np.loadtxt('_init/Re20kf25/pdf_DNS_Re20kf25.dat')
# pdf_DNS = np.loadtxt('_init/Re20kf25/pdf_DNS_Re20kf25.dat')
# pdf_DNS1 = np.loadtxt('_init/pdf_DNS.dat')
pdf_DNS2 = np.loadtxt('_init/pdf_case01_FDNS.dat')

# plt.semilogy(pdf_DNS[:,0]/std_omega_DNS, pdf_DNS[:,1], 'k', linewidth=4.0, alpha=0.25, label='DNS')

# plt.semilogy(pdf_DNS1[:,0]/std_omega_DNS, pdf_DNS1[:,1], 'b', linewidth=4.0, alpha=0.25, label='DNS')
plt.semilogy(pdf_DNS2[:,0], pdf_DNS2[:,1], 'r', linewidth=4.0, alpha=0.25, label='DNS')

plt.legend(loc="upper left")
plt.title(r'Comparison of PDF of $\omega$, '+CASE)


filename_save = '2Dturb_N'+str(NLES)+'_'+METHOD+str(CL)

plt.savefig(filename_save+'_pdf.png', bbox_inches='tight', dpi=450)
stop
#%%
# pdf_dns_= np.hstack((pdf_DNS,pdf_DNS[:,0].reshape(-1,1)/std_omega_DNS))
# np.savetxt(filename_save+"_pdfdns.dat", pdf_dns_, delimiter='\t')


pdf_les_= np.stack((Vecpoints, exp_log_kde,Vecpoints/std_omega),axis=1)
np.savetxt(filename_save+"_pdf.dat", pdf_les_, delimiter='\t')
#%% Plot enstrophy
col1, col2 = 0, 0
ens_M = np.zeros((int(NLES/2)-1,num_file))
tke_M = np.zeros((int(NLES/2)-1,num_file))

for filename in os.listdir(directory):

    if filename.endswith("ens.out"):

        ens_i = np.loadtxt(directory+filename)[:-1,1]
        
        if len(filename)<=41:
            ens_dns = ens_i
            Kplot =  np.loadtxt(directory+filename)[:-1,0]
        else:
            ens_M[:,col1] = ens_i
            col1 +=1
    
    if filename.endswith("tke.out"):
        print(filename)

        tke_i = np.loadtxt(directory+filename)[:-1,1]
        
        if len(filename)<=41:
            tke_dns = tke_i
            Kplot =  np.loadtxt(directory+filename)[:-1,0]
        else:
            tke_M[:,col2] = tke_i
            col2 +=1
#%%
ens_ave = np.mean(ens_M,axis=1)
tke_ave = np.mean(tke_M,axis=1)
#%%
tke_dns =  np.loadtxt('_init/Re20kf4/'+'energy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,1]
Kplot_dns =  np.loadtxt('_init/Re20kf4/'+'energy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,0]

ens_dns =  np.loadtxt('_init/Re20kf4/'+'enstrophy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,1]

#%%
Kplot = np.array(range(int(NLES/2)-1))
#%%
kplot_str = '\kappa_{x}'

plt.figure(figsize=(10,14))

# Energy 
plt.subplot(3,2,3)
plt.loglog(Kplot,tke_ave,'k')
plt.plot([Fn,Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot_dns,tke_dns,':k', alpha=0.5, linewidth=4)
plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')
plt.xlim([1,1e3])
plt.ylim([1e-4,1e0])


# Enstrophy
plt.subplot(3,2,4)
plt.loglog(Kplot, ens_ave,'k')
plt.plot([Fn,Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot_dns, ens_dns,':k', alpha=0.5, linewidth=4)
plt.title(rf'$\varepsilon({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')
plt.xlim([1,1e2])
#plt.ylim([1e-5,1e0])
plt.ylim([1e-4,1e1])

for i in [3,4]:
    plt.subplot(3,2,i)

    plt.grid(which='major', linestyle='--',
             linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
             linewidth='0.5', color='red', alpha=0.25)
    
    
plt.savefig(filename_save+'_spec.png', bbox_inches='tight', dpi=450)
#%%
stop
tke_ave_=np.stack((Kplot,tke_ave)).T
np.savetxt(filename_save+"_tkeave.dat", tke_ave_,delimiter='\t')

ens_ave_=np.stack((Kplot,ens_ave)).T
np.savetxt(filename_save+"_ensave.dat", ens_ave_,delimiter='\t')

# ens_ave_=np.stack((Kplot,tke_dns)).T
# np.savetxt(filename_save+"tkedns.dat", ens_ave_,delimiter='\t')


# ens_ave_=np.stack((Kplot,ens_dns)).T
# np.savetxt(filename_save+"_ensdns.dat", ens_ave_,delimiter='\t')
## Plot loss types
plt.figure(figsize=(10,14))

# Enstrophy
plt.subplot(3,2,1)
plt.plot( Kplot    * ens_dns , 'k')
plt.plot((Kplot**2)*(ens_dns),'--r')
plt.plot((Kplot**3)*(ens_dns),'.-b')
plt.plot((np.log(ens_dns)),'--*k')
plt.title(rf'$\varepsilon({kplot_str})$')

plt.subplot(3,2,2)
plt.plot(Kplot*(ens_dns-ens_ave), 'k')
plt.plot((Kplot**2)*(ens_dns-ens_ave),'--r')
plt.plot((Kplot**3)*(ens_dns-ens_ave),'.-b')
plt.plot((np.log(ens_dns)-np.log(ens_ave)),'--*k')
plt.title(rf'$\varepsilon({kplot_str})$')

# Energy
plt.subplot(3,2,3)
plt.plot( Kplot    * tke_dns , 'k')
plt.plot((Kplot**2)*(tke_dns),'--r')
plt.plot((Kplot**3)*(tke_dns),'.-b')
plt.plot((np.log(tke_dns)),'--*k')
plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')

plt.subplot(3,2,4)
plt.plot(Kplot*(tke_dns-tke_ave), 'k')
plt.plot((Kplot**2)*(tke_dns-tke_ave),'--r')
plt.plot((Kplot**3)*(tke_dns-tke_ave),'.-b')
plt.plot((np.log(tke_dns)-np.log(tke_ave)),'--*k')
plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')

for i in [1,2,3,4]:
    plt.subplot(3,2,i)

    plt.grid(which='major', linestyle='--',
             linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
             linewidth='0.5', color='red', alpha=0.25)
#%% PDF of veRL
num_file = 0
veRL_M = []
omega_M = []# np.zeros((NLES, NLES))
psi_M = []

for filename in os.listdir(directory):

    if METHOD in filename and str(NLES) in filename  and filename.endswith('.mat'):
        print(filename)
        mat_contents = sio.loadmat(directory+filename)
        veRL = mat_contents['veRL']
        w1_hat = mat_contents['w_hat']
        psi_hat = mat_contents['psi_hat']

        omega = np.real(np.fft.ifft2(w1_hat))
        psi = np.real(np.fft.ifft2(psi_hat))

        omega_M = np.append(omega_M, omega)
        veRL_M = np.append(veRL_M, veRL)
        psi_M = np.append(psi_M, psi)

#%%
meanveRL = veRL_M.mean()
Vecpoints, exp_log_kde, log_kde, kde = myKDE(veRL_M,BANDWIDTH=0.1)
plt.semilogy(Vecpoints, exp_log_kde, 'k', alpha=0.75, linewidth=1.0, label=METHOD+r'($C=$'+str(CL)+r')')
plt.semilogy([meanveRL,meanveRL],[min(exp_log_kde),max(exp_log_kde)],'-r')
#%%
from scipy.stats import multivariate_normal

def multivariat_fit(x,y):
    covxy = np.cov(x,y, rowvar=False)
    meanxy=np.mean(x),np.mean(y)
    rv = multivariate_normal(mean=meanxy, cov=covxy, allow_singular=False)
    xv, yv = np.meshgrid(np.linspace(x.min(),x.max(),100),
                         np.linspace(y.min(),y.max(),100), indexing='ij')
    pos = np.dstack((xv, yv))

    return xv, yv, rv, pos, meanxy, covxy
#%%
Lx = 2*np.pi
NX = 16
kx       = (2*np.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),
                                        np.arange((-NX/2+1),0,dtype=np.float64)
                                        ))
[Ky,Kx]  = np.meshgrid(kx,kx)
w1x_hat = -(1j*Kx)*w1_hat
w1y_hat = (1j*Ky)*w1_hat


#%% Plot Dis
xplot, xplot_str = veRL_M, '$C_S^2$'
# yplot, yplot_str= omega_M, '$\omega$'
yplot, yplot_str = psi_M, '$\psi$'
CS2 = 0.17**2
CS2EKI = 0.1**2

xv, yv, rv, pos, meanxy, covxy = multivariat_fit(xplot,yplot)
plt.plot(xplot, yplot,'.k',alpha=0.05, markersize=2)
plt.contour(xv, yv, rv.pdf(pos))
plt.scatter(meanxy[0],meanxy[1], marker="+", color='red',s=100,linewidths=2)
plt.scatter(CS2,meanxy[1], marker="o", color='green',s=50,linewidths=0.15)
plt.scatter(CS2EKI,meanxy[1], marker="*", color='green',s=50,linewidths=0.15)

plt.xlabel(xplot_str)
plt.ylabel(yplot_str)
plt.grid(color='gray', linestyle='dashed')

plt.savefig(filename+'dis.png', bbox_inches='tight', dpi=450)
#%%
lambda_, v = np.linalg.eig(covxy)
lambda_ = np.sqrt(lambda_)
from matplotlib.patches import Ellipse
fig, ax = plt.subplots()#subplot_kw={'aspect': 'equal'})

for j in range(4):
  ell = Ellipse(xy=meanxy,
                width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                angle=np.rad2deg(np.arccos(v[0, 0])),
                alpha=0.05,
                facecolor='blue',
                #facecolor='none',
                edgecolor='blue',
                linestyle='-')


  #ell.set_facecolor('none')
  ax.add_artist(ell)



#plt.plot(xplot, yplot,'.k',alpha=0.05, markersize=0.5)
plt.scatter(xplot, yplot, marker=".", color='black',s=0.1,alpha=0.25)

plt.scatter(meanxy[0],meanxy[1], marker="+", color='blue',s=100)
plt.scatter(CS2,meanxy[1], marker="s", color='red',s=25,linewidths=0.15)
plt.scatter(CS2EKI,meanxy[1], marker="*", color='black',s=50,linewidths=0.15)

plt.xlabel(xplot_str)
plt.ylabel(yplot_str)
plt.grid(color='gray', linestyle='dashed')
plt.savefig(filename+'dis.png', bbox_inches='tight', dpi=450)
