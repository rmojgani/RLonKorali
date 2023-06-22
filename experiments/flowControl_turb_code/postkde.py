#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update : Dec 20
"""
import os
import sys
import re
from natsort import natsorted, ns
try:
    # os.chdir('/home/rm99/Mount/jetstream_volume/docker/RLonKoraliMA/experiments/flowControl_turb_code')
    os.chdir('/home/rm99/Mount/jetstream_volume/docker/RLonKoraliMA/experiments/flowControl_turb_code')  
except:
    print('')

import numpy as np
import scipy.io as sio

from PDE_KDE import myKDE
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
#%%
NumRLSteps = 1e3
EPERU = 1.0
SPIN_UP = 50000
NUM_DATA = 100#900#0
#%%
NLES = 64
nAgents = 16
CASENO = 2
#directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CL_nAgents'+str(nAgents)+'/CLpost/'
#directory = '_result_vracer_C1_N32_R_z1_State_enstrophy_Action_CL_nAgents4_CREWARD0/CLpost/'
#directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_invariantlocalandglobalz_Action_CL_nAgents'+str(nAgents)+'_CREWARD1_Tspin10000.0_Thor10000.0/CLpost/'
directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents'+str(nAgents)+'_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps'+str(NumRLSteps)+'_EPERU'+str(EPERU)+'/CLpost/'

# NLES = 32
# nAgents = 16
# CASENO = 1; 
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CL_nAgents'+str(nAgents)+'/CLpost/'
'''
NLES = 128
nAgents = 144
CASENO = 4;
directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CL_nAgents'+str(nAgents)+'/CLpost/'
'''
# sys.path.append(directory)
# METHOD = 'smagRL' # 'Leith' , 'Smag'
METHOD = 'smagRL'#'smag' # 'Leith' , 'Smag'

# sys.path.append('_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/')
# directory = '_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/'
# METHOD = 'smag0d17' # 'Leith' , 'Smag'

CL = ''
num_file = 0
omega_M = []# np.zeros((NLES, NLES))
#for filename in sorted(os.listdir(directory)):
for filename in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):

    if METHOD in filename and str(NLES) in filename and filename.endswith('.mat'):
        
        numbers_in_file = re.findall(r'\d+', filename)
        file_numstep = int(numbers_in_file[-1])

        if file_numstep>SPIN_UP:        
        # if file_numstep>50000 and file_numstep%20000==1:
            print(str(num_file), filename)

            mat_contents = sio.loadmat(directory+filename)
            w1_hat = mat_contents['w_hat']
            omega = np.real(np.fft.ifft2(w1_hat))
            #if omega.max() > 10:
            #    print(omega.max())
            omega_M = np.append(omega_M, omega)
            # omega_M = np.append(omega_M, -omega)
    
            num_file += 1
            if num_file==NUM_DATA: break
#%%
if CASENO == 4:
    Fn = 25;
    CASE_str = r'Case 4 ($Re = 20x10^3, k_f=25$)'+r', $N_{LES}=$'+str(NLES)+'$, n_{MARL}=$'+str(nAgents)
    pdf_DNS1 = np.loadtxt('_init/Re20kf25/pdf_DNS_Re20kf25.dat')
    pdf_DNS2 = np.loadtxt('_init/Re20kf25/pdf_DNS_Re20kf25.dat')
    std_omega_DNS = 12.85
    XMIN, XMAX = -8, 8
    YMIN, YMAX = 1e-5, 1e-1

elif CASENO == 1:
    Fn = 4;
    CASE_str = r'Case 1 ($Re = 20x10^3, k_f=4$)'+r', $N_{LES}=$'+str(NLES)+'$, n_{MARL}=$'+str(nAgents)
    pdf_DNS1 = np.loadtxt('_init/pdf_DNS.dat')
    pdf_DNS2 = np.loadtxt('_init/pdf_case01_FDNS.dat')
    std_omega_DNS = 6.0705
    XMIN, XMAX = -7, 7
    YMIN, YMAX = 1e-5, 1e-1

elif CASENO == 2:
    Fn = 4;
    CASE_str = r'Case 2 ($Re = 20x10^3, k_f=4, \beta=20$)'+r', $N_{LES}=$'+str(NLES)+'$, n_{MARL}=$'+str(nAgents)
    #pdf_DNS1 = np.loadtxt('_init/pdf_DNS.dat')  # to be updated 
    pdf_DNS1 = np.loadtxt('_init/pdf_case02_FDNS_shading.dat')
    pdf_DNS2 = np.loadtxt('_init/pdf_case02_FDNS.dat') 
    std_omega_DNS = 10.378936
    XMIN, XMAX = -5, 5
    YMIN, YMAX = 1e-7, 1e-1
    
std_omega = std_omega_DNS#np.std(omega_M)
#%%
omega_M_2D = omega_M.reshape(NLES*NLES, -1)
#%%
from PDE_KDE import mybandwidth_scott
plt.figure(figsize=(6,4), dpi=450)

BANDWIDTH = mybandwidth_scott(omega_M_2D)*1#*10
Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M.reshape(-1,1), BANDWIDTH=BANDWIDTH, padding=2)
plt.semilogy(Vecpoints/std_omega, exp_log_kde, 'k', alpha=1.0, linewidth=2, label=METHOD+r'($C=$'+str(CL)+r')')

num_line = 5
div = int(num_file/num_line)#67
for icount in range(int(len(omega_M_2D.T)/div)):
    print('line no:', icount+1)
    omega_M_now = omega_M_2D[:,icount:icount+div]
    Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M_now.reshape(-1,1), BANDWIDTH=BANDWIDTH, padding=2) 
    plt.semilogy(Vecpoints/std_omega, exp_log_kde, 'k', alpha=0.1, linewidth=1, label=METHOD+r'($C=$'+str(CL)+r')')

if CASENO == 1:
    plt.semilogy(pdf_DNS2[:,0], pdf_DNS2[:,1], 'r', linewidth=4.0, alpha=0.5, label='DNS')
    plt.semilogy(-pdf_DNS2[:,0], pdf_DNS2[:,1], 'r', linewidth=4.0, alpha=0.5, label='DNS')

elif CASENO == 4:
    plt.semilogy(pdf_DNS1[:,0]/std_omega_DNS, pdf_DNS1[:,1], 'b', linewidth=4.0, alpha=0.25, label='DNS')
elif CASENO == 2:
    # To be updated
    plt.semilogy(pdf_DNS2[:,0]/std_omega_DNS, pdf_DNS2[:,1], 'c', linewidth=4.0, alpha=0.25, label='DNS')
    plt.semilogy(pdf_DNS1[:,0]/std_omega_DNS, pdf_DNS1[:,2], 'c', linewidth=4.0, alpha=0.25, label='DNS')
    plt.fill_between(pdf_DNS1[:,0]/std_omega_DNS,
                 pdf_DNS1[:,1],
                 pdf_DNS1[:,3],
                 color='c', alpha=0.1)

# plt.legend(loc="upper left")
plt.title(CASE_str+', bw='+ str(np.round(BANDWIDTH,2)))

plt.xlabel(r'$\omega / \sigma(\omega)$, $\sigma(\omega)$=' +
           str(np.round(std_omega, 2)))
plt.ylabel(r'$\mathcal{P}\left(\omega\right)$, w. '+str(num_file)+' samples')

plt.grid(which='major', linestyle='--',
         linewidth='1.0', color='black', alpha=0.25)
plt.grid(which='minor', linestyle='-',
         linewidth='0.5', color='red', alpha=0.25)

# minor_ticks = np.arange(-6, 7, 0.5)
# plt.xticks(minor_ticks, minor=True)
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))

plt.xlim([XMIN, XMAX])
plt.ylim([YMIN, YMAX])

filename_save = '2Dturb_C'+str(CASENO)+'_N'+str(NLES)+'_'+METHOD+str(CL)+'_nAgents'+str(nAgents)
plt.savefig(filename_save+'_pdf.png', bbox_inches='tight', dpi=450)
plt.show()
#%%
# pdf_dns_= np.hstack((pdf_DNS,pdf_DNS[:,0].reshape(-1,1)/std_omega_DNS))
# np.savetxt(filename_save+"_pdfdns.dat", pdf_dns_, delimiter='\t')
pdf_les_= np.stack((Vecpoints, exp_log_kde,Vecpoints/std_omega),axis=1)
np.savetxt(filename_save+"_pdf.dat", pdf_les_, delimiter='\t')
#%%
# plt.hist(omega_M.reshape(-1,1), 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
Vecpoints, exp_log_kde, log_kde, kde = myKDE(omega_M.reshape(-1,1), BANDWIDTH=BANDWIDTH)

fig = plt.figure(figsize=(6,4),dpi=450)

plt.plot(Vecpoints, exp_log_kde, 'k', alpha=0.5, linewidth=2, label=METHOD+r'($C=$'+str(CL)+r')')
hist, bins, patches = plt.hist(omega_M.reshape(-1,1), 300, fc='gray', histtype='stepfilled', alpha=0.3,density=True)

plt.yscale('log')
print((hist * np.diff(bins)).sum())
plt.ylim([1e-6, 1e-1])
plt.savefig(filename_save+'_pdfbin.png', bbox_inches='tight', dpi=450)
plt.show()
#%% Plot enstrophy
col1, col2 = 0, 0
ens_M = []#np.zeros((int(NLES/2)-1,num_file))
tke_M = []#np.zeros((int(NLES/2)-1,num_file))
#
from natsort import natsorted, ns
#dirlist = natsorted(dirlist, alg=ns.PATH | ns.IGNORECASE)
#

num_file = 0
#for filename in sorted(os.listdir(directory)):
for filename in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):

    if METHOD in filename and filename.endswith('ens.out'):

        numbers_in_file = re.findall(r'\d+', filename)
        file_numstep = int(numbers_in_file[-1])

        if file_numstep>SPIN_UP:
            print('Ens files',str(num_file),':',str(col1),':',str(file_numstep))
            ens_i = np.loadtxt(directory+filename)[:-1,1]
            ens_i = np.asarray(ens_i).reshape((-1,))
            #ens_M[:,col1] = ens_i
            ens_M = np.append(ens_M, ens_i)
            col1 +=1
            num_file += 1

        if num_file==NUM_DATA: break
        if col1==NUM_DATA: break

num_file = 0
#for filename in sorted(os.listdir(directory)):
for filename in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):

    if METHOD in filename and filename.endswith('tke.out'):

        numbers_in_file = re.findall(r'\d+', filename)
        file_numstep = int(numbers_in_file[-1])

        if file_numstep>SPIN_UP:   
            print('TKE files',str(num_file),':',str(col2),':',str(file_numstep))
            tke_i = np.loadtxt(directory+filename)[:-1,1]
            tke_i = np.asarray(tke_i).reshape((-1,))
            tke_M = np.append(tke_M, tke_i)
            #tke_M[:,col2] = tke_i
            col2 +=1
            num_file += 1

        if num_file==NUM_DATA: break
        if col2==NUM_DATA: break

#%%
ens_M = ens_M.reshape(-1, int(NLES/2)-1)
tke_M = tke_M.reshape(-1, int(NLES/2)-1)
ens_ave = np.mean(ens_M,axis=0)
tke_ave = np.mean(tke_M,axis=0)
#%%
if CASENO == 4:
    tke_dns =  np.loadtxt('_init/Re20kf25/'+'energy_spectrum_Re20kf25_DNS1024_xy.dat')[:-1,1]
    Kplot_dns =  np.loadtxt('_init/Re20kf25/'+'energy_spectrum_Re20kf25_DNS1024_xy.dat')[:-1,0]
    ens_dns =  np.loadtxt('_init/Re20kf25/'+'enstrophy_spectrum_Re20kf25_DNS1024_xy.dat')[:-1,1]

elif CASENO == 1:
    tke_dns =  np.loadtxt('_init/Re20kf4/'+'energy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,1]
    Kplot_dns =  np.loadtxt('_init/Re20kf4/'+'energy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,0]
    ens_dns =  np.loadtxt('_init/Re20kf4/'+'enstrophy_spectrum_Re20kf4_DNS1024_xy.dat')[:-1,1]
    
elif CASENO == 2:
    tke_dns =  np.loadtxt('_init/Re20kf4beta20/'+'energy_spectrum_Re20kf4beta20_DNS1024_xy.dat')[:-1,1]
    Kplot_dns =  np.loadtxt('_init/Re20kf4beta20/'+'energy_spectrum_Re20kf4beta20_DNS1024_xy.dat')[:-1,0]
    ens_dns =  np.loadtxt('_init/Re20kf4beta20/'+'enstrophy_spectrum_Re20kf4beta20_DNS1024_xy.dat')[:-1,1]
#%%
Kplot = np.array(range(int(NLES/2)-1))
#%%
kplot_str = '\kappa_{x}'

plt.figure(figsize=(10,14),dpi=450)

# Energy 
plt.subplot(3,2,4)
plt.loglog(Kplot,tke_ave,'k')
plt.plot([Fn,Fn],[1e-12,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot_dns,tke_dns,':k', alpha=0.5, linewidth=4)
plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')

if CASENO==1:
    plt.xlim([1,512])
    plt.ylim([1e-12,1e0])    
elif CASENO==4:
    plt.xlim([1,1e3])
    plt.ylim([1e-9,1e-1])
elif CASENO==2:
    plt.xlim([1,1e3])
    plt.ylim([1e-10,1e1])
    
# Enstrophy
plt.subplot(3,2,3)
plt.loglog(Kplot, ens_ave,'k')
plt.plot([Fn,Fn],[1e-11,1e6],':k', alpha=0.5, linewidth=2)
plt.loglog(Kplot_dns, ens_dns,':k', alpha=0.5, linewidth=4)
plt.title(rf'$\varepsilon({kplot_str})$')
plt.xlabel(rf'${kplot_str}$')
if CASENO==1:
    plt.xlim([1,512])
    plt.ylim([1e-6,1e1])
elif CASENO==4:
    plt.xlim([1,1e3])
    plt.ylim([1e-1,1e1])
elif CASENO==2:
    plt.xlim([1,1e3])
    plt.ylim([1e-6,1e1])

for i in [3,4]:
    plt.subplot(3,2,i)

    plt.grid(which='major', linestyle='--',
             linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
             linewidth='0.5', color='red', alpha=0.25)
    
    
plt.savefig(filename_save+'_spec.png', bbox_inches='tight', dpi=450)
#%%
# stop
tke_ave_=np.stack((Kplot,tke_ave)).T
np.savetxt(filename_save+"_tkeave.dat", tke_ave_,delimiter='\t')

ens_ave_=np.stack((Kplot,ens_ave)).T
np.savetxt(filename_save+"_ensave.dat", ens_ave_,delimiter='\t')
stop_spec
# ens_ave_=np.stack((Kplot,tke_dns)).T
# np.savetxt(filename_save+"tkedns.dat", ens_ave_,delimiter='\t')


# ens_ave_=np.stack((Kplot,ens_dns)).T
# np.savetxt(filename_save+"_ensdns.dat", ens_ave_,delimiter='\t')
## Plot loss types
# plt.figure(figsize=(10,14))

# # Enstrophy
# plt.subplot(3,2,1)
# plt.plot( Kplot    * ens_dns , 'k')
# plt.plot((Kplot**2)*(ens_dns),'--r')
# plt.plot((Kplot**3)*(ens_dns),'.-b')
# plt.plot((np.log(ens_dns)),'--*k')
# plt.title(rf'$\varepsilon({kplot_str})$')

# plt.subplot(3,2,2)
# plt.plot(Kplot*(ens_dns-ens_ave), 'k')
# plt.plot((Kplot**2)*(ens_dns-ens_ave),'--r')
# plt.plot((Kplot**3)*(ens_dns-ens_ave),'.-b')
# plt.plot((np.log(ens_dns)-np.log(ens_ave)),'--*k')
# plt.title(rf'$\varepsilon({kplot_str})$')

# # Energy
# plt.subplot(3,2,3)
# plt.plot( Kplot    * tke_dns , 'k')
# plt.plot((Kplot**2)*(tke_dns),'--r')
# plt.plot((Kplot**3)*(tke_dns),'.-b')
# plt.plot((np.log(tke_dns)),'--*k')
# plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')

# plt.subplot(3,2,4)
# plt.plot(Kplot*(tke_dns-tke_ave), 'k')
# plt.plot((Kplot**2)*(tke_dns-tke_ave),'--r')
# plt.plot((Kplot**3)*(tke_dns-tke_ave),'.-b')
# plt.plot((np.log(tke_dns)-np.log(tke_ave)),'--*k')
# plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')

# for i in [1,2,3,4]:
#     plt.subplot(3,2,i)

#     plt.grid(which='major', linestyle='--',
#              linewidth='1.0', color='black', alpha=0.25)
#     plt.grid(which='minor', linestyle='-',
#              linewidth='0.5', color='red', alpha=0.25)
#%% PDF of veRL
num_file = 0
veRL_M = []
omega_M = []# np.zeros((NLES, NLES))
psi_M = []

for filename in sorted(os.listdir(directory)):

    if METHOD in filename and str(NLES) in filename and '01' in filename and filename.endswith('.mat'):
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
        num_file +=1
        if num_file==NUM_DATA: break
#%%
# meanveRL = veRL_M.mean()
# Vecpoints, exp_log_kde, log_kde, kde = myKDE(veRL_M,BANDWIDTH=0.1)
# plt.semilogy(Vecpoints, exp_log_kde, 'k', alpha=0.75, linewidth=1.0)#, label=METHOD+r'($C=$'+str(CL)+r')')
# plt.semilogy([meanveRL,meanveRL],[min(exp_log_kde),max(exp_log_kde)],'-r')
#%%
from multivariat_tools import multivariat_fit
##%%
Lx = 2*np.pi
NX = NLES
kx = (2*np.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),
                                  np.arange((-NX/2+1),0,dtype=np.float64)
                                ))
[Ky,Kx]  = np.meshgrid(kx,kx)
w1x_hat = -(1j*Kx)*psi_hat
w1y_hat = +(1j*Ky)*psi_hat

u1_hat = -(1j*Ky)*psi_hat
v1_hat = +(1j*Kx)*psi_hat
u1 = np.real(np.fft.ifft2(u1_hat))
v1 = np.real(np.fft.ifft2(v1_hat))
#%%
from velincrement import realVelInc_fast
#%%
u = u1
dr = 1
v = v1

u = np.expand_dims(u, axis=2)
du = realVelInc_fast(u, ax=0, r=dr).reshape(-1)
v = np.expand_dims(v, axis=2)
# dv = realVelInc_fast(v, ax=0, r=dr).reshape(-1)
dv = realVelInc_fast(v, ax=1, r=dr).reshape(-1)
incr = np.concatenate((du,dv), axis=None) # / u_rms
##%%
u = u1
dr = 1
dur = u[:,dr:]-u[:,0:-dr]

v = v1
dr = 1
# dvr = v[:,dr:]-v[:,0:-dr] #dv = realVelInc_fast(v, ax=1, r=dr).reshape(-1)
dvr = v[dr:,:]-v[0:-dr,:]# dv = realVelInc_fast(v, ax=0, r=dr).reshape(-1)
duv = np.concatenate((dur,dvr), axis=None) # / u_rms


plt.figure(figsize=(12,6),dpi=400)
plt.subplot(2,2,1)
plt.contourf(u,cmap='bwr',levels=50)
plt.subplot(2,2,2)
plt.contourf(dur,cmap='bwr',levels=50)

plt.subplot(2,2,3)
Vecpoints, exp_log_kde, log_kde, kde = myKDE(u,BANDWIDTH=0.01)
plt.plot(Vecpoints, exp_log_kde, 'r', alpha=0.75, linewidth=2, label=r'$u$')

plt.subplot(2,2,4)
Vecpoints, exp_log_kde, log_kde, kde = myKDE(duv,BANDWIDTH=0.1)
plt.semilogy(Vecpoints, exp_log_kde, 'r', alpha=0.75, linewidth=2, label=r'$u$')

Vecpoints, exp_log_kde, log_kde, kde = myKDE(incr,BANDWIDTH=0.1)
plt.semilogy(Vecpoints, exp_log_kde, 'b', alpha=0.75, linewidth=2, label=r'$u$')
plt.xlabel(r'$\delta u$',fontsize=16)
plt.ylabel(r'$\mathcal{P}( \left( \delta u \right)$',fontsize=16)
plt.ylim([1e-2,2e0])
#%% Plot Dis
xplot, xplot_str = veRL_M, '$C_S^2$'
yplot, yplot_str= omega_M, '$\omega$'
# yplot, yplot_str = psi_M, '$\psi$'
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
ax.set_axisbelow(True)

for j in range(4):
  ell = Ellipse(xy=meanxy,
                width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                angle=np.rad2deg(np.arccos(v[0, 0])),
                alpha=0.05,
                #facecolor='blue',
                #facecolor='none',
                edgecolor='blue',
                linestyle='-')

  ellborder = Ellipse(xy=meanxy,
                width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                angle=np.rad2deg(np.arccos(v[0, 0])),
                facecolor='none',
                edgecolor='blue',
                linestyle='-')
  
  ax.add_artist(ell)
  ax.add_artist(ellborder)

#plt.plot(xplot, yplot,'.k',alpha=0.05, markersize=0.5)
#plt.scatter(xplot, yplot, marker=".", color='black',s=0.1,alpha=0.25)
plt.scatter(xplot, yplot, marker=".", color='black',s=0.1,alpha=0.05)

plt.scatter(meanxy[0],meanxy[1], marker="+", color='blue',s=100)
plt.scatter(CS2,meanxy[1], marker="s", color='red',s=25,linewidths=0.15)
plt.scatter(CS2EKI,meanxy[1], marker="*", color='black',s=50,linewidths=0.15)

plt.xlabel(xplot_str)
plt.ylabel(yplot_str)
plt.grid(color='gray', linestyle='dashed')
plt.xlim([-0.1,0.1])
plt.ylim([-20,20])

plt.savefig(filename+'dis.png', bbox_inches='tight', dpi=450)

#%% Plot Dis point forward
xplot, xplot_str = veRL_M, '$C_S^2$'
yplot, yplot_str= omega_M, '$\omega$'
# yplot, yplot_str = psi_M, '$\psi$'
CS2 = 0.17**2
CS2EKI = 0.1**2
meanxy_M = []

from matplotlib.patches import Ellipse
fig, ax = plt.subplots()#subplot_kw={'aspect': 'equal'})
ax.set_axisbelow(True)
    
for icount in range(1,int(len(xplot)/NLES/NLES)):
    print(icount)
    xploti = xplot[(icount-1)*NLES*NLES:(icount)*NLES*NLES]
    yploti = yplot[(icount-1)*NLES*NLES:(icount)*NLES*NLES]
    
    xv, yv, rv, pos, meanxy, covxy = multivariat_fit(xploti, yploti )
    # # plt.plot(xploti, yploti,'.k',alpha=0.05, markersize=2)
    plt.scatter(meanxy[0],meanxy[1], marker=".", color='red',s=10,linewidths=2)
    plt.scatter(CS2,meanxy[1], marker="o", color='green',s=50,linewidths=0.15)
    plt.scatter(CS2EKI,meanxy[1], marker="*", color='green',s=50,linewidths=0.15)
    
    # plt.xlabel(xplot_str)
    # plt.ylabel(yplot_str)
    # plt.grid(color='gray', linestyle='dashed')
    
    meanxy_M = np.append(meanxy_M, meanxy)

    lambda_, v = np.linalg.eig(covxy)
    lambda_ = np.sqrt(lambda_)

    j=1
    ellborder1 = Ellipse(xy=meanxy,
                    width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                    angle=np.rad2deg(np.arccos(v[0, 0])),
                    facecolor='none',
                    edgecolor='blue',
                    alpha=0.1,
                    linestyle='-',
                    label=r'$\sigma$')
    j=2
    ellborder2 = Ellipse(xy=meanxy,
              width=lambda_[0]*j*2, height=lambda_[1]*j*2,
              angle=np.rad2deg(np.arccos(v[0, 0])),
              facecolor='none',
              edgecolor='pink',
              alpha=0.2,
              linestyle='-',
              label=r'$\sigma$')
       
    ax.add_artist(ellborder1)
    ax.add_artist(ellborder2)
    
meanx_M = meanxy_M.reshape(-1,2)[:,0]
Vecpoints, exp_log_kde, log_kde, kde = myKDE(meanx_M,BANDWIDTH=0.01, padding=0.025)
plt.plot(Vecpoints, 0.5*exp_log_kde, 'r', alpha=0.75, linewidth=2, label=METHOD+r'($C=$'+str(CL)+r')')

_, _, _, _, meanxy_alldata, _ = multivariat_fit(xplot, yplot )
plt.scatter(meanxy_alldata[0],meanxy_alldata[1], marker="+", color='red',s=100)

plt.xlabel(xplot_str)
plt.ylabel(yplot_str)

plt.xlim([-0.1,0.1])
plt.ylim([-20,20])
plt.grid(color='gray', linestyle='dashed')

ax.add_artist(ellborder1)
ax.add_artist(ellborder2)
ax.legend(['Inst RL','Lit.','EKI','$\sigma$','$2\sigma$'])

plt.savefig(filename+'_instant.png', bbox_inches='tight', dpi=450)
#%% 3D Plot Dis point forward
[Ky,Kx]  = np.meshgrid(kx,kx)
NY = NX
Ksq      = (Kx**2 + Ky**2)
Kabs     = np.sqrt(Ksq)
Ksq[0,0] = 1e12
invKsq   = 1/Ksq
Ksq[0,0] = 0
invKsq[0,0] = 0
#%%
from DSMAG import smag_dynamic_cs
ALPHA=1
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, ALPHA),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, ALPHA),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, ALPHA),  # blue  with alpha = 20%
})

xplot, xplot_str = veRL_M, '$C_S^2$'
yplot, yplot_str= omega_M, '$\omega$'
# yplot, yplot_str = psi_M, '$\psi$'
CS2 = 0.17**2
CS2EKI = 0.1**2
meanxy_M = []
cs_Local_SM_M = []
cs_local_SM_withClipping_M = []
S_M = []
omega_M = []
veRL_M = []

from matplotlib.patches import Ellipse
fig, ax = plt.subplots(figsize=(6,6),dpi=400)#subplot_kw={'aspect': 'equal'})
# plt.figure(figsize=(12,6),dpi=400)

ax.set_axisbelow(True)

this_method ='MARL' # 'LDSM', 'LDSMw', 'MARL'
for filename in sorted(os.listdir(directory)):

    if METHOD in filename and str(NLES) in filename and '01' in filename and filename.endswith('.mat'):
        print(filename)
        mat_contents = sio.loadmat(directory+filename)
        w1_hat = mat_contents['w_hat']
        omega = np.real(np.fft.ifft2(w1_hat))
        omega_M = np.append(omega_M, omega)
        veRL = mat_contents['veRL']


        cs_Local_SM, cs_local_SM_withClipping, cs_Averaged_SM, cs_Averaged_SM_withClipping, S =\
            smag_dynamic_cs(w1_hat, invKsq, NX, NY, Kx, Ky)
        
        if this_method=='LDSM':
            cs2_plot = cs_Local_SM
            xploti_str='Dynamic local'
        elif this_method=='LDSMw':
            cs2_plot = cs_local_SM_withClipping
            xploti_str='Dynamic local w. clipping'
        elif this_method=='MARL':
            cs2_plot = mat_contents['veRL']
            xploti_str='MARL'


        cs_Local_SM_M.append(cs_Local_SM)
        cs_local_SM_withClipping_M.append(cs_local_SM_withClipping)
        S_M = np.append(S_M, S)
        veRL_M = np.append(veRL_M, veRL)

        xploti = cs2_plot.reshape(-1,)

        # yploti = np.real(np.fft.ifft2(w1_hat)).reshape(-1,)
        yploti = S.reshape(-1,)
        yplot_str = r'$S$'
        
        xv, yv, rv, pos, meanxy, covxy = multivariat_fit(xploti, yploti )
        # # plt.plot(xploti, yploti,'.k',alpha=0.05, markersize=2)
        plt.scatter(meanxy[0],meanxy[1], marker=".", color='red',s=10,linewidths=2)
        plt.scatter(CS2,meanxy[1], marker="o", color='green',s=50,linewidths=0.15)
        plt.scatter(CS2EKI,meanxy[1], marker="*", color='green',s=50,linewidths=0.15)
        
        # plt.xlabel(xplot_str)
        # plt.ylabel(yplot_str)
        # plt.grid(color='gray', linestyle='dashed')
        meanxy_M = np.append(meanxy_M, meanxy)

        lambda_, v = np.linalg.eig(covxy)
        lambda_ = np.sqrt(lambda_)
    
        j=1
        ellborder1 = Ellipse(xy=meanxy,
                        width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                        angle=np.rad2deg(np.arccos(v[0, 0])),
                        facecolor='none',
                        edgecolor='blue',
                        alpha=0.1,
                        linestyle='-',
                        label=r'$\sigma$')
        j=2
        ellborder2 = Ellipse(xy=meanxy,
                  width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  facecolor='none',
                  edgecolor='pink',
                  alpha=0.2,
                  linestyle='-',
                  label=r'$\sigma$')
           
        ax.add_artist(ellborder1)
        ax.add_artist(ellborder2)
     
        # plt.xlim([-0.1,0.1])
        # plt.ylim([-20,20])
        stop_2dplot
meanx_M = meanxy_M.reshape(-1,2)[:,0]
Vecpoints, exp_log_kde, log_kde, kde = myKDE(meanx_M,BANDWIDTH=0.01,padding=0.1)
plt.plot(Vecpoints, 0.01*exp_log_kde, 'r', alpha=0.75, linewidth=2, label='Model')

_, _, _, _, meanxy_alldata, _ = multivariat_fit(xplot, yplot )
plt.scatter(meanxy_alldata[0],meanxy_alldata[1], marker="+", color='red',s=100)

plt.xlabel(xplot_str)
plt.ylabel(yplot_str)

plt.xlim([-0.1,0.1])
# plt.ylim([-20,20])
plt.grid(color='gray', linestyle='dashed')

ax.add_artist(ellborder1)
ax.add_artist(ellborder2)
ax.legend([xploti_str,'Lit.','EKI','$\sigma$','$2\sigma$'])
#%%
Vecpoints, exp_log_kde, log_kde, kde = myKDE(meanx_M,BANDWIDTH=0.01)
plt.plot(Vecpoints, 0.5*exp_log_kde, 'r', alpha=0.75, linewidth=2, label=METHOD+r'($C=$'+str(CL)+r')')
plt.plot(meanx_M,'.k')
#%% Distribtuion of increments of ve
CS_M= [ np.expand_dims(veRL, axis=2),
np.expand_dims(np.array(cs_Local_SM_M)[-1,:,:], axis=2),
np.expand_dims(np.array(cs_local_SM_withClipping_M)[-1,:,:], axis=2)]
plt.figure(figsize=(12,6),dpi=400)

icount = 0
for CS in CS_M:
    icount += 1
    dCS1 = realVelInc_fast(CS, ax=0, r=dr).reshape(-1)
    dCS2 = realVelInc_fast(CS, ax=1, r=dr).reshape(-1)
    incr = np.concatenate((dCS1,dCS2), axis=None) # / u_rms
    
    
    plt.subplot(2,3,icount)
    plt.contourf(CS[:,:,0],cmap='bwr',levels=50,vmin=-0.12,vmax=0.12);plt.colorbar()
    plt.axis('equal')
    plt.subplot(2,1,2)
    
    Vecpoints, exp_log_kde, log_kde, kde = myKDE(incr,BANDWIDTH=0.01)
    plt.semilogy(Vecpoints, exp_log_kde, 'b', alpha=0.75, linewidth=2, label=r'$C^2_S$')
    plt.xlabel(r'$\delta C^2_S$',fontsize=22)
    plt.ylabel(r'$\mathcal{P}( \left( \delta C^2_S \right)$',fontsize=22)
    plt.ylim([5e-3,5e1])
    plt.xlim([-0.1,0.1])
    
plt.xlim([-1,1])
#%% 3D scatter plot of
import matplotlib.ticker as mticker
ALPHA_S =0.1
fig = plt.figure(figsize=(18,6))
# ax = fig.add_axes([0,0,1,1], projection='3d')
icount = 0

for this_method in ['LDSM','LDSMw','MARL']:
    icount+=1
    if this_method=='LDSM':
        cs2_plot = np.array(cs_Local_SM_M).reshape(-1,)
        xploti_str='Dynamic local'
    elif this_method=='LDSMw':
        cs2_plot = np.array(cs_local_SM_withClipping_M).reshape(-1,)
        xploti_str='Dynamic local w. clipping'
    elif this_method=='MARL':
        cs2_plot =  np.array(veRL_M).reshape(-1,)
        xploti_str='MARL'
    
    ax = fig.add_subplot(1, 3, icount, projection='3d')

    
    # #plot the points
    # ax.scatter(x,y,z*0.4, c="r", facecolor="r", s=60)
    CPLOT = cs2_plot
    # VMAX= max(cs2_plot.max(),-cs2_plot.min())
    # VMIN=-VMAX
    VMAX=0.01
    VMIN=-VMAX
    cax = ax.scatter3D(np.array(S_M),omega_M, (cs2_plot), s=3,\
                       alpha=ALPHA_S, c=CPLOT, cmap='bwr',vmin=VMIN, vmax=VMAX);
    fig.colorbar(cax, location='top')
    
    
    ax.set_xlabel('S',fontsize=16)
    ax.set_ylabel(r'$\omega$',fontsize=16)
    ax.set_zlabel(r'$C_S^2$',fontsize=16)
    
    ax.set_title(xploti_str)
    
    # def log_tick_formatter(val, pos=None):
    #     return f"$10^{{{int(val)}}}$"
    
    # ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    # ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # ax.zaxis.set_scale('log')
    ax.set_zlim(-0.1,0.1)
    ax.set_xlim(-0.2,0.2)

    # ax.view_init(azim=0, elev=90) # x-y 
    ax.view_init(azim=0, elev=0) # z-y
    # ax.view_init(azim=90, elev=90) # x-z
    
    ax.set_proj_type('ortho')  # FOV = 0 deg

#%%
import matplotlib.ticker as mticker

this_method = 'LDSM' 
cs2_plot1 = np.array(cs_Local_SM_M).reshape(-1,)
xplot1_str='Dynamic local'
cs2_plot2 = np.array(cs_local_SM_withClipping_M).reshape(-1,)
xplot2_str='Dynamic local w. clipping'
cs2_plot3 = veRL_M
xplot3_str='MARL'

fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1], projection='3d')

# #plot the points
# ax.scatter(x,y,z*0.4, c="r", facecolor="r", s=60)
cax = ax.scatter3D(cs2_plot1,cs2_plot2, cs2_plot3,'.',alpha=0.15, c=omega_M);
fig.colorbar(cax, location='top')


ax.set_xlabel(xplot1_str,fontsize=16)
ax.set_ylabel(xplot2_str,fontsize=16)
ax.set_zlabel(xplot3_str,fontsize=16)

ax.set_title(xploti_str)

# def log_tick_formatter(val, pos=None):
#     return f"$10^{{{int(val)}}}$"

# ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# ax.zaxis.set_scale('log')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

# for ii in range(1,360,30):
#     ax.view_init(elev=20., azim=ii)
#     plt.savefig("imovie%d.png" % ii)
#%%

plt.pcolor(veRL,cmap='bwr');plt.colorbar()
#%%
plt.pcolor(CS[:,:,0],cmap='bwr',vmin=-1,vmax=1);plt.colorbar()
plt.title(r'$C_S^2$')
#%%
from scipy.interpolate import RectBivariateSpline
#%%
nagents = 1
nActions = 25
# temporary
nActiongrid = int((nActions*nagents)**0.5)
nActiongrid = nActiongrid
# Initlize action
X = np.linspace(0,Lx,nActiongrid, endpoint=True)
Y = np.linspace(0,Lx,nActiongrid, endpoint=True)
xaction = X
yaction = Y


def upsample(arr_action, xaction, yaction, Lx, NX):
    upsample_action = RectBivariateSpline(xaction, yaction, arr_action, kx=1, ky=1)
    
    # Initlize action
    upsamplesize = NX # 1 for testing, will be changed to grid size eventually
    x2 = np.linspace(0,Lx, upsamplesize, endpoint=True)
    y2 = np.linspace(0,Lx,  upsamplesize, endpoint=True)
    forcing = upsample_action(x2, y2)
    return forcing
#%%
arr_action_S = cs2_plot.reshape(128,128)[::26,::26]
forcing_S = upsample((arr_action_S*2*np.pi/NX)**2, xaction, yaction, Lx, NX)

arr_action_ve = ve[::26,::26]
forcing_ve = upsample(arr_action_ve, xaction, yaction, Lx, NX)
#%%
cs = (0.17 * 2*np.pi/NX )**2
S1 = np.real(np.fft.ifft2(-Ky*Kx*psi_hat)) # make sure .* 
S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psi_hat))
S  = 2.0*(S1*S1 + S2*S2)**0.5
ve = cs*S

Smean = (np.mean(S**2.0))**0.5;
vemean = cs*Smean
#%%
plt.figure(figsize=(19,11), dpi=450)
plt.subplot(2,3,1)
plt.pcolor(ve,cmap='bwr'); plt.colorbar()
plt.title(r'$\nu$',fontsize=30)

plt.subplot(2,3,2)
plt.pcolor(S,cmap='bwr'); plt.colorbar()
plt.title(r'$S$',fontsize=30)

plt.subplot(2,3,4)
plt.pcolor(forcing_ve,cmap='bwr'); plt.colorbar()
plt.title(r'$\bar{\nu}$ ',fontsize=20)


plt.subplot(2,3,5)
plt.pcolor(forcing_S,cmap='bwr'); plt.colorbar()
plt.title(r'$\bar{C^2_S}$',fontsize=20)

plt.subplot(2,3,6)
plt.pcolor(forcing_ve*S,cmap='bwr'); plt.colorbar()
plt.title(r'$\nu = \bar{C^2_S} \times S$',fontsize=20)
