#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update : Dec 20
"""
import os
import sys
import re

from postkdearg import postkdearg
args = postkdearg()

locals().update(vars(args))
print(args)

try:
    from natsort import natsorted, ns
except:
    os.system("pip3 install natsort")
    from natsort import natsorted, ns

try:
    # os.chdir('/home/rm99/Mount/jetstream_volume/docker/RLonKoraliMA/experiments/flowControl_turb_code')
    os.chdir('/home/rm99/Mount/jetstream_volume/docker/RLonKoraliMA/experiments/flowControl_turb_code')
except:
    print('')

import numpy as np
import scipy.io as sio

from PDE_KDE import myKDE
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
NUM_DATA = 300
NLES = 64
#%%
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents'+str(nAgents)+'_CREWARD1_Tspin'+str(SPIN_UP)+'_Thor'+str(Thor)+'_NumRLSteps'+str(NumRLSteps)+'_EPERU'+str(EPERU)+'/CLpost/'
directory = '_result_vracer_C2_N64_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin0.0_Thor20000.0_NumRLSteps2000.0_EPERU1.0/CLpost/'

CL = ''
num_file = 0
omega_M = []# np.zeros((NLES, NLES))
w1_hat_M = []# np.zeros((NLES, NLES))
veRL_M = []# np.zeros((NLES, NLES))
psi_hat_M = []

#for filename in sorted(os.listdir(directory)):
for filename in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):

    if str(NLES) in filename and filename.endswith('.mat'):

        numbers_in_file = re.findall(r'\d+', filename)
        file_numstep = int(numbers_in_file[-1])

        if file_numstep>SPIN_UP:
        # if file_numstep>50000 and file_numstep%20000==1:
            print(str(num_file), filename)

            mat_contents = sio.loadmat(directory+filename)

            w1_hat = mat_contents['w_hat']
            veRL = mat_contents['veRL']
            psi_hat = mat_contents['psi_hat']

            omega = np.real(np.fft.ifft2(w1_hat))
            #if omega.max() > 10:
            #    print(omega.max())

            omega_M = np.append(omega_M, omega)
            w1_hat_M = np.append(w1_hat_M, w1_hat)
            veRL_M = np.append(veRL_M, veRL)
            psi_hat_M = np.append(psi_hat_M, psi_hat)


            num_file += 1
            if num_file==NUM_DATA: break
#%%W
omega_M_2D = omega_M.reshape(-1,NLES,NLES,order='C')
w1_hat_M_2D = w1_hat_M.reshape(-1,NLES,NLES,order='C')
veRL_M_2D = veRL_M.reshape(-1,NLES,NLES,order='C')
psi_hat_M_2D = psi_hat_M.reshape(-1,NLES,NLES,order='C')
#%% def PI_from_eddymodel
def PI_from_eddymodel(psi_hat, w1_hat, ve, Kx, Ky, Ksq):
    diffu_hat = -Ksq*w1_hat
    PI_hat = ve*diffu_hat
    PI  = np.fft.ifft2(PI_hat).real
    return PI, PI_hat

#%%
NX = 64
Lx = 2*np.pi
dx = Lx/NX
#-----------------
x        = np.linspace(0, Lx-dx, num=NX)
kx       = (2*np.pi/Lx)*np.concatenate((
                            np.arange(0,NX/2+1,dtype=np.float64),
                            np.arange((-NX/2+1),0,dtype=np.float64)
                            ))
[Y,X]    = np.meshgrid(x,x)
[Ky,Kx]  = np.meshgrid(kx,kx)
Ksq      = (Kx**2 + Ky**2)
Ksq[0,0] = 1e16
Kabs     = np.sqrt(Ksq)
invKsq   = 1/Ksq
#%%

w1_hat = mat_contents['w_hat']
ve = mat_contents['veRL']

psi_hat = mat_contents['psi_hat']

diffu_hat = -Ksq*w1_hat
PI_hat = diffu_hat
PI  = np.fft.ifft2(PI_hat).real
PI, PI_hat = PI_from_eddymodel(psi_hat, w1_hat, veRL, Kx, Ky, Ksq)
#%%
import sys
sys.path.append('/media/rmojgani/hdd/PostDoc/ScratchBook/spectra/experiments/case2')
from spectra import spectrum_angled_average_2DFHIT, TKE_angled_average_2DFHIT, enstrophy_angled_average_2DFHIT

# initialize_wavenumbers_2DFHIT: Initializes the wavenumbers for a 2D field
from initialize import initialize_wavenumbers_2DFHIT

# Omega2Psi_2DFHIT: Converts vorticity to stream function in a 2D field
from convert import Omega2Psi_2DFHIT

from spectra import *
#%%
NSAMPLE = 300
kplot_str ='\kappa'

spec_tke_mean = 0
spec_ens_mean = 0

Eflux_hat_M = []
Zflux_hat_M = []

for tcount in np.linspace(0,NUM_DATA,NSAMPLE,endpoint=False).astype(int):
    print(tcount, '/', NUM_DATA)

    psi_hat = psi_hat_M_2D[tcount]
    w1_hat = w1_hat_M_2D[tcount]
    ve = veRL_M_2D[tcount]

    np.testing.assert_array_equal( -invKsq*w1_hat, psi_hat)

    #if 'DNS_data' in filename and filename.endswith('.mat'):
    #if 'w1_1' in filename and filename.endswith('.mat'):
    # print(tcount, ':', filename)

    #w1Python = sio.loadmat(directory+filename)['w1Python']
    # w1Python = mat_contents['slnWorDNS'][:,:,tcount]

    #
    # w1_hat = np.fft.fft2(w1Python)
    # psi_hat = -invKsq*w1_hat

    # PI_FDNS, PI_hat_FDNS = PI_from_convection_conserved(psi_hat, w1_hat, Kx, Ky, filter)
    # PI, PI_hat = PI_from_eddymodel(psi_hat, w1_hat, veRL, Kx, Ky, Ksq)
    # PI, PI_hat = PI_from_eddymodel(psi_hat, w1_hat, np.mean(veRL), Kx, Ky, Ksq)
    PI, PI_hat = PI_from_eddymodel(psi_hat, w1_hat, 0.12, Kx, Ky, Ksq)

    Omega = np.fft.ifft2(w1_hat).real
    Psi = np.fft.ifft2(psi_hat)

    spec_ens, k = enstrophyTransfer_spectra_2DFHIT(Kx, Ky, Omega=Omega, Sigma1=None, Sigma2=None, PiOmega=PI, method='PiOmega', spectral=False)
    spec_tke, Kplot = energyTransfer_spectra_2DFHIT(Kx, Ky, U=None, V=None, Tau11=None, Tau12=None, Tau22=None, Psi=Psi, PiOmega=PI, method='PiOmega', spectral=False)

    spec_tke_mean +=spec_tke/2
    spec_ens_mean +=spec_ens/2

    Eflux_hat_M = np.append(Eflux_hat_M, spec_tke)
    Zflux_hat_M = np.append(Zflux_hat_M, spec_ens)
#%%
from fractions import Fraction

# Kplot = wavenumber_Z

NSAMPLE = 15
NLES = 64
kmax=int(NLES/2)

Delta_c = 2*np.pi/NLES
Delta = 2*Delta_c


filter = lambda a_hat: filter_guassian(a_hat, Delta, Ksq)
DELTA_STR=Fraction(str(Delta/np.pi)).as_integer_ratio()
title_str = str(DELTA_STR[0])+'\Pi'+'/'+str(DELTA_STR[1])

YMINEflux, YMAXEflux = -0.05*(64/NLES), 0.05*(64/NLES)
YMINZflux, YMAXZflux = -0.25, 0.25
#%%
from filters import myplot_mean_max_std
myplot_mean_max_std(Kplot, Eflux_hat_M, spec_tke_mean, title_str, kplot_str, 'E_{flux}', YMINEflux, YMAXEflux, kmax)
##%%
myplot_mean_max_std(Kplot, Zflux_hat_M, spec_ens_mean, title_str, kplot_str, 'Z_{flux}', YMINZflux, YMAXZflux, kmax)
#%%
NSAMPLE = NUM_DATA#1024/NUM_DATA
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title('Sample size:'+str(NSAMPLE) )
plt.semilogx(k,-spec_tke_mean/NSAMPLE)
plt.ylabel(rf'E',fontsize=24)
# plt.ylim([-0.25,0.25])
# plt.ylim([-0.03,0.03])

plt.subplot(1,2,2)
plt.title('Sample size:'+str(NSAMPLE) )
plt.semilogx(k,spec_ens_mean/NSAMPLE)
plt.ylabel(rf'Z',fontsize=24)

# plt.ylim([-0.3,0.3])
# plt.ylim([-0.02,0.02])

for icount in [1,2]:
    plt.subplot(1,2,icount)
    plt.grid(which='major', linestyle='--',
              linewidth='1.0', color='black', alpha=0.25)
    plt.grid(which='minor', linestyle='-',
              linewidth='0.5', color='red', alpha=0.15)

    plt.xlim([0,64])
    plt.xlabel(rf'${kplot_str}$')

plt.show()