#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update : Dec 20
"""
import os
import sys
import re
try:
    from natsort import natsorted, ns
except:
    os.system("pip3 install natsort")
    from natsort import natsorted, ns

import numpy as np
import scipy.io as sio
from scipy.interpolate import RectBivariateSpline

from PDE_KDE import myKDE
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
DPI = 150
plt.rcParams['figure.dpi'] = DPI

#%%
SPIN_UP = 50000
NUM_DATA = 200#00#0
#%%
NLES = 32
nAgents = 1
CASENO = 1;
# CASENO = 4;
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CS_nAgents'+str(nAgents)+'/CSpost/'
#directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CL_nAgents'+str(nAgents)+'/CLpost/'


#NLES = 128
nAgents = 1
#CASENO = 4;
#directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_ke_State_energy_Action_CS/dsmag_save/'
directory = '_result_vracer_C'+str(CASENO)+'_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps100.0_EPERU1.0/CSpost/'
directory = '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CS_nAgents16_CREWARD1_Tspin0.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost/'

# sys.path.append(directory)
METHOD = 'smagRL' # 'Leith' , 'Smag'
#METHOD = 'dsmag'#'smag' # 'Leith' , 'Smag'

# sys.path.append('_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/')
# directory = '_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/'
# METHOD = 'smag0d17' # 'Leith' , 'Smag'

CL = ''
num_file = 0
omega_M = []# np.zeros((NLES, NLES))
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
omega_M_2D = omega_M.reshape(NLES,NLES, -1,order='F')
w1Python = omega_M_2D[:,:,0]#mat_contents['slnWorDNS'][:,:,i]
plt.contourf(w1Python,vmin=-26,vmax=26,levels=100, cmap='bwr');plt.axis('square')
plt.tight_layout()
plt.grid(False)
plt.axis('off')
#%%
filename_save = '2Dturb_N'+str(NLES)+'_'+METHOD+str(CL)
#%%
# importing required libraries
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display

L = 2*np.pi
xcoarse = np.linspace(0,L,NLES, endpoint=True)
ycoarse = np.linspace(0,L,NLES, endpoint=True)
xfine = np.linspace(0,L,1024, endpoint=True)
yfine = np.linspace(0,L,1024, endpoint=True)

fig,ax = plt.subplots()

def animate(i,LEVELS=100):

    print(i)
    ax.clear()
    w1Python = omega_M_2D[:,:,i]#mat_contents['slnWorDNS'][:,:,i]
    #upsample_action = RectBivariateSpline(xcoarse, ycoarse, w1Python, kx=2, ky=2)
    #w1Python = upsample_action(xfine, yfine)
    plt.contourf(w1Python,vmin=-26,vmax=26,levels=LEVELS, cmap='bwr');plt.axis('square')
    #ax.set_title('%03d'%(i))
    ax.grid(False)
    ax.axis('off')

MAX_FRAMES= 200
interval = 0.005#in seconds
ani = animation.FuncAnimation(fig,animate,save_count=MAX_FRAMES,blit=False)
FFwriter = animation.FFMpegWriter()
ani.save(filename_save+'.mp4', writer = FFwriter)
plt.show()
