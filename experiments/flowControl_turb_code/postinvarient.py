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

from PDE_KDE import myKDE
import matplotlib
BACKEND = matplotlib.get_backend()
#matplotlib.use('Agg')
import matplotlib.pylab as plt
DPI = 250
plt.rcParams['figure.dpi'] = DPI

#%%
SPIN_UP = 50000
NUM_DATA = 5#0#200#00#0
#%%
NLES = 32
nAgents = 16
CASENO = 2;
# CASENO = 4;
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CS_nAgents'+str(nAgents)+'/CSpost/'
#directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_enstrophy_Action_CL_nAgents'+str(nAgents)+'/CLpost/'
# root = '/mnt/Mount/jetstream_volume3/RLonKoraliMA_gradgrad_betacase_smag/experiments/flowControl_turb_code_CS_Py2D/'
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_psiomega_Action_CL_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost/'

root='/mnt/Mount/jetstream_volume3/RLonKoraliMA_gradgrad_betacase/experiments/flowControl_turb_code/'
directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps100.0_EPERU1.0/CLpost/'

# root=''
# directory = '_result_vracer_C'+str(CASENO)+'_N'+str(NLES)+'_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps100.0_EPERU1.0/CLpost/'

# directory = '_result_vracer_C1_N32_R_z1_State_psiomega_Action_CS_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CSpost/'
# directory = '_result_vracer_C1_N32_R_z1_State_psiomega_Action_CL_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost/'

# root = '/mnt/Mount/jetstream_volume3/RLonKoraliMA_gradgrad_betacase_smag/experiments/flowControl_turb_code_CL_Py2D/'
# directory = '_result_vracer_C1_N32_R_z1_State_invariantlocalandglobalgradgrad_Action_CL_nAgents16_CREWARD1_Tspin10000.0_Thor10000.0_NumRLSteps1000.0_EPERU1.0/CLpost/'

directory = root+directory
# sys.path.append(directory)
METHOD = 'smagRL' # 'Leith' , 'Smag'
#METHOD = 'dsmag'#'smag' # 'Leith' , 'Smag'

# sys.path.append('_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/')
# directory = '_result_vracer_C1_N'+str(NLES)+'_R_k1_State_enstrophy_Action_CL/smag/'
# METHOD = 'smag0d17' # 'Leith' , 'Smag'

CL = ''
num_file = 0
omega_M = []# np.zeros((NLES, NLES))
ve_M = []
for filename in natsorted(os.listdir(directory), alg=ns.PATH | ns.IGNORECASE):

    if METHOD in filename and str(NLES) in filename and filename.endswith('.mat'):

        numbers_in_file = re.findall(r'\d+', filename)
        file_numstep = int(numbers_in_file[-1])

        if file_numstep>SPIN_UP:
        # if file_numstep>50000 and file_numstep%20000==1:
            print(str(num_file), filename)

            mat_contents = sio.loadmat(directory+filename)
            ve = mat_contents['veRL']
            w1_hat = mat_contents['w_hat']
            omega = np.real(np.fft.ifft2(w1_hat))
            #if omega.max() > 10:
            #    print(omega.max())
            omega_M = np.append(omega_M, omega)
            ve_M = np.append(ve_M, ve)

            # omega_M = np.append(omega_M, -omega)

            num_file += 1
            if num_file==NUM_DATA: break
#%%
omega_M_2D = omega_M.reshape(NLES,NLES, -1,order='F')
ve_M_2D = omega_M.reshape(NLES,NLES, -1,order='F')

w1Python = omega_M_2D[:,:,0]#mat_contents['slnWorDNS'][:,:,i]
plt.contourf(w1Python,vmin=-26,vmax=26,levels=100, cmap='bwr');plt.axis('square')
plt.tight_layout()
plt.grid(False)
plt.axis('off')
#%%
filename_save = '2Dturb_N'+str(NLES)+'_'+METHOD+str(CL)
#%%
sys.path.append('_model')
from _model.turb import turb
from _model.split2d import *

Lx = 2*np.pi
nActiongrid = 16

TURB=turb(     Lx=Lx, Ly=Lx,
     NX=NLES,       NY=NLES,
     nagents = nActiongrid
)
matplotlib.use('module://matplotlib_inline.backend_inline')

NX, NY = TURB.NX, TURB.NY
Ky, Kx = TURB.Ky, TURB.Kx
Ksq    = TURB.Ksq
invKsq = TURB.invKsq
#%%

# fig, ax = plt.subplots(2,2)#,
                       # sharex=False, sharey=True,
                       # gridspec_kw={'hspace': 0,'wspace':0.01})
Q_M=[]
R_M=[]
O_M=[]
ve_M=[]

for tcount in range(NUM_DATA):
    w1Python = omega_M_2D[:,:,tcount]#mat_contents['slnWorDNS'][:,:,i]
    ve = ve_M_2D[:,:,tcount]#mat_contents['slnWorDNS'][:,:,i]


    w1_hat = np.fft.fft2(w1Python)
    psi_hat = -invKsq*w1_hat

    u1_hat = TURB.D_dir(psi_hat,Ky) # u_hat = (1j*Ky)*psi_hat
    v1_hat = -TURB.D_dir(psi_hat,Kx) # v_hat = -(1j*Kx)*psi_hat

    dudx_hat = TURB.D_dir(u1_hat,Kx)
    dudy_hat = TURB.D_dir(u1_hat,Ky)

    dvdx_hat = TURB.D_dir(v1_hat,Kx)
    dvdy_hat = TURB.D_dir(v1_hat,Ky)

    dudx = np.fft.ifft2(dudx_hat).real
    dudy = np.fft.ifft2(dudy_hat).real
    dvdx = np.fft.ifft2(dvdx_hat).real
    dvdy = np.fft.ifft2(dvdy_hat).real


    list1 =  pickcenter(dudx, NX, NY, nActiongrid)
    list2 =  pickcenter(dudy, NX, NY, nActiongrid)
    list3 =  pickcenter(dvdx, NX, NY, nActiongrid)
    list4 =  pickcenter(dvdy, NX, NY, nActiongrid)

    list5 =  pickcenter(w1Python, NX, NY, nActiongrid)
    list6 =  pickcenter(ve, NX, NY, nActiongrid)


    mystatelist = []
    ncount = 0
    for dudx,dudy,dvdx,dvdy,omega,ve in zip(list1, list2, list3, list4, list5, list6):


        gradV = np.array([[dudx[0], dudy[0]],
	                    [dvdx[0], dvdy[0]]])

        _, Q,R = TURB.invariant(gradV)


        Q_M = np.append(Q_M, Q)
        R_M = np.append(R_M, R)
        O_M = np.append(O_M, omega[0])
        ve_M = np.append(ve_M, omega[0])

        # ax[0,0].plot(Q,R,'.k',alpha=0.1)
        # # plt.xlabel('Q')
        # # plt.ylabel('R')
        # # plt.xlim([0,50])
        # # plt.ylim([-100,0])

        # ax[1,0].plot(Q,omega[0],'.r',alpha=0.1)
        # # plt.xlabel('Q',fontsize=12)
        # # plt.ylabel('$\omega$',fontsize=12)

        # ax[0,1].plot(omega[0],R,'.r',alpha=0.1)
        # # plt.xlabel('$\omega$',fontsize=12)
        # # plt.ylabel('R',fontsize=12)
        ncount += 1


plt.savefig(filename_save+'_QR.png', bbox_inches='tight', dpi=150)
plt.show()

#%%
import pandas as pd
df = pd.DataFrame(np.array([Q_M,R_M,O_M,ve_M]).T, columns=["Q", "R", "$\omega$","nu"])
#%%
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex' : True}, style='ticks')
#%%
sns.jointplot(data=df, x="Q", y="R", kind="kde")
#%%
sns.jointplot(data=df, x="Q", y="$\omega$", kind="kde")
#%%
sns.jointplot(data=df, x="Q", y="nu", kind="kde")

#%%
g = sns.jointplot(data=df, x="Q", y="nu")
g.plot_joint(sns.kdeplot, color="r")
plt.ylim([-25,25])
plt.xlim([-50,250])

# stop
#%%

Rz=np.linspace(0,100,1000)
Qz = (27/4*Rz)**(1/3)

kdeplot = sns.jointplot(x = "R", y = "Q",
              kind = "kde", data = df, cbar=True, cmap="Greys", fill=True)#cmap='RdGy'# Reds
plt.xlabel(r'$R$')#, size=20)
plt.ylabel(r'$Q$')#, size=20)
##%%
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = kdeplot.ax_joint.get_position()
pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# reposition the joint ax so it has the same width as the marginal x ax
kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

sns.lineplot(x=Rz, y=-Qz, sort=False, lw=1,color='r')
sns.lineplot(x=-Rz, y=-Qz, sort=False, lw=1,color='r')
plt.xlim([-100,100])
plt.ylim([-10,30])

# Show the plot
plt.show()
#%%
plt.figure(figsize=(10,10))
kdeplot = sns.jointplot(x = "R", y = "$\omega$",
              kind = "kde", data = df, cbar=True, cmap="Greys", fill=True)#cmap='RdGy'# Reds
plt.xlabel(r'$R$')#, size=20)
plt.ylabel(r'$\omega$')#, size=20)
##%%
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = kdeplot.ax_joint.get_position()
pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# reposition the joint ax so it has the same width as the marginal x ax
kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

plt.xlim([-100,100])
plt.ylim([-10,10])

# Show the plot
plt.show()

#%%
plt.figure(figsize=(10,10))
kdeplot = sns.jointplot(x = "R", y = "$\omega$",
              kind = "kde", data = df, cbar=True, cmap="Greys", fill=True)#cmap='RdGy'# Reds
plt.xlabel(r'$R$')#, size=20)
plt.ylabel(r'$\omega$')#, size=20)
##%%
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = kdeplot.ax_joint.get_position()
pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# reposition the joint ax so it has the same width as the marginal x ax
kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

plt.xlim([-100,100])
plt.ylim([-10,10])

# Show the plot
plt.show()