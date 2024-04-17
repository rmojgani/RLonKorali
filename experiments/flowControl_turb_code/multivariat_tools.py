#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:56:02 2023

@author: rm99
"""
import numpy as np
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
if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, PathPatch
    from matplotlib.patches import Ellipse
    import mpl_toolkits.mplot3d.art3d as art3d
    
    ALPHA=1
    plt.rcParams.update({
        "figure.facecolor":  (1.0, 1.0, 1.0, ALPHA),  # red   with alpha = 30%
        "axes.facecolor":    (1.0, 1.0, 1.0, ALPHA),  # green with alpha = 50%
        "savefig.facecolor": (1.0, 1.0, 1.0, ALPHA),  # blue  with alpha = 20%
    })
    
    x = np.array([1,2,3])
    y = np.array([2,3,1])
    z = np.array([1,1,1])
    
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], projection='3d')
    
    # #plot the points
    # ax.scatter(x,y,z*0.4, c="r", facecolor="r", s=60)
    # ax.scatter(y,x,z*0.9, c="b", facecolor="b", s=60)
    # ax.scatter(x,y,z*1.6, c="g", facecolor="g", s=60)
    # #plot connection lines
    # ax.plot([x[0],y[0],x[0]],[y[0],x[0],y[0]],[0.4,0.9,1.6], color="k")
    # ax.plot([x[2],y[2],x[2]],[y[2],x[2],y[2]],[0.4,0.9,1.6], color="k")
    # #plot planes
    # p = Rectangle((0,0), 4,4, color="r", alpha=0.2)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0.4, zdir="z")
    
    # p = Rectangle((0,0), 4,4, color="b", alpha=0.2)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0.9, zdir="z")
    
    p = Rectangle((0,0), 4,4, color="g", alpha=0.2)
      
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=1.6, zdir="z")
    
    
    ellborder2 = Ellipse(xy=meanxy,
              width=lambda_[0]*j*2, height=lambda_[1]*j*2,
              angle=np.rad2deg(np.arccos(v[0, 0])),
              facecolor='none',
              edgecolor='pink',
              alpha=0.2,
              linestyle='-',
              label=r'$\sigma$')
    
    # ax.add_patch(ellborder2)
    
    # # ax.set_aspect('equal')
    ax.view_init(13,-63)
    ax.set_xlim3d([0,4])
    ax.set_ylim3d([0,4])
    ax.set_zlim3d([0,4])
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.text import TextPath
    from matplotlib.transforms import Affine2D
    import mpl_toolkits.mplot3d.art3d as art3d
    
    def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
        """
        https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html#sphx-glr-gallery-mplot3d-pathpatch3d-py
        Plots the string *s* on the axes *ax*, with position *xyz*, size *size*,
        and rotation angle *angle*. *zdir* gives the axis which is to be treated as
        the third dimension. *usetex* is a boolean indicating whether the string
        should be run through a LaTeX subprocess or not.  Any additional keyword
        arguments are forwarded to `.transform_path`.
    
        Note: zdir affects the interpretation of xyz.
        """
        x, y, z = xyz
        if zdir == "y":
            xy1, z1 = (x, z), y
        elif zdir == "x":
            xy1, z1 = (y, z), x
        else:
            xy1, z1 = (x, y), z
    
        text_path = TextPath((0, 0), s, size=size, usetex=usetex)
        trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])
    
        p1 = PathPatch(trans.transform_path(text_path), **kwargs)
        ax.add_patch(p1)
        art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)
        # Manually label the axes
        # text3d(ax, (4, -2, 0), "X-axis", zdir="z", size=.5, usetex=False,
        #    ec="none", fc="k")
        # text3d(ax, (12, 4, 0), "Y-axis", zdir="z", size=.5, usetex=False,
        #    angle=np.pi / 2, ec="none", fc="k")
        # text3d(ax, (12, 10, 4), "Z-axis", zdir="y", size=.5, usetex=False,
        #    angle=np.pi / 2, ec="none", fc="k")
    
        plt.show()
    #%%
    delta_t = 1e-5
    ALPHA_L = 0.95
    ALPHA_S = 0.01
    plt.figure(figsize=(12,6),dpi=400)
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    meanxy_3d=[]
    Z = 0   
    this_method ='MARL' # 'LDSM', 'LDSMw', 'MARL'
    for filename in sorted(os.listdir(directory)):

        if METHOD in filename and str(NLES) in filename and '01' in filename and filename.endswith('.mat'):
            print(filename)
            Z += delta_t
            mat_contents = sio.loadmat(directory+filename)
            w1_hat = mat_contents['w_hat']

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

            xploti = cs2_plot.reshape(-1,)

            # yploti = np.real(np.fft.ifft2(w1_hat)).reshape(-1,)
            yploti = S.reshape(-1,)
            yplot_str = r'$S$'
            
            
        # for icount in range(1,int(len(xplot)/NLES/NLES)):
        #     print(icount)
        #     Z = Z+delta_t
        
        #     xploti = xplot[(icount-1)*NLES*NLES:(icount)*NLES*NLES]
        #     yploti = yplot[(icount-1)*NLES*NLES:(icount)*NLES*NLES]
            
            xv, yv, rv, pos, meanxy, covxy = multivariat_fit(xploti, yploti )
            # # plt.plot(xploti, yploti,'.k',alpha=0.05, markersize=2)
            # ax.scatter3D(meanxy[0],meanxy[1],Z, marker=".", color='red',s=10,linewidths=2)
            # ax.scatter3D(CS2,meanxy[1],Z, marker="o", color='green',s=50,linewidths=0.15)
            # ax.scatter3D(CS2EKI,meanxy[1],Z, marker="*", color='green',s=50,linewidths=0.15)
            
            # plt.xlabel(xplot_str)
            # plt.ylabel(yplot_str)
            # plt.grid(color='gray', linestyle='dashed')
            
            meanxy_M = np.append(meanxy_M, meanxy)
            meanxy_3d.append(meanxy+(Z,))
        
            lambda_, v = np.linalg.eig(covxy)
            lambda_ = np.sqrt(lambda_)
        
            ZDIR='z'
            j=1
            p = Ellipse(xy=meanxy,
                            width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                            angle=np.rad2deg(np.arccos(v[0, 0])),
                            facecolor='none',
                            edgecolor='blue',
                            alpha=ALPHA_L)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=Z*1.01, zdir=ZDIR)
            
            # j=2
            # p = Ellipse(xy=meanxy,
            #           width=lambda_[0]*j*2, height=lambda_[1]*j*2,
            #           angle=np.rad2deg(np.arccos(v[0, 0])),
            #         facecolor='none',
            #         edgecolor='red',
            #         alpha=ALPHA_L)
            
            # ax.add_patch(p)
            # art3d.pathpatch_2d_to_3d(p, z=Z, zdir=ZDIR)
            
            # p = Rectangle((-1,-20), 1,40, color="g", alpha=0.2)
            # ax.add_patch(p)
            # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
            
            yy, xx = np.meshgrid(np.linspace(-20,20,3), np.linspace(-0.20,0.30,3))
            zz = xx*0+Z
            ax = plt.subplot(projection='3d')
            ax.plot_surface(xx, yy, zz, alpha=ALPHA_S)
        
        
        # ax.view_init(azim=0, elev=0)
        # ax.view_init(azim=0, elev=90) # top view
        # ax.view_init(azim=10, elev=10)
        
    ax.set_xlabel(r'$c_s^2$')
    ax.set_ylabel(yplot_str)
    ax.set_zlabel(r'$t$')
    
    ax.set_title(xploti_str)

    # plt.ylim([-1,1])
    # plt.ylim([-0.5,0.5])
    
    
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-delta_t, Z)
    
    # Line plot of center of Cs2 in time
    meanxy_3d = np.array(meanxy_3d)
    ax.plot(meanxy_3d[:,0],meanxy_3d[:,1],meanxy_3d[:,2], marker=".")#, color='red',linewidths=2)
    
    
    yy, zz = np.meshgrid(np.linspace(-20,20,3), np.linspace(0,Z,3))
    xx = yy*0
    ax.plot_surface(xx, yy, zz, alpha=0.5+ALPHA_S)
    
    ##%%
    ii=270.5
    ax.view_init(elev=20., azim=ii)

    # for ii in range(1,360,30):
        # ax.view_init(elev=20., azim=ii)
        # plt.savefig("movie%d.png" % ii)
    # plt.show()