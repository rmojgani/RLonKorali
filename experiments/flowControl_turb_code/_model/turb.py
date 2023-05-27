#from numpy import pi
from scipy.fftpack import fft, ifft
import numpy as np
import time as time
from scipy.io import loadmat,savemat
import scipy as sp
from scipy.interpolate import RectBivariateSpline
from split2d import split2d
from split2d import pickcenter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr'
from scipy.stats import multivariate_normal

np.seterr(over='raise', invalid='raise')

# ---------------------- Forced turb
#lim = int(SPIN_UP ) # Start saving
#st = int( 1. / dt ) # How often to save data
NNSAVE = 10 
#
class turb:
    #
    # Solution of the 2D tub
    # Solve 2D turbulence by Fourier-Fourier pseudo-spectral method
    # Navier-Stokes equation is in the vorticity-stream function form
    #
    # u_t + ...  = 0,
    # with ... doubly periodic BC on ... .
    #
    # The nature of the solution depends on the Re=1/nu
    # condition u(x,0).  Energy enters the system

    #
    def __init__(self, 
                Lx=2.0*np.pi, Ly=2.0*np.pi, 
                NX=128,       NY=128, 
                dt=5e-4, nu=1e-4, rho=1.0, alpha=0.1, 
				nsteps=None, tend=1.5000, iout=1, u0=None, v0=None, 
                RL=False, 
                nActions=1, sigma=0.4,
                case='1', 
                rewardtype='k1', 
                statetype='enstrophy',
                actiontype='CL',
                nagents=2):
        #
        print('__init__')
        print('rewardtype', rewardtype[0:2])
        print('number of Actions=', nActions)
        print('number of Agents=', nagents)

        self.tic = time.time()
        if rewardtype[0]=='z':
            self.rewardtype ='enstrophy'
        elif rewardtype[0]=='k':
            self.rewardtype ='energy'
        if rewardtype[1]=='1':
            self.rewardfunc = '1'
        elif rewardtype[1]=='e':
            self.rewardfunc = 'e'
        elif rewardtype[1]=='c':
            self.rewardfunc = 'c'

        self.statetype= statetype
        self.actiontype= actiontype
        self.nagents= nagents
        self.nActions = nActions

        # Choose reward type function
        #if rewardtype[0] =='k':
        #    order = int(rewardtype[1])
        #    def myreward(self):
        #        return self.rewardk(self.mykrange(order), rewardtype[-1] )
        #elif rewardtype[0:2] == 'ratio':
        #    def mreward(self):
        #        return self.rewardratio()

        # Initialize
        L  = float(Lx); dt = float(dt); tend = float(tend)

        if (nsteps is None):
            nsteps = int(tend/dt)
            nsteps = 10
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps

        print('> tend=',tend,', nsteps=', nsteps, 'NxN',str(NX),'x',str(NY))

        self.case=case
        #
        # save to self
        self.Lx     = Lx
        self.Ly     = Ly
        self.NX     = NX
        self.NY     = NY
    	# ----------
        self.dt     = dt
        self.nu     = 1/(20e3)
        self.alpha  = 0.1
        self.nsteps = nsteps
        self.iout   = iout
        self.nout   = int(nsteps/iout)
        self.RL     = RL
        # ----------
        self.stepsave = 15000
        print('Init, ---->nsteps=', nsteps)
        # Operators and grid generator
        self.operatorgen()
	#
        # Grid gen
        self.v=0

        # set initial condition
        self.IC()

        # get targets for control:
        if self.RL:
#            self.nActions = nActions
            self.sigma = sigma
            self.x = np.arange(self.NX)*self.Lx/(self.NX-1)
        self.case = case
        #self.nActions = nActions

	    # SAVE SIZE
        slnU = np.zeros([NX,NNSAVE])
        slnV = np.zeros([NX,NNSAVE])
	    
        Energy = np.zeros([NNSAVE])
        Enstrophy = np.zeros([NNSAVE])
        onePython = np.zeros([NNSAVE])
	
        # precompute Gaussians for control:
        if self.RL:
            self.nActions = nActions
            self.sigma = sigma
            self.x = np.arange(self.N)*self.L/(self.N-1)
            self.veRL = 0
            #print('RL to run:', nActions)
   
    def mykrange(self, order):
        NX = int(self.NX)
        kmax = self.kmax
        krange = np.array(range(0, kmax))
        return krange**order
    
    def setup_reference(self):
        NX = self.NX
        kmax = self.kmax
        rewardtype = self.rewardtype
        if rewardtype == 'enstrophy':
            #print('Enstrophy as reference')
            spec_ref = self.ref_ens[0:kmax,1]
        elif rewardtype == 'energy':
            #print('Energy as reference')
            spec_ref = self.ref_tke[0:kmax,1]
        self.spec_ref = spec_ref

    def setup_target(self):
        NX = self.NX
        kmax = self.kmax
        rewardtype = self.rewardtype
        if rewardtype == 'enstrophy':
            #print('Enstrophy as reference')
            spec_now = self.enstrophy_spectrum()
        elif rewardtype == 'energy':
            #print('Energy as reference')
            spec_now = self.energy_spectrum()
        return spec_now

    def setup_reward(self):
        rewardtype = self.rewardtype
        krange = self.krange
        rewardfunc = self.rewardfunc

        reference  = self.spec_ref
        target = self.setup_target()

        if rewardfunc == '1' or rewardfunc == '3':
            myreward = 1/( np.linalg.norm( krange*(target-reference)  )**2 )
            #print(myreward)
        elif rewardfunc == 'c':
            myreward = - np.linalg.norm( (target-reference)  )**2
        elif rewardfunc == 'e':
            myreward = - np.linalg.norm( np.exp( (np.log(target)-np.log(reference))**2) )

        return myreward

    def mySGS(self, action):
        actiontype = self.actiontype
        if actiontype=='CL':
            nu = self.leith_cs(action)
        elif actiontype=='CS':
            nu = self.smag_cs(action)
        return nu

    def step( self, action=None ):
        '''
        2D Turbulence: One time step simulation of 2D Turbulence
        '''
        NX=self.NX

        forcing  = np.zeros(self.nActions)
        if (action is not None):
            #assert len(action) == self.nActions, print("Wrong number of actions. provided: {}, expected:{}".format(len(action), self.nActions))
            forcing = self.upsample(action)
            self.veRL = forcing#forcing[0]# For test
            #print(self.veRL)
            #stop_veRL
        #else:
        #    #self.veRL=0.17**2

        if self.stepnum % self.stepsave == 0:
            print(self.stepnum)
            #self.myplot()
            #savemat('N'+str(self.NX)+'_t='+str(self.stepnum)+'.mat',dict([('psi_hat', self.psi_hat),('w_hat', self.w1_hat)]))
            print('time:', np.round((time.time()-self.tic)/60.0,4),' min.')

        self.stepturb(action)
        self.sol = [self.w1_hat, self.psiCurrent_hat, self.w1_hat, self.psiPrevious_hat]
        self.stepnum += 1
        self.t       += self.dt
   

    def simulate(self, nsteps=None, iout=None, restart=False, correction=[]):
        nsteps=self.nsteps#int(1e4)
        #
        # If not provided explicitly, get internal values
        if (nsteps is None):
            nsteps = self.nsteps
        else:
            nsteps = int(nsteps)
            self.nsteps = nsteps
        if (iout is None):
            iout = self.iout
            nout = self.nout
        else:
            self.iout = iout
        if restart:
            # update nout in case nsteps or iout were changed
            nout      = int(nsteps/iout)
            self.nout = nout
            # reset simulation arrays with possibly updated size
            self.setup_timeseries(nout=self.nout)
        #
        # advance in time for nsteps steps
        for n in range(1,self.nsteps+1):
            self.step()

    def state(self):
        NX= int(self.NX)
        kmax= self.kmax
        statetype=self.statetype
        nagents=self.nagents
        # --------------------------------------
        STATE_GLOBAL=True
        # --------------------------------------
        if statetype=='psiomegadiag':
            s1= np.diag(np.real(np.fft.ifft2(self.w1_hat))).reshape(-1,)
            s2= np.diag(np.real(np.fft.ifft2(self.psiCurrent_hat))).reshape(-1,)
            mystate= np.hstack((s1,s2))
        # --------------------------
        elif statetype=='enstrophy':
            enstrophy= self.enstrophy_spectrum()
            mystate= np.log(enstrophy[0:kmax])
        # --------------------------
        elif statetype=='energy':
            energy= self.energy_spectrum()
            mystate= np.log(energy[0:kmax])
        # --------------------------
        elif statetype=='psiomega':
           '''
           self.sol = [self.w1_hat, self.psiCurrent_hat, self.w1_hat, self.psiPrevious_hat]

           '''
           STATE_GLOBAL=False
           s1 = np.real(np.fft.ifft2(self.sol[0])) #w1
           s2 = np.real(np.fft.ifft2(self.sol[1])) #psi
        # --------------------------
        elif statetype=='omega':
           '''
           self.sol = [self.w1_hat, self.psiCurrent_hat, self.w1_hat, self.psiPrevious_hat]

           '''
           STATE_GLOBAL=False
           s1 = np.real(np.fft.ifft2(self.sol[0])) #w1
        # --------------------------
        elif statetype=='psiomegalocal':
           STATE_GLOBAL=False
           s1 = np.real(np.fft.ifft2(self.sol[0])) #w1
           s2 = np.real(np.fft.ifft2(self.sol[1])) #psi
           # --------------------------
        elif statetype=='invariantlocal':
           STATE_GLOBAL=False
           #s1 = np.real(np.fft.ifft2(self.sol[0])) #w1
           s2 = np.real(np.fft.ifft2(self.sol[1])) #psi


        if STATE_GLOBAL:
            mystatelist = [mystate.tolist()]
            for _ in range(nagents-1):
                mystatelist.append(mystate.tolist())

        elif not STATE_GLOBAL:
            if statetype=='psiomega':
                mystatelist1 =  split2d(s1, self.nActiongrid)
                mystatelist2 =  split2d(s2, self.nActiongrid)
                mystatelist = [x+y for x,y in zip(mystatelist1, mystatelist2)]
            elif statetype=='omega':
                mystatelist =  split2d(s1, self.nActiongrid)
            elif statetype=='psiomegalocal':
                NX = self.NX
                NY = self.NY
                mystatelist1 =  pickcenter(s1, NX, NY, self.nActiongrid)
                mystatelist2 =  pickcenter(s2, NY, NY, self.nActiongrid)
                mystatelist = [x+y for x,y in zip(mystatelist1, mystatelist2)]
            elif statetype=='invariantlocal':
                NX = self.NX
                NY = self.NY
                Kx = self.Kx
                Ky = self.Ky

                mystatelist2 =  pickcenter(s2, NX, NY, self.nActiongrid)

                lambdalist = []

                for psi_hat in mystatelist2:
                    u1_hat = self.D_dir(psi_hat,Ky) # u_hat = (1j*Ky)*psi_hat
                    v1_hat = -self.D_dir(psi_hat,Kx) # v_hat = -(1j*Kx)*psi_hat

                    dudx_hat = self.D_dir(u1_hat,Kx)
                    dudy_hat = self.D_dir(u1_hat,Ky)
    
                    dvdx_hat = self.D_dir(v1_hat,Kx)
                    dvdy_hat = self.D_dir(v1_hat,Ky)
                    
                    dudx = np.fft.ifft2(dudx_hat)
                    dudy = np.fft.ifft2(dudy_hat)
                    dvdx = np.fft.ifft2(dvdx_hat)
                    dvdy = np.fft.ifft2(dvdy_hat)

                    gradV = np.array([[dudx,dudy],[dvdx,dvdy]])
                    
                    lambdalist.append(self.invariant(gradV))

        if mystatelist[0][0]>1000: raise Exception("State diverged!")
        return mystatelist

   
    def reward(self):
        nagents=self.nagents
        # --------------------------------------
        try:
            myreward=self.setup_reward()
        except:
            myreward=-10000
        # --------------------------
        myrewardlist = [myreward.tolist()]
        for _ in range(nagents-1):
            myrewardlist.append(myreward.tolist())
        return myrewardlist 

    def convection_conserved(self, psiCurrent_hat, w1_hat):#, Kx, Ky):
        Kx = self.Kx
        Ky = self.Ky
        
        # Velocity
        u1_hat = -(1j*Ky)*psiCurrent_hat
        v1_hat = (1j*Kx)*psiCurrent_hat
        
        # Convservative form
        w1 = np.real(np.fft.ifft2(w1_hat))
        conu1 = 1j*Kx*np.fft.fft2((np.real(np.fft.ifft2(u1_hat))*w1))
        conv1 = 1j*Ky*np.fft.fft2((np.real(np.fft.ifft2(v1_hat))*w1))
        convec_hat = conu1 + conv1
     
        # Non-conservative form
        w1x_hat = 1j*Kx*w1_hat
        w1y_hat = 1j*Ky*w1_hat
        conu1 = np.fft.fft2(np.real(np.fft.ifft2(u1_hat))*np.real(np.fft.ifft2(w1x_hat)))
        conv1 = np.fft.fft2(np.real(np.fft.ifft2(v1_hat))*np.real(np.fft.ifft2(w1y_hat)))
        convecN_hat = conu1 + conv1
  
        convec_hat = 0.5*(convec_hat + convecN_hat)
        return convec_hat

    def stepturb(self, action):
        #psiCurrent_hat = self.psiCurrent_hat
        #w1_hat = self.w1_hat
       	Ksq = self.Ksq
        invKsq = self.invKsq
        dt = self.dt
        nu = self.nu
        alpha = self.alpha
        Fk = self.Fk
        # ---------------
        psiCurrent_hat = self.psiCurrent_hat
        w1_hat = self.w1_hat
        convec0_hat = self.convec1_hat
        # 2 Adam bash forth Crank Nicolson
        convec1_hat = self.convection_conserved(psiCurrent_hat, w1_hat)
       	diffu_hat = -Ksq*w1_hat
       
        # Calculate SGS diffusion 
        ve = self.mySGS(action)
#        ve = 0
        RHS = w1_hat + dt*(-1.5*convec1_hat+0.5*convec0_hat) + dt*0.5*(nu+ve)*diffu_hat+dt*Fk
       	RHS[0,0] = 0
    
       	psiTemp = RHS/(1+dt*alpha+0.5*dt*(nu+ve)*Ksq)
    
        w0_hat = w1_hat
        w1_hat = psiTemp
        convec0_hat = convec1_hat

        # Poisson equation for Psi
        psiPrevious_hat = psiCurrent_hat
        psiCurrent_hat = -w1_hat*invKsq

        # Update this step
        self.update(w0_hat, w1_hat, convec0_hat, convec1_hat, psiPrevious_hat, psiCurrent_hat, ve )

    def update(self, w0_hat, w1_hat, convec0_hat, convec1_hat, psiPrevious_hat, psiCurrent_hat, ve):
        # write to self
        self.w0_hat = w0_hat
        self.w1_hat = w1_hat
        self.convec0_hat = convec0_hat
        self.convec1_hat = convec1_hat # it is never used, consider deleting 
        self.psiPrevious_hat = psiPrevious_hat
        self.psiCurrent_hat = psiCurrent_hat
        self.ve = ve
        #self.velist.append(self.veRL)
        self.myrewardlist=[]
        self.mystatelist=[]

    def IC(self, u0=None, v0=None, SEED=42):
        X = self.X
        Y = self.Y
        NX = self.NX
        NY = self.NY
        Kx = self.Kx
        Ky = self.Ky
        invKsq = self.invKsq
        # ------------------
        np.random.seed(SEED)
        # ------------------
        kp = 10.0
        A  = 4*np.power(kp,(-5))/(3*np.pi)  
        absK = np.sqrt(Kx*Kx+Ky*Ky)
        #
        Ek = A*np.power(absK,4)*np.exp(-np.power(absK/kp,2))
        coef1 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
        coef2 = np.random.uniform(0,1,[NX//2+1,NX//2+1])*np.pi*2
        #
        perturb = np.zeros([NX,NX])
        perturb[0:NX//2+1, 0:NX//2+1] = coef1[0:NX//2+1, 0:NX//2+1]+coef2[0:NX//2+1, 0:NX//2+1]
        perturb[NX//2+1:, 0:NX//2+1] = coef2[NX//2-1:0:-1, 0:NX//2+1] - coef1[NX//2-1:0:-1, 0:NX//2+1]
        perturb[0:NX//2+1, NX//2+1:] = coef1[0:NX//2+1, NX//2-1:0:-1] - coef2[0:NX//2+1, NX//2-1:0:-1]
        perturb[NX//2+1:, NX//2+1:] = -(coef1[NX//2-1:0:-1, NX//2-1:0:-1] + coef2[NX//2-1:0:-1, NX//2-1:0:-1])
        perturb = np.exp(1j*perturb)
	    # omega
        w1_hat = np.sqrt(absK/np.pi*Ek)*perturb*np.power(NX,2)
        # psi
        psi_hat         = -w1_hat*invKsq
        psiPrevious_hat = psi_hat.astype(np.complex128)
        psiCurrent_hat  = psi_hat.astype(np.complex128)
        # Forcing
        if self.case=='1':
            n = 4
        elif self.case=='4':
            n = 25

        Xi = 1
        Fk = -n*Xi*np.cos(n*Y)-n*Xi*np.cos(n*X)
        Fk = np.fft.fft2(Fk)
        #
        time = 0.0
        slnW = []
        
        if self.case =='1':
            folder_path = '_init/Re20kf4/iniWor_Re20kf4_'
        elif self.case == '4':
            folder_path = '_init/Re20kf25/iniWor_Re20kf25_'

        filenum_str=str(1)
        data_Poi = loadmat(folder_path+str(NX)+'_'+filenum_str+'.mat')
        w1 = data_Poi['w1']
        
        if self.case =='4':
            ref_tke = np.loadtxt("_init/Re20kf25/energy_spectrum_Re20kf25_DNS1024_xy.dat")
            ref_ens = np.loadtxt("_init/Re20kf25/enstrophy_spectrum_Re20kf25_DNS1024_xy.dat")

        if self.case == '1':
            ref_tke = np.loadtxt("_init/Re20kf4/energy_spectrum_DNS1024_xy.dat")
            ref_ens = np.loadtxt("_init/Re20kf4/enstrophy_spectrum_DNS1024_xy.dat")
 
        w1_hat = np.fft.fft2(w1)
        psiCurrent_hat = -invKsq*w1_hat
        psiPrevious_hat = psiCurrent_hat
    
        # ... and save to self
        self.w1_hat = w1_hat
        self.psi_hat = psi_hat
        self.psiCurrent_hat = psiCurrent_hat
        self.psiPrevious_hat = psiPrevious_hat
        self.t = 0.0
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        self.sol = [self.w1_hat, self.psiCurrent_hat, self.w1_hat, self.psiPrevious_hat]
        # 
        convec0_hat = self.convection_conserved(psiCurrent_hat, w1_hat)
        self.convec0_hat = convec0_hat
        self.convec1_hat = convec0_hat
        # 
        self.Fk = Fk
        self.Fn = n # Forcing k
        # Aux reward 
        kmax = self.kmax
        krange = np.array(range(0, kmax))
        self.krange = krange
        # SGS Model
        self.ve = 0
        #self.velist = []
        # Reference files 
        self.ref_tke = ref_tke
        self.ref_ens = ref_ens
        # temporary
        self.N = NX
        self.L = 2*np.pi
        # 
        self.setup_reference()
        self.setup_MAagents()

    def setup_MAagents(self):
        # Copied from:   f36df60 on main  
        # temporary
        nActiongrid = int((self.nActions*self.nagents)**0.5)
        self.nActiongrid = nActiongrid
        # Initlize action
        X = np.linspace(0,self.L,nActiongrid, endpoint=True)
        Y = np.linspace(0,self.L,nActiongrid, endpoint=True)
        self.xaction = X
        self.yaction = Y

    def upsample(self, action): 
        action_flat = [item for sublist in action for item in sublist]
        arr_action = np.array(action_flat).reshape(self.nActiongrid, self.nActiongrid)
        upsample_action = RectBivariateSpline(self.xaction, self.yaction, arr_action, kx=1, ky=1)

        # Initlize action
        upsamplesize = self.NX # 1 for testing, will be changed to grid size eventually
        x2 = np.linspace(0,self.L, upsamplesize, endpoint=True)
        y2 = np.linspace(0,self.L,  upsamplesize, endpoint=True)
        forcing = upsample_action(x2, y2)
        return forcing

    def operatorgen(self):
        Lx = self.Lx
        NX = self.NX
        dx = Lx/NX
        #-----------------  
        x        = np.linspace(0, Lx-dx, num=NX)
        kx       = (2*np.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),
                                                np.arange((-NX/2+1),0,dtype=np.float64)
                                                ))   
        [Y,X]    = np.meshgrid(x,x)
        [Ky,Kx]  = np.meshgrid(kx,kx)
        Ksq      = (Kx**2 + Ky**2)
        Kabs     = np.sqrt(Ksq)
        Ksq[0,0] = 1e12
        invKsq   = 1/Ksq
        Ksq[0,0] = 0
        invKsq[0,0] = 0
        kmax = int(NX/2)
	    # .... and save to self
        self.X = X
        self.Y = Y
        self.dx = dx
        self.kx = kx
        self.Ky = Ky
        self.Kx = Kx
        self.Ksq = Ksq
        self.Kabs = Kabs
        self.invKsq = invKsq
        self.kmax = kmax
    #-----------------------------------------
    # ============= SGS Models ===============
    #-----------------------------------------
    def leith_cs(self, action=None):
        '''
        ve =(Cl * \delta )**3 |Grad omega|  LAPL omega ; LAPL := Grad*Grad
        '''
        #print('action is:', action_leith)
        if action != None:
        #    if self.veRL !=0:
            CL3 = self.veRL#action_leith[0]
        else:
            CL3 = 0.17**3# (Lit)
        #else:
        Kx = self.Kx
        Ky = self.Ky
        w1_hat = self.w1_hat

        w1x_hat = -(1j*Kx)*w1_hat
        w1y_hat = (1j*Ky)*w1_hat
        w1x = np.real(np.fft.ifft2(w1x_hat))
        w1y = np.real(np.fft.ifft2(w1y_hat))
        abs_grad_omega = np.mean(np.sqrt( w1x**2+w1y**2  ))
        # 
        delta3 = (2*np.pi/self.NX)**3
        ve = CL3*delta3*abs_grad_omega
        return ve

    def smag_cs(self, action=None):
        Kx = self.Kx
        Ky = self.Ky
        NX = self.NX
        psiCurrent_hat = self.psiCurrent_hat
        w1_hat = self.w1_hat

        if action != None:
            cs = (self.veRL) * ((2*np.pi/NX )**2)  # for LX = 2 pi
        else:
            #self.veRL = 0.17 * 2
            #cs = (self.veRL) * ((2*np.pi/NX )**2)  # for LX = 2 pi
            cs = (0.17 * 2*np.pi/NX )**2  # for LX = 2 pi

        S1 = np.real(np.fft.ifft2(-Ky*Kx*psiCurrent_hat)) # make sure .* 
        S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psiCurrent_hat))
        S  = 2.0*(S1*S1 + S2*S2)**0.5
#        cs = (0.17 * 2*np.pi/NX )**2  # for LX = 2 pi
        S = (np.mean(S**2.0))**0.5;
        ve = cs*S
        return ve
    #-----------------------------------------
    def enstrophy_spectrum(self):
        NX = self.NX
        NY = self.NY # Square for now
        w1_hat = self.w1_hat
        #-----------------------------------
        signal = np.power(abs(w1_hat),2)/2;
    
        spec_x = np.mean(np.abs(signal),axis=0)
        spec_y = np.mean(np.abs(signal),axis=1)
        spec = (spec_x + spec_y)/2
        spec = spec/ (NX**2)/NX
        spec = spec[0:int(NX/2)]
    
        self.enstrophy_spec = spec
        return spec
    #-----------------------------------------
    def energy_spectrum(self):
        NX = self.NX
        NY = self.NY # Square for now
        Ksq = self.Ksq
        w1_hat = self.w1_hat
    
        Ksq[0,0]=1
        w_hat = np.power(np.abs(w1_hat),2)/NX/NY/Ksq
        w_hat[0,0]=0;
        spec_x = np.mean(np.abs(w_hat),axis=0)
        spec_y = np.mean(np.abs(w_hat),axis=1)
        spec = (spec_x + spec_y)/2
        spec = spec /NX
        
        spec=spec[0:int(NX/2)]
        return  spec
    #-----------------------------------------
    def myplot(self, append_str='', prepend_str=''):
        NX = int(self.NX)
        Kplot = self.Kx; kplot_str = '\kappa_{x}'; kmax = self.kmax
        #Kplot = self.Kabs; kplot_str = '\kappa_{sq}'; kmax = int(np.sqrt(2)*NX/2)+1
        #kplot_str = '\kappa_{sq}'
        stepnum = self.stepnum
        ve = self.ve
        Fn = self.Fn
        dt = self.dt
        # --------------
        energy = self.energy_spectrum()
        enstrophy = self.enstrophy_spectrum()
        #
        plt.figure(figsize=(8,14))
 
        omega = np.real(np.fft.ifft2(self.sol[0]))
        VMAX, VMIN = np.max(omega), np.min(omega)
        VMAX = max(np.abs(VMIN), np.abs(VMAX))
        VMIN = -VMAX
        levels = np.linspace(VMIN,VMAX,100)

        plt.subplot(3,2,1)
        plt.contourf(omega, levels, vmin=VMIN, vmax=VMAX); plt.colorbar()
        plt.title(r'$\omega$')

        psi = np.real(np.fft.ifft2(self.sol[1]))
        VMAX, VMIN = np.max(psi), np.min(psi)
        VMAX = max(np.abs(VMIN), np.abs(VMAX))
        VMIN = -VMAX
        levels = np.linspace(VMIN,VMAX,100)
 
        plt.subplot(3,2,2)
        plt.contourf(psi, levels, vmin=VMIN, vmax=VMAX); plt.colorbar()
        plt.title(r'$\psi$')
        
        ref_tke = self.ref_tke#np.loadtxt("tke.dat")
        # Energy 
        plt.subplot(3,2,3)
        energy = self.energy_spectrum()
        plt.loglog(Kplot[0:kmax,0], energy[0:kmax],'k')
        plt.plot([self.Fn,self.Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
        plt.plot(ref_tke[:,0],ref_tke[:,1],':k', alpha=0.25, linewidth=4)
        plt.title(r'$\hat{E}$'+rf'$({kplot_str})$')
        plt.xlabel(rf'${kplot_str}$')
        plt.xlim([1,1e3])
        plt.ylim([1e-6,1e0])
        
        ref_ens = self.ref_ens#np.loadtxt("ens.dat")
        # Enstrophy
        plt.subplot(3,2,4)
        enstrophy = self.enstrophy_spectrum()
        plt.plot([self.Fn,self.Fn],[1e-6,1e6],':k', alpha=0.5, linewidth=2)
        plt.plot(ref_ens[:,0],ref_ens[:,1],':k', alpha=0.25, linewidth=4)
        plt.loglog(Kplot[0:kmax,0], enstrophy[0:kmax],'k')
        plt.title(rf'$\varepsilon({kplot_str})$')
        plt.xlabel(rf'${kplot_str}$')
        plt.xlim([1,1e2])
        #plt.ylim([1e-5,1e0])
        plt.ylim([1e-3,1e1])
        #plt.pcolor(np.real(sim.w1_hat));plt.colorbar()
        
        #plt.subplot(3,2,5)
        #omega = np.real(np.fft.ifft2(self.w1_hat))
        #Vecpoints, exp_log_kde, log_kde, kde = self.KDEof(omega)
        #plt.semilogy(Vecpoints,exp_log_kde)  
        #plt.xlabel(r'$\omega$')
        #plt.ylabel('PDF')
        #plt.title('$t=$'+f"{stepnum*dt:.2E}"+r'$, \nu=$'+f"{ve:.2E}")
 

        Kx = self.Kx
        Ky = self.Ky
        psi_hat = self.psiCurrent_hat
        v_hat = -(1j*Kx)*psi_hat
        u_hat = (1j*Ky)*psi_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))

        plt.subplot(3,2,5)
        plt.pcolor(u)
        plt.subplot(3,2,6)
        plt.pcolor(v)
        plt.title('v')
        plt.colorbar()
        #plt.subplot(3,2,6)
        #plt.semilogy(Vecpoints,log_kde) 
        filename = prepend_str+'2Dturb_'+str(stepnum)+append_str
        plt.savefig(filename+'.png', bbox_inches='tight', dpi=450)
        plt.close('all')
#        print(filename)
#        print(Kplot[0:kmax,0].shape)
#        print( energy[0:kmax].shape)
#        print( np.stack((Kplot[0:kmax,0], energy[0:kmax]),axis=0).T.shape   )
        np.savetxt(filename+'_tke.out', np.stack((Kplot[0:kmax,0], energy[0:kmax]),axis=0).T, delimiter='\t')
        np.savetxt(filename+'_ens.out', np.stack((Kplot[0:kmax,0], enstrophy[0:kmax]),axis=0).T, delimiter='\t')

    #-----------------------------------------
    def myplotforcing(self, append_str='', prepend_str=''):
        NX = int(self.NX)
        Kx = self.Kx
        Ky = self.Ky
        w1_hat = self.w1_hat
        omega = np.real(np.fft.ifft2(self.sol[0]))
        w1x_hat = -(1j*Kx)*w1_hat
        w1y_hat = (1j*Ky)*w1_hat
        w1x = np.real(np.fft.ifft2(w1x_hat))
        w1y = np.real(np.fft.ifft2(w1y_hat))
        grad_omega = np.sqrt( w1x**2+w1y**2)

        veRL=self.veRL
        stepnum = self.stepnum

        plt.figure(figsize=(8,14))
        levels = np.linspace(-30,30,100)

        plt.subplot(3,2,1)
        plt.contourf(veRL)
        plt.colorbar()
        plt.title(r'forcing')

        plt.subplot(3,2,3)
        plt.contourf(grad_omega)
        plt.colorbar()
        plt.title(r'$\nabla \omega$')

        plt.subplot(3,2,2)
        xplot = veRL.reshape(-1,1)
        yplot = omega.reshape(-1,1)

        xv, yv, rv, pos, meanxy = self.multivariat_fit(xplot,yplot)
        plt.plot(xplot, yplot,'.k',alpha=0.5)
        plt.scatter(meanxy[0],meanxy[1], marker="+", color='red',s=100)
        plt.contour(xv, yv, rv.pdf(pos))

        plt.xlabel(r'$forcing$')
        plt.ylabel(r'$\omega$')
        plt.grid(color='gray', linestyle='dashed')

        plt.subplot(3,2,4)
        xplot = veRL.reshape(-1,1)
        yplot = grad_omega.reshape(-1,1)
        xv, yv, rv, pos, meanxy = self.multivariat_fit(xplot,yplot)
        plt.plot(xplot, yplot,'.k',alpha=0.5)
        plt.scatter(meanxy[0],meanxy[1], marker="+", color='red',s=100)
        plt.contour(xv, yv, rv.pdf(pos))

        plt.xlabel(r'$forcing$')
        plt.ylabel(r'$\nabla \omega$')
        plt.grid(color='gray', linestyle='dashed')

        filename = prepend_str+'2Dturb_'+str(stepnum)+'forcing'+append_str
        plt.savefig(filename+'.png', bbox_inches='tight', dpi=450)
        plt.close('all')
    #-----------------------------------------  
    def multivariat_fit(self,x,y):
        covxy = np.cov(x,y, rowvar=False)
        meanxy=np.mean(x),np.mean(y)
        rv = multivariate_normal(mean=meanxy, cov=covxy, allow_singular=False)
        xv, yv = np.meshgrid(np.linspace(x.min(),x.max(),50), 
                             np.linspace(y.min(),y.max(),50), indexing='ij')
        pos = np.dstack((xv, yv))

        return xv, yv, rv, pos, meanxy 
    #-----------------------------------------
    def KDEof(self, u):
        from PDE_KDE import myKDE
        Vecpoints, exp_log_kde, logkde, kde = myKDE(u)
        return Vecpoints, exp_log_kde, logkde, kde
    #-----------------------------------------
    def D_dir(u_hat, K_dir):
        Du_Ddir = 1j*K_dir*u_hat
        return Du_Ddir  
    #-----------------------------------------
    def decompose_sym(A):
        S = 0.5*(A+A.T)
        R = 0.5*(A-A.T)
        return S, R
    #-----------------------------------------
    def invariant(A):
        S, R = self.decompose_sym(A)
        lambda1 = np.trace(S)
        lambda2 = np.trace(S@S)
        lambda3 = np.trace(R@R)
        return [lambda1, lambda2, lambda3]
