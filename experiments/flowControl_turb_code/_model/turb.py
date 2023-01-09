#from numpy import pi
from scipy.fftpack import fft, ifft
import numpy as np
import time as time
from scipy.io import loadmat,savemat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr'

np.seterr(over='raise', invalid='raise')

# ---------------------- Forced turb
import math
# ---------------------- QG

#ts = int(SPIN_UP*2) # Total timesteps
#tot_time = dt*ts    # Length of run
#lim = int(SPIN_UP ) # Start saving
#st = int( 1. / dt ) # How often to save data
NNSAVE = 10 
#
def gaussian( x, mean, sigma ):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*( (x-mean)/sigma )**2)

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
    # Spatial  discretization: Spectral (Fourier)
    # Temporal discretization: 
    #
    def __init__(self, 
                Lx=2.0*math.pi, Ly=2.0*math.pi, 
                NX=128, NY=128, 
                dt=5e-4, nu=1e-4, rho=1.0, alpha=0.1, 
				nsteps=None, tend=1.5000, iout=1, u0=None, v0=None, 
                RL=False, 
                nActions=1, sigma=0.4, case='1', rewardtype='k1' ):
        #
        print('__init__')
        print('rewardtype', rewardtype[0:2])
        self.tic = time.time()
        

        # Choose reward type function
        if rewardtype[0] =='k':
            order = int(rewardtype[1])
            self.reward = lambda :self.rewardk(self.mykrange(order), rewardtype[-1] )
        elif rewardtype[0:2] == 'ratio':
            self.reward = lambda :self.rewardratio()

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
        #
        self.stepsave = 15000
        print('Init, ---->nsteps=', nsteps)
        # Operators and grid generator
        self.operatorgen()
	#
        # Grid gen
        self.v=0

        # set initial condition
#        if (u0 is None) or (v0 is None):
#			# turb:
#			# Initial condition
        self.IC()


        # get targets for control:
        if self.RL:
            self.nActions = nActions
            self.sigma = sigma
            self.x = np.arange(self.NX)*self.Lx/(self.NX-1)
            self.case = case
            self.setup_targets()
            self.setup_gaussians()
	
	    # SAVE SIZE
        slnU = np.zeros([NX,NNSAVE])
        slnV = np.zeros([NX,NNSAVE])
	    
        Energy = np.zeros([NNSAVE])
        Enstrophy = np.zeros([NNSAVE])
        onePython = np.zeros([NNSAVE])
	
        #
        # initialize simulation arrays
        # self.setup_timeseries()
        #
        # precompute Fourier-related quantities
        #KS: self.setup_fourier()
        #
        # precompute ETDRK4 scalar quantities:
        #KS: self.setup_etdrk4()
        #
        # precompute Gaussians for control:
        if self.RL:
            self.nActions = nActions
            self.sigma = sigma
            self.x = np.arange(self.N)*self.L/(self.N-1)
            self.setup_gaussians()
            self.veRL = 0
            #print('RL to run:', nActions)
   
    def mykrange(self, order):
        NX = int(self.NX)
        kmax = int(NX/2)+1
        krange = np.array(range(0, kmax))
        return krange**order
    
    def setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        # nout+1 so we store the IC as well
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        #
        # store the IC in [0]
        self.vv[0,:] = self.v0
        self.tt[0]   = 0.


        #KS:def setup_fourier(self, coeffs=None):
        
        #KS: def setup_etdrk4(self):

    def setup_gaussians(self):
        print('setup_gaussian')
        self.gaussians = np.zeros((self.nActions, self.NX))
        #self.gaussians = np.zeros((self.nActions, 1))
        for i in range(self.nActions):
            mean = i*self.Lx/4
            #self.gaussians[i,:] = gaussian( self.x, mean, self.sigma )
            self.gaussians[i,:] = gaussian( mean, mean, self.sigma )
        #print(self.gaussians)
        #stop

    def setup_targets(self):
        NX = self.NX
        kmax = int(NX/2)+1
        self.targets = np.zeros((2,kmax))
        self.targets[0,:] = self.ref_tke[0:kmax,1] #np.loadtxt("_model/tke.dat")[0:kmax,1]
        self.targets[1,:] = self.ref_ens[0:kmax,1] #np.loadtxt("_model/ens.dat")[0:kmax,1]
    
    def step( self, action=None ):
        '''
        2D Turbulence: One time step simulation of 2D Turbulence
        '''
        forcing  = np.zeros(self.N)
        #Fforcing = np.zeros(self.N)#QG?
        if (action is not None):
            assert len(action) == self.nActions, print("Wrong number of actions. provided {}/{}".format(len(action), self.nActions))
            for i, a in enumerate(action):
                forcing += a #*self.gaussians[i,:]
                #Fforcing = fft( forcing )
            #print('forcing',forcing)
            #stop
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        # Action
        if (action is not None):
            #print(forcing.shape)
            self.veRL = forcing[0]
            #print(self.veRL)
            #stop_veRL
#        else:
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
        if (correction==[]):
            for n in range(1,self.nsteps+1):
                #try:
                self.step()
                #except FloatingPointError:
                ##
                ## something exploded
                ## cut time series to last saved solution and return
                #    self.nout = self.ioutnum
                #    self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
                #    self.tt.resize(self.nout+1)      # nout+1 because the IC is in [0]
                #return -1
        #    if ( (self.iout>0) and (n%self.iout==0) ):
        #        self.ioutnum += 1
        #        self.vv[self.ioutnum,:] = self.v
        #        self.tt[self.ioutnum]   = self.t
        else:
            # lots of code duplication here, but should improve speed instead of having the 'if correction' at every time step
            for n in range(1,self.nsteps+1):
                try:
                     self.step()
                     self.v += correction
                except FloatingPointError:
                    #
                    # something exploded
                    # cut time series to last saved solution and return
                    self.nout = self.ioutnum
                    self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
                    self.tt.resize(self.nout+1)      # nout+1 because the IC is in [0]
                    return -1
            if ( (self.iout>0) and (n%self.iout==0) ):
                self.ioutnum += 1
                self.vv[self.ioutnum,:] = self.v
                self.tt[self.ioutnum]   = self.t


    def fou2real(self):
        # Convert from spectral to physical space
        self.uu = np.real(ifft(self.vv))

    def state(self):
        s1 = np.diag(np.real(np.fft.ifft2(self.w1_hat))).reshape(-1,)
        s2 = np.diag(np.real(np.fft.ifft2(self.psiCurrent_hat))).reshape(-1,)
        return np.hstack((s1,s2))
    
        #NX = int(self.NX)
        #kmax = int(NX/2)+1

        #energy = self.energy_spectrum()

        #return np.log(energy[0:kmax])

    def rewardk(self, krange, rewardtype):
        # use some self.variable to calculate the reward
        NX = int(self.NX)
        kmax = int(NX/2)+1
        
        if rewardtype == 'z':
            #print('enssssssssssssssssssss')
            spec_now = self.enstrophy_spectrum()[0:kmax]
            spec_ref = self.ref_ens[0:kmax,1]
        elif rewardtype == 'e':
            #print('energyyyyyyyyyyyyyyyy')
            spec_now = self.energy_spectrum()[0:kmax]
            spec_ref = self.ref_ens[0:kmax,1]

        myreward = 1/( np.linalg.norm( krange*(spec_now - spec_ref)  )**2 )
        return myreward

    def rewardratio(self,rewardtype):
        # use some self.variable to calculate the reward
        NX = int(self.NX)
        kmax = int(NX/2)+1

        if rewardtype == 'z':
            #print('enssssssssssssssssssss')
            spec_now = self.enstrophy_spectrum()[0:kmax]
            spec_ref = self.ref_ens[0:kmax,1]
        elif rewardtype == 'e':
            #print('energyyyyyyyyyyyyyyyy')
            spec_now = self.energy_spectrum()[0:kmax]
            spec_ref = self.ref_ens[0:kmax,1]

        myreward = -np.linalg.norm( (spec_now - spec_ref)  / np.linalg.norm( spec_ref)  )
        return myreward

    def compute_Ek(self):
        #
        # compute all forms of kinetic energy
        #
        # Kinetic energy as a function of wavenumber and time
        self.compute_Ek_kt()
        # Time-averaged energy spectrum as a function of wavenumber
        self.Ek_k = np.sum(self.Ek_kt, 0)/(self.ioutnum+1) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0]
        # Total kinetic energy as a function of time
        self.Ek_t = np.sum(self.Ek_kt, 1)
        # Time-cumulative average as a function of wavenumber and time
        self.Ek_ktt = np.cumsum(self.Ek_kt, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero
        # Time-cumulative average as a function of time
        self.Ek_tt = np.cumsum(self.Ek_t, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

    def compute_Ek_kt(self):
        try:
            self.Ek_kt = 1./2.*np.real( self.vv.conj()*self.vv / self.N ) * self.dx
        except FloatingPointError:
            #
            # probable overflow because the simulation exploded, try removing the last solution
            problem=True
            remove=1
            self.Ek_kt = np.zeros([self.nout+1, self.N]) + 1e-313
            while problem:
                try:
                    self.Ek_kt[0:self.nout+1-remove,:] = 1./2.*np.real( self.vv[0:self.nout+1-remove].conj()*self.vv[0:self.nout+1-remove] / self.N ) * self.dx
                    problem=False
                except FloatingPointError:
                    remove+=1
                    problem=True
            return self.Ek_kt


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
        ve = self.leith_cs(w1_hat, action)
#        ve = self.smag_cs(w1_hat, action)
#        print(ve1, ve2)
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
        self.convec1_hat = convec1_hat
        self.psiPrevious_hat = psiPrevious_hat
        self.psiCurrent_hat = psiCurrent_hat
        self.ve = ve
        self.velist.append(self.veRL)

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
        
        if NX==128:
            if self.case =='1':
                data_Poi = loadmat('iniWor_128x128.mat')
            elif self.case == '4':
                data_Poi = loadmat('iniWor_Re20kf25_128_1.mat')


        if NX==64:
            if self.case == '1':
                data_Poi = loadmat('iniWor_64x64.mat')
            elif self.case == '4':
                data_Poi = loadmat('iniWor_Re20kf25_64_1.mat')

        if NX==16:
#            if self.case == '1':
           data_Poi = loadmat('iniWor_16_5.mat')

        w1 = data_Poi['w1']
        
        if self.case =='4':
#             ref_tke = np.loadtxt("tke_case04.dat") #instantaneous
#             ref_ens = np.loadtxt("ens_case04.dat") #instantaneous
             ref_tke = np.loadtxt("tke_case04_fdns.dat")#Averaged 
             ref_ens = np.loadtxt("ens_case04_fdns.dat")#Averaged

        if self.case == '1':
            ref_tke = np.loadtxt("tke.dat")
            ref_ens = np.loadtxt("ens.dat")
        
 
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
        # SGS Model
        self.ve = 0
        self.velist = []
        # Reference files 
        self.ref_tke = ref_tke
        self.ref_ens = ref_ens
        print('init: omega, psi')
        #print(w1_hat.shape)
        #print(psi_hat.shape)

        # temporary
        self.N = NX
        self.L = 2*np.pi
        #
        # Set initial condition
        '''
        if (v0 is None):
            if (u0 is None):
                # set u0
                if self.RL:
                    # initial condition for chosen RL case
                    #if self.case == "E31":
                    #    u0 = self.targets[2,:]
                    #elif self.case == "E12":
                    #    u0 = self.targets[0,:]
                    #elif self.case == "E23":
                    #    u0 = self.targets[1,:]
                    #else:
                    #    assert False, print("RL case {} unknown...".format(self.case))
                    u0 = self.targets[1,:]
                else:
                    print("Using random initial condition...")
                    # uniform noise
                    # u0 = (np.random.rand(self.N) -0.5)*0.01
                    # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                    np.random.seed( SEED )
                    u0 = np.random.normal(0., 1e-4, self.N)
            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    print('Error: wrong IC array size')
                    return -1
                else:
                    print("Using given (real) flow field...")
                    # if ok cast to np.array
                    u0 = np.array(u0)
            # in any case, set v0:
            v0 = fft(u0)
        else:
            # the initial condition is provided in v0
            # check the input size
            if (np.size(v0,0) != self.N):
                print('Error: wrong IC array size')
                return -1
            else:
                print("Using given (Fourier) flow field...")
                # if ok cast to np.array
                v0 = np.array(v0)
                # and transform to physical space
                u0 = ifft(v0)
        #
        # and save to self
        self.u0  = u0
        self.v0  = v0
        self.v   = v0
        '''

    def operatorgen(self):
        Lx = self.Lx
        NX = self.NX
        dx = Lx/NX
        #-----------------  
        x        = np.linspace(0, Lx-dx, num=NX)
        kx       = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=np.float64),np.arange((-NX/2+1),0,dtype=np.float64)))   
        [Y,X]    = np.meshgrid(x,x)
        [Ky,Kx]  = np.meshgrid(kx,kx)
        Ksq      = (Kx**2 + Ky**2)
        Kabs     = np.sqrt(Ksq)
        Ksq[0,0] = 1e12
        invKsq   = 1/Ksq
        Ksq[0,0] = 0
        invKsq[0,0] = 0

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
    
    def leith_cs(self, w1_hat, action_leith=None):
        '''
        ve =(Cl * \delta )**3 |Grad omega|  LAPL omega ; LAPL := Grad*Grad
        '''
        #print('action is:', action_leith)
        if action_leith != None:
            if self.veRL !=0:
                CL3 = self.veRL#action_leith[0]
        #       print('CL3', self.veRL, action_leith)
        #     with open("test.txt", "a") as myfile:
        #        myfile.write(str(CL3)+"\n")
        else:
            CL3 = 0.17**3# (EKI)
        #else:
        Kx = self.Kx
        Ky = self.Ky
  
        w1x_hat = -(1j*Kx)*w1_hat
        w1y_hat = (1j*Ky)*w1_hat
        w1x = np.real(np.fft.ifft2(w1x_hat))
        w1y = np.real(np.fft.ifft2(w1y_hat))
        abs_grad_omega = np.mean(np.sqrt( w1x**2+w1y**2  ))
        # 
        #LAPL_omega = np.real(np.fft.ifft2(diffu_hat))
        #LAPL_omega = np.abs(LAPL_omega)**(0.5)
        
        delta3 = (2*math.pi/self.NX)**3
        ve = CL3*delta3*abs_grad_omega
        return ve

    def smag_cs(self, w1_hat, action=None):
        Kx = self.Kx
        Ky = self.Ky
        NX = self.NX
        psiCurrent_hat = self.psiCurrent_hat
        S1 = np.real(np.fft.ifft2(-Ky*Kx*psiCurrent_hat)) # make sure .* 
        S2 = 0.5*np.real(np.fft.ifft2(-(Kx*Kx - Ky*Ky)*psiCurrent_hat))
        S  = 2.0*(S1*S1 + S2*S2)**0.5
        cs = (0.17 * 2*np.pi/NX )**2  # for LX = 2 pi
#        print(S**2.0.shape)
#        print(S1.shape)
#        print(S.shape)
        S = (np.mean(S**2.0))**0.5;
        ve = cs*S
#        print(cs, ve)
#        stop
        return ve

    def enstrophy_spectrum(self):
        omega=np.real(np.fft.ifft2(self.w1_hat))
        NX = self.NX
        NY = self.NY # Square for now
#        enstrophy_spectrum = self.myspectrum(np.power(omega,2))
        enstrophy_spectrum = self.myspectrum2(omega)
        self.enstrophy_spec = enstrophy_spectrum#/NX/NY
        return  enstrophy_spectrum#/NX/NY

    def energy_spectrum(self):
        omega=np.real(np.fft.ifft2(self.w1_hat))
        psi=np.real(np.fft.ifft2(self.psi_hat))
        NX = self.NX
        NY = self.NY # Square for now
        Ksq = self.Ksq
        #energy_spectrum = self.myspectrum(psi*omega)
        #self.energy_spec = energy_spectrum/NX/NY
        #np.power(np.abs(np.fft.fft2(a)),2)/NX/NY/Ksq
        Ksq[0,0]=1
        w_hat = np.power(np.abs(np.fft.fft2(omega)),2)/NX/NY/Ksq 
        w_hat[0,0]=0;
        spec_x = np.mean(np.abs(w_hat),axis=0)
        spec_y = np.mean(np.abs(w_hat),axis=1)
        tke = (spec_x + spec_y)/2
        return  tke

    def myspectrum(self, a):
        return np.mean(np.abs(np.fft.fft(a,axis=0)),axis=1)
        
    def myspectrum2(self, a):
        NX = self.NX
        NY = self.NY
        a_hat = np.fft.fft2(a)/NX/NY
        a_hat_sq = np.power(a_hat,2)
        spec_x = np.mean(np.abs(a_hat_sq),axis=0)
        spec_y = np.mean(np.abs(a_hat_sq),axis=1)
        tke = (spec_x + spec_y)/2
        return tke


    def myplot(self, append_str=''):
        NX = int(self.NX)
        Kplot = self.Kx; kplot_str = '\kappa_{x}'; kmax = int(NX/2)+1
        #Kplot = self.Kabs; kplot_str = '\kappa_{sq}'; kmax = int(np.sqrt(2)*NX/2)+1
        #kplot_str = '\kappa_{sq}'
        stepnum = self.stepnum
        ve = self.ve
        Fn = self.Fn
        dt = self.dt
        # --------------
        plt.figure(figsize=(8,14))
        levels = np.linspace(-30,30,100)
 
        plt.subplot(3,2,1)
        plt.contourf(np.real(np.fft.ifft2(self.sol[0])),levels, vmin=-30,vmax=30)
        plt.colorbar()
        plt.title(r'$\omega$')


        levels = np.linspace(-30,3,100)
 
        plt.subplot(3,2,2)
        plt.contourf(np.real(np.fft.ifft2(self.sol[1])),levels);plt.colorbar()
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
        plt.ylim([1e-4,1e2])
        
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
        plt.ylim([1e-4,1e-1])
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
        #plt.subplot(3,2,6)
        #plt.semilogy(Vecpoints,log_kde) 

        filename = '2Dturb_N'+str(NX)+'_'+str(stepnum)+append_str
        plt.savefig(filename+'.png', bbox_inches='tight', dpi=450)
        
#        print(filename)
#        print(Kplot[0:kmax,0].shape)
#        print( energy[0:kmax].shape)
#        print( np.stack((Kplot[0:kmax,0], energy[0:kmax]),axis=0).T.shape   )
        
        np.savetxt(filename+'_tke.out', np.stack((Kplot[0:kmax,0], energy[0:kmax]),axis=0).T, delimiter='\t')
        np.savetxt(filename+'_ens.out', np.stack((Kplot[0:kmax,0], enstrophy[0:kmax]),axis=0).T, delimiter='\t')

 
    def KDEof(self, u):
        from PDE_KDE import myKDE
        Vecpoints, exp_log_kde, logkde, kde = myKDE(u)
        return Vecpoints, exp_log_kde, logkde, kde

