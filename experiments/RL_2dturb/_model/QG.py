from numpy import pi
#from scipy.fftpack import fft, ifft
import numpy as np

np.seterr(over='raise', invalid='raise')

# ---------------------- QG
from QGlib import case_select
from QGlib import operator_gen

ts = int(SPIN_UP*2) # Total timesteps
tot_time = dt*ts    # Length of run
lim = int(SPIN_UP ) # Start saving
st = int( 1. / dt ) # How often to save data

from QGlib import QGinitialize, QGinitialize2, QGinitialize3
from QGlib import my_load
from QGlib import initialize_psi_to_param
from QGloop import QGloop
from QGlib import QGfun
#
def gaussian( x, mean, sigma ):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*( (x-mean)/sigma )**2)

class QG:
    #
    # Solution of the 2D quasi-geostrophic (2-layer)  equation
    #
    # u_t + ...  = 0,
    # with ... BCs on ... .
    #
    # The nature of the solution depends on the system size L and on the initial
    # condition u(x,0).  Energy enters the system

    #
    # Spatial  discretization: spectral (Fourier)
    # Temporal discretization: 
    #

    def __init__(self, Lx=46.0, Ly=68.0, N=96, N2=192, dt=0.025, nsteps=None, tend=150, iout=1, u0=None, v0=None, RL=False, nActions=4, sigma=0.4, case=None ):
        #
        # Initialize
        L  = float(L); dt = float(dt); tend = float(tend)
        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps
        #
        # save to self
        self.Lx     = Lx
        self.Ly     = Ly
        self.N      = N
        self.N2     = N2
        #self.dx     = 2*pi*L/N
        self.dt     = dt
        self.nsteps = nsteps
        self.iout   = iout
        self.nout   = int(nsteps/iout)
        self.RL     = RL
        #
        # Operators 
        self.kk, self.ll, self.Grad, self.Lapl, self.Grad8, self.D_Dx, self.D_Dy = self.operator_gen(N, Lx, N2, Ly)
        #
        # Grid gen
        xx, yy = self.gridgen(Lx, Ly, N, N2)
        #
        CONSTANTS_r, topographicPV_ref  , x_mountain_r, y_mountain_r, H_r,\
        CONSTANTS_m, topographicPV_model, x_mountain_m, y_mountain_m, H_m,\
        sigma, beta, sx, sy = case_select( CASE_NO, CASE_NO_R, xx, yy)
        #
        self.g = 0.04 #leapfrog filter coefficient
        self.U_1 = 1.
        #
        psi_R, psi_Rc, sponge, u_eq, u_eq2 =  self.extraparam(N2,y,sigma,U_1)
        #
        # get targets for control:
        if self.RL:
            self.case = case
            self.setup_targets()
        #
        # set initial condition
        if (u0 is None) or (v0 is None):
            #self.IC()
            # QG:
            psic_1, psic_2, vorc_1, vorc_2, q_1, q_2, qc_1, qc_2, psi_1, psi_2 =\
            QGinitialize(N, N2, ll, kk, Lapl, CONSTANTS_r['beta'], y, topographicPV_ref)
            pv, psiAll, Uall, Vall = QGinitialize2(N, N2, ts, lim, st)
        elif (u0 is not None):
            self.IC(u0 = u0)
        elif (v0 is not None):
            self.IC(v0 = v0)
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
        self.gaussians = np.zeros((self.nActions, self.N))
        for i in range(self.nActions):
            mean = i*self.L/4
            self.gaussians[i,:] = gaussian( self.x, mean, self.sigma )

    def setup_targets(self):
        self.targets = np.zeros((3,self.N))
        for i in range(3):
            self.targets[i,:] = np.loadtxt("_model/u{}.dat".format(i+1))

    def IC(self, u0=None, v0=None, seed=42):
        #
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
            # set u0
            if self.RL:
                # initial condition for chosen RL case
                if self.case == "E31":
                u0 = self.targets[2,:]
                elif self.case == "E12":
                u0 = self.targets[0,:]
                elif self.case == "E23":
                u0 = self.targets[1,:]
                else:
                assert False, print("RL case {} unknown...".format(self.case))
            else:
                print("Using random initial condition...")
                # uniform noise
                # u0 = (np.random.rand(self.N) -0.5)*0.01
                # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                np.random.seed( seed )
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
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        

    def step( self, action=None ):
        '''
        QG: One time step simulation of QG
        '''
        forcing  = np.zeros(self.N)#QG?
        Fforcing = np.zeros(self.N)#QG?
        if (action is not None):#QG?
            assert len(action) == self.nActions, print("Wrong number of actions. provided {}/{}".format(len(action), self.nActions))
            for i, a in enumerate(action):
            forcing += a*self.gaussians[i,:]
            Fforcing = fft( forcing )
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        # Action
        if (action is not None):
            self.v = self.E*v + (Nv + Fforcing)*self.f1 + 2.*(Na + Nb + 2*Fforcing)*self.f2 + (Nc + Fforcing)*self.f3
        else:
            #self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3
            psic_1, psic_2,\
            vorc_1, vorc_2,\
            psi_1 , psi_2 ,\
            q_1   , q_2   ,\
            qc_1  , qc_2  ,\
            x =\
            QGfun(self.stepnum, kk, ll, D_Dx, D_Dy, Grad, Lapl, Grad8,
              psic_1, vorc_1, beta,
              psic_2, vorc_2,
              psi_1, psi_2, psi_R, tau_d, tau_f,
              sponge, q_1, q_2,
              R,
              qc_1, qc_2,
              dt, nu, y, g,
              opt,
              c_me_jac, c_non_linear)
        
           self.stepnum += 1
           self.t       += self.dt
   

    def simulate(self, nsteps=None, iout=None, restart=False, correction=[]):
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
            try:
                self.step()
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
        #
        # Convert from spectral to physical space
        self.uu = np.real(ifft(self.vv))


    def state(self):
        u = np.real(ifft(self.v))
        state = np.full(8,fill_value=np.inf)
        for i in range(1,17,2):
            indexState = int( i/2 )
            indexField = int( i*self.N/16 )
            state[indexState] = u[indexState]
        return state


    def reward(self):
        u = np.real(ifft(self.v))
        if self.case == "E31":
            return np.linalg.norm( u - self.targets[0,:] )
        elif self.case == "E12":
            return np.linalg.norm( u - self.targets[1,:] )
        elif self.case == "E23":
            return np.linalg.norm( u - self.targets[2,:] )


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


    def space_filter(self, k_cut=2):
        #
        # spatially filter the time series
        self.uu_filt  = np.zeros([self.nout+1, self.N])
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])    # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 0 # set to zero wavenumbers > k_cut
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt


    def space_filter_int(self, k_cut=2, N_int=10):
        #
        # spatially filter the time series
        self.N_int    = N_int
        self.uu_filt      = np.zeros([self.nout+1, self.N])
        self.uu_filt_int  = np.zeros([self.nout+1, self.N_int])
        self.x_int    = 2*pi*self.L*np.r_[0:self.N_int]/self.N_int
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])   # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 313e6
            v_filt_int = v_filt[v_filt != 313e6] * self.N_int/self.N
            self.uu_filt_int[n,:] = np.real(ifft(v_filt_int))
            v_filt[np.abs(self.k)>=k_cut] = 0
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt

        # -------------------------- QG:
    def gridgen(Lx, Ly, N, N2):
        x = np.linspace( -Lx / 2, Lx / 2, N , endpoint=False)
        y = np.linspace( -Ly / 2, Ly / 2, N2, endpoint=False)

        xx, yy = np.meshgrid(x, y)

        return xx, yy

    def extraparam(N2,y,sigma,U_1):
        sponge = np.zeros( N2)
        u_eq = np.zeros( N2)
        u_eq2 = np.zeros( N2)

        for i in range( N2 ):
            y1 = float( i - N2 /2) * (y[1] - y[0] )
            y2 = float(min(i, N2 -i - 1)) * (y[1] - y[0] )
            sponge[i] = U_1 / (np.cosh(abs(y2/sigma)))**2
            u_eq[i] = U_1 * ( 1. / (np.cosh(abs(y1/sigma)))**2 - 1. / (np.cosh(abs(y2/sigma)))**2  )
            u_eq2[i] = U_1 * ( (1./np.cosh(abs(y1/sigma)))**2  )

        psi_Rc = -np.fft.fft( u_eq ) / 1.j / ll
        psi_Rc[0] = 0.
        psi_R = np.fft.ifft( psi_Rc )

        return psi_R, psi_Rc, sponge, u_eq, u_eq2
