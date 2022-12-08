#from KS import *
#from QGr import *
from turb import *

import numpy as np
def environment( args, s ):
    L    = 2*np.pi
    N    = 64
    dt   = 1.0e-5
    case = args["case"]
    #sim = KS()
    #sim  = QG()#L=L, N=N, dt=dt, tend=tEnd, RL=True, case=case)
    #sim  = turb(RL=True)
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + 10e-3  #0.025*(2500*4+1000
    nInitialSteps = int(tInit/dt)
    sim  = turb(RL=True, case=case)
    sim.simulate( nsteps=nInitialSteps )

    #print(vars(sim))

    #for var in vars(sim):
    #    print(getattr(sim, var))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    plt.rcParams['image.cmap'] = 'bwr_r'
    #print('------------------')
    #print(sim.w1_hat.shape)
    #print(sim.psiPrevious_hat.shape)
    #print(sim.psi_hat.shape)
    #print('------------------')
    sim.myplot()    

    print('file saved')
    #print(sim.state())
    #print(sim.state().tolist())
    #print('xxxx       \n')
    #print(s)
    ## get initial state
    s["State"] = sim.state().tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    nContolledSteps = int((tEnd-tInit)/dt)
    step = 0
    while step < nContolledSteps:
        # Getting new action
        s.update()

        # apply action and advance environment
        sim.step( s["Action"] )
        #print("action:", s["Action"])

        # get reward
        s["Reward"] = sim.reward()
        # print("state:", sim.reward())

        # get new state
        s["State"] = sim.state().tolist()
        # print("state:", sim.state())
        
        # print()
        #print(sim.veRL)    
        step += 1

    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
