from turb import *

import numpy as np
def environment( args, s ):
    L    = 2*np.pi
    N    = 128 #128# 64
    dt   = 5.0e-4
    case = args["case"]
    rewardtype = args["rewardtype"]
    IF_RL = True #False
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + int(10e3)*dt# 30e-3  #0.025*(2500*4+1000
    nInitialSteps = int(tEnd/dt)
    print('Initlize sim.')
    sim  = turb(RL=IF_RL, 
                NX=N, NY=N,
                case=case,
                rewardtype=rewardtype,
                nsteps=nInitialSteps)
    print('================================')
    print('Simulate, nsteps=', nInitialSteps)
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
    nContolledSteps = int(10e3)#(tEnd-tInit)/dt)
    print('run controlled simulation with nControlledSteps=', nContolledSteps)
    step = 0
    while step < nContolledSteps:
        # Getting new action
        s.update()

        # apply action and advance environment
        sim.step( s["Action"] )
        #print("action:", s["Action"])

        # get reward
        s["Reward"] = sim.reward()
        #print("Reward", s["Reward"])

        # get new state
        s["State"] = sim.state().tolist()
        # print("state:", sim.state())
        
        # print()
        #print(sim.veRL)    
        step += 1

        #print( "Reward sum", np.sum(np.array(s["Reward"])) )

    sim.myplot('_controlled')
    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
