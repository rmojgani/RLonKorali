from turb import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr_r'

import numpy as np
def environment( args, s ):
    L    = 2*np.pi
    N    = args['NLES'] #128# 64
    dt   = 5.0e-4
    case = args["case"]
    rewardtype = args["rewardtype"]
    statetype = args['statetype']
    actiontype = args['actiontype']
    casestr = '_C'+args['case']+'_N'+str(N)+'_R_'+args['rewardtype']+'_State_'+args['statetype']+'_Action_'+args['actiontype']

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
                statetype=statetype,
                actiontype=actiontype,
                nsteps=nInitialSteps)
    print('================================')
    print('Simulate, nsteps=', nInitialSteps)
    sim.simulate( nsteps=nInitialSteps )

    #print('------------------')
    #print(sim.w1_hat.shape)
    #print(sim.psiPrevious_hat.shape)
    #print(sim.psi_hat.shape)
    #print('------------------')
    sim.myplot(casestr)    
    print('PNG file saved')

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

    sim.myplot(casestr+'_RL')
    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
