from turb import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr_r'
import numpy as np
import os

import copy
import time

def environmentpost( args, initSim, s ):

    sim = copy.deepcopy(initSim)
    startSim = time.time()
    
    L    = 2*np.pi
    N    = args['NLES'] #128# 64
    dt   = 5.0e-4
    case = args["case"]
    action_size = args["nActions"]
    rewardtype = args["rewardtype"]
    statetype = args['statetype']
    actiontype = args['actiontype']
    nagents = args['nagents']
    IF_REWARD_CUM = args['IF_REWARD_CUM']
    Thorizon = args['Thorizon']
    Tspinup = args['Tspinup']
    NumRLSteps = args['NumRLSteps']
    
    runFolder = args["runFolder"]

    casestr = '_C'+case+'_N'+str(N)+'_R_'+rewardtype+'_State_'+statetype+'_Action_'+actiontype+'_nAgents_'+str(nagents)
    casestr = casestr + '_CREWARD'+str( IF_REWARD_CUM )

    print(casestr)

    IF_RL = True #False
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + int(Tspinup)*dt# 30e-3  #0.025*(2500*4+1000
    nInitialSteps = int(tEnd/dt)
    
    #print('------------------')
    cmd="(awk \'$3==\"kB\"{$2=$2/1024^2;$3=\"GB\";} 1\' /proc/meminfo | head -n 3 | grep Mem)"#| column -t 
    SYSMEM = os.system(cmd);
    SYSDATE = os.system('date')
    #print('------------------')
    #sim.myplot(casestr)    
    #print('PNG file saved')

    ## get initial state
    s["State"] = sim.state()#.tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    nSteps = int(Thorizon) #int((tEnd-tInit)/dt)
    nControlledSteps = int(NumRLSteps)
    nIntermediateSteps = int(nSteps / nControlledSteps)
    nSteps = int(1e7)
    print(f'run controlled simulation with nSteps {nSteps} and nControlledSteps {nControlledSteps}, updating state every {nIntermediateSteps}')
        
    mystr = "smagRL"#'smag0d17'

    step = 0
    while step < nSteps:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        for i in range(nIntermediateSteps):
            if step % int(5e3) == 1 and step>2:
                print('Save at time step=', step)
                sim.myplot('_ctrled_'+mystr+'_'+str(step), runFolder)
                '''
                try:
                    sim.myplotforcing('_ctrled_'+mystr+'_'+str(step), runFolder)
                except:
                    print("not plotted")
                '''
                savemat(runFolder+'N'+str(sim.NX)+'_t='+str(step)+'_'+mystr+'.mat',
                     dict([
                           ('psi_hat', sim.psiCurrent_hat),
                           ('w_hat', sim.w1_hat),
                           ('convec1_hat', sim.convec1_hat),
                           ('c_dynamic', sim.c_dynamic),
                           ('ve', sim.ve)
                        ])
                     )

            sim.step( s["Action"] )
            step += 1

        # get reward
        #s["Reward"] = sim.reward()

        # get new state
        s["State"] = sim.state()#.tolist()
        # print("state:", sim.state())
        

    #sim.myplot(casestr+'_RL')
    #sim.myplotforcing(casestr+'_RL_f')
    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
