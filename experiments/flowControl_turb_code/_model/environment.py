from turb import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr_r'
import numpy as np
import os

import copy
import time

def environment( args, initSim, s ):

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

    casestr = '_C'+case+'_N'+str(N)+'_R_'+rewardtype+'_State_'+statetype+'_Action_'+actiontype+'_nAgents_'+str(nagents)
    casestr = casestr + '_CREWARD'+str( IF_REWARD_CUM )

    print(casestr)

    IF_RL = True #False
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + int(Tspinup)*dt# 30e-3  #0.025*(2500*4+1000
    nInitialSteps = int(tEnd/dt)

    #print('------------------')
    #print(sim.w1_hat.shape)
    #print(sim.psiPrevious_hat.shape)
    #print(sim.psi_hat.shape)
    cmd="(awk \'$3==\"kB\"{$2=$2/1024^2;$3=\"GB\";} 1\' /proc/meminfo | head -n 3 | grep Mem)"#| column -t 
    SYSMEM = os.system(cmd)
    SYSDATE = os.system('date')
    #print(SYSMEM, SYSDATE)
    #print('------------------')
    #sim.myplot(casestr) 
    #print('PNG file saved')

    ## get initial state
    s["State"] = sim.state()#.tolist()
    # print("state:", sim.state())

    try:
        ## run controlled simulation
        nSteps = int(Thorizon) #int((tEnd-tInit)/dt)
        nControlledSteps = int(NumRLSteps)
        nIntermediateSteps = int(nSteps / nControlledSteps)
        print(f'run controlled simulation with nSteps {nSteps} and nControlledSteps {nControlledSteps}')

        step = 0
        while step < nSteps:
            #print(step)
            # Getting new action
            s.update()

            # apply action and advance environment
            for i in range(nIntermediateSteps):
                sim.step( s["Action"] )
                step += 1

            #print("action:", s["Action"])

            # get reward
            s["Reward"] = sim.reward()
            #print("Reward", s["Reward"])
            if not IF_REWARD_CUM or step == 0:
                cumulativeReward = sim.reward()
            else:
                cumulativeReward = [x + y for x, y in zip(cumulativeReward, sim.reward())]

            # get new state
            s["State"] = sim.state()#.tolist()
            # print("state:", sim.state())
        
            #print( "Reward sum", np.sum(np.array(s["Reward"])) )

        if not IF_REWARD_CUM:
            s["Reward"] = cumulativeReward # which is already equal to sim.reward()
        else:
            cumulativeReward_normalized =  [x/nControlledSteps for x in cumulativeReward]
            s["Reward"] = cumulativeReward_normalized

        s["Termination"] = "Terminal"

        #sim.myplot(casestr+'_RL')
        #sim.myplotforcing(casestr+'_RL_f')
        # TODO?: Termination in case of divergence
    except Exception as err:
        print(f'Exception: {err}')
        print('s["Termination"]: Truncated')
        s["Termination"] = "Truncated"

    endSim = time.time()
    print(f'Sim loop {(endSim-startSim):.3}s')


