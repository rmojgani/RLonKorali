from turb import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr_r'
import numpy as np
import os

def environmentpost( args, s ):
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

    runFolder = args["runFolder"]

    casestr = '_C'+case+'_N'+str(N)+'_R_'+rewardtype+'_State_'+statetype+'_Action_'+actiontype+'_nAgents_'+str(nagents)
    casestr = casestr + '_CREWARD'+str( IF_REWARD_CUM )

    print(casestr)

    IF_RL = True #False
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + int(10e3)*dt# 30e-3  #0.025*(2500*4+1000
    nInitialSteps = int(tEnd/dt)
    print('Initlize sim.')
    sim  = turb(RL=IF_RL, 
                NX=N, NY=N,
                case=case,
                nActions=action_size,
                rewardtype=rewardtype,
                statetype=statetype,
                actiontype=actiontype,
                nagents=nagents,
                nsteps=nInitialSteps)
    print('================================')
    print('Simulate, nsteps=', nInitialSteps)
    sim.simulate( nsteps=nInitialSteps )

    #print('------------------')
    #print(sim.w1_hat.shape)
    #print(sim.psiPrevious_hat.shape)
    #print(sim.psi_hat.shape)
    SYSMEM = os.system('cat /proc/meminfo | grep Mem | head -n 3')
    SYSDATE = os.system('date')
    print(SYSMEM, SYSDATE)
    #print('------------------')
    #sim.myplot(casestr)    
    print('PNG file saved')

    ## get initial state
    s["State"] = sim.state()#.tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    nContolledSteps = int(1e7)#(tEnd-tInit)/dt)
    print('run controlled simulation with nControlledSteps=', nContolledSteps)
    mystr = "smagRL"#'smag0d17'

    step = 0
    while step < nContolledSteps:
        if step % int(5e3) == 1 :
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
                           ('veRL', sim.veRL)
                        ])
                     )

        # Getting new action
        s.update()

        # apply action and advance environment
        sim.step( s["Action"] )
        #print("action:", s["Action"])

        # get reward
        s["Reward"] = sim.reward()
        #print("Reward", s["Reward"])
        '''
        if not IF_REWARD_CUM or step == 0:
            cumulativeReward = sim.reward()
        else:
            cumulativeReward = [x + y for x, y in zip(cumulativeReward, sim.reward())]
        '''
        # get new state
        s["State"] = sim.state()#.tolist()
        # print("state:", sim.state())
        
        # print()
        #print(sim.veRL)    
        step += 1

        #print( "Reward sum", np.sum(np.array(s["Reward"])) )

    '''
    cumulativeReward_normalized =  [x/nContolledSteps for x in cumulativeReward]
    s["Reward"] = cumulativeReward_normalized
    '''

    #sim.myplot(casestr+'_RL')
    #sim.myplotforcing(casestr+'_RL_f')
    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
