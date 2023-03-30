from turb import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr_r'
import numpy as nnp
import jax.numpy as np
from jax import grad, jit, vmap, pmap
from jax.lib import xla_bridge
import jax

# Loading files
import glob
import re
import os
import scipy.io as sio

def environmentpost(args, s):
    L    = 2*np.pi
    N    = args['NLES'] #128# 64
    dt   = 5.0e-4
    case = args["case"]
    action_size = args["nActions"]
    rewardtype = args["rewardtype"]
    statetype = args['statetype']
    actiontype = args['actiontype']
    nagents = args['nagents']

    runFolder = args["runFolder"]
    casestr = '_C'+case+'_N'+str(N)+'_R_'+rewardtype+'_State_'+statetype+'_Action_'+actiontype+'_nAgents_'+str(nagents)
    print(casestr)
    IF_RL = True #False
    # simulate up to T=20
    tInit = 0
    tEnd = tInit + int(1)*dt# 30e-3  #0.025*(2500*4+1000
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
    #print('------------------')
    sim.myplot(casestr)    
    print('PNG file saved')

    ## get initial state
    s["State"] = sim.state()#.tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    nContolledSteps = int(1e7)#(tEnd-tInit)/dt)
    print('run controlled simulation with nControlledSteps=', nContolledSteps)
    mystr = "smagRL"#'smag0d17'
    step = 0
    print('Type of init:', type(sim.psiCurrent_hat), type(sim.w1_hat), type(sim.w1_hat) )
    try:
        list_of_files = glob.glob(runFolder+'*.mat')
        latest_file = max(list_of_files, key=os.path.getctime)
        print('Last file loaded:', latest_file)
        mat_contents = sio.loadmat(latest_file)

        sim.psiCurrent_hat = np.array(mat_contents['psi_hat'])
        sim.w1_hat = np.array(mat_contents['w_hat'])
        sim.convec0_hat = np.array(mat_contents['convec1_hat'])

        numbers_in_file = re.findall(r'\d+', latest_file)
        print(numbers_in_file)
        step = int(numbers_in_file[-1])
        sim.stepnum = step
        print('Step number set to', step)
        print('Type of loaded:', type(sim.psiCurrent_hat), type(sim.w1_hat), type(sim.w1_hat) )
    
    except:
        print('Simulation initialized')

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
                           ('psi_hat', nnp.array(sim.psiCurrent_hat)),
                           ('w_hat', nnp.array(sim.w1_hat)),
                           ('convec1_hat', nnp.array(sim.convec1_hat)),
                           ('veRL', nnp.array(sim.veRL))
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

        # get new state
        s["State"] = sim.state()#.tolist()
        # print("state:", sim.state())
        
        # print()
        #print(sim.veRL)    
        step += 1

        #print( "Reward sum", np.sum(np.array(s["Reward"])) )

    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"
