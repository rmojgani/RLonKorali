import argparse
import sys
import os
import numpy as np
import time
sys.path.append('_model')
from turb import *
#sys.path.append('_init')
#from environment import *
#from mpi4py import MPI

## Parameters

### Parsing arguments
parser = argparse.ArgumentParser()
# LES
parser.add_argument('--case', help='Reinforcement learning case considered. Choose one from the following list: "1", or "4"', type=str, default='4')
parser.add_argument('--NLES', help='', type=int, default=32)

# RL
parser.add_argument('--rewardtype', help='Reward type [k1,k2,k3,log,] ', type=str, default='k1')
parser.add_argument('--statetype', help='State type [enstrophy, energy, psidiag, psiomegadiag, psiomega, psiomegalocal] ', type=str, default='psiomega')
parser.add_argument('--actiontype', help='Action type [CS,CL,CLxyt,nuxyt,] ', type=str, default='CL')
parser.add_argument('--action0', help='initial noise for the action', type=float, default=0.25**3)

parser.add_argument('--EPERU', help='Number of experiences per policy update', type=float, default=1.0) # This can be eg. 0.1, to have a policy update for every 10 experiences
parser.add_argument('--gensize', help='', type=int, default=10)
parser.add_argument('--solver', help='training/postprocess ', type=str, default='training')
parser.add_argument('--runstrpost', help='nosgs, Leith, Smag,', type=str, default='')
parser.add_argument('--nagents', help='Number of Agents', type=int, default=4)
parser.add_argument('--nconcurrent', help='Number of concurrent jobs', type=int, default=1)
parser.add_argument('--IF_REWARD_CUM', type=int, default=False, choices=[0, 1], help='If reward to accumalated?')
parser.add_argument('--Tspinup',  help='number of time steps for spin up', type=float, default=1e4)
parser.add_argument('--Thorizon', help='number of tiem steps for training horizon', type=float, default=1e4)
parser.add_argument('--NumRLSteps', help='number of RL steps during simulation', type=float, default=1e3)

args = vars(parser.parse_args())

args['runstrpost']=args['actiontype']+'post'

NLES = args['NLES']
EPERU = args['EPERU'] 

casestr = '_C'+args['case']+'_N'+str(NLES)+'_R_'+args['rewardtype']+'_State_'+args['statetype']+'_Action_'+args['actiontype']+'_nAgents'+str(args['nagents'])
casestr = casestr + '_CREWARD'+str(args['IF_REWARD_CUM'])
casestr = casestr + '_Tspin'+str(args['Tspinup'])+'_Thor'+str(args['Thorizon'])+'_NumRLSteps'+str(args['NumRLSteps'])
casestr = casestr + '_EPERU'+str(EPERU)

print('args:', args)
print ('case:', casestr)

# Case selection
if args['case']=='1':
    sys.path.append('_init/Re20kf4')
elif args['case']=='4':
    sys.path.append('_init/Re20kf25')

# State type
if args['statetype'] == 'enstrophy' or args['statetype'] == 'energy':
    state_size = int(NLES/2)#+1
elif args['statetype'] == 'psiomegadiag':
    state_size = int(NLES*2)
elif args['statetype'] == 'psiomega':
    nagents_h = np.sqrt(args['nagents'])
    state_size = int(2*(NLES/nagents_h)**2)
elif args['statetype'] == 'omega':
    nagents_h = np.sqrt(args['nagents'])
    state_size = int((NLES/nagents_h)**2)
elif args['statetype'] == 'psiomegalocal':
    state_size = 2 #np.sqrt(args['nagents'])
elif args['statetype'] == 'invariantlocal':
    state_size = 6
elif args['statetype'] == 'invariantlocalandglobalgradgrad':
    state_size = 6 +  int(NLES/2)
elif args['statetype'] == 'invariantlocalandglobalgradgradeps':
    state_size = 6 +  int(NLES/2) + 1

# Type of the action
if args['actiontype'] == 'CL':
    action_size=1
elif args['actiontype'] == 'CS':
    action_size=1
else:
    action_size=8**2
print('Racer: Action size is:', action_size)
args['nActions']=action_size

N    = args['NLES']
case = args["case"]
rewardtype = args["rewardtype"]
statetype = args['statetype']
actiontype = args['actiontype']
nagents = args['nagents']
Tspinup = args['Tspinup']
action0 = args['action0']

dt   = 5.0e-4
tInit = 0
tEnd = tInit + int(Tspinup)*dt# 30e-3  #0.025*(2500*4+1000
nInitialSteps = int(tEnd/dt)

# Initialize simulation here
startInit = time.time()
print('Initlize sim.')
sim  = turb(RL=True, 
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
endInit = time.time()
print(f'Init time {(endInit-startInit):.3}s')


### Defining Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any
resultFolder = '_result_vracer'+casestr+'/'
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n");

# Mode settings
if args['solver']=='training':
    print('Enviroment loaded')
    from environment import *
else:
    print('Postprocess enviroment loaded')
    from environmentpost import environmentpost as environment
    args['runFolder'] = resultFolder+args['runstrpost']+'/'
    os.makedirs(args['runFolder'], exist_ok = True)
    print(args['runFolder'])

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, sim, x )
e["Problem"]["Agents Per Environment"] = args['nagents']
e["Problem"]["Policies Per Environment"] = 1
print('Number of agents', args['nagents'])
if args['nagents']>1:
    e["Solver"]["Multi Agent Relationship"] = "Individual" #"Individual" (or "Cooperation")
    e["Solver"]["Multi Agent Correlation"] = False # (False or True)

### Defining Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Concurrent Workers"] = 2
e["Solver"]["Episodes Per Generation"] = args['gensize'] #--> 10, 25, 50
e["Solver"]["Experiences Between Policy Updates"] = EPERU
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables
# States
for i in range(state_size):
	e["Variables"][i]["Name"] = "Sensor " + str(i)
	e["Variables"][i]["Type"] = "State"

# Actions
for i in range(action_size): # size of action 
	e["Variables"][state_size+i]["Name"] = "Actuator " + str(i)
	e["Variables"][state_size+i]["Type"] = "Action"
	e["Variables"][state_size+i]["Lower Bound"] = -0.5**3
	e["Variables"][state_size+i]["Upper Bound"] = 0.5**3
	e["Variables"][state_size+i]["Initial Exploration Noise"] = action0 #0.25**3 #0.15**3

### Setting Experience Replay and REFER settings
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Start Size"] = 10000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000

e["Solver"]["Policy"]["Distribution"] = "Squashed Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
#e["Solver"]["Reward"]["Outbound Penalization"]["Enabled"] = True
#e["Solver"]["Reward"]["Outbound Penalization"]["Factor"] = 0.5
  
### Configuring the neural network and its hidden layers
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = True
e["Solver"]["L2 Regularization"]["Importance"] = 1.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration
e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
e["Solver"]["Termination Criteria"]["Max Generations"] = 100 #201
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["File Output"]["Path"] = resultFolder

### Over ride the settings for post process
if args['solver'] == 'postprocess':
    print('Postprocess ---> Generation ........... ')
    e["Solver"]["Episodes Per Generation"] = 1
    e["Solver"]["Termination Criteria"]["Max Generations"] = 202
#    e["Solver"]["Mode"] = "Testing" #"Training / Testing"
#    e["Solver"]["Testing"]["Sample Ids"] = [0]
'''
if args['nconcurrent'] > 1:
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = 1#args['nconcurrent']
    e["Solver"]["Concurrent Workers"] = args['nconcurrent'];

if args['nconcurrent'] > 1:
    print(MPI.COMM_WORLD)
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed";
    k["Conduit"]["Ranks Per Worker"] = 1;
    e["Solver"]["Concurrent Workers"] = 7;# args['nconcurrent'];
'''
### Running Experiment
print("Running the experiment ---- ")
k.run(e)

### Checking if we reached a minimum performance

bestReward = e["Solver"]["Training"]["Best Reward"]
if (bestReward < 1000.0):
 print("Flow Control example did not reach minimum training performance.")
 exit(-1)

