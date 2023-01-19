import argparse
import sys
sys.path.append('_model')
from environmentpost import environmentpost as environment
import math
## Parameters
state_size = int(128*2) # For spectrum as state: N/2+1; 9 , 33, 65
action_size = 1

### Parsing arguments

parser = argparse.ArgumentParser()

parser.add_argument('--case', 
                    help='Reinforcement learning case considered. Choose one from the following list: "1", or "4"', 
                    required=True, type=str)

args = vars(parser.parse_args())

### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_vracer/'
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n");

### Defining Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Problem"]["Agents Per Environment"] = 1

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1 #--> 10 
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Defining Variables

statesize = state_size # For spectrum as state: N/2+1; 9 , 33, 65
# States (flow at sensor locations)
for i in range(statesize):
	e["Variables"][i]["Name"] = "Sensor " + str(i)
	e["Variables"][i]["Type"] = "State"

# Actions (amplitude of gaussian actuation)
for i in range(action_size): # size of action 
	e["Variables"][statesize+i]["Name"] = "Actuator " + str(i)
	e["Variables"][statesize+i]["Type"] = "Action"
	e["Variables"][statesize+i]["Lower Bound"] = -0.5**3
	e["Variables"][statesize+i]["Upper Bound"] = 0.5**3
	e["Variables"][statesize+i]["Initial Exploration Noise"] = 0.15**3

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
e["Solver"]["Reward"]["Outbound Penalization"]["Enabled"] = True
e["Solver"]["Reward"]["Outbound Penalization"]["Factor"] = 0.5
  
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
e["Solver"]["Termination Criteria"]["Max Generations"] = 1000
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)

### Checking if we reached a minimum performance

bestReward = e["Solver"]["Training"]["Best Reward"]
if (bestReward < 1000.0):
 print("Flow Control example did not reach minimum training performance.")
 exit(-1)
