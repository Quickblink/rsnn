
import torch
import torch.optim as optim
import os
import sys
import random
import matplotlib.pyplot as plt
# hack to perform relative imports
sys.path.append('../')
from ChrisCode import train_agent, SQN



torch.backends.cudnn.deterministic = True
import numpy as np
np.random.seed(1)


#CartPole
env = 'CartPole-v0'

#hyperparameters
BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.999
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQUENCY = 10
LEARNING_RATE = 0.001
REPLAY_MEMORY_SIZE = 4*10**4
# minimum size of the replay memory before the training starts
INITIAL_REPLAY_SIZE = 0
# the gym standard for CartPole ("solving" it) is to achieve a 100-episode average of <=195 for 100 consecutive episodes
GYM_TARGET_AVG = 195
GYM_TARGET_DURATION = 100
# maximum number of steps before the environment is reset
MAX_STEPS = 200
# number of episodes to train the agent
NUM_EPISODES = 6 #TODO: change
# whether to use Double Q Learning and Gradient Clipping
DOUBLE_Q = True
GRADIENT_CLIPPING = True
# whether to render the environment
RENDER = False

# device: automatically runs on GPU, if a GPU is detected, else uses CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# First we set up a new sub directory

# We use a non-leaky integrate-and-fire neuron
ALPHA = 0
BETA = 1
# Simulation time is chosen relatively short, such that the network does not need too much time to run, but not too short,
# such that it can still learn something
SIMULATION_TIME = 20
# We also have to define the input/output and reset methods, to our knowledge, SpyTorch supports only potential outputs
# and reset-by-subtraction. As input method we use constant input currents. It would be interesting to see if SpyTorch
# can also use reset-to-zero, as this would make it more similar to the iaf_delta models in NEST and SpyNNaker
ENCODING = 'constant'
DECODING = 'potential'
RESET = 'subtraction'
# SpyTorch uses a fixed threshold of one, we didn't test other thresholds, but should be possible
THRESHOLD = 1


# alternatively use the seeds gym: 240, torch: 18, random: 626 and a learning rate of 0.0005
# to get the same results as figure 4.9 in the thesis
torch.manual_seed(467)
random.seed(208)
gym_seed = 216




architecture = [4,17,17,2]
policy_net = SQN.SQN(architecture,device,alpha=ALPHA,beta=BETA,simulation_time=SIMULATION_TIME,add_bias_as_observation=True,
                  encoding=ENCODING,decoding=DECODING,reset=RESET,threshold=THRESHOLD)
# load the fixed initial weights, remove this line to get random initial weights
#policy_net.load_state_dict(torch.load('./../../CartPole-v0/DSQN-Surrogate-Gradients/initial/model.pt'))

target_net = SQN.SQN(architecture,device,alpha=0,beta=1,simulation_time=SIMULATION_TIME,add_bias_as_observation=True,
                  encoding=ENCODING,decoding=DECODING,reset=RESET,threshold=THRESHOLD)
target_net.load_state_dict(policy_net.state_dict())

# initialize optimizer
#optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)





#%%

for l in policy_net.parameters():#.state_dict()[0]:
    print(l.shape)

#%%

from Code import SNN

#%%

architecture = [4,17,17,2]

neuron_params = {
    'threshold': 1,
    'alpha': ALPHA,
    'beta': BETA,
    'reset_zero': False
}

my_net = SNN.FeedForwardSNN(architecture, SNN.LIFNeuron, neuron_params, SNN.SuperSpike.apply, SNN.PotentialNeuron)
my_target_net = SNN.FeedForwardSNN(architecture, SNN.LIFNeuron, neuron_params, SNN.SuperSpike.apply, SNN.PotentialNeuron)

#%%
'''
mp = list(my_net.parameters())
hp = list(policy_net.parameters())
mp[0].data = hp[0].data.T.detach().clone()
mp[1].data = hp[1].data.T.detach().clone()
mp[2].data = hp[2].data.T.detach().clone()
mp = list(my_target_net.parameters())
mp[0].data = hp[0].data.T.detach().clone()
mp[1].data = hp[1].data.T.detach().clone()
mp[2].data = hp[2].data.T.detach().clone()
'''

mp = list(my_net.parameters())
hp = list(policy_net.parameters())
mp[0].data = hp[0].data.detach().clone()
mp[1].data = hp[1].data.detach().clone()
mp[2].data = hp[2].data.detach().clone()
mp = list(my_target_net.parameters())
mp[0].data = hp[0].data.detach().clone()
mp[1].data = hp[1].data.detach().clone()
mp[2].data = hp[2].data.detach().clone()

#%%

testinp = torch.tensor([[.1,.1,.1,1], [.1,.3,.2,1]], dtype=torch.float)
myout = my_net(testinp, None)
print("-------------------------------------------------------------------------")
chrisout = policy_net.forward(testinp)
my_net.zero_grad()
policy_net.zero_grad()
myout.sum().backward()
print("-------------------------------------------------------------------------")
chrisout.sum().backward()
print(myout-chrisout)
print('Outcomp: ', myout[0,0].item(), chrisout[0,0].item())
print(list(my_net.parameters())[0].grad[0,0].item())
print(list(policy_net.parameters())[0].grad[0,0].item())