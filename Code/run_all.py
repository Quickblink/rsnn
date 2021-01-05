import sys
sys.path.append('.')
from Code.envs.SequentialMNIST import SequentialMNIST
from Code.train import train, OptWrapper
import torch
import torch.nn as nn
import numpy as np
from Code.everything6 import OuterWrapper, DynNetwork, ParallelNetwork, SequenceWrapper, MeanModule, BaseNeuron, build_standard_loop
from Code.envs.statemachine import SuccessiveLookups
import json

#TODO: this is same decay!!!!
#MAIN_DECAY = np.exp(-1/(700)*0.5)
#ADAP_DECAY = np.exp(-1/(700*2))

#MAIN_DECAY = np.exp(-1/700)

ex_spec = {
    'control_config': {
        'neuron_type': 'LIF',
        'n_neurons': 120,
        'BETA': 0.8,
        '1-beta': 'improved',
        'SPIKE_FN': 'bellec'
    },
    'mem_config': {
        'neuron_type': 'NoReset2',
        'n_neurons': 100,
        'BETA': np.exp(-1/1400),
        '1-beta': 'improved',
        'SPIKE_FN': 'bellec',
        'ADAPSCALE': 30,
        'ADAPDECAY': None, #TODO: set this
        'OFFSET': None,
        'DECAY': None#MAIN_DECAY
    },
    'exp_config': {
        'n_sequence': 30,
        'val_sequence': 100,
        'round_length': 20
    },
    'experiment': 'SequentialMNIST',
    'lr': 0.001,
    'lr_decay': 0.9,
    'iterations': 5000,
    'batch_size': 64,
    'architecture': '1L'
}

run_id = sys.argv[1]
with open('configs/'+run_id+'.json', 'r') as config_file:
    config = json.load(config_file)
    spec = config['params']

#%%

DEVICE = torch.device('cuda')

if spec['experiment'] == 'SequentialMNIST':
    train_problem = SequentialMNIST(spec['iterations'], spec['batch_size'], DEVICE, '.')
    val_problem = SequentialMNIST(-1, spec['batch_size'], DEVICE, '.', validate=True)
elif spec['experiment'] == 'SuccessiveLookups':
    train_problem = SuccessiveLookups(spec['iterations'], spec['batch_size'], spec['exp_config']['n_sequence'],
                                      spec['exp_config']['round_length'], DEVICE)
    val_problem = SuccessiveLookups(1, spec['batch_size'], spec['exp_config']['val_sequence'],
                                    spec['exp_config']['round_length'], DEVICE)
else:
    raise Exception('Experiment unknown!')


n_in, n_out, input_rate = train_problem.get_infos()




loop = build_standard_loop(spec, n_in, input_rate)
loop_model = SequenceWrapper(ParallelNetwork(loop))
out_neuron_size = loop_model.out_size

if spec['experiment'] == 'SequentialMNIST':
    outer = {
        'input': n_in,
        'loop': [['input'], loop_model, None],
        'mean': [['loop'], MeanModule(out_neuron_size, -56), None],
        'output': [['mean'], BaseNeuron(n_out, None), nn.Linear]
    }
else:
    outer = {
        'input': n_in,
        'loop': [['input'], loop_model, None],
        'output': [['loop'], BaseNeuron(n_out, None), nn.Linear]
    }

model = OuterWrapper(DynNetwork(outer))
model.to(DEVICE)


optimizer = OptWrapper(model.parameters(), spec['lr'], spec['lr_decay'], 2500)


train(train_problem, val_problem, optimizer, model, run_id)




