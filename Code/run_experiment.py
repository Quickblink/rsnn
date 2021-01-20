import sys
sys.path.append('.')
from Code.envs.SequentialMNIST import SequentialMNIST
from Code.train import train, OptWrapper
import torch
import torch.nn as nn
from Code.networks import OuterWrapper, SequentialNetwork, ParallelNetwork, SequenceWrapper, MeanModule, BaseNeuron, build_standard_loop, LSTMWrapper
from Code.envs.SuccessiveLookups import SuccessiveLookups
import json


run_id = sys.argv[1]
with open('configs/'+run_id+'.json', 'r') as config_file:
    config = json.load(config_file)
    spec = config['params']



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



loop_model = LSTMWrapper(n_in, spec['mem_config']['n_neurons']) if spec['architecture'] == 'LSTM' else SequenceWrapper(ParallelNetwork(build_standard_loop(spec, n_in)))
if spec['architecture'] == 'LSTM':
    with torch.no_grad():
        loop_model.lstm.bias_hh_l0[spec['mem_config']['n_neurons']:spec['mem_config']['n_neurons']*2] += 1
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

model = OuterWrapper(SequentialNetwork(outer))
model.to(DEVICE)


optimizer = OptWrapper(model.parameters(), spec['lr'], spec['lr_decay'], 2500)


train(train_problem, val_problem, optimizer, model, run_id)




