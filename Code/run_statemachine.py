#%%

import sys
sys.path.append('.')
from Code.envs.statemachine import run, make_rythm
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import OrderedDict

from PIL import Image
import numpy as np


run_id = sys.argv[1]
#print(sys.argv)
with open('configs/'+run_id+'.json', 'r') as config_file:
    config = json.load(config_file)
    spec = config['params']

#%%

standard = {'beta': 0.8,
   'lr': 0.001,
   'NoBias': False, #no effect
   'iterations': 5000,
   'batch_size': 128,
   'spkfn': 'bellec',
   'decay_out': False,
   'control_neuron': 'LIF',
}

example_spec = {
    **standard,
   '1-beta': 'improved',
   'decay_change': 1,
   'architecture': '1L',
   'mem_neuron': 'Cooldown',
   'char_dur': 20,
   'n_mem': 100
}

spec = {
    **standard,
    **spec
}

#%%

BATCH_SIZE = spec['batch_size']

device = torch.device('cuda')

SEQ_LEN = 30
CHAR_DUR = spec['char_dur']

perm_num = 8
n_input = perm_num + CHAR_DUR
n_out = perm_num
n_control = 100
n_mem = spec['n_mem']

MAIN_DECAY = np.exp(-1/(CHAR_DUR*spec['decay_change']))
INPUT_RATE = 1.5/n_input

#%%

MAIN_DECAY# = 0.99

#%%

from Code.everything4 import DynNetwork, OuterWrapper, BaseNeuron, SequenceWrapper, ParallelNetwork, \
 SeqOnlySpike, CooldownNeuron, OutputNeuron, LIFNeuron, NoResetNeuron, AdaptiveNeuron, FlipFlopNeuron, ParallelNetwork2


built_config = {
    'BETA': spec['beta'],
    'OFFSET': -np.log(1-spec['beta']),#3
    'SPIKE_FN': spec['spkfn'],
    '1-beta': spec['1-beta'],
    'ADAPDECAY': MAIN_DECAY, #0.9985,
    'ADAPSCALE': 180
}

#built_config['ADAPDECAY'] = 0.99

mem_config = {
    **built_config,
    'BETA': spec['beta'] if spec['mem_neuron'] in ['Adaptive', 'LIF'] else MAIN_DECAY
}



control_lookup = {
    'LIF': LIFNeuron,
    'Disc': SeqOnlySpike,
    'NoReset': NoResetNeuron
}

mem_lookup = {
    'Adaptive': AdaptiveNeuron,
    'Cooldown': CooldownNeuron,
    'NoReset': NoResetNeuron,
    'FlipFlop': FlipFlopNeuron,
    'LIF': LIFNeuron
}

control_neuron = control_lookup[spec['control_neuron']](n_control, built_config)
mem_neuron = mem_lookup[spec['mem_neuron']](n_mem, mem_config)
out_neuron_size = n_mem if spec['architecture'] == '2L' else n_mem+n_control
out_neuron = OutputNeuron(out_neuron_size, built_config) if spec['decay_out'] else BaseNeuron(out_neuron_size, built_config)
#out_neuron = LIFNeuron(n_control, built_config)

#from mem only
loop_2L = OrderedDict([
    ('input', (n_input, INPUT_RATE)),
    ('control', [['input', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['control'], mem_neuron, nn.Linear]),
    ('output', [['mem'], out_neuron, None]),
])


loop_1L = OrderedDict([
    ('input', (n_input, INPUT_RATE)),
    ('control', [['input', 'control', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['input', 'control', 'mem'], mem_neuron, nn.Linear]),
    ('output', [['control', 'mem'], out_neuron, None]),
])

loop = loop_1L if spec['architecture'] == '1L' else loop_2L

outer = OrderedDict([
    ('input', n_input),
    ('loop', [['input'], SequenceWrapper(ParallelNetwork2(loop, bias=(not spec['NoBias']))), None]),
    ('output', [['loop'], BaseNeuron(n_out, None), nn.Linear]),
])

model = OuterWrapper(DynNetwork(outer), device)


params = list(model.parameters())

model.to(device)


#%%

lr = spec['lr']
optimizer = optim.Adam(params, lr=lr)
#bce = nn.BCEWithLogitsLoss(reduction='none')
#ce = nn.CrossEntropyLoss() #reduction='none'


ITERATIONS = spec['iterations']#36000


lookup = torch.tensor([[6, 1, 4, 5, 7, 2, 0, 3],
        [7, 0, 4, 2, 3, 1, 5, 6],
        [0, 5, 6, 2, 4, 3, 7, 1],
        [2, 7, 6, 4, 3, 1, 5, 0],
        [0, 6, 4, 5, 2, 1, 7, 3],
        [5, 1, 0, 6, 4, 7, 3, 2],
        [4, 6, 1, 2, 5, 7, 0, 3],
        [2, 7, 4, 3, 5, 6, 0, 1]], dtype=torch.long, device=device)

#%%



stats = {
    'grad_norm': [],
    'loss': [],
    'acc': [],
    'batch_var': [],
    'val': []
}

grad_norm_history = []
def record_norm():
    norms = []
    for p in params:
        norms.append(p.grad.norm().item())
    stats['grad_norm'].append(torch.tensor(norms).norm().item())


val_rythm = make_rythm(BATCH_SIZE, 100, CHAR_DUR, device)
def validate():
    acc, _, _ = run(model, lookup, val_rythm, BATCH_SIZE, 100, CHAR_DUR, perm_num, device)
    stats['val'].append((acc).item())
    print('Validation: ', acc)

#%%

train_rythm = make_rythm(BATCH_SIZE, SEQ_LEN, CHAR_DUR, device)


start = time.time()
i = 1
sumloss = 0
sumacc = 0

while i < ITERATIONS:
    batchstart = time.time()
    optimizer.zero_grad()
    acc, loss, _ = run(model, lookup, train_rythm, BATCH_SIZE, SEQ_LEN, CHAR_DUR, perm_num, device)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        record_norm()
        stats['loss'].append(loss.item())
        stats['acc'].append(acc)
        #batch_var = 3 #out_final.var(0).mean().item()
        #stats['batch_var'].append(batch_var)

        #print(loss.item(), acc1, acc2)


    sumloss += loss.item()
    sumacc += acc
    if i%20 == 0:
        print(sumacc/20, i)
        #print(loss.item(), sumloss/20, sumacc/20, time.time()-batchstart, batch_var) #torch.argmax(outputs[-1], 1).float().var()
        sumloss = 0
        sumacc = 0
    if i%100 == 0:
        validate()
        config['stats'] = stats
        config['progress'] = i
        with open('configs/' + run_id + '.json', 'w') as config_file:
            json.dump(config, config_file, indent=2)
        model.save('models/' + run_id)
    i += 1



print('Total time: ', time.time()-start)
