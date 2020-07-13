
import sys
#TODO: path
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
import pickle

#TODO: change to 256
BATCH_SIZE = 256#128

USE_JIT = False

device = torch.device('cuda')


mnist = MNIST('.', transform=transforms.ToTensor(), download=True) #distortion_transform([0,15], 3)
test = MNIST('.', transform=transforms.ToTensor(), train=False)


data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, drop_last=True, num_workers=0, shuffle=True)

test_loader = DataLoader(test, batch_size=1024, drop_last=False, num_workers=0)

like_bellec = {
    'spkfn' : 'bellec',
    'spkconfig' : 0,
    'architecture': '1L',
    'beta': 0.95,
    'control_neuron': 'LIF',
    'mem_neuron' : 'Adaptive',
    'lr' : 1e-2,
    '1-beta': True,
    'decay_out': True
}

spec = like_bellec

from Code.Networks import Selector, DynNetwork, OuterWrapper, LSTMWrapper, ReLuWrapper, DummyNeuron, make_SequenceWrapper, ParallelNetwork
from Code.NewNeurons2 import SeqOnlySpike, CooldownNeuron, OutputNeuron, LIFNeuron, NoResetNeuron, AdaptiveNeuron

built_config = {
    'BETA': spec['beta'],
    'OFFSET': 2, # TODO: this?
    'SPIKE_FN': spec['spkfn'],
    '1-beta': spec['1-beta'],
    'ADAPDECAY': 0.9985,
    'ADAPSCALE': 180
}

n_control = 120
n_mem = 100

control_lookup = {
    'LIF': LIFNeuron,
    'Disc': SeqOnlySpike,
    'NoReset': NoResetNeuron
}

mem_lookup = {
    'Adaptive': AdaptiveNeuron,
    'Cooldown': CooldownNeuron,
    'NoReset': NoResetNeuron
}

control_neuron = control_lookup[spec['control_neuron']](n_control, built_config)
mem_neuron = mem_lookup[spec['mem_neuron']](n_mem, built_config)
out_neuron = OutputNeuron(n_control+n_mem, built_config) if spec['decay_out'] else DummyNeuron(n_control+n_mem, built_config)


loop_2L = OrderedDict([
    ('input', 81),
    ('control', [['input', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['control'], mem_neuron, nn.Linear]),
    ('output', [['control', 'mem'], out_neuron, None]),
])

loop_1L = OrderedDict([
    ('input', 81),
    ('control', [['input', 'control', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['input', 'control', 'mem'], mem_neuron, nn.Linear]),
    ('output', [['control', 'mem'], out_neuron, None]),
])

loop = loop_1L if spec['architecture'] == '1L' else loop_2L

loop_model = OuterWrapper(make_SequenceWrapper(ParallelNetwork(loop), USE_JIT), device, USE_JIT)

final_linear = nn.Linear(n_control+n_mem, 10).to(device)



params = list(loop_model.parameters())+list(final_linear.parameters())
lr = spec['lr']
optimizer = optim.Adam(params, lr=lr)
ce = nn.CrossEntropyLoss()




'''
with torch.no_grad():
    for i in range(100):
        model.pretrace.model.layers.adaptive_synapse.weight[i, i+81] = 0

    for i in range(120):
        model.pretrace.model.layers.regular_synapse.weight[i, i+181] = 0

'''


trigger_signal = torch.ones([783+56, 1, 1], device=device)
trigger_signal[:783] = 0
def encode_input(curr, last):
    out = torch.zeros([783+56, curr.shape[1], 2,40], device=curr.device)
    out[:783, :, 0, :] = ((torch.arange(40, device=curr.device) < 40 * last) & (torch.arange(40, device=curr.device) > 40 * curr)).float()
    out[:783, :, 1, :] = ((torch.arange(40, device=curr.device) > 40 * last) & (torch.arange(40, device=curr.device) < 40 * curr)).float()
    out = torch.cat((out.view([783+56, curr.shape[1], 80]), trigger_signal.expand([783+56, curr.shape[1], 1])), dim=-1)
    return out

stats = {
    'grad_norm': [],
    'loss': [],
    'acc': [],
    'batch_var': []
}

grad_norm_history = []
def record_norm():
    norms = []
    for p in params:
        norms.append(p.grad.norm().item())
    stats['grad_norm'].append(torch.tensor(norms).norm().item())


ITERATIONS = 36000

start = time.time()
i = 1
sumloss = 0
sumacc = 0
k = 0
while i < ITERATIONS:
    print('Epoch: ', k)
    k = k + 1
    for inp, target in data_loader:
        batchstart = time.time()
        x = inp.view(BATCH_SIZE, -1, 1).transpose(0,1).to(device)
        x = encode_input(x[1:], x[:-1])
        #print(x.shape)
        target = target.to(device)
        optimizer.zero_grad()
        outputs, _ = loop_model(x)
        meaned = outputs[-56:].mean(dim=0) #TODO: what is this value really in bellec?
        out_final = final_linear(meaned)
        loss = ce(out_final, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            record_norm()
            stats['loss'].append(loss.item())
            acc = (torch.argmax(out_final, 1) == target).float().mean().item()
            stats['acc'] = acc
            batch_var = out_final.var(0).mean().item()
            stats['batch_var'] = batch_var

        sumloss += loss.item()
        sumacc += acc
        if i%20 == 0:
            print(loss.item(), sumloss/20, sumacc/20, time.time()-batchstart, batch_var) #torch.argmax(outputs[-1], 1).float().var()
            sumloss = 0
            sumacc = 0
        if i%2500 == 0:
            lr = lr * 0.8
            optimizer = optim.Adam(params, lr=lr)
            print('Learning Rate: ', lr)
        i += 1
    pickle.dump(stats, open('stats', 'wb'))
    #model.save('../../models/adap_clip5_'+str(k))
    #post_model.save('../../models/post_big11_'+str(k))


print('Total time: ', time.time()-start)




#TODO: what about data augmentation?