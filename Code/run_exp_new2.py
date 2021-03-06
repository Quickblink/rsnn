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
import json


run_id = sys.argv[1]
#print(sys.argv)
with open('configs/'+run_id+'.json', 'r') as config_file:
    config = json.load(config_file)
    spec = config['params']


BATCH_SIZE = spec['batch_size']


device = torch.device('cuda')



mnist = MNIST('.', transform=transforms.ToTensor(), download=True) #distortion_transform([0,15], 3)
test = MNIST('.', transform=transforms.ToTensor(), train=False)


data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, drop_last=True, num_workers=0, shuffle=True)

test_loader = DataLoader(test, batch_size=BATCH_SIZE, drop_last=False, num_workers=0)



sample_settings = {
    'spkfn' : 'bellec',
    'architecture': '1L',
    'beta': 0.95,
    'control_neuron': 'LIF',
    'mem_neuron' : 'Adaptive',
    'lr' : 1e-2,
    'lr_decay': 0.8,
    '1-beta': True,
    'decay_out': True,
}




from Code.everything4 import ParallelNetwork2, MeanModule, SequenceWrapper, BaseNeuron, OuterWrapper, DynNetwork,\
    CooldownNeuron, OutputNeuron, LIFNeuron, NoResetNeuron, AdaptiveNeuron, FlipFlopNeuron, SeqOnlySpike

built_config = {
    'BETA': spec['beta'],
    'OFFSET': 7, # was 3 for config24
    'SPIKE_FN': spec['spkfn'],
    '1-beta': spec['1-beta'],
    'ADAPDECAY': 0.9985,
    'ADAPSCALE': 180
}

mem_config = {
    **built_config,
    'BETA': spec['mem_beta']
}

#TODO: revert this
n_control = 220#120
n_mem = 1#100
input_rate = 0.03
n_input = 80+28+30

control_lookup = {
    'LIF': LIFNeuron,
    'Disc': SeqOnlySpike,
    'NoReset': NoResetNeuron
}

mem_lookup = {
    'Adaptive': AdaptiveNeuron,
    'Cooldown': CooldownNeuron,
    'NoReset': NoResetNeuron,
    'FlipFlop': FlipFlopNeuron
}

control_neuron = control_lookup[spec['control_neuron']](n_control, built_config)
mem_neuron = mem_lookup[spec['mem_neuron']](n_mem, mem_config)
out_neuron = OutputNeuron(n_control+n_mem, built_config) if spec['decay_out'] else BaseNeuron(n_control+n_mem, built_config)


loop_2L = OrderedDict([
    ('input', (n_input, input_rate)),
    ('control', [['input', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['control'], mem_neuron, nn.Linear]),
    ('output', [['control', 'mem'], out_neuron, None]),
])

loop_1L = OrderedDict([
    ('input', (n_input, input_rate)),
    ('control', [['input', 'control', 'mem'], control_neuron, nn.Linear]),
    ('mem', [['input', 'control', 'mem'], mem_neuron, nn.Linear]),
    ('output', [['control', 'mem'], out_neuron, None]),
])

loop = loop_1L if spec['architecture'] == '1L' else loop_2L

outer = OrderedDict([
    ('input', n_input),
    ('loop', [['input'], SequenceWrapper(ParallelNetwork2(loop)), None]),
    ('mean', [['loop'], MeanModule(n_control+n_mem, -56), None]),
    ('output', [['mean'], BaseNeuron(10, None), nn.Linear]),
])

model = OuterWrapper(DynNetwork(outer), device)


#loop_model = OuterWrapper(make_SequenceWrapper(ParallelNetwork(loop), USE_JIT), device, USE_JIT)

#final_linear = nn.Linear(n_control+n_mem, 10).to(device)
'''
if spec['ported_weights']:
    o_weights = pickle.load(open('weight_transplant_enc', 'rb'))

    o1 = torch.tensor(o_weights['RecWeights/RecurrentWeight:0']).t()
    o2 = torch.tensor(o_weights['InputWeights/InputWeight:0']).t()
    o3 = torch.cat((o2, o1), dim=1)
    with torch.no_grad():
        model.pretrace.layers.loop.model.layers.control_synapse.weight.data[:,:300] = o3[:120] if spec['architecture'] == '1L' else o3[:120, :181]
        model.pretrace.layers.loop.model.layers.mem_synapse.weight.data[:,:300] = o3[120:] if spec['architecture'] == '1L' else o3[120:, 180:]
        model.pretrace.layers.output_synapse.weight.data = torch.tensor(o_weights['out_weight:0']).t()
'''
params = list(model.parameters())

'''
if spec['NoBias']:
    with torch.no_grad():
        model.pretrace.layers.loop.model.layers.control_synapse.bias *= 0
        model.pretrace.layers.loop.model.layers.mem_synapse.bias *= 0
        model.pretrace.layers.output_synapse.bias *= 0
    params = [model.pretrace.layers.loop.model.layers.control_synapse.weight,
              model.pretrace.layers.loop.model.layers.mem_synapse.weight, model.pretrace.layers.output_synapse.bias,
              model.pretrace.layers.output_synapse.weight]
    if spec['NoBias'] == 'addBias':
        params += [model.pretrace.layers.loop.model.layers.control_synapse.bias,
                   model.pretrace.layers.loop.model.layers.mem_synapse.bias]

'''
model.to(device)



lr = spec['lr']
optimizer = optim.Adam(params, lr=lr)
ce = nn.CrossEntropyLoss()

ITERATIONS = spec['iterations']#36000


'''

#TODO: check correctness here

with torch.no_grad():
    for i in range(100):
        loop_model.pretrace.model.layers.mem_synapse.weight[i, i+201] = 0

    for i in range(120):
        loop_model.pretrace.model.layers.control_synapse.weight[i, i+81] = 0

'''
with torch.no_grad():
    rythm = torch.diag(torch.ones([28], device=device))
    rythm = rythm.expand(30, 28, 28).reshape(30 * 28, 1, 28)[1:]
    rythm2 = torch.diag(torch.ones([30], device=device))
    rythm2 = rythm2.view(30, 1, 30).expand(30, 28, 30).reshape(30 * 28, 1, 30)[1:]

#trigger_signal = torch.ones([783+56, 1, 1], device=device)
#trigger_signal[:783] = 0
def encode_input(curr, last):
    out = torch.zeros([783+56, curr.shape[1], 2,40], device=curr.device)
    out[:783, :, 0, :] = ((torch.arange(40, device=curr.device) < 40 * last) & (torch.arange(40, device=curr.device) > 40 * curr)).float()
    out[:783, :, 1, :] = ((torch.arange(40, device=curr.device) > 40 * last) & (torch.arange(40, device=curr.device) < 40 * curr)).float()
    #out = torch.cat((out.view([783+56, curr.shape[1], 80]), trigger_signal.expand([783+56, curr.shape[1], 1])), dim=-1)
    out = torch.cat((out.view([783+56, curr.shape[1], 80]), rythm.expand([783+56, curr.shape[1], 28]), rythm2.expand([783+56, curr.shape[1], 30])), dim=-1)

    return out

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


def validate():
    with torch.no_grad():
        i = 0
        acc = 0
        for inp, target in test_loader:
            x = inp.view(inp.shape[0], -1, 1).transpose(0, 1).to(device)
            x = encode_input(x[1:], x[:-1])
            target = target.to(device)
            outputs, _ = model(x)
            choice = torch.argmax(outputs, 1)
            acc += (choice == target).float().mean()
            i += 1
        stats['val'].append((acc/i).item())
        #print('Acc: ' + str(acc / i))


start = time.time()
i = 1
sumloss = 0
sumacc = 0
k = 0
while i < ITERATIONS:
    print('Epoch: ', k)
    k = k + 1
    validate()
    for inp, target in data_loader:
        batchstart = time.time()
        x = inp.view(BATCH_SIZE, -1, 1).transpose(0,1).to(device)
        x = encode_input(x[1:], x[:-1])
        #print(x.shape)
        target = target.to(device)
        optimizer.zero_grad()
        out_final, _ = model(x)
        #meaned = outputs[-56:].mean(dim=0) #TODO: what is this value really in bellec?
        #out_final = final_linear(meaned)
        #test_norm = out_final.norm().item()
        loss = ce(out_final, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            record_norm()
            stats['loss'].append(loss.item())
            acc = (torch.argmax(out_final, 1) == target).float().mean().item()
            stats['acc'].append(acc)
            batch_var = out_final.var(0).mean().item()
            stats['batch_var'].append(batch_var)

        #print(loss.item(), acc, batch_var, test_norm, loop_model.pretrace.model.layers.control_synapse.weight.grad.norm().item(), target[0].item())


        sumloss += loss.item()
        sumacc += acc
        if i%20 == 0:
            print(loss.item(), sumloss/20, sumacc/20, time.time()-batchstart, batch_var) #torch.argmax(outputs[-1], 1).float().var()
            sumloss = 0
            sumacc = 0
        if i%2500 == 0:
            lr = lr * spec['lr_decay']
            optimizer = optim.Adam(params, lr=lr)
            print('Learning Rate: ', lr)
        i += 1
    #pickle.dump(stats, open('stats', 'wb'))
    config['stats'] = stats
    config['progress'] = i
    #config['mem_req'] = torch.cuda.max_memory_allocated()
    with open('configs/' + run_id + '.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)
    model.save('models/'+run_id)
    #post_model.save('../../models/post_big11_'+str(k))


print('Total time: ', time.time()-start)


