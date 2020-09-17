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
import json


run_id = sys.argv[1]
#print(sys.argv)
with open('configs/'+run_id+'.json', 'r') as config_file:
    config = json.load(config_file)
    spec = config['params']


#TODO: was 256
BATCH_SIZE = spec['batch_size']

USE_JIT = False

device = torch.device('cuda')



mnist = MNIST('.', transform=transforms.ToTensor(), download=True) #distortion_transform([0,15], 3)
test = MNIST('.', transform=transforms.ToTensor(), train=False)


data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, drop_last=True, num_workers=0, shuffle=True)

test_loader = DataLoader(test, batch_size=BATCH_SIZE, drop_last=False, num_workers=0)





from Code.everything3 import DynNetwork, OuterWrapper, LSTMWrapper, MeanModule, BaseNeuron


outer = OrderedDict([
    ('input', 81),
    ('lstm', [['input'], LSTMWrapper(81, 128), None]),
    ('mean', [['lstm'], MeanModule(128, -56), None]),
    ('output', [['mean'], BaseNeuron(10, None), nn.Linear]),
])

model = OuterWrapper(DynNetwork(outer), device, USE_JIT)

with torch.no_grad():
    model.model.layers.lstm.lstm.bias_hh_l0[:256] += 3



params = list(model.parameters())


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




#TODO: what about data augmentation?