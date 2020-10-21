from time import time
from torch.optim import Adam
import torch
import json
from torch.optim.lr_scheduler import StepLR


class OptWrapper:
    def __init__(self, params, lr, decay, stepsize):
        self.opt = Adam(params, lr=lr)
        self.scheduler = StepLR(self.opt, stepsize, decay)
        self.params = list(params)

    def get_grad_norm(self):
        norm = 0
        for p in self.params:
            norm += (p.grad.norm().item()) ** 2
        return norm ** (1 / 2)

    def step(self):
        self.opt.step()
        self.scheduler.step()

    def zero_grad(self):
        self.opt.zero_grad()

#TODO: remove
'''
class MyAdam(Adam):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.params = list(params)

    def get_grad_norm(self):
        norm = 0
        for p in self.params:
            norm += (p.grad.norm().item())**2
        return norm**(1/2)
'''

def save_results(run_id, stats, progress, model):
    with open('configs/' + run_id + '.json', 'r') as config_file:
        config = json.load(config_file)
    config['stats'] = stats
    config['progress'] = progress
    with open('configs/' + run_id + '.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)
    model.save('models/' + run_id)


def train(train_problem, val_problem, optimizer, model, run_id, print_every=20, validate_every=100):
    stats = {
        'grad_norm': [],
        'loss': [],
        'acc': [],
        'batch_var': [],
        'val': []
    }

    start = time()
    i = 0
    sumloss = 0
    sumacc = 0
    epstart = time()
    lastit = -1

    for input in train_problem:

        optimizer.zero_grad()
        out, _ = model(input)
        loss, acc = train_problem.loss_and_acc(out)
        loss.backward()
        optimizer.step()

        stats['grad_norm'].append(optimizer.get_grad_norm())
        stats['loss'].append(loss.item())
        stats['acc'].append(acc)


        sumloss += loss.item()
        sumacc += acc
        if i%print_every == print_every-1:
            print(f'It:{i:>5} Loss: {sumloss/print_every:>3.3f} Acc: {sumacc/print_every*100:2.2f}%')
            sumloss = 0
            sumacc = 0
        if i%validate_every == 0:
            k = 0
            val_acc = 0
            for input in val_problem:
                out, _ = model(input)
                _, acc = val_problem.loss_and_acc(out)
                val_acc += acc
                k += 1
            val_acc = val_acc / k
            stats['val'].append(val_acc)
            print(f'Val Acc: {val_acc} Time per it: {(time()-epstart)/(i-lastit):>2.3f}')
            epstart = time()
            lastit = i
            if run_id:
                save_results(run_id, stats, i, model)

        i += 1
    print('Total time: ', time.time() - start)
    return stats
