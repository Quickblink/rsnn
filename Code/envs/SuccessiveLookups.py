
import torch
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
#do same thing or different?


# distance exponentially distributed?
def make_permutations(length, device, perm_num):
    sequence = torch.empty([length, perm_num], dtype=torch.long, device=device)
    for i in range(length):
        sequence[i] = torch.randperm(perm_num, device=device)
    return sequence

def encode_perm(perm, perm_num):
    data = torch.zeros(list(perm.shape)+[perm_num], dtype=torch.float, device=perm.device)
    ones = torch.ones([1], dtype=torch.float, device=perm.device).expand(data.shape)
    data.scatter_(-1, perm.unsqueeze(-1), ones)
    return data

def make_reset(length, prob, device):
    data = Bernoulli(torch.tensor([prob], device=device)).sample([length]).squeeze()
    return data


def make_sequence(length, prob, device, perm_num):
    data = torch.zeros((length, perm_num**2+1), device=device) # recall, store, data, targets
    data[:, 0] = make_reset(length, prob, device)
    targets = torch.empty((2, length, perm_num), dtype=torch.long, device=device)
    cur_perm = torch.tensor(range(perm_num), dtype=torch.long, device=device)
    factors = make_permutations(length, device, perm_num)
    data[:, 1:] = encode_perm(factors, perm_num).view(length, perm_num**2)
    for i in range(length):
        targets[1, i, cur_perm] = torch.tensor(range(perm_num), dtype=torch.long, device=device)
        targets[0, i] = cur_perm
        cur_perm = factors[i] if data[i, 0] > 0 else cur_perm[factors[i]]
    return data, targets



def make_batch(batch_size, length, device, n_in):
    data = torch.zeros((length, batch_size, n_in), device=device)
    ind = torch.randint(n_in, (length, batch_size, 1), device=device)
    data.scatter_(-1, ind, torch.ones([1], device=device).expand_as(data))
    return data, ind

def make_rythm(batch_size, seql, char_dur, device):
    with torch.no_grad():
        rythm = torch.diag(torch.ones([char_dur], device=device))
        rythm = rythm.expand(seql, char_dur, char_dur).reshape(seql * char_dur, 1, char_dur).expand(seql * char_dur, batch_size, char_dur)
        return rythm


def run1(model, lookup, rythm, batch_size, seql, char_dur, perm_num, device, logging=False):

    input, raw_inp = make_batch(batch_size, seql, device, perm_num)

    input = input.repeat_interleave(char_dur, 0)
    input = torch.cat((input, rythm), dim=-1)


    dir_out = model(input, logging=logging)
    output = dir_out[0]
    output = output.view(seql, char_dur, batch_size, perm_num).mean(dim=1)
    out_class = output.argmax(dim=-1, keepdim=True)
    targets = lookup[out_class, raw_inp]

    loss = nn.CrossEntropyLoss()(output[1:].view(-1, perm_num), targets[:-1].view(-1))

    acc = (out_class[1:] == targets[:-1]).float().mean().item()

    return acc, loss, ((input, dir_out[2]) if logging else None)

def run(model, lookup, rythm, batch_size, seql, char_dur, perm_num, device, logging=False):
    inp_dur = char_dur // 2
    pause_dur = char_dur - inp_dur
    input, raw_inp = make_batch(batch_size, seql, device, perm_num)
    input = input.view(seql, 1, batch_size, perm_num).expand(seql, inp_dur, batch_size, perm_num)
    input = torch.cat((input, torch.zeros((seql, pause_dur, batch_size, perm_num), device=device)), dim=1).view(seql*char_dur, batch_size, perm_num)
    #input = input.repeat_interleave(char_dur, 0)
    input = torch.cat((input, rythm), dim=-1)


    dir_out = model(input, logging=logging)
    output = dir_out[0]
    output = output.view(seql, char_dur, batch_size, perm_num).mean(dim=1)
    out_class = output.argmax(dim=-1, keepdim=True)
    targets = lookup[out_class, raw_inp]

    loss = nn.CrossEntropyLoss()(output[1:].view(-1, perm_num), targets[:-1].view(-1))

    acc = (out_class[1:] == targets[:-1]).float().mean().item()

    return acc, loss, ((input, dir_out[2]) if logging else None)


class SuccessiveLookups:
    def __init__(self, iterations, batch_size, n_sequence, round_length, device):
        self.batch_size = batch_size
        self.n_sequence = n_sequence
        #self.val_sequence = val_sequence
        self.round_length = round_length
        self.device = device
        self.max_iter = iterations
        self.lookup = torch.tensor([[6, 1, 4, 5, 7, 2, 0, 3],
                               [7, 0, 4, 2, 3, 1, 5, 6],
                               [0, 5, 6, 2, 4, 3, 7, 1],
                               [2, 7, 6, 4, 3, 1, 5, 0],
                               [0, 6, 4, 5, 2, 1, 7, 3],
                               [5, 1, 0, 6, 4, 7, 3, 2],
                               [4, 6, 1, 2, 5, 7, 0, 3],
                               [2, 7, 4, 3, 5, 6, 0, 1]], dtype=torch.long, device=device)
        self.lookup_size = 8
        with torch.no_grad():
            self.rythm = torch.diag(torch.ones([self.round_length], device=device))
            self.rythm = self.rythm.expand(self.n_sequence, self.round_length, self.round_length).reshape(
                self.n_sequence * self.round_length, 1, self.round_length).expand(
                self.n_sequence * self.round_length, batch_size, self.round_length)

        #self.val_rythm = None

    def __iter__(self):
        self.cur_iter = 0
        return self

    def __next__(self):
        self.cur_iter += 1
        if self.cur_iter > self.max_iter:
            raise StopIteration
        return self.make_inputs()

    def loss_and_acc(self, model_output):
        output = model_output.view(self.n_sequence, self.round_length, self.batch_size, self.lookup_size).mean(dim=1)
        out_class = output.argmax(dim=-1, keepdim=True)
        targets = self.lookup[out_class, self.raw_inp]

        loss = nn.CrossEntropyLoss()(output[1:].view(-1, self.lookup_size), targets[:-1].view(-1))

        acc = (out_class[1:] == targets[:-1]).float().mean().item()

        return loss, acc


    def get_infos(self):
        n_in = self.lookup_size + self.round_length
        return n_in, self.lookup_size, 1.5/n_in


    def make_inputs(self):
        data = torch.zeros((self.n_sequence, self.batch_size, self.lookup_size), device=self.device)
        ind = torch.randint(self.lookup_size, (self.n_sequence, self.batch_size, 1), device=self.device)
        data.scatter_(-1, ind, torch.ones([1], device=self.device).expand_as(data))
        self.raw_inp = ind
        inp_dur = self.round_length // 2
        pause_dur = self.round_length - inp_dur
        data = data.view(self.n_sequence, 1, self.batch_size, self.lookup_size).expand(
            self.n_sequence, inp_dur, self.batch_size, self.lookup_size)
        empty_second_half = torch.zeros((self.n_sequence, pause_dur, self.batch_size, self.lookup_size), device=self.device)
        data = torch.cat((data, empty_second_half), dim=1).view(
            self.n_sequence * self.round_length, self.batch_size, self.lookup_size)
        data = torch.cat((data, self.rythm), dim=-1)
        return data