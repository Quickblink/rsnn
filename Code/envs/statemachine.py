
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
    ind = torch.randint(n_in, (length, batch_size, 1))
    data.scatter_(-1, ind, torch.ones([1]).expand_as(data))
    return data, ind



def run(model, lookup, batch_size, seql, char_dur, perm_num, device, logging=False):
    with torch.no_grad():
        rythm = torch.diag(torch.ones([char_dur], device=device))
        # rythm *= 0
        rythm = rythm.expand(seql, char_dur, char_dur).reshape(seql * char_dur, 1, char_dur).expand(seql * char_dur, batch_size, char_dur)

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

    return acc, loss, (dir_out[2] if logging else None)