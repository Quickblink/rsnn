
import torch
from torch.distributions.bernoulli import Bernoulli

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


def make_batch(batch_size, length, prob, device, perm_num):
    data = torch.empty((length, batch_size, perm_num**2+1), device=device)
    targets = torch.empty((2, length, batch_size, perm_num), device=device, dtype=torch.long)
    for i in range(batch_size):
        tmp_data, tmp_targets = make_sequence(length, prob, device, perm_num)
        data[:, i] = tmp_data
        targets[:, :, i] = tmp_targets
    return data, targets

    # test for at least one recall because otherwise not relevant