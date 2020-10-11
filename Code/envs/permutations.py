
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

def make_sequence(length, store_dist, device, perm_num):
    last_pos = 0
    cur_pos = store_dist() # 0
    data = torch.zeros((length, perm_num**2+1), device=device) # recall, store, data, targets
    targets = torch.empty((2, length, perm_num), dtype=torch.long, device=device)
    cur_perm = torch.tensor(range(perm_num), dtype=torch.long, device=device)
    factors = make_permutations(length, device, perm_num)
    data[:, 1:] = encode_perm(factors, perm_num).view(length, perm_num**2)
    while cur_pos < length:
        targets[1, last_pos:cur_pos, cur_perm] = torch.tensor(range(perm_num), dtype=torch.long, device=device)
        targets[0, last_pos:cur_pos] = cur_perm
        cur_perm = cur_perm[factors[cur_pos]]
        data[cur_pos, 0] = 1 # "store"
        last_pos = cur_pos
        cur_pos += store_dist()
    targets[1, last_pos:length, cur_perm] = torch.tensor(range(perm_num), dtype=torch.long, device=device)
    targets[0, last_pos:length] = cur_perm
    return data, targets


def make_batch(batch_size, length, store_dist, device, perm_num):
    data = torch.empty((length, batch_size, perm_num**2+1), device=device)
    targets = torch.empty((2, length, batch_size, perm_num), device=device, dtype=torch.long)
    for i in range(batch_size):
        valid = False
        while not valid:
            tmp_data, tmp_targets = make_sequence(length, store_dist, device, perm_num)
            valid = tmp_data[:, 0].max() > 0
        data[:, i] = tmp_data
        targets[:, :, i] = tmp_targets
    return data, targets

    # test for at least one recall because otherwise not relevant