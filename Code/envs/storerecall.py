
import torch
from torch.distributions.bernoulli import Bernoulli

#do same thing or different?


# distance exponentially distributed?

def make_sequence(length, store_dist, recall_dist, device):
    last_store = False
    cur_pos = store_dist() # 0
    data = torch.zeros((length, 4), device=device) # recall, store, data, targets
    data[:, 2] = Bernoulli(torch.tensor([0.5], device=device)).sample([length]).squeeze()
    while cur_pos < length:
        if last_store:
            data[cur_pos, 0] = 1 # doing recall here
            data[cur_pos, 3] = cur_val
            x = store_dist()
            cur_pos += x
        else:
            data[cur_pos, 1] = 1 # doing store here
            cur_val = data[cur_pos, 2]
            x = recall_dist()
            cur_pos += x
        last_store = not last_store
    return data


def make_batch(batch_size, length, store_dist, recall_dist, device):
    data = torch.empty((length, batch_size, 4), device=device)
    for i in range(batch_size):
        valid = False
        while not valid:
            tmp = make_sequence(length, store_dist, recall_dist, device)
            valid = tmp[:, 0].max() > 0
        data[:, i] = tmp

    return data

    # test for at least one recall because otherwise not relevant