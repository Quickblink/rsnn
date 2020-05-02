import torch
import torch.nn.functional as F
from torch.distributions import Normal

'''
def make_dataset(num_batches, batch_size, max_iter, v_model, p_model, m_model, device, env, var, gamma):
    with torch.no_grad():
        obs = torch.zeros((max_iter, num_batches * batch_size, 2), dtype=torch.float, requires_grad=False, device=device)
        mem = torch.zeros((max_iter, num_batches * batch_size, 128), dtype=torch.float, requires_grad=False, device=device)
        value = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        advantage = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        for i in range(num_batches):
            base = i * batch_size
            obs[base:(base+batch_size), 0] = env.reset()
            h = None
            v_old, reward = 0, 0
            for k in range(max_iter - 1):
                mem[base:(base+batch_size), k], h = m_model(obs[base:(base+batch_size), k], h)
                v = v_model(mem[base:(base+batch_size), k])
                vpr = gamma * v + reward
                value[base:(base+batch_size), k] = vpr
                advantage[base:(base+batch_size), k] = vpr - v_old
                v_old = v
                mean = p_model(mem[base:(base+batch_size), k])
                action = Normal(mean, var).sample()
                obs[base:(base + batch_size), k+1], reward, _ = env.step(action)
            k = max_iter - 1
            mem[base:(base + batch_size), k], h = m_model(obs[base:(base + batch_size), k], h)
            value[base:(base + batch_size), k] = reward
            advantage[base:(base + batch_size), k] = reward - v_old
    return obs, mem, value, advantage
    #reduce everything by 1

'''

def make_dataset(num_batches, batch_size, max_iter, v_model, p_model, m_model, device, env, var, gamma, dims):
    with torch.no_grad():
        obs = torch.zeros((max_iter, num_batches * batch_size, dims+1), dtype=torch.float, requires_grad=False, device=device)
        mem = torch.zeros((max_iter, num_batches * batch_size, 128), dtype=torch.float, requires_grad=False, device=device)
        value = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        advantage = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        actions = torch.zeros((max_iter, num_batches * batch_size, dims), dtype=torch.float, requires_grad=False, device=device)

        r_sum = 0
        for i in range(num_batches):
            base = i * batch_size
            cobs = env.reset()
            cmem, h = m_model(cobs, None)
            v_old = 0
            for k in range(max_iter):
                obs[k, base:(base + batch_size)] = cobs
                mem[k, base:(base+batch_size)] = cmem
                p_out, _ = p_model(mem[k, base:(base + batch_size)], None)
                mean = torch.sigmoid(p_out)
                action = Normal(mean, var).sample()
                actions[k, base:(base + batch_size)] = action
                cobs, reward, _ = env.step(action)
                r_sum += reward.mean()
                if k < max_iter - 1:
                    cmem, h = m_model(cobs, h)
                    v, _ = v_model(cmem, None)
                else:
                    v = 0

                vpr = gamma * v + reward
                #print(reward.shape, v.shape, vpr.shape, value[base:(base+batch_size), k].shape)
                value[k, base:(base+batch_size)] = vpr
                advantage[k, base:(base+batch_size)] = vpr - v_old
                v_old = v

    return obs, mem, value, advantage, actions, r_sum/num_batches

'''
def make_dataset_simple(num_batches, batch_size, max_iter, model, var, gamma, device):
    data = torch.empty((num_batches * batch_size, max_iter, 4), dtype=torch.float, requires_grad=False, device=device)
    env = MultiEnv(batch_size, max_iter, device)
    for i in range(num_batches):
        base = i * batch_size
        data[base:(base+batch_size), 0, :2] = env.reset()
        with torch.no_grad():
            out, v, hidden, var = model(data[base:(base+batch_size), 0, :2], None)
            action = Normal(out, var).sample()
            v_old = v.squeeze()
        for k in range(1, max_iter-1):
            data[base:(base+batch_size), k, :2], reward = env.step(action)
            with torch.no_grad():
                out, v, hidden, var = model(data[base:(base+batch_size), k, :2], hidden)
                action = Normal(out, var).sample()
                v = v.squeeze()
                vpr = gamma * v + reward
                data[base:base + batch_size, k, 2] = vpr
                data[base:base + batch_size, k, 3] = vpr - v_old
                v_old = v
        k = max_iter - 1
        data[base:(base + batch_size), k, :2], reward = env.step(action)
        data[base:base + batch_size, k, 2] = reward
        data[base:base + batch_size, k, 3] = reward - v_old
    return data
    
    #value+reward of all states, real and shadow
#compare real and shadow for advantage
#
def backward_batch_simple(batch0, model, max_iter, gamma, varf, device):
    lossv = torch.zeros((1), dtype=torch.float, device=device)
    lossp = torch.zeros((1), dtype=torch.float, device=device)
    lossvar = torch.zeros((1), dtype=torch.float, device=device)


    act0, v0, hidden, var = model(batch0[:, 0, :2], None)
    v_old = v0.squeeze()
    act_old = act0.squeeze()
    var_old = var.squeeze()
    for k in range(1, max_iter-1):
        act0, v0, hidden, var = model(batch0[:, k, :2], hidden)
        act0 = act0.squeeze()
        v0 = v0.squeeze()
        lossv += F.mse_loss(v_old, batch0[:, k, 2])#.backward(retain_graph=True)
        adv = batch0[:, k, 3]
        loss = -adv * torch.exp(Normal(act_old, var_old).log_prob(batch0[:, k, 0]))
        lossp += loss.mean()#.backward(retain_graph=True)
        lossvar += var_old.mean() * varf
        v_old = v0
        act_old = act0
        var_old = var
    k = max_iter - 1
    lossv += F.mse_loss(v_old, batch0[:, k, 2])#.backward(retain_graph=True)
    adv = batch0[:, k, 3]
    loss = adv * torch.exp(Normal(act_old, var_old).log_prob(batch0[:, k, 0]))
    lossp += loss.mean()#.backward()
    lossvar += var_old.mean() * varf
    #print('Loss:', lossv.item(), lossp.item())
    tloss = lossp + lossv + lossvar
    tloss.backward()
    return lossv.item(), lossp.item(), lossvar.item()
    
'''

def backward_one(action, mem, value, advantage, v_model, p_model, var, device):
    v, _ = v_model(mem, None)
    lossv = F.mse_loss(v, value)
    p_out, _ = p_model(mem, None)
    mean = torch.sigmoid(p_out)
    lossp = (-advantage * torch.exp(Normal(mean, var).log_prob(action))).mean()
    tloss = lossv + lossp
    tloss.backward()
    return lossv.item(), lossp.item()

#def train(num_epochs, num_batches, batch_size, max_iter, v_model, p_model, m_model, var, gamma, opt, device):






def trainold(num_bigsteps, num_epochs, num_batches, batch_size, max_iter, model, varf, gamma, opt, device):
    for bs in range(num_bigsteps):
        data = make_dataset_simple(num_batches, batch_size, max_iter, model, None, gamma, device)
        print('Bigstep: ', bs) #,', Avarage Advantage: ',data[:,:,3].sum()/data.shape[1]
        for e in range(num_epochs):
            idc = torch.randperm(data.shape[0], device=device)
            for i in range(num_batches):
                base = i*batch_size
                #batch0 = data[0, idc[base:base + batch_size]]
                #batch1 = data[1, idc[base:base + batch_size]]
                batch = data[idc[base:base + batch_size]]
                model.zero_grad()
                #lossv, lossp = backward_batch(batch0, batch1, model, max_iter, gamma, var, device)
                lossv, lossp, lossvar = backward_batch_simple(batch, model, max_iter, gamma, varf, device)
                opt.step()
                if i % 10 == 0:
                    print('Loss:', lossv, lossp, lossvar)
            validate(1, batch_size, max_iter, model, device)


def validate(num_batches, batch_size, max_iter, model, device, render=False):
    env = MultiEnv(batch_size, max_iter, device)
    traj = []
    obs = env.reset()
    totr = 0
    hidden = None  # model.get_initial_state()
    for k in range(1, max_iter):
        with torch.no_grad():
            out, _, hidden, var = model(obs, hidden)
        obs, r = env.step(out)
        totr += r.sum()
        traj.append('%.2f' % out[0, 0].item())
    print('Policy Reward:', totr/(batch_size))
    print('Trajectory: ', traj)
    print('Last Action: ', out.squeeze()[:10])
    print('Last Var: ', var.squeeze()[:10])
    if render:
        env.render()
