import torch
import torch.nn.functional as F
from torch.distributions import Normal


def make_dataset_simple(num_batches, batch_size, max_iter, model, teacher, device, env):
    #reihenFolge!!!!
    obs = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
    mask = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
    target = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
    mask[0] = torch.ones([1], device=device)

    for i in range(num_batches):
        base = i * batch_size
        fobs = env.reset(batch_size)
        obs[0, base:(base+batch_size), 0] = fobs[:, 0]
        hidden = None
        for k in range(max_iter-1):
            with torch.no_grad():
                out, hidden = model(obs[k, base:(base + batch_size)].expand([1, batch_size, 1]), hidden)
                action = (out.squeeze() > 0) * 2.0
                target[k, base:(base + batch_size)] = teacher(fobs).unsqueeze(1)/2
            fobs, _, done, _ = env.step(action)
            obs[k + 1, base:(base + batch_size), 0] = fobs[:, 0]
            mask[k+1, base:(base+batch_size)] = (~done.unsqueeze(1)).float()
            if done.all():
                break
        k += 1
        with torch.no_grad():
            target[k, base:(base + batch_size)] = teacher(fobs).unsqueeze(1)/2
    return obs, target, mask



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



def train(num_bigsteps, num_epochs, num_batches, batch_size, max_iter, model, varf, gamma, opt, device):
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
