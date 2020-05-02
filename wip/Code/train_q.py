import torch
import torch.nn.functional as F
from torch.distributions import Normal



def make_dataset(num_batches, batch_size, max_iter, s_model, q_model, p_model, m_model, device, env, gamma, num_actions, dims, rand_prob):
    with torch.no_grad():
        obs = torch.zeros((max_iter, num_batches * batch_size, dims+1), dtype=torch.float, requires_grad=False, device=device)
        mem = torch.zeros((max_iter, num_batches * batch_size, 128+max_iter), dtype=torch.float, requires_grad=False, device=device) #128
        value = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        actions = torch.zeros((max_iter, num_batches * batch_size, dims), dtype=torch.float, requires_grad=False, device=device)
        targets = torch.zeros((max_iter, num_batches * batch_size, dims), dtype=torch.float, requires_grad=False, device=device)


        r_sum = 0
        idc_sum = 0
        for i in range(num_batches):
            base = i * batch_size
            cobs = env.reset()
            cmem, h = m_model(cobs, None)
            for k in range(max_iter):
                counterinp = torch.zeros([batch_size, max_iter], device=device)
                counterinp[:, k] = 1
                cmem = torch.cat([cmem, counterinp], dim=1)
                obs[k, base:(base + batch_size)] = cobs
                mem[k, base:(base+batch_size)] = cmem
                p_out, _ = p_model(cmem, None)
                proposal = torch.sigmoid(p_out)
                rand_actions = torch.rand([batch_size, num_actions-1, dims], device=device)
                all_actions = torch.cat([proposal.unsqueeze(1), rand_actions], dim=1)
                processed_state, _ = s_model(cmem, None)
                processed_state = processed_state.unsqueeze(1).expand(batch_size, num_actions, -1)
                q_input = torch.cat([processed_state, all_actions], dim=2).view(batch_size*num_actions, -1)
                q_values, _ = q_model(q_input, None)
                q_values = q_values.view(batch_size, num_actions)
                v, idc = torch.max(q_values, 1, keepdim=True)
                idc_sum += (idc == 0).float().mean()
                action = all_actions[range(batch_size), idc.squeeze()]
                #print(all_actions.shape, idc.shape, action.shape)
                targets[k, base:(base + batch_size)] = action
                action = torch.where(torch.rand_like(action) < rand_prob, torch.rand_like(action), action)
                actions[k, base:(base + batch_size)] = action
                if k > 0:
                    value[k-1, base:(base + batch_size)] = gamma * v + reward
                cobs, reward, _ = env.step(action)
                cmem, h = m_model(cobs, h)
                r_sum += reward.mean()

            value[max_iter - 1, base:(base+batch_size)] = reward

    return obs, mem, value, actions, targets, r_sum/num_batches, idc_sum/(num_batches*max_iter)



def backward_one(action, target, mem, value, s_model, q_model, p_model, device):
    processed_state, _ = s_model(mem, None)
    q_input = torch.cat([processed_state, action], dim=1)
    q_value, _ = q_model(q_input, None)
    lossv = F.mse_loss(q_value, value)
    p_out, _ = p_model(mem, None)
    proposal = torch.sigmoid(p_out)
    lossp = F.mse_loss(proposal, target)
    tloss = lossv + lossp
    tloss.backward()
    return lossv.item(), lossp.item()

#def train(num_epochs, num_batches, batch_size, max_iter, v_model, p_model, m_model, var, gamma, opt, device):



