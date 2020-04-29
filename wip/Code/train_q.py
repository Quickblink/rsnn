import torch
import torch.nn.functional as F
from torch.distributions import Normal



def make_dataset(num_batches, batch_size, max_iter, s_model, q_model, p_model, m_model, device, env, var, gamma, num_actions):
    with torch.no_grad():
        obs = torch.zeros((max_iter, num_batches * batch_size, 2), dtype=torch.float, requires_grad=False, device=device)
        mem = torch.zeros((max_iter, num_batches * batch_size, 128), dtype=torch.float, requires_grad=False, device=device)
        value = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)
        actions = torch.zeros((max_iter, num_batches * batch_size, 1), dtype=torch.float, requires_grad=False, device=device)

        r_sum = 0
        for i in range(num_batches):
            base = i * batch_size
            cobs = env.reset()
            cmem, h = m_model(cobs, None)
            for k in range(max_iter):
                obs[k, base:(base + batch_size)] = cobs
                mem[k, base:(base+batch_size)] = cmem
                p_out, _ = p_model(cmem, None)
                proposal = torch.sigmoid(p_out)
                rand_actions = torch.rand([batch_size, num_actions-1], device=device)
                all_actions = torch.cat([proposal, rand_actions], dim=1).unsqueeze(2)
                processed_state, _ = s_model(cmem, None)
                processed_state = processed_state.unsqueeze(1).expand(batch_size, num_actions, -1)
                q_input = torch.cat([processed_state, all_actions], dim=2).view(batch_size*num_actions, -1)
                q_values, _ = q_model(q_input, None)
                q_values = q_values.view(batch_size, num_actions)
                v, idc = torch.max(q_values, 1, keepdim=True)
                action = all_actions[idc]
                actions[k, base:(base + batch_size)] = action
                r_sum += reward.mean()
                if k > 0:
                    value[k-1, base:(base + batch_size)] = gamma * v + reward
                cobs, reward, _ = env.step(action)
            value[max_iter - 1, base:(base+batch_size)] = reward

    return obs, mem, value, actions, r_sum/num_batches



def backward_one(action, mem, value, s_model, q_model, p_model, var, device):
    processed_state, _ = s_model(mem, None)
    q_input = torch.cat([processed_state, action], dim=1)
    q_value, _ = q_model(q_input, None)
    lossv = F.mse_loss(q_value, value)
    p_out, _ = p_model(mem, None)
    proposal = torch.sigmoid(p_out)
    lossp = F.mse_loss(proposal, action)
    tloss = lossv + lossp
    tloss.backward()
    return lossv.item(), lossp.item()

#def train(num_epochs, num_batches, batch_size, max_iter, v_model, p_model, m_model, var, gamma, opt, device):



