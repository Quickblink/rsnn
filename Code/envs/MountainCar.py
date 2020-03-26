import torch
import torch.nn as nn
import gzip
from torch.distributions.uniform import Uniform
import os


class LookupPolicy(nn.Module):
    def __init__(self):
        super(LookupPolicy, self).__init__()
        self.data = torch.load(gzip.open(os.path.dirname(__file__)+'/lookup_policy', 'rb'))
        env = MultiMountainCar()
        self.b = torch.tensor([-env.min_position, env.max_speed], dtype=torch.float)
        self.m = torch.tensor([1023.999 / (env.max_position - env.min_position), 1023.999 / (2 * env.max_speed)], dtype=torch.float)

    def forward(self, inp):
        idc = ((inp + self.b) * self.m).long()
        return self.data[idc[..., 0], idc[..., 1]]


class MultiMountainCar:
    def __init__(self):
        self.state = None
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0
        self.force = 0.001
        self.gravity = 0.0025

    def reset(self, num_env):
        self.state = torch.zeros((num_env, 2))
        self.state[:, 0] = Uniform(-0.6, -0.4).sample((num_env,))
        self.done = torch.zeros((num_env), dtype=torch.bool)
        return self.state


    def step(self, action):
        # assert
        #print(action.shape, self.state[:, 1].shape)
        newstate = torch.empty_like(self.state)
        newstate[:, 1] = torch.clamp(self.state[:, 1] + (action-1)*self.force + torch.cos(3*self.state[:, 0])*(-self.gravity), -self.max_speed, self.max_speed)
        newstate[:, 0] = torch.clamp(self.state[:, 0] + newstate[:, 1], self.min_position, self.max_position)
        newstate[:, 1] = torch.where(((newstate[:, 0] == self.min_position) & (newstate[:, 1] < 0)), torch.zeros([1]), newstate[:, 1])
        self.state = newstate
        self.done = self.done | ((self.state[:, 0] >= self.goal_position) & (self.state[:, 1] >= self.goal_velocity))
        reward = -1 + self.done
        return self.state, reward, self.done, {}


class PassiveEnv:
    def __init__(self):
        self.env = MultiMountainCar()
        self.agent = LookupPolicy()

    def getBatch(self, bsz):
        data = torch.empty((200, bsz, 1))
        mask = torch.empty((200, bsz))
        targets = torch.empty((200, bsz))
        obs = self.env.reset(bsz)
        action = torch.ones((1, 1))
        done = torch.zeros((1,1))
        for t in range(200):
            data[t, :, 0] = obs[:, 0]
            #data[t, :, 1] = action
            mask[t] = 1.0 - done.float()
            targets[t] = obs[:, 1]
            action = self.agent(obs)
            obs, _, done, _ = self.env.step(action)
            if done.all():
                break
        return data[:(t+1)], targets[:(t+1)], mask[:(t+1)]
