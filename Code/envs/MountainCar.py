import torch
import torch.nn as nn
import gym
import gzip
from torch.distributions.uniform import Uniform


class LookupPolicy(nn.Module):
    def __init__(self):
        super(LookupPolicy, self).__init__()
        self.data = torch.load(gzip.open('lookup_policy', 'rb'))
        env = gym.make('MountainCar-v0')
        self.b = torch.tensor([-env.min_position, env.max_speed], dtype=torch.float)
        self.m = torch.tensor([1023.9999 / (env.max_position - env.min_position), 1023.9999 / (2 * env.max_speed)], dtype=torch.float)

    def forward(self, inp):
        idc = ((inp + self.b) * self.m).long()
        return self.data[idc[..., 0], idc[..., 1]]


class MultiMountainCar():
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
        self.state[:, 0] = Uniform(-0.6, -0.4).sample((num_env))
        self.done = torch.zeros((num_env), dtype=torch.bool)


    def step(self, action):
        # assert
        self.state[:, 1] += (action-1)*self.force + torch.cos(3*self.state[:, 0])*(-self.gravity)
        self.state[:, 1] = torch.clamp(self.state[:, 1], -self.max_speed, self.max_speed)
        self.state[:, 0] += self.state[:, 1]
        self.state[:, 0] = torch.clamp(self.state[:, 0], self.min_position, self.max_position)
        self.state[:, 0] = torch.where(((self.state[:, 0]==self.min_position) & (self.state[:, 1]<0)), 0, self.state[:, 0])

        self.done = self.done | ((self.state[:, 0] >= self.goal_position) & (self.state[:, 1] >= self.goal_velocity))
        reward = -1 + self.done
        return self.state, reward, self.done, {}
