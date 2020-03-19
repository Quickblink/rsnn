from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from gym.spaces import Box
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

#TODO: clamp in multi?

class MultiEnv():
    def __init__(self, batch_size, num_points, device):
        self.data = torch.empty((batch_size, 2, num_points), dtype=torch.float, device=device)
        self.best = torch.empty((batch_size), dtype=torch.float, device=device)
        self.nstep = 0
        self.batch_size = batch_size
        self.device = device

    def reset(self):
        self.data[:, 0, 0] = torch.rand((self.batch_size), dtype=torch.float, device=self.device)
        self.best = torch.normal(torch.zeros((1), dtype=torch.float, device=self.device).expand((self.batch_size, 1)),
                                 torch.ones((1), dtype=torch.float, device=self.device).expand((self.batch_size, 1))).squeeze()
        self.nstep = 1
        self.data[:, 1, 0] = self.best
        return self.data[:, :, 0]



    def step(self, actions, shadow=False):
        x = self.data[:, 0, :self.nstep]
        y = self.data[:, 1, :self.nstep]
        K = torch.exp(-1 / 2 * ((x.view(self.batch_size, self.nstep, 1) - x.view(self.batch_size, 1, self.nstep)) / 0.1) ** 2) + torch.eye(self.nstep, dtype=torch.float, device=self.device) * 1e-5
        u = torch.cholesky(K)
        k = torch.exp(-1 / 2 * ((x.view(self.batch_size, self.nstep) - actions.view(self.batch_size,1)) / 0.1) ** 2)
        sol = torch.cholesky_solve(torch.cat((k.view(self.batch_size, self.nstep, 1), y.view(self.batch_size, self.nstep, 1)), dim=2), u)
        #print(sol)
        mav = torch.matmul(k.view(self.batch_size, 1, self.nstep), sol).view(self.batch_size, 2) #check shapes!
        #print(mav)
        newy = torch.normal(mav[:, 1], 1-mav[:, 0])
        newbest = torch.max(self.best, newy)
        reward = newbest - self.best
        if not shadow:
            self.best = newbest
        self.data[:, 0, self.nstep] = actions.view(self.batch_size)
        self.data[:, 1, self.nstep] = newy
        self.nstep = self.nstep + 1
        return self.data[:, :, self.nstep-1], reward

    def render(self):
        plt.scatter(self.data[0, 0, :self.nstep].cpu(), self.data[0, 1, :self.nstep].cpu(), c=range(self.nstep))
        for i in range(self.nstep):
            r = np.random.rand() * 2 * np.pi
            xoff = np.sin(r) * 0.1
            yoff = np.cos(r) * 0.1
            plt.text(self.data[0,0,i].item()+xoff-0.01, self.data[0,1,i].item()+yoff-0.01, i)
            plt.plot([self.data[0,0,i].item()+xoff*0.2, self.data[0,0,i].item()+xoff*0.8], [self.data[0,1,i].item()+yoff*0.2, self.data[0,1,i].item()+yoff*0.8], c='black')


class GPEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def render(self, mode='human'):
        pass

    def __init__(self, num_points):
        self.kernel = RBF(0.1)
        self.gp = None #GaussianProcessRegressor(kernel=self.kernel,random_state=None, optimizer=None)  # random?
        self.data = np.empty((num_points, 2), dtype=np.float)
        #self.observation_space = Box(
        #    -np.inf, np.inf, shape=(2,)) #, dtype=np.float32
        #self.action_space = Box(0, 1, shape=(1,)) #, dtype=np.float32
        self.best = 0
        self.nstep = 0

    def reset(self):
        self.nstep = 0
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None)  # random?
        y = self.gp.sample_y([[0.5]], random_state=np.random.randint(100000))
        self.gp.fit([[0.5]], y)
        y = y[0, 0]
        self.best = y
        self.data[0] = [0.5, y]
        return torch.tensor([0.5, y])

    def step(self, action, shadow=False):
        # assert 0 <= action <= 1, action
        self.nstep = self.nstep + 1
        action = torch.clamp(action, 0, 1).view(1, -1)
        #print('Action: ', action)
        y = self.gp.sample_y(action, random_state=np.random.randint(100000))[0]
        #print('y: ', y)
        #self.gp.fit(action, y)
        y = y[0, 0]
        self.data[self.nstep] = [action[0,0], y]
        self.gp.fit(self.data[:self.nstep+1, :1], self.data[:self.nstep+1, 1:])
        reward = 0
        if y > self.best:
            reward = y - self.best
            if not shadow:
                self.best = y
        done = False #self.nstep >= 20
        return torch.tensor([action[0], y]), reward, done, {}  # never done
