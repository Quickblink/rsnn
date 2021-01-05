import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def loop_dataloader(dataloader):
    while True:
        for x, y in dataloader:
            yield x, y

class SequentialMNIST:
    def __init__(self, iterations, batch_size, device, mnist_location, validate=False):
        self.batch_size = batch_size
        self.device = device
        self.max_iter = iterations
        mnist = MNIST(mnist_location, transform=transforms.ToTensor(), download=True, train=not validate)  # distortion_transform([0,15], 3)
        if validate:
            dl = DataLoader(mnist, batch_size=batch_size, drop_last=False, num_workers=0)
            self.loop_loader = loop_dataloader(dl)
            self.max_iter = len(dl)
        else:
            self.loop_loader = loop_dataloader(DataLoader(mnist, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True))
        self.trigger_signal = torch.ones([783+56, 1, 1], device=device)
        self.trigger_signal[:783] = 0

    def __iter__(self):
        self.cur_iter = 0
        return self

    def __next__(self):
        self.cur_iter += 1
        if self.cur_iter > self.max_iter:
            raise StopIteration
        return self.make_inputs()

    def loss_and_acc(self, model_output):
        #TODO: uses different model
        loss = nn.CrossEntropyLoss()(model_output, self.targets)
        acc = (torch.argmax(model_output, 1) == self.targets).float().mean().item()
        return loss, acc

    def get_infos(self):
        return 81, 10, 0.03  # Measured empirically

    def make_inputs(self):
        inp, self.targets = self.loop_loader.__next__()
        self.targets = self.targets.to(self.device)
        x = inp.view(inp.shape[0], 784, 1).transpose(0, 1).to(self.device)
        curr = x[1:]
        last = x[:-1]
        out = torch.zeros([783 + 56, curr.shape[1], 2, 40], device=curr.device)
        out[:783, :, 0, :] = ((torch.arange(40, device=curr.device) < 40 * last) & (
                    torch.arange(40, device=curr.device) > 40 * curr)).float()
        out[:783, :, 1, :] = ((torch.arange(40, device=curr.device) > 40 * last) & (
                    torch.arange(40, device=curr.device) < 40 * curr)).float()
        out = torch.cat((out.view([783 + 56, curr.shape[1], 80]), self.trigger_signal.expand([783 + 56, curr.shape[1], 1])),
                        dim=-1)
        return out