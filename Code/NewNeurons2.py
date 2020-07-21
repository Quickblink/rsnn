import torch
import torch.nn as nn


#devide by threshold, 1-beta factor

class BellecSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.max(torch.zeros([1], device=input.device), 1 - torch.abs(input)) * 0.3



class SuperSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 2.0#100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        #print(input[0,0].item())
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        #print(grad_output[0,0].item())
        input, = ctx.saved_tensors
        #clamped = torch.clamp(grad_output, -1e-3, 1e-3)
        out = grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2

        #out = torch.clamp((grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2), -1, 1)
        return out #torch.where((out == 0), torch.ones([1]) * 0.001, out)


class SeqOnlySpike(nn.Module):
    def __init__(self, size, params):
        super().__init__()
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return self.spike_fn(x), ()


class CooldownNeuron(nn.Module):
    def __init__(self, size, params):
        super(CooldownNeuron, self).__init__()
        self.beta = params['BETA']
        self.offset = params['OFFSET']
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.elu = torch.nn.ELU()
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.register_buffer('sgn', torch.ones([size], requires_grad=False))
        self.sgn[(size//2):] *= -1
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return self.initial_mem.expand([batch_size, self.in_size])

    def get_initial_output(self, batch_size):
        return (self.sgn < 0).float().expand([batch_size, self.in_size])

    def forward(self, x, h):
        h = self.beta * h + self.elu(x-self.offset) + 1
        spikes = self.spike_fn(self.sgn * (h - 1))

        return spikes, h


class NoResetNeuron(nn.Module):
    def __init__(self, size, params):
        super().__init__()
        self.beta = params['BETA']
        self.omb = params['1-beta']
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return self.initial_mem.expand([batch_size, self.in_size])


    def get_initial_output(self, batch_size):
        return self.spike_fn(self.initial_mem.expand([batch_size, self.in_size]) - 1)

    def forward(self, x, h):
        if self.omb:
            h = self.beta * h + (1-self.beta) * x
        else:
            h = self.beta * h + x
        spikes = self.spike_fn(h - 1)
        return spikes, h

class AdaptiveNeuron(nn.Module):
    def __init__(self, size, params):
        super().__init__()
        self.beta = params['BETA']
        self.omb = params['1-beta']
        self.decay = params['ADAPDECAY']
        self.scale = params['ADAPSCALE']
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        h = [self.initial_mem.expand([batch_size, self.in_size]), torch.zeros([batch_size, self.in_size], device=self.initial_mem.device), self.get_initial_output(batch_size)]
        return tuple(h)

    def get_initial_output(self, batch_size):
        return self.spike_fn(self.initial_mem.expand([batch_size, self.in_size]) - 1)

    #@printcode
    def forward(self, x, h):
        new_h = [None, None, None]
        old_mem = h[0]
        rel_thresh = h[1]
        old_spike = h[2]
        rel_thresh = self.decay * rel_thresh + (1-self.decay) * old_spike
        threshold = 1 + rel_thresh * self.scale
        if self.omb:
            mem = self.beta * old_mem + (1-self.beta) * x - old_spike * threshold
        else:
            mem = self.beta * old_mem + x - old_spike * threshold
        spikes = self.spike_fn((old_mem - threshold)/threshold)
        #spikes = torch.where(old_spike > 0, torch.zeros_like(spikes), spikes)
        #threshold = 1 + self.decay * (threshold - 1) + spikes
        new_h[0] = mem
        new_h[1] = rel_thresh #threshold
        new_h[2] = spikes
        return spikes, tuple(new_h)

class LIFNeuron(nn.Module):
    def __init__(self, size, params):
        super(LIFNeuron, self).__init__()
        self.beta = params['BETA']
        self.omb = params['1-beta']
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        h = [self.initial_mem.expand([batch_size, self.in_size]), self.get_initial_output(batch_size)]
        return tuple(h)

    def get_initial_output(self, batch_size):
        return self.spike_fn(self.initial_mem.expand([batch_size, self.in_size]) - 1)

    def forward(self, x, h):
        new_h = [None, None]
        old_mem = h[0]
        old_spike = h[1]
        if self.omb:
            mem = self.beta * old_mem + (1-self.beta) * x - old_spike
        else:
            mem = self.beta * old_mem + x - old_spike
        spikes = self.spike_fn(old_mem - 1)
        #spikes = torch.where(old_spike > 0, torch.zeros_like(spikes), spikes)
        #print(mem[0, 0])
        new_h[0] = mem
        new_h[1] = spikes

        return spikes, tuple(new_h)


class OutputNeuron(nn.Module):
    def __init__(self, size, params):
        super(OutputNeuron, self).__init__()
        self.beta = params['BETA']
        self.omb = params['1-beta']
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return self.initial_mem.expand([batch_size, self.in_size])

    def forward(self, x, h):
        if self.omb:
            h = self.beta * h + (1-self.beta) * x
        else:
            h = self.beta * h + x
        return h, h
