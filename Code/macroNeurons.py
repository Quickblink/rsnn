import numpy as np
import torch
import torch.nn as nn
from .iff_macro import macros, mif, melse, printcode

#TODO: make neuron models more modular

class AdaptiveBellec(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.max(torch.zeros([1], device=input.device), 1 - torch.abs(input/ctx.threshold)) * 0.3, torch.zeros_like(ctx.threshold)



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

    scale = 100.0  # controls steepness of surrogate gradient

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
        input, = ctx.saved_tensors
        return grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2


class AdaptiveNeuron(nn.Module):
    def __init__(self, params):
        super(AdaptiveNeuron, self).__init__()
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        self.spike_fn = AdaptiveBellec.apply
        self.reset_zero = params['RESET_ZERO']
        self.thresh_add = params['THRESH_ADD']
        self.thresh_decay = params['THRESH_DECAY']

    #@printcode
    def forward(self, x, h):
        if not h:
            h = {'mem': torch.zeros_like(x), 'threshold': torch.ones_like(x)}
            with mif('ALPHA > 0'):
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        with mif('BETA < 1'):
            mem = self.beta * h['mem']
        with melse:
            mem = h['mem']
        with mif('ALPHA > 0'):
            new_h['syn'] = self.alpha * h['syn'] + x
            mem += new_h['syn']
        with melse:
            mem += x
        spikes = self.spike_fn(mem - h['threshold'], h['threshold'].detach())
        with mif('RESET_ZERO'):
            new_h['mem'] = mem * (1.0 - spikes.detach())
        with melse:
            new_h['mem'] = mem - (spikes * h['threshold']).detach()
        with mif('THRESH_DECAY < 1'):
            new_h['threshold'] = 1 + self.thresh_decay * (h['threshold'] - 1) + self.thresh_add * spikes
        with melse:
            new_h['threshold'] = h['threshold'] + self.thresh_add * spikes

        return spikes, new_h


class LIFNeuron(nn.Module):
    def __init__(self, params):
        super(LIFNeuron, self).__init__()
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        with mif('SPIKE_FN == "bellec"'):
            self.spike_fn = BellecSpike.apply
        with melse:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']

    #@printcode
    def forward(self, x, h):
        if not h:
            h = {'mem': torch.zeros_like(x)}
            with mif('ALPHA > 0'):
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        # Order of operations unclear. Update Membrane before or after spike calculation? Synapse -> Membrane -> Spike apparently?
        with mif('BETA == 1'):
            mem = h['mem']
        with melse:
            mem = self.beta * h['mem']
        with mif('ALPHA > 0'):
            new_h['syn'] = self.alpha * h['syn'] + x
            mem += new_h['syn']
        with melse:
            mem += x
        spikes = self.spike_fn(mem - 1)
        with mif('RESET_ZERO'):
            new_h['mem'] = mem * (1.0 - spikes.detach())
        with melse:
            new_h['mem'] = mem - spikes.detach()

        return spikes, new_h


class OutputNeuron(nn.Module):
    def __init__(self, params):
        super(OutputNeuron, self).__init__()
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        with mif('SPIKE_FN == "bellec"'):
            self.spike_fn = BellecSpike.apply
        with melse:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']

    #@printcode
    def forward(self, x, h):
        if not h:
            h = {'mem': torch.zeros_like(x)}
            with mif('ALPHA > 0'):
                h['syn'] = torch.zeros_like(x)
            with mif('COUNT_SPIKES'):
                h['spike_count'] = torch.zeros_like(x)

        new_h = {}

        with mif('BETA == 1'):
            new_h['mem'] = h['mem']
        with melse:
            new_h['mem'] = self.beta * h['mem']
        with mif('ALPHA > 0'):
            new_h['syn'] = self.alpha * h['syn'] + x
            new_h['mem'] += new_h['syn']
        with melse:
            new_h['mem'] += x
        with mif('COUNT_SPIKES'):
            spikes = self.spike_fn(new_h['mem'] - 1)
            new_h['spike_count'] = h['spike_count'] + spikes
            with mif('RESET_ZERO'):
                new_h['mem'] = new_h['mem'] * (1.0 - spikes.detach())
            with melse:
                new_h['mem'] = new_h['mem'] - spikes.detach()
            return new_h['spike_count'], new_h
        return new_h['mem'], new_h
