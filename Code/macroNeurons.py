import numpy as np
import torch
import torch.nn as nn
import importlib
from .iff_macro import macros, preprocess, set_config, container

if container['first_load']:
    def preprocess(func):
        return func


class FlipFlopSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, last):
        gto = (input >= 1).float()
        decision = gto - (input <= -1).float()
        use_old = decision.abs() < 0.5
        ctx.save_for_backward(input, last, use_old)
        out = torch.where(use_old, last, gto)
        return out, decision

    @staticmethod
    def backward(ctx, grad_output, grad_dec):
        #print(grad_output[0,0].item())
        input, last, use_old = ctx.saved_tensors
        clamped = torch.clamp(grad_output, -1e-3, 1e-3)
        to_input = clamped * 0.9 * (1 / (2*torch.abs(input + 1) + 1.0) ** 2 + 1 / (2*torch.abs(input - 1) + 1.0) ** 2) / 3#10 #work out doubly spiked function
        to_last = torch.where(use_old, 0.9 * grad_output, torch.zeros([1]))
        return to_input, to_last



class NaNtoZero(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return torch.where(torch.isnan(grad_output), torch.zeros([1]), grad_output)


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

@preprocess
class CooldownNeuron(nn.Module):
    def __init__(self, params, size):
        super(CooldownNeuron, self).__init__()
        self.spike_fn = SuperSpike.apply
        self.beta = params['BETA']
        self.config = params
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.elu = torch.nn.ELU()
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.sgn = torch.ones([size], requires_grad=False)
        self.sgn[(size//2):] *= -1
        self.size = size


    def get_initial_spike(self, batch_size):
        return (self.sgn < 0).float().expand([batch_size, self.size]) #torch.zeros([batch_size, self.size])#

    #@printcode
    def forward(self, x, h):
        #x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x)} #, 'sgn': torch.ones([x.shape[1]])
            #h['sgn'][(x.shape[1]//2):] *= -1
            h['mem'] = self.initial_mem.expand(x.shape)

        new_h = {}
        #new_h['sgn'] = h['sgn']
        new_h['mem'] = self.beta * h['mem'] + self.elu(x-2) + 1#torch.sigmoid(x-2) #torch.tanh(x) torch.sigmoid(x-4)
        spikes = self.spike_fn(self.sgn * (new_h['mem'] - 1))

        return spikes, new_h

@preprocess
class FlipFlopNeuron(nn.Module):
    #@printcode
    def __init__(self, params):
        super(FlipFlopNeuron, self).__init__()
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        self.spike_fn = FlipFlopSpike.apply
        self.reset_zero = params['RESET_ZERO']
        self.config = params


    #@printcode
    def forward(self, x, h):
        #x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x), 'spikes': torch.zeros_like(x)}
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        if self.config['BETA'] < 1:
            mem = self.beta * h['mem']
        else:
            mem = h['mem']
        if self.config['ALPHA'] > 0:
            new_h['syn'] = self.alpha * h['syn'] + x
            mem = mem + new_h['syn']
        else:
            mem = mem + x
        spikes, decisions = self.spike_fn(mem, h['spikes']) #.detach() # / (threshold)
        new_h['spikes'] = spikes
        if self.config['RESET_ZERO']:
            new_h['mem'] = mem * (1.0 - decisions.abs())
        else:
            new_h['mem'] = mem - decisions

        return spikes, new_h


@preprocess
class MagicNeuron(nn.Module):
    #@printcode
    def __init__(self, params):
        super(MagicNeuron, self).__init__()
        self.config = params
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']
        self.thresh_add = params['THRESH_ADD']
        self.thresh_decay = params['THRESH_DECAY']


    #@printcode
    def forward(self, x, p, h):
        #x = NaNtoZero.apply(x)
        #p = NaNtoZero.apply(p)
        if not h:
            h = {'mem': torch.zeros_like(x), 'threshold': torch.zeros_like(x)}
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        if self.config['BETA'] < 1:
            mem = self.beta * h['mem']
        else:
            mem = h['mem']
        if self.config['ALPHA'] > 0:
            new_h['syn'] = self.alpha * h['syn'] + x
            mem = mem + new_h['syn']
        else:
            mem = mem + x
        if self.config['THRESH_DECAY'] < 1:
            new_h['threshold'] = self.thresh_decay * h['threshold'] + p
        else:
            new_h['threshold'] = h['threshold'] + p
        threshold = torch.where((new_h['threshold'] < 0), (1/(1 - new_h['threshold'])), (new_h['threshold'] + 1))
        spikes = self.spike_fn((mem - threshold) / threshold) #.detach() # / (threshold)
        if self.config['RESET_ZERO']:
            new_h['mem'] = mem * (1.0 - spikes.detach())
        else:
            new_h['mem'] = mem - (spikes * new_h['threshold']).detach()

        return spikes, new_h

@preprocess
class AdaptiveNeuron(nn.Module):
    def __init__(self, params):
        super(AdaptiveNeuron, self).__init__()
        self.config = params
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']
        self.thresh_add = params['THRESH_ADD']
        self.thresh_decay = params['THRESH_DECAY']


    #@printcode
    def forward(self, x, h):
        #x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x), 'threshold': torch.ones_like(x)}
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        if self.config['BETA'] < 1:
            mem = self.beta * h['mem']
        else:
            mem = h['mem']
        if self.config['ALPHA'] > 0:
            new_h['syn'] = self.alpha * h['syn'] + x
            mem = mem + new_h['syn']
        else:
            mem = mem + x
        spikes = self.spike_fn((mem - h['threshold']) / h['threshold']) #.detach()
        if self.config['RESET_ZERO']:
            new_h['mem'] = mem * (1.0 - spikes.detach())
        else:
            new_h['mem'] = mem - (spikes * h['threshold']).detach()
        if self.config['THRESH_DECAY'] < 1:
            new_h['threshold'] = 1 + self.thresh_decay * (h['threshold'] - 1) + self.thresh_add * spikes
        else:
            new_h['threshold'] = h['threshold'] + self.thresh_add * spikes

        return spikes, new_h

@preprocess
class LIFNeuron(nn.Module):
    def __init__(self, params, size):
        super(LIFNeuron, self).__init__()
        self.config = params
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)



    #@printcode
    def forward(self, x, h):
        #x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x)}
            h['mem'] = self.initial_mem.expand(x.shape)
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)

        new_h = {}
        # Order of operations unclear. Update Membrane before or after spike calculation? Synapse -> Membrane -> Spike apparently?
        if self.config['BETA'] < 1:
            mem = self.beta * h['mem']
        else:
            mem = h['mem']
        if self.config['ALPHA'] > 0:
            new_h['syn'] = self.alpha * h['syn'] + x
            mem = mem + new_h['syn']
        else:
            mem = mem + x
        spikes = self.spike_fn(mem - 1)
        if self.config['RESET_ZERO']:
            new_h['mem'] = mem * (1.0 - spikes.detach())
        else:
            new_h['mem'] = mem - spikes.detach()

        return spikes, new_h

@preprocess
class OutputNeuron(nn.Module):
    def __init__(self, params, size):
        super(OutputNeuron, self).__init__()
        self.config = params
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.reset_zero = params['RESET_ZERO']
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)



    #@printcode
    def forward(self, x, h):
        if not h:
            h = {'mem': torch.zeros_like(x)}
            h['mem'] = self.initial_mem.expand(x.shape)
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)
            if self.config['DECODING'] == "spike_cnt":
                h['spike_count'] = torch.zeros_like(x)

        new_h = {}

        if self.config['BETA'] == 1:
            new_h['mem'] = h['mem']
        else:
            new_h['mem'] = self.beta * h['mem']
        if self.config['ALPHA'] > 0:
            new_h['syn'] = self.alpha * h['syn'] + x
            new_h['mem'] = new_h['mem'] + new_h['syn']
        else:
            new_h['mem'] = new_h['mem'] + x
        if self.config['DECODING'] == "spike_cnt" or self.config['DECODING'] == "spikes":
            spikes = self.spike_fn(new_h['mem'] - 1)
            if self.config['RESET_ZERO']:
                new_h['mem'] = new_h['mem'] * (1.0 - spikes.detach())
            else:
                new_h['mem'] = new_h['mem'] - spikes.detach()
            if self.config['DECODING'] == "spike_cnt":
                new_h['spike_count'] = h['spike_count'] + spikes
                return new_h['spike_count'], new_h
            return spikes, new_h
        return new_h['mem'], new_h
