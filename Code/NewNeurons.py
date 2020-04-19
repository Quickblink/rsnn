import torch
import torch.nn as nn


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
        if params == 'bellec':
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
        #self.alpha = params['ALPHA']
        self.beta = params['BETA']
        self.offset = params['OFFSET']
        self.config = params
        if self.config['SPIKE_FN'] == 'bellec':
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
        self.config = params
        self.alpha = params['ALPHA']
        self.beta = params['BETA']
        if self.config['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        if self.config['ALPHA'] > 0:
            self.initial_syn = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        h = [self.initial_mem.expand([batch_size, self.in_size])]
        if self.config['ALPHA'] > 0:
            h.append(self.initial_syn.expand([batch_size, self.in_size]))
        return tuple(h)

    def get_initial_output(self, batch_size):
        return torch.zeros([batch_size, self.in_size])

    #@printcode
    def forward(self, x, h):

        new_h = [None]
        mem = h[0]
        if self.config['BETA'] < 1:
            mem = self.beta * mem
        if self.config['ALPHA'] > 0:
            syn = self.alpha * h[1] + x
            mem = mem + syn
            new_h.append(syn)
        else:
            mem = mem + x
        spikes = self.spike_fn(mem - 1)
        new_h[0] = mem

        return spikes, tuple(new_h)



class LIFNeuron(nn.Module):
    def __init__(self, size, params):
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
        if self.config['ALPHA'] > 0:
            self.initial_syn = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        h = [self.initial_mem.expand([batch_size, self.in_size])]
        if self.config['ALPHA'] > 0:
            h.append(self.initial_syn.expand([batch_size, self.in_size]))
        return tuple(h)

    def get_initial_output(self, batch_size):
        return torch.zeros([batch_size, self.in_size])

    #@printcode
    def forward(self, x, h):

        new_h = [None]
        mem = h[0]
        if self.config['BETA'] < 1:
            mem = self.beta * mem
        if self.config['ALPHA'] > 0:
            syn = self.alpha * h[1] + x
            mem = mem + syn
            new_h.append(syn)
        else:
            mem = mem + x
        spikes = self.spike_fn(mem - 1)
        if self.config['RESET_ZERO']:
            mem = mem * (1.0 - spikes.detach())
        else:
            mem = mem - spikes.detach()
        new_h[0] = mem

        return spikes, tuple(new_h)

class OutputNeuron(nn.Module):
    def __init__(self, size, params):
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
        if self.config['ALPHA'] > 0:
            self.initial_syn = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        h = [self.initial_mem.expand([batch_size, self.in_size])]
        if self.config['ALPHA'] > 0:
            h.append(self.initial_syn.expand([batch_size, self.in_size]))
        return tuple(h)

    # @printcode
    def forward(self, x, h):
        if not h:
            h = {'mem': torch.zeros_like(x)}
            h['mem'] = self.initial_mem.expand(x.shape)
            if self.config['ALPHA'] > 0:
                h['syn'] = torch.zeros_like(x)
            if self.config['DECODING'] == "spike_cnt":
                h['spike_count'] = torch.zeros_like(x)

        new_h = [None]
        mem = h[0]
        if self.config['BETA'] < 1:
            mem = self.beta * mem
        if self.config['ALPHA'] > 0:
            syn = self.alpha * h[1] + x
            mem = mem + syn
            new_h.append(syn)
        else:
            mem = mem + x
        new_h[0] = mem
        '''
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
        '''
        return mem, tuple(new_h)