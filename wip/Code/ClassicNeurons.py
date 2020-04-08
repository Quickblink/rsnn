import torch
import torch.nn as nn


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


class AdaptiveNonLeaky(nn.Module):

    def __init__(self, spike_fn, params):
        super(AdaptiveNonLeaky, self).__init__()
        self.spike_fn = spike_fn
        self.reset_zero = params['reset_zero']
        self.decay = params['decay']
        self.beta = params['beta']


    def forward(self, x, h):
        h = h or {'mem': torch.zeros_like(x), 'threshold': torch.ones_like(x)}
        new_h = {}

        spikes = self.spike_fn(h['mem'] - h['threshold'], h['threshold'].detach())
        new_h['mem'] = (h['mem'] + x) * (1.0 - spikes.detach()) if self.reset_zero else \
                        h['mem'] + x - (spikes * h['threshold']).detach()
        new_h['threshold'] = 1 + self.decay * (h['threshold'] - 1) + self.beta * spikes

        return spikes, new_h


class NonLeaky(nn.Module):

    def __init__(self, spike_fn, params):
        super(NonLeaky, self).__init__()
        self.spike_fn = spike_fn
        self.reset_zero = params['reset_zero']
        self.threshold = params['threshold']

    def forward(self, x, h):
        h = h or {'mem': torch.zeros_like(x)}
        new_h = {}

        spikes = self.spike_fn(h['mem'] - self.threshold)
        new_h['mem'] = (h['mem'] + x) * (1.0 - spikes.detach()) if self.reset_zero else \
                        h['mem'] + x - spikes.detach() * self.threshold

        return spikes, new_h

class NLPotential(nn.Module):

    def __init__(self, spike_fn, params):
        super(NLPotential, self).__init__()


    def forward(self, x, h):
        h = h or {'mem': torch.zeros_like(x)}
        new_h = {}

        new_h['mem'] = h['mem'] + x

        return new_h


class LIFNeuron(nn.Module):

    def __init__(self, spike_fn, params):
        super(LIFNeuron, self).__init__()
        self.beta = params['beta']
        self.alpha = params['alpha']
        self.spike_fn = spike_fn
        self.reset_zero = params['reset_zero']
        self.threshold = params['threshold']

    def forward(self, x, h):
        h = h or {'mem': torch.zeros_like(x), 'syn': torch.zeros_like(x)}
        new_h = {}


        # Order of operations unclear. Update Membrane before or after spike calculation? Synapse -> Membrane -> Spike apparently?
        spikes = self.spike_fn(h['mem'] - self.threshold)
        new_h['mem'] = (self.beta * h['mem'] + h['syn']) * (1.0 - spikes.detach()) if self.reset_zero else \
                   self.beta * h['mem'] + h['syn'] - spikes.detach() * self.threshold
        new_h['syn'] = self.alpha * h['syn'] + x

        return spikes, new_h

class PotentialNeuron(nn.Module):

    def __init__(self, spike_fn, params):
        super(PotentialNeuron, self).__init__()
        self.beta = params['beta']
        self.alpha = params['alpha']


    def forward(self, x, h):
        h = h or {'mem': torch.zeros_like(x), 'syn': torch.zeros_like(x)}
        new_h = {}

        # Order of operations unclear. Update Membrane before or after spike calculation? Synapse -> Membrane -> Spike apparently?
        new_h['mem'] = self.beta * h['mem'] + h['syn']
        new_h['syn'] = self.alpha * h['syn'] + x

        return new_h


class RSNN(nn.Module):

    def __init__(self, architecture, neuron, neuron_params, spike_fn, output_neuron):
        super(RSNN, self).__init__()

        self.main_linear = nn.Linear(architecture[0]+architecture[1], architecture[1], bias=True)
        self.lif_layer = neuron(spike_fn, neuron_params)
        self.output_linear = nn.Linear(architecture[1], architecture[2], bias=True)
        self.output_layer = output_neuron(spike_fn, neuron_params)
        self.hidden_size = architecture[1]

    def forward(self, inp, h=None):
        inp = inp.view(-1, 4)
        inp = inp.expand((20, *inp.shape)).detach()
        T = inp.shape[0]
        bsz = inp.shape[1]
        h = h or {}
        new_h = {}
        x = torch.cat((inp[0], torch.zeros((bsz, self.hidden_size))), dim=1)
        x = self.main_linear(x)
        x, new_h[self.lif_layer] = self.lif_layer(x, h.get(self.lif_layer))
        o = self.output_linear(x)
        new_h[self.output_layer] = self.output_layer(o, h.get(self.output_layer))
        for t in range(1, T):
            x = torch.cat((inp[0], x), dim=1)
            x = self.main_linear(x)
            x, new_h[self.lif_layer] = self.lif_layer(x, h.get(self.lif_layer))
            o = self.output_linear(x)
            new_h[self.output_layer] = self.output_layer(o, h.get(self.output_layer))
            h = new_h

        return new_h[self.output_layer]['mem']#, new_h

class FeedForwardSNN(nn.Module):

    def __init__(self, architecture, neuron, neuron_params, spike_fn, output_neuron):
        super(FeedForwardSNN, self).__init__()

        #self.input_layer = neuron(spike_fn, neuron_params)
        self.input_linear = nn.Linear(architecture[0], architecture[1], bias=True)
        self.linear_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        for i in range(1, len(architecture)-1):
            self.lif_layers.append(neuron(spike_fn, neuron_params))
            self.linear_layers.append(nn.Linear(architecture[i], architecture[i+1], bias=True))
        self.output_layer = output_neuron(spike_fn, neuron_params)

    def forward(self, inp, h=None):
        #inp = inp.view(-1, 4)
        #inp = torch.cat((inp.detach().clone(), torch.ones((inp.shape[0], 1), device=inp.device)), dim=1)
        inp = inp.expand((20, *inp.shape)).detach()
        T = inp.shape[0]
        h = h or {}
        new_h = {}
        for t in range(T):
            x = inp[t]
            #x, new_h[self.input_layer] = self.input_layer(x, h.get(self.input_layer))
            x = self.input_linear(x)
            #x = torch.einsum("ab,bc->ac", [x, self.input_linear.weight])
            for i in range(len(self.linear_layers)):
                x, new_h[self.lif_layers[i]] = self.lif_layers[i](x, h.get(self.lif_layers[i]))
                x = self.linear_layers[i](x)
                #x = torch.einsum("ab,bc->ac", [x, self.linear_layers[i].weight])
                #if i == 0:
                    #print(x[0,0].item())

            new_h[self.output_layer] = self.output_layer(x, h.get(self.output_layer))
            h = new_h

        return new_h[self.output_layer]['mem']#, new_h
