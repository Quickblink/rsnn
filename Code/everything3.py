import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

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

    scale = 2.0#2.0#100.0  # controls steepness of surrogate gradient #TODO: make this a config parameter

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


class FlipFlopSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, last):
        ctx.save_for_backward(input)
        return ((input > 1) | ((input > -1) & (last > 0))).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        to_input = grad_output * 0.9 * (1 / (2*torch.abs(input + 1) + 1.0) ** 2 + 1 / (2*torch.abs(input - 1) + 1.0) ** 2) #10 #work out doubly spiked function
        return to_input, torch.zeros_like(to_input)




# inputs, neuron, synapse
class ParallelNetwork2(nn.Module):
    def __init__(self, architecture, bias=True):
        super().__init__()
        self.architecture = copy.deepcopy(architecture)
        self.in_size, front_input_rate = self.architecture['input']
        del self.architecture['input']
        self.layers = nn.ModuleDict()
        for layer, params in self.architecture.items():
            self.layers[layer] = params[1]
            for i in range(len(params[0])):
                params[0][i] = params[0][i] if type(params[0][i]) is tuple else (params[0][i], 1)
            if params[2]:
                n = 0
                total_contribution = 0
                for ilay, contrib in params[0]:
                    # = klay if type(klay) is tuple else (klay, 1)
                    n += self.in_size if ilay == 'input' else self.architecture[ilay][1].out_size
                    total_contribution += contrib
                self.layers[layer+'_synapse'] = params[2](n, params[1].in_size, bias=True)
                with torch.no_grad():
                    self.layers[layer+'_synapse'].bias.data = Uniform(0, (1 - params[1].beta + 0.03)/params[1].factor).sample(self.layers[layer+'_synapse'].bias.shape)
                k0 = 0
                k1 = 0
                target_var = self.layers[layer].target_var
                for ilay, contrib in params[0]:
                    # = klay if type(klay) is tuple else (klay, 1)
                    k1 += self.in_size if ilay == 'input' else self.architecture[ilay][1].out_size
                    with torch.no_grad():
                        input_rate = front_input_rate if ilay == 'input' else self.architecture[ilay][1].est_rate
                        var = target_var * contrib / total_contribution / (k1-k0) / input_rate
                        self.layers[layer+'_synapse'].weight[:, k0:k1] *= (3*var)**(0.5)
                    k0 = k1
        self.out_size = self.layers['output'].out_size
        self.is_logging = False
        print(self.architecture)


    def forward(self, inp, h):
        state, spikes = h
        new_state = {}
        new_spikes = {}
        log = {}
        l_inputs = {**spikes, 'input': inp}
        for layer, params in self.architecture.items():
            if len(params[0]) > 1:
                inputs = []
                for p in params[0]:
                    inputs.append(l_inputs[p[0]])
                x = torch.cat(inputs, dim=-1)
            else:
                x = l_inputs[params[0][0][0]]
            if params[2]:
                x = self.layers[layer+'_synapse'](x)
            self.layers[layer].is_logging = self.is_logging
            tmp = self.layers[layer](x, state[layer])
            new_spikes[layer], new_state[layer] = tmp[:2]
            if self.is_logging:
                log[layer] = tmp[2] if len(tmp) > 2 else tmp[0]
        if self.is_logging:
            return new_spikes['output'], (new_state, new_spikes), log
        else:
            return new_spikes['output'], (new_state, new_spikes)

    def get_initial_state(self, batch_size):
        state = {}
        spikes = {}
        for layer in self.architecture:
            state[layer] = self.layers[layer].get_initial_state(batch_size)
            spikes[layer] = self.layers[layer].get_initial_output(batch_size)
        return state, spikes



# inputs, neuron, synapse
class ParallelNetwork(nn.Module):
    def __init__(self, architecture, bias=True):
        super().__init__()
        self.architecture = copy.deepcopy(architecture)
        self.in_size = self.architecture['input']
        del self.architecture['input']
        self.layers = nn.ModuleDict()
        for layer, params in self.architecture.items():
            if params[2]:
                n = 0
                for ilay in params[0]:
                    n += self.in_size if ilay == 'input' else self.architecture[ilay][1].out_size
                self.layers[layer+'_synapse'] = params[2](n, params[1].in_size, bias=bias)
            self.layers[layer] = params[1]
        self.out_size = self.layers['output'].out_size
        self.is_logging = False


    def forward(self, inp, h):
        state, spikes = h
        new_state = {}
        new_spikes = {}
        log = {}
        l_inputs = {**spikes, 'input': inp}
        for layer, params in self.architecture.items():
            if len(params[0]) > 1:
                inputs = []
                for p in params[0]:
                    inputs.append(l_inputs[p])
                x = torch.cat(inputs, dim=-1)
            else:
                x = l_inputs[params[0][0]]
            if params[2]:
                x = self.layers[layer+'_synapse'](x)
            self.layers[layer].is_logging = self.is_logging
            tmp = self.layers[layer](x, state[layer])
            new_spikes[layer], new_state[layer] = tmp[:2]
            if self.is_logging:
                log[layer] = tmp[2] if len(tmp) > 2 else tmp[0]
        if self.is_logging:
            return new_spikes['output'], (new_state, new_spikes), log
        else:
            return new_spikes['output'], (new_state, new_spikes)

    def get_initial_state(self, batch_size):
        state = {}
        spikes = {}
        for layer in self.architecture:
            state[layer] = self.layers[layer].get_initial_state(batch_size)
            spikes[layer] = self.layers[layer].get_initial_output(batch_size)
        return state, spikes


class DynNetwork(nn.Module):
    def __init__(self, architecture):
        super(DynNetwork, self).__init__()
        self.architecture = copy.deepcopy(architecture)
        self.in_size = self.architecture['input']
        del self.architecture['input']
        self.layers = nn.ModuleDict()
        processed = ['input']
        self.recurrent_layers = []
        for layer, params in self.architecture.items():
            if params[2]:
                n = 0
                for ilay in params[0]:
                    n += self.in_size if ilay == 'input' else self.architecture[ilay][1].out_size
                self.layers[layer+'_synapse'] = params[2](n, params[1].in_size)
            self.layers[layer] = params[1]
            for p in params[0]:
                if p not in processed and p not in self.recurrent_layers:
                    self.recurrent_layers.append(p)
            processed.append(layer)
        self.out_size = self.layers['output'].out_size
        self.is_logging = False

    def forward(self, inp, h):
        state, spikes = h
        log = {}
        new_state = []
        new_spikes = []
        idxState = 0
        results = {'input': inp}
        for layer, params in self.architecture.items():
            if len(params[0]) > 1:
                inputs = []
                for p in params[0]:
                    inputs.append(results[p] if p in results else spikes[self.recurrent_layers.index(p)])
                x = torch.cat(inputs, dim=-1)
            else:
                x = results[params[0][0]]
            if params[2]:
                x = self.layers[layer+'_synapse'](x)
            self.layers[layer].is_logging = self.is_logging
            tmp = self.layers[layer](x, state[idxState])
            results[layer], hi = tmp[:2]
            if self.is_logging:
                log[layer] = tmp[2] if len(tmp) > 2 else tmp[0]
            new_state.append(hi)
            idxState += 1
        for layer in self.recurrent_layers:
            new_spikes.append(results[layer])
        if self.is_logging:
            return results['output'], (tuple(new_state), tuple(new_spikes)), log
        else:
            return results['output'], (tuple(new_state), tuple(new_spikes))

    def get_initial_state(self, batch_size):
        state = []
        spikes = []
        for layer in self.architecture:
            state.append(self.layers[layer].get_initial_state(batch_size))
        for layer in self.recurrent_layers:
            spikes.append(self.layers[layer].get_initial_output(batch_size))
        return tuple(state), tuple(spikes)





class SequenceWrapper(nn.Module):
    def __init__(self, model):
        super(SequenceWrapper, self).__init__()
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.is_logging = False

    def make_log(self, length, first_entry):
        if type(first_entry) is dict:
            new_log = {}
            for k, v in first_entry.items():
                new_log[k] = self.make_log(length, v)
        elif type(first_entry) is torch.Tensor:
            new_log = torch.empty(length + first_entry.shape, device=first_entry.device)
        else:
            raise Exception('Unknown type in logging!')
        return new_log

    def enter_log(self, log, entry, t):
        if type(log) is dict:
            for k in log:
                self.enter_log(log[k], entry[k], t)
        elif type(log) is torch.Tensor:
            log[t] = entry
        else:
            raise Exception('Unknown type in logging!')

    def forward(self, inp, h):
        self.model.is_logging = self.is_logging
        if self.is_logging:
            out1, h, first_entry = self.model(inp[0], h)
            log = self.make_log(inp.shape[:1], first_entry)
            self.enter_log(log, first_entry, 0)
        else:
            out1, h = self.model(inp[0], h)
        output = torch.empty(inp.shape[:1]+out1.shape, device=inp.device)
        output[0] = out1
        for t in range(1, inp.shape[0]):
            if self.is_logging:
                output[t], h, entry = self.model(inp[t], h)
                self.enter_log(log, entry, t)
            else:
                output[t], h = self.model(inp[t], h)
        if self.is_logging:
            return output, h, log
        else:
            return output, h

    def get_initial_state(self, batch_size):
        return self.model.get_initial_state(batch_size)

class OuterWrapper(nn.Module):
    def __init__(self, model, device, two_dim=False):
        super().__init__()
        self.model = model.to(device)
        self.two_dim = two_dim


    def forward(self, inp, h=None, logging=False):
        if not h:
            h = self.model.get_initial_state(inp.shape[0] if self.two_dim else inp.shape[1])
        self.model.is_logging = logging
        return self.model(inp, h)

    def save(self, addr):
        # self.model.save(addr+'_traced')
        torch.save(self.model, addr)  # +'_og'


class Selector(nn.Module):
    def __init__(self, start, size):
        super().__init__()
        self.start = start
        self.end = start + size
        self.out_size = size

    def forward(self, x, h):
        return x[..., self.start:self.end], ()

    def get_initial_state(self, batch_size):
        return ()

class BaseNeuron(nn.Module):
    def __init__(self, size, _):
        super().__init__()
        self.in_size = size
        self.out_size = size
        self.register_buffer('device_zero', torch.zeros(1, requires_grad=False))

    def get_initial_state(self, batch_size):
        return ()

    def get_initial_output(self, batch_size):
        return self.device_zero.expand([batch_size, self.in_size])

    def forward(self, x, h):
        return x, ()



class LSTMWrapper(BaseNeuron):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, None)
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size)

    def get_initial_state(self, batch_size):
        h = torch.zeros(self.lstm.get_expected_hidden_size(None, [batch_size]), device=self.device_zero.device)
        return h, h.clone()

    def forward(self, x, h):
        return self.lstm(x, h)


class ReLuWrapper(BaseNeuron):
    def __init__(self, size):
        super().__init__(size, None)

    def forward(self, x, h):
        return F.relu(x), ()



class MeanModule(BaseNeuron):
    def __init__(self, size, last_index):
        super().__init__(size, None)
        self.last_index = last_index

    def forward(self, x, h):
        return x[self.last_index:].mean(dim=0), ()



class NoResetNeuron(BaseNeuron):
    def __init__(self, size, params):
        super().__init__(size, None)
        self.target_rate = params['target_rate']
        self.beta = params['BETA']
        if params['1-beta'] == 'improved':
            self.factor = (1 - self.beta ** 2) ** (0.5)
        elif params['1-beta']:
            self.factor = (1-self.beta)
        else:
            self.factor = 1
        if params['SPIKE_FN'] == 'bellec':
            self.spike_fn = BellecSpike.apply
        else:
            self.spike_fn = SuperSpike.apply
        self.initial_mem = nn.Parameter(torch.zeros([size]), requires_grad=True)
        self.target_var = 1
        self.est_rate = 0.06


    def get_initial_state(self, batch_size):
        return {
            'mem': self.initial_mem.expand([batch_size, self.in_size]),
            'spikes': self.get_initial_output(batch_size)
        }

    def get_initial_output(self, batch_size):
        return self.spike_fn(self.initial_mem.expand([batch_size, self.in_size]) - 1)

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + self.factor * x
        new_h['spikes'] = self.spike_fn(h['mem'] - 1)
        return new_h['spikes'], new_h


class FlipFlopNeuron(BaseNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.spike_fn = FlipFlopSpike.apply
        self.register_buffer('initial_out', torch.ones([size], requires_grad=False))
        self.initial_out[(size//2):] *= 0
        #TODO: do mirrored outputs?

    def get_initial_state(self, batch_size):
        state = {
            'mem': self.initial_mem.expand([batch_size, self.in_size]),
            'spikes': self.initial_out.expand([batch_size, self.in_size])
        }
        return state

    def get_initial_output(self, batch_size):
        return self.initial_out.expand([batch_size, self.in_size])

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + self.factor * x
        new_h['spikes'] = self.spike_fn(n_state['mem'], old_spikes.detach())
        return new_h['spikes'], new_h


class CooldownNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.offset = params['OFFSET']
        self.elu = torch.nn.ELU()
        self.register_buffer('sgn', torch.ones([size], requires_grad=False))
        self.sgn[(size//2):] *= -1

    def get_initial_output(self, batch_size):
        return (self.sgn < 0).float().expand([batch_size, self.in_size])

    def forward(self, x, h):
        new_h = {}
        new_h['mem'] = self.beta * h['mem'] + self.elu(x-self.offset) + 1
        new_h['spikes'] = self.spike_fn(self.sgn * (h['mem'] - 1))
        return new_h['spikes'], new_h



class LIFNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)

    def forward(self, x, h):
        out, new_h = super().forward(x, h)
        new_h['mem'] = new_h['mem'] - h['spikes']
        return out, new_h


class AdaptiveNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.decay = params['ADAPDECAY']
        self.scale = params['ADAPSCALE']

    def get_initial_state(self, batch_size):
        h = super().get_initial_state(batch_size)
        h['rel_thresh'] = torch.zeros([batch_size, self.in_size], device=self.initial_mem.device)
        return h

    def forward(self, x, h):
        new_h = {}
        new_h['rel_thresh'] = self.decay * h['rel_thresh'] + (1-self.decay) * h['spikes']
        threshold = 1 + new_h['rel_thresh'] * self.scale
        new_h['mem'] = self.beta * h['mem'] + self.factor * x - h['spikes'] * threshold
        new_h['spikes'] = self.spike_fn((h['mem'] - threshold)/threshold)
        return new_h['spikes'], new_h



class SeqOnlySpike(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)


    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return self.spike_fn(x-1), () #had no -1 before




class OutputNeuron(NoResetNeuron):
    def __init__(self, size, params):
        super().__init__(size, params)

    def forward(self, x, h):
        _, new_h = super().forward(x, h)
        return new_h['mem'], new_h