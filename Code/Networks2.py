import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


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
        torch.save(self.pretrace, addr)  # +'_og'


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


class LSTMWrapper(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = nn.LSTM(in_size, out_size)
        self.register_buffer('dummy', torch.zeros([1], requires_grad=False))

    def get_initial_state(self, batch_size):
        h = torch.zeros(self.lstm.get_expected_hidden_size(None, [batch_size]), device=self.dummy.device)
        return h, h.clone()

    def forward(self, x, h):
        return self.lstm(x, h)


class ReLuWrapper(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return F.relu(x), ()


class DummyNeuron(nn.Module):
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

class MeanModule(nn.Module):
    def __init__(self, size, last_index):
        super().__init__()
        self.in_size = size
        self.out_size = size
        self.last_index = last_index

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return x[self.last_index:].mean(dim=0), ()