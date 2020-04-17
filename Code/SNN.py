import torch
import torch.nn as nn


def getInputs(dict, layer):
    n = 0
    for ilay in dict[layer][1]:
        n += dict[ilay][0]
    return n

class DynNetwork(nn.Module):

    def __init__(self, architecture, sim_time):
        super(DynNetwork, self).__init__()
        self.layers = nn.ModuleDict()
        processed = []
        self.architecture = architecture
        self.recurrent_layers = []
        for layer, params in self.architecture.items():
            if layer == 'input':
                processed.append(layer)
                continue
            if params[4]:
                self.layers[layer+'_linear'] = params[4](getInputs(self.architecture, layer), params[0])
            self.layers[layer] = params[2](params[3], params[0])
            for p in params[1]:
                if p not in processed:
                    self.recurrent_layers.append(p)
            processed.append(layer)
        self.sim_time = sim_time



    def forward(self, inp, h):
        state, spikes = h
        for i in range(self.sim_time):
            new_state = []
            new_spikes = []
            idxState = 0
            results = {'input': inp}
            for layer, params in self.architecture.items():
                if layer == 'input':
                    continue
                if len(params[1]) > 1:
                    inputs = []
                    for p in params[1]:
                        inputs.append(results[p] if p in results else spikes[self.recurrent_layers.index(p)])
                    x = torch.cat(inputs, dim=1)
                else:
                    x = results[params[1][0]]
                if params[4]:
                    x = self.layers[layer+'_linear'](x)
                results[layer], hi = self.layers[layer](x, state[idxState])
                new_state.append(hi)
                idxState += 1
            for layer in self.recurrent_layers:
                new_spikes.append(results[layer])
            state = new_state
            spikes = new_spikes
        return results['output'], (tuple(state), tuple(spikes))

    def get_initial_state(self, batch_size):
        state = []
        spikes = []
        for layer in self.architecture:
            if layer == 'input':
                continue
            state.append(self.layers[layer].get_initial_state(batch_size))
        for layer in self.recurrent_layers:
            spikes.append(self.layers[layer].get_initial_output(batch_size))
        return tuple(state), tuple(spikes)


    # handle hidden state abstraction and time dimension
class SequenceWrapper(nn.Module):
    def __init__(self, model, batch_size, device, trace):
        super(SequenceWrapper, self).__init__()
        self.pretrace = model.to(device)
        self.model = torch.jit.trace(model, (torch.zeros([batch_size, self.pretrace.architecture['input'][0]], device=device), self.pretrace.get_initial_state(batch_size)), optimize=True) if trace else self.pretrace
        #self.model = self.pretrace

    def forward(self, inp, h=None):
        if not h:
            h = self.pretrace.get_initial_state(inp.shape[1])
        out1, h = self.model(inp[0], h)
        if inp.shape[0] == 1:
            return out1.unsqueeze(0), h
        output = torch.empty(inp.shape[:1]+out1.shape, device=inp.device)
        output[0] = out1
        for t in range(1, inp.shape[0]):
            output[t], h = self.model(inp[t], h)
        return output, h


class DynNetworkold(nn.Module):

    def __init__(self, architecture, sim_time):
        super(DynNetworkold, self).__init__()
        self.layers = nn.ModuleDict()
        processed = []
        self.architecture = architecture
        self.recurrent_layers = []
        for layer, params in self.architecture.items():
            if layer == 'input':
                processed.append(layer)
                continue
            self.layers[layer+'_linear'] = nn.Linear(getInputs(self.architecture, layer), params[0])
            self.layers[layer] = params[2](params[3], params[0])
            for p in params[1]:
                if p not in processed:
                    self.recurrent_layers.append(p)
            processed.append(layer)
        self.sim_time = sim_time



    def forward(self, inp, h):
        new_h = {'spikes': {}}
        for i in range(self.sim_time):
            results = {'input': inp}
            for layer, params in self.architecture.items():
                if layer == 'input':
                    continue
                if len(params[1]) > 1:
                    inputs = []
                    for p in params[1]:
                        inputs.append(results[p] if p in results else h['spikes'][p])
                    x = torch.cat(inputs, dim=1)
                else:
                    x = results[params[1][0]]
                x = self.layers[layer+'_linear'](x)
                results[layer], new_h[layer] = self.layers[layer](x, h[layer])
                if layer in self.recurrent_layers:
                    new_h['spikes'][layer] = results[layer]
            h = new_h
        return results['output'], (tuple([results['mem']]),)#new_h

    def get_initial_state(self, batch_size):
        state = {'spikes': {}}
        for layer in self.architecture:
            if layer == 'input':
                continue
            state[layer] = self.layers[layer].get_initial_state(batch_size)
        for layer in self.recurrent_layers:
            state['spikes'][layer] = self.layers[layer].get_initial_output(batch_size)
        return state