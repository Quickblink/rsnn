from .macroNeurons import CooldownNeuron, LIFNeuron
import torch
import torch.nn as nn

lif_params = {}

sampleNetwork = [
    ('input', [1]),
    ('pre_mem', [32, ['input', 'mem'], LIFNeuron, lif_params]),
    ('mem', [16, ['pre_mem'], LIFNeuron, lif_params]),
    ('output', [1, ['mem'], LIFNeuron, lif_params]),
]

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
            self.layers[layer+'_linear'] = nn.Linear(getInputs(self.architecture, layer), params[0])
            self.layers[layer] = params[2](params[3], params[0])
            for p in params[1]:
                if p not in processed:
                    self.recurrent_layers.append(p)
            processed.append(layer)
        self.sim_time = sim_time



    def forward(self, inp, h):
        new_h = {}
        for i in range(self.sim_time):
            results = {}
            for layer, params in self.architecture.items():
                if layer == 'input':
                    continue
                if len(params[1]) > 1:
                    inputs = []
                    for p in params[1]:
                        inputs.append(results[p] if p in results else h[p+'_out'])
                    x = torch.cat(inputs, dim=1)
                else:
                    x = results[params[1][0]]
                x = self.layers[layer+'_linear'](x)
                results[layer], new_h[layer] = self.layers[layer](x, h[layer])
                if layer in self.recurrent_layers:
                    new_h[layer + '_out'] = results[layer]
            h = new_h
        return results['output'], new_h

    def get_initial_state(self):
        state = {}
        for layer in self.architecture:
            state[layer] = self.layers[layer].get_initial_state()
        for layer in self.recurrent_layers:
            state[layer+'_out'] = self.layers[layer].get_initial_output()
        return state


    # handle hidden state abstraction and time dimension
class SequenceWrapper(nn.Module):
    def __init__(self, model):
        super(SequenceWrapper, self).__init__()
        self.model = model

    def forward(self, inp, h=None):
        if not h:
            h = self.model.get_intial_state()
        out1, h = self.model(inp[0], h)
        if inp.shape[0] == 1:
            return out1.unsqueeze(0), h
        output = torch.empty(inp.shape[:1]+out1.shape)
        output[0] = out1
        for t in range(1, inp.shape[0]):
            output[t], h = self.model(inp[t], h)
        return output, h
