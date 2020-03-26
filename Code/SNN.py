import numpy as np
import torch
import torch.nn as nn
from Code.utils import LoggerFn


class SNNWrapper(nn.Module):
    def __init__(self, model, config):
        super(SNNWrapper, self).__init__()
        self.model = model
        self.sim_time = config['SIM_TIME']

    def forward(self, inp):
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
        inp = inp.expand((self.sim_time, *inp.shape)).detach()
        out, _ = self.model(inp)
        return out



class RSNN(nn.Module):

    def __init__(self, neuron_params, num_inputs, num_static, num_adaptive, num_outputs, static_neuron, adaptive_neuron, output_neuron):
        super(RSNN, self).__init__()

        #self.main_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_static + num_adaptive, bias=True)
        self.static_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_static, bias=True)
        self.adaptive_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_adaptive, bias=True)
        self.static_layer = static_neuron(neuron_params)
        self.adaptive_layer = adaptive_neuron(neuron_params)
        self.output_linear = nn.Linear(num_static + num_adaptive, num_outputs, bias=True)
        self.output_layer = output_neuron(neuron_params)
        self.hidden_size = num_static + num_adaptive
        self.num_outputs = num_outputs
        self.sim_time = neuron_params['SIM_TIME']

    def forward(self, inp, h=None):
        T = inp.shape[0]
        bsz = inp.shape[1]
        h = h or {'spikes': torch.zeros((bsz, self.hidden_size), device=inp.device)}
        new_h = {}
        output = torch.empty((T, bsz, self.num_outputs))
        for t in range(T):
            h_out = None
            for i in range(self.sim_time):
                x = torch.cat((inp[t], h['spikes']), dim=1)
                #x = self.main_linear(x)
                sx = self.static_linear(x)
                sx, new_h[self.static_layer] = self.static_layer(sx, h.get(self.static_layer))
                ax = self.adaptive_linear(x)
                ax, new_h[self.adaptive_layer] = self.adaptive_layer(ax, h.get(self.adaptive_layer))
                new_h['spikes'] = torch.cat((sx, ax), dim=1)
                o = self.output_linear(new_h['spikes'])
                #o, new_h[self.output_layer] = self.output_layer(o, h.get(self.output_layer))
                o, h_out = self.output_layer(o, h_out)
                h = new_h
            output[t] = o
        return output, new_h

class magicRSNN(nn.Module):

    def __init__(self, neuron_params, num_inputs, num_static, num_adaptive, num_outputs, static_neuron, adaptive_neuron, output_neuron):
        super(magicRSNN, self).__init__()

        #self.main_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_static + num_adaptive, bias=True)
        self.static_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_static, bias=True)
        self.adaptive_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_adaptive, bias=True)
        self.magic_linear = nn.Linear(num_inputs + num_static + num_adaptive, num_adaptive, bias=True)

        self.static_layer = static_neuron(neuron_params)
        self.adaptive_layer = adaptive_neuron(neuron_params)
        self.output_linear = nn.Linear(num_static + num_adaptive, num_outputs, bias=True)
        self.output_layer = output_neuron(neuron_params)
        self.hidden_size = num_static + num_adaptive
        self.num_outputs = num_outputs
        self.sim_time = neuron_params['SIM_TIME']

    def forward(self, inp, h=None):
        T = inp.shape[0]
        bsz = inp.shape[1]
        h = h or {'spikes': torch.zeros((bsz, self.hidden_size), device=inp.device)}
        new_h = {}
        output = torch.empty((T, bsz, self.num_outputs))
        for t in range(T):
            h_out = None
            for i in range(self.sim_time):
                x = torch.cat((inp[t], h['spikes']), dim=1)
                #x = self.main_linear(x)
                sx = self.static_linear(x)
                sx, new_h[self.static_layer] = self.static_layer(sx, h.get(self.static_layer))
                ax = self.adaptive_linear(x)
                mx = self.magic_linear(x)
                ax, new_h[self.adaptive_layer] = self.adaptive_layer(ax, mx, h.get(self.adaptive_layer))
                new_h['spikes'] = torch.cat((sx, ax), dim=1)
                o = self.output_linear(new_h['spikes'])
                #o, new_h[self.output_layer] = self.output_layer(o, h.get(self.output_layer))
                o, h_out = self.output_layer(o, h_out)
                h = new_h
            output[t] = o
        return output, new_h

class FeedForwardSNN(nn.Module):

    def __init__(self, neuron_params, architecture, neuron, output_neuron):
        super(FeedForwardSNN, self).__init__()

        #self.input_layer = neuron(spike_fn, neuron_params)
        self.input_linear = nn.Linear(architecture[0], architecture[1], bias=True)
        self.linear_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        for i in range(1, len(architecture)-1):
            self.lif_layers.append(neuron(neuron_params))
            self.linear_layers.append(nn.Linear(architecture[i], architecture[i+1], bias=True))
        self.output_layer = output_neuron(neuron_params)
        self.num_outputs = architecture[-1]
        self.sim_time = neuron_params['SIM_TIME']


    def forward(self, inp, h=None):
        T = inp.shape[0]
        bsz = inp.shape[1]
        h = h or {}
        new_h = {}
        output = torch.empty((T, bsz, self.num_outputs))
        for t in range(T):
            h_out = None
            for k in range(self.sim_time):
                x = inp[t]
                x = self.input_linear(x)
                for i in range(len(self.linear_layers)):
                    x, new_h[self.lif_layers[i]] = self.lif_layers[i](x, h.get(self.lif_layers[i]))
                    x = self.linear_layers[i](x)
                o, h_out = self.output_layer(x, h_out)
                h = new_h
            output[t] = o
        return output, new_h


class TestNN(nn.Module):

    def __init__(self, neuron_params, architecture, neuron, output_neuron):
        super(TestNN, self).__init__()

        #self.input_layer = neuron(spike_fn, neuron_params)
        self.input_linear = nn.Linear(architecture[0], architecture[-1], bias=True)


    def forward(self, inp, h=None):
        x = self.input_linear(inp[0])#.detach()
        return x.unsqueeze(0), {} #output