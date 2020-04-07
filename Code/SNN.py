import numpy as np
import torch
import torch.nn as nn
from Code.utils import LoggerFn

class Dampener(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / 3#10


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

    def forward(self, inp, h=None, logger=None):
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
                if logger:
                    logger(h, t, i)
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

    def forward(self, inp, h=None, logger=None):
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
                mx = self.magic_linear(x)/10
                ax, new_h[self.adaptive_layer] = self.adaptive_layer(ax, mx, h.get(self.adaptive_layer))
                new_h['spikes'] = torch.cat((sx, ax), dim=1)
                o = self.output_linear(new_h['spikes'])
                #o, new_h[self.output_layer] = self.output_layer(o, h.get(self.output_layer))
                o, h_out = self.output_layer(o, h_out)
                h = new_h
                if logger:
                    logger(h, t, i)
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



class AdaptiveFF(nn.Module):

    def __init__(self, neuron_params, num_inputs, num_static1, num_adaptive, num_static2, num_outputs, static_neuron, adaptive_neuron, output_neuron):
        super(AdaptiveFF, self).__init__()

        #self.input_layer = neuron(spike_fn, neuron_params)
        self.input_linear = nn.Linear(num_inputs, num_static1, bias=True)
        self.static1 = static_neuron(neuron_params)
        self.static_to_adaptive = nn.Linear(num_static1, num_adaptive)
        self.adaptive_layer = adaptive_neuron(neuron_params)
        self.static_linear = nn.Linear(num_adaptive+num_static1, num_static2)
        self.static2 = static_neuron(neuron_params)
        self.output_linear = nn.Linear(num_static2, num_outputs)
        self.output_layer = output_neuron(neuron_params)
        self.num_outputs = num_outputs
        self.sim_time = neuron_params['SIM_TIME']


    def forward(self, inp, h=None, logger=None):
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
                x, new_h[self.static1] = self.static1(x, h.get(self.static1))
                ax = self.static_to_adaptive(x)
                ax, new_h[self.adaptive_layer] = self.adaptive_layer(ax, h.get(self.adaptive_layer))
                x = torch.cat((x, ax), dim=1)
                x = self.static_linear(x)
                x, new_h[self.static2] = self.static2(x, h.get(self.static2))
                x = self.output_linear(x)
                o, h_out = self.output_layer(x, h_out)
                h = new_h
                #if logger:
                #    logger(h, t, k)
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



class newRSNN(nn.Module):

    def __init__(self, neuron_params, num_inputs, num_pre, num_adaptive, num_post, num_post2, num_outputs, static_neuron, adaptive_neuron, output_neuron):
        super(newRSNN, self).__init__()

        self.pre_layer = static_neuron(neuron_params, num_pre)
        self.adaptive_layer = adaptive_neuron(neuron_params, num_adaptive)
        self.post_layer = static_neuron(neuron_params, num_post)
        #self.post_layer2 = static_neuron(neuron_params, num_post2)
        self.output_layer = output_neuron(neuron_params, num_outputs)

        self.pre_linear = nn.Linear(num_inputs + num_adaptive, num_pre, bias=True)
        self.adaptive_linear = nn.Linear(num_pre, num_adaptive, bias=True)
        self.post_linear = nn.Linear(num_inputs + num_adaptive, num_post, bias=True)
        #self.post_linear2 = nn.Linear(num_post, num_post2, bias=True)
        #num_post = num_post2
        self.output_linear = nn.Linear(num_post, num_outputs, bias=True)

        self.hidden_size = num_adaptive
        self.num_outputs = num_outputs
        self.sim_time = neuron_params['SIM_TIME']

    def forward(self, inp, h=None, logger=None):
        T = inp.shape[0]
        bsz = inp.shape[1]
        #if not h:
        h = {'spikes': self.adaptive_layer.get_initial_spike(bsz)} #torch.zeros((bsz, self.hidden_size), device=inp.device)
        new_h = {}
        output = torch.empty((T, bsz, self.num_outputs))
        for t in range(T):
            h_out = None
            for i in range(self.sim_time):
                x = torch.cat((inp[t], h['spikes']), dim=1)
                x = self.pre_linear(x)
                x, new_h[self.pre_layer] = self.pre_layer(x, h.get(self.pre_layer))
                x = self.adaptive_linear(x)
                x, new_h[self.adaptive_layer] = self.adaptive_layer(x, h.get(self.adaptive_layer))
                new_h['spikes'] = x
                #x = Dampener.apply(x)
                x = torch.cat((inp[t], x), dim=1)
                x = self.post_linear(x)
                x, new_h[self.post_layer] = self.post_layer(x, h.get(self.post_layer))
                #x = self.post_linear2(x)
                #x, new_h[self.post_layer2] = self.post_layer2(x, h.get(self.post_layer2))
                o = self.output_linear(x)
                #o, h_out = self.output_layer(o, h_out)
                h = new_h
                #if logger:
                #    logger(h, t, i)
            output[t] = o
        return output, new_h