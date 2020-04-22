import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


# inputs, neuron, synapse



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
                if p not in processed:
                    self.recurrent_layers.append(p)
            processed.append(layer)
        self.out_size = self.layers['output'].out_size


# TODO: handle sim_time somewhere else

    def forward(self, inp, h):
        state, spikes = h
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
            results[layer], hi = self.layers[layer](x, state[idxState])
            new_state.append(hi)
            idxState += 1
        for layer in self.recurrent_layers:
            new_spikes.append(results[layer])
        return results['output'], (tuple(new_state), tuple(new_spikes))

    def get_initial_state(self, batch_size):
        state = []
        spikes = []
        for layer in self.architecture:
            state.append(self.layers[layer].get_initial_state(batch_size))
        for layer in self.recurrent_layers:
            spikes.append(self.layers[layer].get_initial_output(batch_size))
        return tuple(state), tuple(spikes)


def make_SequenceWrapper(model, do_trace):
    return SequenceTracer(SequenceWrapper(model)) if do_trace else SequenceWrapper(model)

def getTypeName(obj):
    if type(obj) is tuple:
        return 'Tuple['+', '.join([getTypeName(x) for x in obj])+']'
    return 'Tensor'

class SequenceTracer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.trace_run = 0
        self.og_model = model
        self.og_model_model = model.model

    def forward(self, inp, h):
        if self.trace_run == 0:
            self.model.model = torch.jit.trace(self.model.model, (inp[0], h))
            _, ih = self.model.model(inp[0], h)
            typehint = '# type: (torch.Tensor, ' + getTypeName(h) + ') -> Tuple[torch.Tensor, ' + getTypeName(ih) + ']'
            og_build_def = torch.jit.frontend.build_def
            def my_build_def(ctx, py_def, type_line, self_name=None):
                return og_build_def(ctx, py_def, typehint, self_name)
            torch.jit.frontend.build_def = my_build_def
            self.model = torch.jit.script(self.model)
            torch.jit.frontend.build_def = og_build_def
        elif self.trace_run == 3:
            self.model = self.og_model
            self.model.model = self.og_model_model
            self.og_model = None
            self.og_model_model = None
        self.trace_run += 1
        return self.model(inp, h)


    def get_initial_state(self, batch_size):
        return self.model.get_initial_state(batch_size)

class SequenceWrapper(nn.Module):
    def __init__(self, model):
        super(SequenceWrapper, self).__init__()
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size

    def forward(self, inp, h):
        # type: (torch.Tensor, int) -> (int, int)
        out1, h = self.model(inp[0], h)
        #if inp.shape[0] == 1:
        #    return out1.unsqueeze(0), h
        output = torch.zeros(inp.shape[:1]+out1.shape, device=inp.device) #empty
        output[0] = out1
        for t in range(1, inp.shape[0]):
            output[t], h = self.model(inp[t], h)
        return output, h

    def get_initial_state(self, batch_size):
        return self.model.get_initial_state(batch_size)

class OuterWrapper(nn.Module):
    def __init__(self, model, device, do_trace):
        super().__init__()
        self.pretrace = model.to(device)
        example_input = (torch.zeros([1, 1, self.pretrace.in_size], device=device), self.pretrace.get_initial_state(1))
        self.pretrace(*example_input)
        self.model = torch.jit.trace(model, example_input) if do_trace else self.pretrace
        self.pretrace(*example_input)


    def forward(self, inp, h=None):
        if not h:
            h = self.pretrace.get_initial_state(inp.shape[1])
        return self.model(inp, h)

    def save(self, addr):
        # self.model.save(addr+'_traced')
        torch.save(self.pretrace, addr)  # +'_og'

    '''
    @staticmethod
    def load(addr, device):
        traced = torch.jit.load(addr+'_traced')
        original = torch.load(addr) #+'_og'
        og_par = dict(original.named_parameters())
        for name, par in traced.named_parameters():
            og_par[name].data = par.data
        og_buf = dict(original.named_buffers())
        for name, par in traced.named_buffers():
            og_buf[name].data = par.data
        return OuterWrapper(original, device, True, traced=traced)
    '''

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
    def __init__(self, size):
        super().__init__()
        self.in_size = size
        self.out_size = size

    def get_initial_state(self, batch_size):
        return ()

    def forward(self, x, h):
        return x, ()

