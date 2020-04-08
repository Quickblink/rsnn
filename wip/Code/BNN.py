import torch
import torch.nn as nn


class FlipFlopSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, last):
        gto = (input >= 1).float()
        decision = gto - (input <= -1).float()
        use_old = decision.abs() < 0.5
        ctx.save_for_backward(input, last, use_old)
        to_out = torch.where(use_old, last, gto)
        to_next = to_out.clone()
        return to_out, to_next

    @staticmethod
    def backward(ctx, grad_output, grad_next):
        #print(grad_output[0,0].item())
        input, last, use_old = ctx.saved_tensors
        #clamped = torch.clamp(grad_output, -1e-3, 1e-3)
        to_input = (grad_output + 0.5 * grad_next) * 0.2 * (1 / (2*torch.abs(input + 1) + 1.0) ** 2 + 1 / (2*torch.abs(input - 1) + 1.0) ** 2)#10 #work out doubly spiked function
        to_last = 0.9 * grad_next + grad_output
        to_last = torch.where(use_old, to_last, torch.zeros([1]))
        #print('########', to_input[0,0].item())
        return to_input, to_last


class SuperSpike(torch.autograd.Function):

    scale = 2.0#100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):

        #print(input[0,0].item())
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        #clamped = torch.clamp(grad_output, -1e-3, 1e-3)
        out = grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2

        #out = torch.clamp((grad_output / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2), -1, 1)
        return out #torch.where((out == 0), torch.ones([1]) * 0.001, out)


class Dampener(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output[0,0].item())
        return 1.0 * grad_output #0.1


class Profiler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print('------------', grad_output[0,0].item())
        return grad_output


class FlipSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        to_out = input
        to_next = to_out.clone()
        return to_out, to_next

    @staticmethod
    def backward(ctx, grad_output, grad_next):
        return torch.where((grad_output < 0) & (grad_next > 0), grad_output, grad_output + grad_next)

class FlipNeuron(nn.Module):
    def __init__(self):
        super(FlipNeuron, self).__init__()
        self.spike_fn = SuperSpike.apply

    #@printcode
    def forward(self, x, h):
        #x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x)}

        new_h = {}

        new_h['mem'] = 0.95 * h['mem'] + torch.nn.ELU()(x-2)#torch.sigmoid(x-2) #torch.tanh(x) torch.sigmoid(x-4)
        spikes = self.spike_fn(new_h['mem'] - 1)

        return spikes, new_h


class FlopNeuron(nn.Module):
    def __init__(self):
        super(FlopNeuron, self).__init__()
        self.spike_fn = SuperSpike.apply

    # @printcode
    def forward(self, x, h):
        # x = NaNtoZero.apply(x)
        if not h:
            h = {'mem': torch.zeros_like(x)}

        new_h = {}

        new_h['mem'] = 0.95 * h['mem'] + torch.nn.ELU()(x-2) #torch.sigmoid(x-2) #torch.tanh(x)
        spikes = self.spike_fn(1 - new_h['mem'])

        return spikes, new_h


class RBNN(nn.Module):

    def __init__(self, num_inputs, num_pre, num_adaptive, num_post, num_outputs, spk_fn, ff_spk):
        super(RBNN, self).__init__()

        self.spk_fn = spk_fn
        self.ff_spk = ff_spk

        self.pre_linear = nn.Linear(num_inputs + num_adaptive, num_pre, bias=True)
        self.adaptive_linear = nn.Linear(num_pre, num_adaptive, bias=True)
        self.post_linear = nn.Linear(num_inputs + num_adaptive, num_post, bias=True)
        self.output_linear = nn.Linear(num_post, num_outputs, bias=True)

        self.hidden_size = num_adaptive
        self.hidden_zeros = num_adaptive // 2
        self.hidden_ones = num_adaptive - self.hidden_zeros
        self.num_outputs = num_outputs

    def getMemoryParameters(self):
        return list(self.pre_linear.parameters())+list(self.adaptive_linear.parameters())

    def getOutputParameters(self):
        return list(self.post_linear.parameters())+list(self.output_linear.parameters())

    def forward(self, inp, h=None, logger=None):
        T = inp.shape[0]
        bsz = inp.shape[1]
        h = h or {'spikes': torch.cat((torch.zeros((bsz, self.hidden_zeros), device=inp.device), torch.ones((bsz, self.hidden_ones), device=inp.device)), dim=1)}
        new_h = {}
        output = torch.empty((T, bsz, self.num_outputs))
        for t in range(T):
            oldspikes = Dampener.apply(h['spikes'])
            x = torch.cat((inp[t], oldspikes), dim=1)#.detach() #Detached, so gradient only flows through FlipFlop
            x = self.pre_linear(x)
            x = self.spk_fn(x)
            x = self.adaptive_linear(x)
            x, new_h['spikes'] = self.ff_spk(x, h['spikes'])
            #x = Dampener.apply(x)
            x = torch.cat((inp[t], x), dim=1)
            x = self.post_linear(x)
            #x = Profiler.apply(x)
            x = self.spk_fn(x)
            o = self.output_linear(x)
            h = new_h
            if logger:
                logger(h, t)
            output[t] = o
        return output, new_h

class newRBNN(nn.Module):

        def __init__(self, num_inputs, num_pre, num_adaptive, num_post, num_outputs, spk_fn, ff_spk):
            super(newRBNN, self).__init__()

            self.hidden_size = num_adaptive
            self.hidden_zeros = num_adaptive // 2
            self.hidden_ones = num_adaptive - self.hidden_zeros
            self.num_outputs = num_outputs

            self.spk_fn = spk_fn

            self.flips = FlipNeuron()
            self.flops = FlopNeuron()

            self.pre_linear = nn.Linear(num_inputs + num_adaptive, num_pre, bias=True)
            self.flip_linear = nn.Linear(num_pre, self.hidden_zeros, bias=True)
            self.flop_linear = nn.Linear(num_pre, self.hidden_ones, bias=True)
            self.post_linear = nn.Linear(num_inputs + num_adaptive, num_post, bias=True)
            self.output_linear = nn.Linear(num_post, num_outputs, bias=True)



        def getMemoryParameters(self):
            return list(self.pre_linear.parameters()) + list(self.adaptive_linear.parameters())

        def getOutputParameters(self):
            return list(self.post_linear.parameters()) + list(self.output_linear.parameters())

        def forward(self, inp, h=None, logger=None):
            T = inp.shape[0]
            bsz = inp.shape[1]
            h = h or {'spikes': torch.cat((torch.zeros((bsz, self.hidden_zeros), device=inp.device),
                                           torch.ones((bsz, self.hidden_ones), device=inp.device)), dim=1)}
            new_h = {}
            output = torch.empty((T, bsz, self.num_outputs))
            for t in range(T):
                oldspikes = Dampener.apply(h['spikes'])
                x = torch.cat((inp[t], oldspikes), dim=1)  # .detach() #Detached, so gradient only flows through FlipFlop
                x = self.pre_linear(x)
                x = self.spk_fn(x)
                ix = self.flip_linear(x)
                ix, new_h[self.flips] = self.flips(ix, h.get(self.flips))

                ox = self.flop_linear(x)
                ox, new_h[self.flops] = self.flops(ox, h.get(self.flops))

                x = torch.cat((ix, ox), dim=1)
                new_h['spikes'] = x
                # x = Dampener.apply(x)
                x = torch.cat((inp[t], x), dim=1)
                x = self.post_linear(x)
                # x = Profiler.apply(x)
                x = self.spk_fn(x)
                o = self.output_linear(x)
                h = new_h
                if logger:
                    logger(h, t)
                output[t] = o
            return output, new_h