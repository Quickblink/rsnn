import torch

class LoggerFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return grad_output