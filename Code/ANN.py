import torch
import torch.nn as nn
import torch.nn.functional as F

class lstmPolicyPredictor(nn.Module):

    def __init__(self, inp_dim, hidden_dim, out_hidden_dim):
        super(lstmPolicyPredictor, self).__init__()
        #self.hidden_dim = hidden_dim


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(inp_dim, hidden_dim)
        self.out1 = nn.Linear(hidden_dim+inp_dim, out_hidden_dim)
        self.out2 = nn.Linear(out_hidden_dim, 1)


    def forward(self, inp, oldstate=None):
        lstmout, newstate = self.lstm(inp, oldstate)
        allforout = torch.cat((lstmout, inp), dim=2)
        h1 = F.relu(self.out1(allforout))
        out = self.out2(h1)  # F.sigmoid(
        return out, newstate



class FullyConnected(nn.Module):
    """
    A generic fully connected network with ReLu activations, biases and no activation function for
    the output
    """
    def __init__(self,architecture):
        """
        Architecture needs to be a list that describes the architecture of the  network,
        e.g. [4,16,16,2] is a network with 4 inputs, 2 outputs and two hidden layers with 16 neurons
        each
        """
        super(FullyConnected, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(0,len(architecture)-1):
            self.layers.append(nn.Linear(architecture[i], architecture[i+1], bias=True))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # no ReLu activation in the output layer
        x = self.layers[-1](x)
        return x, None