
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StackedRNN(nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, cell='LSTM', dropout=0.0, bias=True):
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.cell = cell
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        super(StackedRNN, self).__init__()

        for i in range(num_layers):
            layer = getattr(nn, cell + 'Cell')(in_dim, hid_dim, bias=bias)
            self.add_module(cell + 'Cell_%d' % i, layer)
            in_dim = hid_dim

    def forward(self, inp, hidden):
        if self.cell.startswith('LSTM'):
            h_0, c_0 = hidden
            h_1, c_1 = [], []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + 'Cell_%d' % i)
                h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
                h_1.append(h_1_i), c_1.append(c_1_i)
                inp = h_1_i
                # only add dropout to hidden layer (not output)
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(inp, p=self.dropout, training=self.training)
            output, hidden = inp, (torch.stack(h_1), torch.stack(c_1))
        else:
            h_0, h_1 = hidden, []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + 'Cell_%d' % i)
                h_1_i = layer(inp, h_0[i])
                h_1.append(h_1_i)
                inp = h_1_i
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(inp, p=self.dropout, training=self.training)
            output, hidden = inp, torch.stack(h_1)
        return output, hidden
