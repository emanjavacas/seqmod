
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StackedRNN(nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, cell='LSTM', dropout=0.0):
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.cell = cell
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        super(StackedRNN, self).__init__()

        for i in range(self.num_layers):
            layer = getattr(nn, cell + 'Cell')(in_dim, hid_dim)
            self.add_module(cell + 'Cell_%d' % i, layer)
            in_dim = hid_dim

    def forward(self, inp, hidden):
        """
        Parameters:
        -----------
        inp: torch.Tensor (batch x inp_dim),
            Tensor holding the target for the previous decoding step,
            inp_dim = emb_dim or emb_dim + hid_dim if self.add_pred is True.
        hidden: tuple (h_c, c_0), output of previous step or init hidden at 0,
            h_c: (num_layers x batch x hid_dim)
            n_c: (num_layers x batch x hid_dim)
        Returns: output, (h_n, c_n)
        --------
        output: torch.Tensor (batch x hid_dim)
        h_n: torch.Tensor (num_layers x batch x hid_dim)
        c_n: torch.Tensor (num_layers x batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_0, c_0 = hidden
            h_1, c_1 = [], []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + ('Cell_%d' % i))
                h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
                h_1.append(h_1_i), c_1.append(c_1_i)
                inp = h_1_i
                # only add dropout to hidden layer (not output)
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(
                        inp, p=self.dropout, training=self.training)
            output, hidden = inp, (torch.stack(h_1), torch.stack(c_1))
        else:
            h_0, h_1 = hidden, []
            for i in range(self.num_layers):
                layer = getattr(self, self.cell + ('Cell_%d' % i))
                h_1_i = layer(inp, h_0[i])
                h_1.append(h_1_i)
                inp = h_1_i
                if i + 1 != self.num_layers and self.has_dropout:
                    inp = F.dropout(
                        inp, p=self.dropout, training=self.training)
            output, hidden = inp, torch.stack(h_1)
        return output, hidden
