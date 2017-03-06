
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MaxOut(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        """
        Implementation of MaxOut:
            h_i^{maxout} = max_{j \in [1, ..., k]} x^T W_{..., i, j} + b_{i, j}
        where W is in R^{D x M x K}, D is the input size, M is the output size
        and K is the number of pieces to max-pool from. (i.e. i ranges over M,
        j ranges over K and ... corresponds to the input dimension)

        Parameters:
        -----------
        in_dim: int, Input dimension
        out_dim: int, Output dimension
        k: int, number of "pools" to max over

        Returns:
        --------
        out: torch.Tensor (batch x k)
        """
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        super(MaxOut, self).__init__()
        self.projection = nn.Linear(in_dim, k * out_dim)

    def forward(self, inp):
        """
        Because of the linear projection we are bound to 1-d input
        (excluding batch-dim), therefore there is no need to generalize
        the implementation to n-dimensional input.
        """
        batch, in_dim = inp.size()
        # (batch x self.k * self.out_dim) -> (batch x self.out_dim x self.k)
        out = self.projection(inp).view(batch, self.out_dim, self.k)
        out, _ = out.max(2)
        return out.squeeze(2)


class TiedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, weight, **kwargs):
        super(TiedEmbedding, self).__init__(
            num_embeddings, embedding_dim, **kwargs)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight


class TiedLinear(nn.Linear):
    def __init__(self, in_features, out_features, weight, bias=True):
        super(TiedLinear, self).__init__(in_features, out_features, bias=bias)
        assert isinstance(weight, nn.parameter.Parameter)
        self.weight = weight
