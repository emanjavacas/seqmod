
import torch
import torch.nn as nn
from torch.autograd import Variable


# Stateful Modules
class _StackedRNN(nn.Module):
    def __init__(self, cell, num_layers, in_dim, hid_dim, dropout=0.0):
        super(_StackedRNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.has_dropout = False
        if dropout:
            self.has_dropout = True
            self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        cell = getattr(nn, cell)
        for i in range(num_layers):
            self.layers.append(cell(in_dim, hid_dim))
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
        raise NotImplementedError


class StackedLSTM(_StackedRNN):
    def __init__(self, *args, **kwargs):
        super(StackedLSTM, self).__init__('LSTMCell', *args, **kwargs)

    def forward(self, inp, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
            inp = h_1_i
            # dropout on all but last layer
            if i + 1 != self.num_layers and self.has_dropout:
                inp = self.dropout(inp)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        return inp, (torch.stack(h_1), torch.stack(c_1))


class StackedGRU(_StackedRNN):
    def __init__(self, *args, **kwargs):
        super(StackedGRU, self).__init__('GRUCell', *args, **kwargs)

    def forward(self, inp, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(inp, hidden[0][i])
            inp = h_1_i
            # dropout on all but last layer
            if i + 1 != self.num_layers and self.has_dropout:
                inp = self.dropout(inp)
            h_1 += [h_1_i]

        return inp, torch.stack(h_1)


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


# Stateless modules
def variable_length_dropout_mask(X, dropout_rate, reserved_codes=()):
    """
    Computes a binary mask across batch examples based on a
    bernoulli distribution with mean equal to dropout
    """
    probs = X.new(*X.size()).float().zero_() + dropout_rate
    # zeroth reserved_codes
    probs[sum((X == x) for x in reserved_codes)] = 0
    return probs.bernoulli().byte()


def word_dropout(
        inp, target_code, p=0.0, reserved_codes=(), training=True):
    """
    Applies word dropout to an input Variable. Dropout isn't constant
    across batch examples.

    Parameters:
    -----------
    - inp: torch.Tensor
    - target_code: int, code to use as replacement for dropped timesteps
    - dropout: float, dropout rate
    - reserved_codes: tuple of ints, ints in the input that should never
        be dropped
    - training: bool
    """
    if not training or not p > 0:
        return inp
    inp = Variable(inp.data.new(*inp.size()).copy_(inp.data))
    mask = variable_length_dropout_mask(
        inp.data, dropout_rate=p, reserved_codes=reserved_codes)
    inp.masked_fill_(Variable(mask), target_code)
    return inp
