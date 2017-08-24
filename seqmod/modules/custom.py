
import torch
import torch.nn as nn
from torch.autograd import Variable


# Stateful Modules
class _StackedRNN(nn.Module):
    def __init__(self, cell, num_layers, in_dim, hid_dim,
                 dropout=0.0, **kwargs):
        """
        cell: str or custom cell class
        """
        super(_StackedRNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.has_dropout = False
        if dropout:
            self.has_dropout = True
            self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if isinstance(cell, str):
            cell = getattr(nn, cell)
        for i in range(num_layers):
            self.layers.append(cell(in_dim, hid_dim, **kwargs))
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
            h_1_i = layer(inp, hidden[i])
            inp = h_1_i
            # dropout on all but last layer
            if i + 1 != self.num_layers and self.has_dropout:
                inp = self.dropout(inp)
            h_1 += [h_1_i]

        return inp, torch.stack(h_1)


class StackedNormalizedGRU(_StackedRNN):
    def __init__(self, *args, **kwargs):
        super(StackedNormalizedGRU, self).__init__(
            NormalizedGRUCell, *args, **kwargs)

    def forward(self, inp, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(inp, hidden[i])
            inp = h_1_i
            # dropout on all but last layer
            if i + 1 != self.num_layers and self.has_dropout:
                inp = self.dropout(inp)
            h_1 += [h_1_i]

        return inp, torch.stack(h_1)


class NormalizedGRUCell(nn.GRUCell):
    """
    Layer-normalized GRUCell

    kindly taken from here [https://github.com/pytorch/pytorch/issues/1959]
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(NormalizedGRUCell, self).__init__(input_size, hidden_size, bias)
        # match GRUCell params for gates (reset, update) and input
        self.gamma_ih = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.eps = 0

    def _layer_norm_x(self, x, g, bias=None):
        mean = x.mean(1).expand_as(x)
        std = x.std(1).expand_as(x)
        output = g.expand_as(x) * ((x - mean) / (std + self.eps))
        if bias is not None:
            output += bias.expand_as(x)
        return output

    def _layer_norm_h(self, x, g, bias=None):
        mean = x.mean(1).expand_as(x)
        output = g.expand_as(x) * (x - mean)
        if bias is not None:
            output += bias.expand_as(x)

        return output

    def forward(self, x, h):
        """
        x: (batch x input_size)
        h: (batch x hidden_size)
        """
        # 1. GATES
        # 1.1. input to hidden gates
        weight_ih_rz = self.weight_ih.narrow(  # update & reset gate params
            0, 0, 2 * self.hidden_size
        ).transpose(0, 1)
        ih_rz = self._layer_norm_x(
            torch.mm(x, weight_ih_rz),
            self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
            None if not self.bias
            else self.bias_ih.narrow(0, 0, 2 * self.hidden_size))

        # 1.2 hidden to hidden gates
        weight_hh_rz = self.weight_hh.narrow(
            0, 0, 2 * self.hidden_size
        ).transpose(0, 1)
        hh_rz = self._layer_norm_h(
            torch.mm(h, weight_hh_rz),
            self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
            None if not self.bias
            else self.bias_hh.narrow(0, 0, 2 * self.hidden_size))

        rz = torch.sigmoid(ih_rz + hh_rz)
        r = rz.narrow(1, 0, self.hidden_size)
        z = rz.narrow(1, self.hidden_size, self.hidden_size)

        # 1. PROJECTIONS
        # 1.1. input to hidden projection
        weight_ih_n = self.weight_ih.narrow(
            0, 2 * self.hidden_size, self.hidden_size
        ).transpose(0, 1)
        ih_n = self._layer_norm_x(
            torch.mm(x, weight_ih_n),
            self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
            None if not self.bias
            else self.bias_ih.narrow(0, 2*self.hidden_size, self.hidden_size))

        # 1.2. hidden to hidden projection
        weight_hh_n = self.weight_hh.narrow(
            0, 2 * self.hidden_size, self.hidden_size
        ).transpose(0, 1)
        hh_n = self._layer_norm_h(
            torch.mm(h, weight_hh_n),
            self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
            None if not self.bias
            else self.bias_hh.narrow(0, 2*self.hidden_size, self.hidden_size))

        # h' = (1 - z) * n + (z * h)
        n = torch.tanh(ih_n + r * hh_n)
        h = (1 - z) * n + z * h
        return h


class NormalizedGRU(StackedNormalizedGRU):
    def __init__(self, input_size, hid_dim, num_layers,
                 dropout=0.0, bias=True, **kwargs):
        if 'batch_first' in kwargs or 'bidirectional' in kwargs:
            raise NotImplementedError
        super(NormalizedGRU, self).__init__(
            num_layers, input_size, hid_dim, bias=bias, dropout=dropout)

    def forward(self, xs, h_0):
        """
        xs: (seq_len x batch x input_size)
        h_0: (num_layers * 1 x batch x hidden_size)
        """
        outputs, h_t = [], h_0
        for x_t in xs:
            output, h_t = super(NormalizedGRU, self).forward(x_t, h_t)
            outputs.append(output)
        return torch.stack(outputs), h_t


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
    bernoulli distribution with mean equal to dropout.
    """
    probs = X.new(*X.size()).float().zero_() + dropout_rate
    # zeroth reserved_codes
    probs[sum((X == x) for x in reserved_codes)] = 0
    return probs.bernoulli().byte()


def word_dropout(
        inp, target_code, p=0.0, reserved_codes=(), training=True):
    """
    Applies word dropout to an input Variable. Dropout isn't constant
    across batch examples. This is only to be used to drop input symbols
    (i.e. before the embedding layer)

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
