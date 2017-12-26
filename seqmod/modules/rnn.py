
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseStackedRNN(nn.Module):
    def __init__(self, cell, num_layers, in_dim, hid_dim,
                 dropout=0.0, **kwargs):
        """
        cell: str or custom cell class
        """
        super(BaseStackedRNN, self).__init__()
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


class StackedLSTM(BaseStackedRNN):
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


class StackedGRU(BaseStackedRNN):
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


class StackedNormalizedGRU(BaseStackedRNN):
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
        self.gamma_ih.custom, self.gamma_hh.custom = True, True
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
            0, 0, 2 * self.hidden_size).transpose(0, 1)
        ih_rz = self._layer_norm_x(
            torch.mm(x, weight_ih_rz),
            self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
            None if not self.bias else self.bias_ih.narrow(
                0, 0, 2 * self.hidden_size))

        # 1.2 hidden to hidden gates
        weight_hh_rz = self.weight_hh.narrow(
            0, 0, 2 * self.hidden_size).transpose(0, 1)
        hh_rz = self._layer_norm_h(
            torch.mm(h, weight_hh_rz),
            self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
            None if not self.bias else self.bias_hh.narrow(
                0, 0, 2 * self.hidden_size))

        rz = torch.sigmoid(ih_rz + hh_rz)
        r = rz.narrow(1, 0, self.hidden_size)
        z = rz.narrow(1, self.hidden_size, self.hidden_size)

        # 1. PROJECTIONS
        # 1.1. input to hidden projection
        weight_ih_n = self.weight_ih.narrow(
            0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)
        ih_n = self._layer_norm_x(
            torch.mm(x, weight_ih_n),
            self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
            None if not self.bias else self.bias_ih.narrow(
                0, 2*self.hidden_size, self.hidden_size))

        # 1.2. hidden to hidden projection
        weight_hh_n = self.weight_hh.narrow(
            0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)
        hh_n = self._layer_norm_h(
            torch.mm(h, weight_hh_n),
            self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
            None if not self.bias else self.bias_hh.narrow(
                0, 2*self.hidden_size, self.hidden_size))

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


def _custom_rhn_init(module):
    for m_name, m in module.named_modules():
        if 'T' in m_name:      # initialize transform gates bias to -3
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant(m.bias, -3)


def make_dropout_mask(inp, p, size):
    """
    Make dropout mask variable
    """
    keep_p = 1 - p
    mask = inp.data.new(*size)
    mask.bernoulli_(keep_p).div_(keep_p)
    return Variable(mask)


class _RHN(nn.Module):
    """
    Implementation of the full RHN in which the recurrence is computed as:
        s_l^t = (h_l^t * t_l^t) + (s_{l-1}^t * c_l^t)
    Input embeddings to this module should not have dropout, since the
    input dropout is controlled by the RHN in order to (optionally) enforce
    same dropout mask is applied to all linear transformations (see argument)
    `input_dropout`.

    Parameters:
    -----------
    in_dim: int, number of features in the input
    hid_dim: int, number of features in the output
    num_layers: int, number of mini-steps inside a recurrence step.
    tied_noise: bool, whether to use the same mask for all gates
    input_dropout: float, dropout applied to the input layer
    hidden_dropout: float, dropout applied to the recurrent layers
    kwargs: extra arguments to general RNN cells that are ignored by RHN
    """
    def __init__(self, in_dim, hid_dim, num_layers=1, tied_noise=True,
                 input_dropout=0.75, hidden_dropout=0.25, **kwargs):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.depth = num_layers
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.tied_noise = tied_noise
        super(_RHN, self).__init__()
        self.add_module('input_H', nn.Linear(in_dim, hid_dim))
        self.add_module('input_T', nn.Linear(in_dim, hid_dim))
        self.add_module('input_C', nn.Linear(in_dim, hid_dim))
        self.rnn_h, self.rnn_t, self.rnn_c = [], [], []
        for layer in range(self.depth):
            bias = False if layer == 0 else True
            rnn_h = nn.Linear(hid_dim, hid_dim, bias=bias)
            rnn_t = nn.Linear(hid_dim, hid_dim, bias=bias)
            rnn_c = nn.Linear(hid_dim, hid_dim, bias=bias)
            self.add_module('rnn_H_{}'.format(layer + 1), rnn_h)
            self.add_module('rnn_T_{}'.format(layer + 1), rnn_t)
            self.add_module('rnn_C_{}'.format(layer + 1), rnn_c)
            self.rnn_h.append(rnn_h)
            self.rnn_t.append(rnn_t)
            self.rnn_c.append(rnn_c)

    def custom_init(self):
        _custom_rhn_init(self)

    def _step(self, H_t, T_t, C_t, h0, h_mask, t_mask, c_mask):
        s_lm1, rnns = h0, [self.rnn_h, self.rnn_t, self.rnn_c]
        for l, (rnn_h, rnn_t, rnn_c) in enumerate(zip(*rnns)):
            s_lm1_H = h_mask.expand_as(s_lm1) * s_lm1
            s_lm1_T = t_mask.expand_as(s_lm1) * s_lm1
            s_lm1_C = c_mask.expand_as(s_lm1) * s_lm1
            if l == 0:
                H_t = F.tanh(H_t + rnn_h(s_lm1_H))
                T_t = F.sigmoid(T_t + rnn_t(s_lm1_T))
                C_t = F.sigmoid(C_t + rnn_t(s_lm1_C))
            else:
                H_t = F.tanh(rnn_h(s_lm1_H))
                T_t = F.sigmoid(rnn_t(s_lm1_T))
                C_t = F.sigmoid(rnn_t(s_lm1_C))
            s_l = H_t * T_t + s_lm1 * C_t
            s_lm1 = s_l

        return s_l

    def forward(self, inp, hidden):
        """
        Parameters:
        -----------

        inp: FloatTensor (seq_len x batch_size x in_dim)
        hidden: FloatTensor (batch_size x hidden_size)
        """
        seq_len, batch_size, _ = inp.size()
        mask_size = (batch_size, self.in_dim)
        h_mask = make_dropout_mask(inp, self.input_dropout, mask_size)
        if self.tied_noise:
            t_mask, c_mask = h_mask, h_mask
        else:
            t_mask = make_dropout_mask(inp, self.input_dropout, (mask_size))
            c_mask = make_dropout_mask(inp, self.input_dropout, (mask_size))
        H = (h_mask.expand_as(inp) * inp).view(seq_len * batch_size, -1)
        T = (t_mask.expand_as(inp) * inp).view(seq_len * batch_size, -1)
        C = (c_mask.expand_as(inp) * inp).view(seq_len * batch_size, -1)
        H = self.input_H(H).view(seq_len, batch_size, -1)
        T = self.input_T(T).view(seq_len, batch_size, -1)
        C = self.input_C(C).view(seq_len, batch_size, -1)

        mask_size = (batch_size, self.hid_dim)
        s_h_mask = make_dropout_mask(inp, self.hidden_dropout, mask_size)
        if self.tied_noise:
            s_t_mask, s_c_mask = s_h_mask, s_h_mask
        else:
            s_t_mask = make_dropout_mask(inp, self.hidden_dropout, mask_size)
            s_c_mask = make_dropout_mask(inp, self.hidden_dropout, mask_size)

        outs = []
        for H_t, T_t, C_t in zip(H, T, C):
            out = self._step(H_t, T_t, C_t, hidden,
                             s_h_mask, s_t_mask, s_c_mask)
            outs.append(out)
            hidden = out

        return torch.stack(outs), outs[-1]


class _RHNCoupled(nn.Module):
    """
    Simple variant of the RHN from https://arxiv.org/abs/1607.03474, in which
    the carry gate is omitted and the highway transformation is defined as:
        s_l^t = (h_l^t - s_{l-1}^t) * t_l^t + s_{l-1}^t
    which corresponds to setting the carry gate as:
        c = 1 - t

    Input embeddings to this module should not have dropout, since the
    input dropout is controlled by the RHN in order to (optionally) enforce
    same dropout mask is applied to all linear transformations (see argument)
    `input_dropout`.

    Parameters: (See _RHN)
    """
    def __init__(self, in_dim, hid_dim, num_layers=1, tied_noise=True,
                 input_dropout=0.75, hidden_dropout=0.25, **kwargs):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.depth = num_layers
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.tied_noise = tied_noise
        super(_RHNCoupled, self).__init__()
        self.add_module('input_H', nn.Linear(in_dim, hid_dim))
        self.add_module('input_T', nn.Linear(in_dim, hid_dim))
        self.rnn_h, self.rnn_t = [], []
        for layer in range(self.depth):
            bias = False if layer == 0 else True
            rnn_h = nn.Linear(hid_dim, hid_dim, bias=bias)
            rnn_t = nn.Linear(hid_dim, hid_dim, bias=bias)
            self.add_module('rnn_H_{}'.format(layer + 1), rnn_h)
            self.add_module('rnn_T_{}'.format(layer + 1), rnn_t)
            self.rnn_h.append(rnn_h), self.rnn_t.append(rnn_t)

    def custom_init(self):
        _custom_rhn_init(self)

    def _step(self, H_t, T_t, h0, h_mask, t_mask):
        s_lm1 = h0
        for l, (rnn_h, rnn_t) in enumerate(zip(self.rnn_h, self.rnn_t)):
            s_lm1_H = h_mask.expand_as(s_lm1) * s_lm1
            s_lm1_T = t_mask.expand_as(s_lm1) * s_lm1
            if l == 0:
                H_t = F.tanh(H_t + rnn_h(s_lm1_H))
                T_t = F.sigmoid(T_t + rnn_t(s_lm1_T))
            else:
                H_t = F.tanh(rnn_h(s_lm1_H))
                T_t = F.sigmoid(rnn_t(s_lm1_T))
            s_l = (H_t - s_lm1) * T_t + s_lm1
            s_lm1 = s_l

        return s_l

    def forward(self, inp, hidden=None):
        """
        Parameters:
        -----------

        inp: FloatTensor (seq_len x batch_size x in_dim)
        hidden: FloatTensor (batch_size x hidden_size)
        """
        seq_len, batch_size, _ = inp.size()
        mask_size = (batch_size, self.in_dim)
        h_mask = make_dropout_mask(inp, self.input_dropout, mask_size)
        if self.tied_noise:
            t_mask = h_mask
        else:
            t_mask = make_dropout_mask(inp, self.input_dropout, (mask_size))
        H = (h_mask.expand_as(inp) * inp).view(seq_len * batch_size, -1)
        T = (t_mask.expand_as(inp) * inp).view(seq_len * batch_size, -1)
        H = self.input_H(H).view(seq_len, batch_size, -1)
        T = self.input_T(T).view(seq_len, batch_size, -1)

        mask_size = (batch_size, self.hid_dim)
        s_h_mask = make_dropout_mask(inp, self.hidden_dropout, mask_size)
        if self.tied_noise:
            s_t_mask = s_h_mask
        else:
            s_t_mask = make_dropout_mask(inp, self.hidden_dropout, mask_size)

        outs = []
        for H_t, T_t in zip(H, T):
            out = self._step(H_t, T_t, hidden, s_h_mask, s_t_mask)
            outs.append(out)
            hidden = out

        return torch.stack(outs), outs[-1]


class RHN(_RHN):
    """
    Wrapper class for a stacked RHN. Note that although not necessary,
    the RHN is limited to a single stacked layer (reinterpreting parameter
    `num_layers` horizontally as RHN depth).
    """
    def forward(self, inp, hidden):
        """
        Parameters:
        -----------

        inp: FloatTensor (seq_len x batch_size x in_dim)
        hidden: FloatTensor (num_layers x batch_size x hidden_size)
        """
        out, hidden = super(RHN, self).forward(inp, hidden.squeeze(0))
        return out, hidden.unsqueeze(0)


class RHNCoupled(_RHNCoupled):
    """
    Wrapper class or a stacked RHNCoupled. Note that although not necessary,
    the RHN is limited to a single stacked layer (reinterpreting parameter
    `num_layers` horizontally as RHN depth).
    """
    def forward(self, inp, hidden):
        """
        Parameters:
        -----------

        inp: FloatTensor (seq_len x batch_size x in_dim)
        hidden: FloatTensor (num_layers x batch_size x hidden_size)
        """
        out, hidden = super(RHNCoupled, self).forward(inp, hidden.squeeze(0))
        return out, hidden.unsqueeze(0)
