
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils as u
import attention as attn


class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, hid_dim, num_layers, cell,
                 att_dim, att_type='Bahdanau', dropout=0.0,
                 add_prev=True, project_init=False):
        """
        Parameters:
        -----------
        project_init: bool, optional
            Whether to use an extra projection on last encoder hidden state to
            initialize decoder hidden state.
        """
        in_dim = enc_hid_dim if not add_prev else enc_hid_dim + emb_dim
        enc_num_layers, dec_num_layers = num_layers
        self.num_layers = dec_num_layers
        self.cell = cell
        self.hid_dim = hid_dim
        self.add_prev = add_prev
        self.project_init = project_init

        super(Decoder, self).__init__()
        # decoder layers
        self.layers = []
        for layer in range(dec_num_layers):
            rnn = getattr(nn, cell + 'Cell')(in_dim, hid_dim)
            # since layer isn't an attribute of this module, we add it manually
            self.add_module('rnn_%d' % layer, rnn)
            self.layers.append(rnn)
            in_dim = hid_dim

        # dropout
        self.has_dropout = bool(dropout)
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout)
        # attention network
        self.att_type = att_type
        if att_type == 'Bahdanau':
            self.attn = attn.BahdanauAttention(att_dim, in_dim, hid_dim)
        elif att_type == 'Global':
            assert att_dim == in_dim == hid_dim, \
                "For global, Encoder, Decoder & Attention must have same size"
            self.attn = attn.GlobalAttention(hid_dim)
        else:
            raise ValueError("unknown attention network [%s]" % att_type)
        # init state matrix (Bahdanau)
        if self.project_init:
            assert self.att_type != "Global", \
                "GlobalAttention doesn't support project_init yet"
            # normally dec_hid_dim == enc_hid_dim, but if not we project
            self.W_h = nn.Parameter(torch.Tensor(
                enc_hid_dim * enc_num_layers, hid_dim * dec_num_layers))
            if self.cell.startswith('LSTM'):
                self.W_c = nn.Parameter(torch.Tensor(
                    enc_hid_dim * enc_num_layers, hid_dim * dec_num_layers))

    def rnn_step(self, inp, hidden):
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
            for i, layer in enumerate(self.layers):
                h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
                h_1.append(h_1_i), c_1.append(c_1_i)
                inp = h_1_i
                # only add dropout to hidden layer (not output)
                if i < (len(self.layers) - 1) and self.has_dropout:
                    inp = self.dropout(inp)
            output, hidden = inp, (torch.stack(h_1), torch.stack(c_1))
        else:
            h_0, h_1 = hidden, []
            for i, layer in enumerate(self.layers):
                h_1_i = layer(inp, h_0[i])
                h_1.append(h_1_i)
                inp = h_1_i
                if i < (len(self.layers) - 1) and self.has_dropout:
                    inp = self.dropout(inp)
            output, hidden = inp, torch.stack(h_1)
        return output, hidden

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable at decoding step 0 to be fed as init hidden step

        Returns (h_0, c_0):
        --------
        h_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)
        c_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_t, c_t = enc_hidden
        else:
            h_t = enc_hidden
        num_layers, bs, hid_dim = h_t.size()
        size = self.num_layers, bs, self.hid_dim
        # use a projection of last encoder hidden state
        if self.project_init:
            # -> (batch x 1 x num_layers * enc_hid_dim)
            h_t = h_t.t().contiguous().view(-1, num_layers * hid_dim).unsqueeze(1)
            dec_h0 = torch.bmm(h_t, u.tile(self.W_h, bs)).view(*size)
            if self.cell.startswith('LSTM'):
                c_t = c_t.t().contiguous().view(-1, num_layers * hid_dim).unsqueeze(1)
                dec_c0 = torch.bmm(c_t, u.tile(self.W_c, bs)).view(*size)
        else:
            assert num_layers == self.num_layers, \
                "encoder and decoder need equal depth if project_init not set"
            assert hid_dim == self.hid_dim, \
                "encoder and decoder need equal size if project_init not set"
            if self.cell.startswith('LSTM'):
                dec_h0, dec_c0 = enc_hidden
            else:
                dec_h0 = enc_hidden
        if self.cell.startswith('LSTM'):
            return dec_h0, dec_c0
        else:
            return dec_h0

    def init_output_for(self, dec_hidden):
        """
        Creates a variable to be concatenated with previous target embedding
        as input for the current rnn step

        Parameters:
        -----------
        hidden: tuple (h_0, c_0)
        h_0: torch.Tensor (num_layers x batch x hid_dim)
        c_0: torch.Tensor (num_layers x batch x hid_dim)

        Returns:
        --------
        torch.Tensor (batch x dec_hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_0, c_0 = dec_hidden
        else:
            h_0 = dec_hidden
        data = h_0.data.new(h_0.size(1), self.hid_dim).zero_()
        return Variable(data, requires_grad=False)

    def forward(self, prev, enc_outs, enc_hidden, out=None,
                hidden=None, enc_att=None, mask=None):
        """
        Parameters:
        -----------

        prev: torch.Tensor (batch x emb_dim),
            Previously decoded output.

        enc_outs: torch.Tensor (seq_len x batch x enc_hid_dim),
            Output of the encoder at the last layer for all encoding steps.

        enc_hidden: tuple (h_t, c_t)
            h_t: (num_layers x batch x hid_dim)
            c_t: (num_layers x batch x hid_dim)
            Can be used to use to specify an initial hidden state for the
            decoder (e.g. the hidden state at the last encoding step.)
        """
        hidden = hidden or self.init_hidden_for(enc_hidden)
        out = out or self.init_output_for(hidden)
        if self.add_prev:
            inp = torch.cat([out, prev], 1)
        else:
            inp = out
        out, hidden = self.rnn_step(inp, hidden)
        out, att_weight = self.attn(out, enc_outs, enc_att=enc_att, mask=mask)
        if self.has_dropout:
            out = self.dropout(out)
        return out, hidden, att_weight
