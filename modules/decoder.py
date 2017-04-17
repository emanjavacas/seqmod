
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules import utils as u
from modules import attention as attn
from modules.custom import StackedRNN, MaxOut


class Decoder(nn.Module):
    """
    Attentional decoder for the EncoderDecoder architecture.

    Parameters:
    -----------
    num_layers: tuple (enc_num_layers, dec_num_layers)
    project_init: bool, optional
        Whether to use an extra projection on last encoder hidden state to
        initialize decoder hidden state.
    """
    def __init__(self, emb_dim, hid_dim, num_layers, cell, att_dim,
                 att_type='Bahdanau', dropout=0.0, maxout=2,
                 add_prev=True, project_init=False):
        in_dim = emb_dim if not add_prev else hid_dim + emb_dim
        if isinstance(num_layers, tuple):
            enc_num_layers, dec_num_layers = num_layers
        else:
            enc_num_layers, dec_num_layers = num_layers, num_layers
        self.num_layers = dec_num_layers
        self.hid_dim = hid_dim
        self.cell = cell
        self.add_prev = add_prev
        self.dropout = dropout
        self.project_init = project_init
        super(Decoder, self).__init__()

        # rnn layers
        self.rnn_step = StackedRNN(
            self.num_layers, in_dim, hid_dim, cell=cell, dropout=dropout)

        # attention network
        self.att_type = att_type
        if att_type == 'Bahdanau':
            self.attn = attn.BahdanauAttention(att_dim, hid_dim)
        elif att_type == 'Global':
            assert att_dim == hid_dim, \
                "For global, Encoder, Decoder & Attention must have same size"
            self.attn = attn.GlobalAttention(hid_dim)
        else:
            raise ValueError("unknown attention network [%s]" % att_type)
        if self.project_init:
            assert self.att_type != "Global", \
                "GlobalAttention doesn't support project_init yet"
            # normally dec_hid_dim == enc_hid_dim, but if not we project
            self.project_h = nn.Linear(hid_dim * enc_num_layers,
                                       hid_dim * dec_num_layers)
            if self.cell.startswith('LSTM'):
                self.project_c = nn.Linear(hid_dim * enc_num_layers,
                                           hid_dim * dec_num_layers)

        # maxout
        self.has_maxout = False
        if bool(maxout):
            self.has_maxout = True
            self.maxout = MaxOut(att_dim + emb_dim, att_dim, maxout)

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable at decoding step 0 to be fed as init hidden step.

        Returns (h_0, c_0):
        --------
        h_0: torch.Tensor (dec_num_layers x batch x hid_dim)
        c_0: torch.Tensor (dec_num_layers x batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_t, c_t = enc_hidden
        else:
            h_t = enc_hidden
        enc_num_layers, bs, hid_dim = h_t.size()

        if not self.project_init:
            # use last encoder hidden state
            assert enc_num_layers == self.num_layers, \
                "encoder and decoder need equal depth if project_init not set"
            if self.cell.startswith('LSTM'):
                dec_h0, dec_c0 = enc_hidden
            else:
                dec_h0 = enc_hidden
        else:
            # use a projection of last encoder hidden state
            # TODO: alternatively we could use state at lastest encoder layer
            # -> (batch x 1 x enc_num_layers * hid_dim)
            h_t = h_t.t().contiguous().view(-1, enc_num_layers * hid_dim)
            dec_h0 = self.project_h(h_t)
            dec_h0 = dec_h0.view(bs, self.num_layers, self.hid_dim).t()
            if self.cell.startswith('LSTM'):
                c_t = c_t.t().contiguous().view(-1, enc_num_layers * hid_dim)
                dec_c0 = self.project_c(c_t)
                dec_c0 = dec_c0.view(bs, self.num_layers, self.hid_dim).t()

        if self.cell.startswith('LSTM'):
            return dec_h0, dec_c0
        else:
            return dec_h0

    def init_output_for(self, dec_hidden):
        """
        Creates a variable to be concatenated with previous target
        embedding as input for the first rnn step. This is used
        for the first decoding step when using the add_prev flag.

        Parameters:
        -----------
        hidden: tuple (h_0, c_0)
        h_0: torch.Tensor (dec_num_layers x batch x hid_dim)
        c_0: torch.Tensor (dec_num_layers x batch x hid_dim)

        Returns:
        --------
        torch.Tensor (batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            dec_hidden = dec_hidden[0]
        data = dec_hidden.data.new(dec_hidden.size(1), self.hid_dim).zero_()
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
            h_t: (enc_num_layers x batch x hid_dim)
            c_t: (enc_num_layers x batch x hid_dim)
            Can be used to use to specify an initial hidden state for the
            decoder (e.g. the hidden state at the last encoding step.)
        """
        hidden = hidden or self.init_hidden_for(enc_hidden)
        if self.add_prev:
            inp = torch.cat([out or self.init_output_for(hidden), prev], 1)
        else:
            inp = prev
        out, hidden = self.rnn_step(inp, hidden)
        out, att_weight = self.attn(out, enc_outs, enc_att=enc_att, mask=mask)
        # out (batch x att_dim)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if self.has_maxout:
            out = self.maxout(torch.cat([out, prev], 1))
        return out, hidden, att_weight
