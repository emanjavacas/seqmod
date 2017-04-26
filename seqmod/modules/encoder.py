
import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod.modules import utils as u


class Encoder(nn.Module):
    """
    RNN Encoder that computes a sentence matrix representation
    of the input using an RNN.
    """
    def __init__(self, in_dim, hid_dim, num_layers, cell,
                 dropout=0.0, bidi=True):
        self.cell = cell
        self.num_layers = num_layers
        self.num_dirs = 2 if bidi else 1
        self.bidi = bidi
        self.hid_dim = hid_dim // self.num_dirs
        assert hid_dim % self.num_dirs == 0, \
            "Hidden dimension must be even for BiRNNs"
        super(Encoder, self).__init__()
        self.rnn = getattr(nn, cell)(in_dim, self.hid_dim,
                                     num_layers=self.num_layers,
                                     dropout=dropout,
                                     bidirectional=self.bidi)

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_dirs * self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp, hidden=None, compute_mask=False, mask_symbol=None):
        """
        Paremeters:
        -----------
        inp: torch.Tensor (seq_len x batch x emb_dim)

        hidden: tuple (h_0, c_0)
            h_0: ((num_layers * num_dirs) x batch x hid_dim)
            n_0: ((num_layers * num_dirs) x batch x hid_dim)

        Returns: output, (h_t, c_t)
        --------
        output: (seq_len x batch x hidden_size * num_directions)
        h_t: (num_layers x batch x hidden_size * num_directions)
        c_t: (num_layers x batch x hidden_size * num_directions)
        """
        if compute_mask:        # fixme, somehow not working
            seqlen, batch, _ = inp.size()
            outs, hidden = [], hidden or self.init_hidden_for(inp)
            for inp_t in inp.chunk(seqlen):
                out_t, hidden = self.rnn(inp_t, hidden)
                mask_t = inp_t.data.squeeze(0).eq(mask_symbol).nonzero()
                if mask_t.nelement() > 0:
                    mask_t = mask_t.squeeze(1)
                    if self.cell.startswith('LSTM'):
                        hidden[0].data.index_fill_(1, mask_t, 0)
                        hidden[1].data.index_fill_(1, mask_t, 0)
                    else:
                        hidden.data.index_fill_(1, mask_t, 0)
                outs.append(out_t)
            outs = torch.cat(outs)
        else:
            outs, hidden = self.rnn(inp, hidden or self.init_hidden_for(inp))
        if self.bidi:
            # BiRNN encoder outputs (num_layers * 2 x batch x hid_dim)
            # but decoder expects   (num_layers x batch x hid_dim * 2)
            if self.cell.startswith('LSTM'):
                hidden = (u.repackage_bidi(hidden[0]),
                          u.repackage_bidi(hidden[1]))
            else:
                hidden = u.repackage_bidi(hidden)
        return outs, hidden
