
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import seqmod.utils as u
from seqmod.modules.ff import grad_reverse, MLP


class BaseEncoder(nn.Module):
    """
    Base abstract class
    """
    def forward(self, inp, **kwargs):
        raise NotImplementedError

    @property
    def conditional(self):
        return False

    @property
    def encoding_size(self):
        """
        Return a tuple specifying number of dimensions and size of the encoding
        computed by the Encoder
        """
        raise NotImplementedError

    def loss(self, enc_outs, enc_trg, test=False):
        return tuple()


class RNNEncoder(BaseEncoder):
    """
    RNN Encoder that computes a sentence matrix representation with a RNN.
    """
    def __init__(self, embeddings, hid_dim, num_layers, cell, bidi=True,
                 dropout=0.0, summary='full', train_init=False,
                 add_init_jitter=False):

        super(BaseEncoder, self).__init__()

        super(BaseEncoder, self).__init__()

        if bidi and hid_dim % 2 != 0:
            raise ValueError("Hidden dimension must be even for BiRNNs")

        self.embeddings = embeddings
        self.bidi = bidi
        self.num_dirs = 2 if self.bidi else 1
        self.hid_dim = hid_dim // self.num_dirs
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.summary = summary
        self.train_init = train_init
        self.add_init_jitter = add_init_jitter

        self.rnn = getattr(nn, cell)(self.embeddings.embedding_dim,
                                     self.hid_dim, num_layers=self.num_layers,
                                     dropout=dropout, bidirectional=self.bidi)

        if self.train_init:
            init_size = self.num_layers * self.num_dirs, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*init_size).zero_())

        if self.summary == 'inner-attention':
            # Challenges with Variational Autoencoders for Text
            # https://pdfs.semanticscholar.org/dbb6/73a811bf4519b668ee481365c6a8f297ad60.pdf
            # Compute feature-wise attention on projected concatenation of
            # input embedding and RNN hidden states
            enc_dim = self.hid_dim * self.num_dirs  # output dimension
            self.attention = nn.Linear(
                self.embeddings.embedding_dim + enc_dim, enc_dim)

        elif self.summary == 'structured-attention':
            # A structured self-attentive sentence embedding
            # https://arxiv.org/pdf/1703.03130.pdf
            raise NotImplementedError

    def init_hidden_for(self, inp):
        batch_size = inp.size(1)
        size = (self.num_dirs * self.num_layers, batch_size, self.hid_dim)

        if self.train_init:
            h_0 = self.h_0.repeat(1, batch_size, 1)
        else:
            h_0 = inp.data.new(*size).zero_()
            h_0 = Variable(h_0, volatile=not self.training)

        if self.add_init_jitter:
            h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

        if self.cell.startswith('LSTM'):
            # compute memory cell
            c_0 = inp.data.new(*size).zero_()
            c_0 = Variable(c_0, volatile=not self.training)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp, hidden=None, lengths=None):
        """
        Paremeters:
        -----------

        - inp: torch.LongTensor (seq_len x batch)
        - hidden: torch.FloatTensor (num_layers * num_dirs x batch x hid_dim)

        Returns: output, hidden
        --------
        - output: depending on the summary type output will have different
            shapes.
            - full: (seq_len x batch x hid_dim * num_dirs)
            - mean: (batch x hid_dim * num_dirs)
            - mean-concat: (batch x hid_dim * num_dirs * 2)
            - inner-attention: (batch x hid_dim * num_dirs)
            - structured-attention: TODO
        - hidden: (num_layers x batch x hid_dim * num_dirs)
        """
        inp = self.embeddings(inp)

        if hidden is None:
            hidden = self.init_hidden_for(inp)

        if lengths is not None:  # pack if lengths given
            inp = pack(inp, lengths=lengths.data.tolist())

        outs, hidden = self.rnn(inp, hidden)

        if lengths is not None:  # unpack if lengths given
            (outs, _), (inp, _) = unpack(outs), unpack(inp)

        if self.bidi:
            # BiRNN encoder outputs:   (num_layers * 2 x batch x hid_dim)
            # but RNN decoder expects: (num_layers x batch x hid_dim * 2)
            if self.cell.startswith('LSTM'):
                hidden = (u.repackage_bidi(hidden[0]),
                          u.repackage_bidi(hidden[1]))
            else:
                hidden = u.repackage_bidi(hidden)

        if self.summary == 'full':
            outs = outs         # do nothing

        elif self.summary == 'mean':
            outs = outs.mean(0)

        elif self.summary == 'mean-concat':
            outs = torch.cat([outs[:-1].mean(0), outs[-1]], 1)

        elif self.summary == 'inner-attention':
            seq_len, batch_size, _ = inp.size()
            # combine across feature dimension and project to hid_dim
            weights = self.attention(
                torch.cat(
                    [inp.view(-1, self.embeddings.embedding_dim),
                     outs.view(-1, self.hid_dim * self.num_dirs)], 1))
            # apply softmax over the seq_len dimension
            weights = F.softmax(weights.view(seq_len, batch_size, -1), 0)
            # weighted sum of encoder outputs (feature-wise)
            outs = (weights * outs).sum(0)

        return outs, hidden

    @property
    def encoding_size(self):
        if self.summary == 'full':
            return 3, self.hid_dim * self.num_dirs

        elif self.summary == 'mean':
            return 2, self.hid_dim * self.num_dirs

        elif self.summary == 'mean-concat':
            return 2, self.hid_dim * self.num_dirs * 2

        elif self.summary == 'inner-attention':
            return 2, self.hid_dim * self.num_dirs

    @property
    def conditional(self):
        return False


class GRLRNNEncoder(BaseEncoder):
    def __init__(self, *args, cond_dims, cond_vocabs, **kwargs):

        super(GRLRNNEncoder, self).__init__(*args, **kwargs)

        if self.encoding_size > 2:
            raise ValueError("GRLRNNEncoder can't regularize 3D summaries")

        if len(cond_dims) != len(cond_vocabs):
            raise ValueError("cond_dims & cond_vocabs must be same length")

        # MLPs regularizing on input conditions
        self.grls = nn.ModuleList()
        _, hid_dim = self.encoding_size  # size of the encoder output
        for cond_vocab, cond_dim in zip(cond_vocabs, cond_dims):
            self.grls.append(MLP(hid_dim, hid_dim, cond_vocab))

    def loss(self, out, conds, test=False):
        grl_loss = []
        for cond, grl in zip(conds, self.grls):
            cond_out = F.log_softmax(grad_reverse(grl(out)), 1)
            grl_loss.append(F.nll_loss(cond_out, cond, size_average=True))

        if not test:
            (sum(grl_loss) / len(self.grls)).backward(retain_graph=True)

        return [l.data[0] for l in grl_loss]

    @property
    def conditional(self):
        return True
