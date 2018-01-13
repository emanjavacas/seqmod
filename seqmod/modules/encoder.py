
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from seqmod.modules.torch_utils import init_hidden_for
from seqmod.modules.ff import grad_reverse, MLP
from seqmod.modules.torch_utils import pack_sort, repackage_bidi


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
        return [], None


class RNNEncoder(BaseEncoder):
    """
    RNN Encoder that computes a sentence matrix representation with a RNN.
    """
    def __init__(self, embeddings, hid_dim, num_layers, cell, bidi=True,
                 dropout=0.0, summary='full', train_init=False,
                 add_init_jitter=False):

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

        self.h_0 = None
        if self.train_init:
            init_size = self.num_layers * self.num_dirs, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*init_size).zero_())

        if self.summary not in ('mean', 'mean-concat', 'full', 'inner-attention'):
            raise ValueError("Unknown summary type [{}]".format(self.summary))

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
        return init_hidden_for(
            inp, self.num_dirs, self.num_layers, self.hid_dim, self.cell,
            h_0=self.h_0, add_init_jitter=self.add_init_jitter,
            training=self.training)

    def forward(self, inp, lengths=None):
        """
        Paremeters:
        -----------

        - inp: torch.LongTensor (seq_len x batch)
        - lengths: torch.LongTensor (batch)

        Returns: output, hidden
        --------
        - output: depending on the summary type output will have different
            shapes.
            - full: (seq_len x batch x hid_dim * num_dirs)
            - mean: (batch x hid_dim * num_dirs)
            - mean-concat: (batch x hid_dim * num_dirs * 2)
            - inner-attention: (batch x hid_dim * num_dirs)

        - hidden: (num_layers x batch x hid_dim * num_dirs)
        """
        if hasattr(self.embeddings, 'is_complex'):
            # complex embeddings require and return lengths
            inp, lengths = self.embeddings(inp, lengths)
        else:
            inp = self.embeddings(inp)

        hidden = self.init_hidden_for(inp)

        rnn_inp = inp
        if lengths is not None:  # pack if lengths given
            rnn_inp, unsort = pack_sort(rnn_inp, lengths.data)

        outs, hidden = self.rnn(rnn_inp, hidden)

        if lengths is not None:  # unpack & unsort
            outs, _ = unpack(outs)
            outs = outs[:, unsort]
            if self.cell.startswith('LSTM'):
                hidden = (hidden[0][:, unsort, :], hidden[1][:, unsort, :])
            else:
                hidden = hidden[:, unsort, :]

        if self.bidi:
            # BiRNN encoder outputs:   (num_layers * 2 x batch x hid_dim)
            # but RNN decoder expects: (num_layers x batch x hid_dim * 2)
            if self.cell.startswith('LSTM'):
                hidden = (repackage_bidi(hidden[0]),
                          repackage_bidi(hidden[1]))
            else:
                hidden = repackage_bidi(hidden)

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


class CNNEncoder(BaseEncoder):

    def __init__(self, embeddings, filter_sizes, ):
        super(CNNEncoder, self).__init__()

    def forward(inp):
        pass

    @property
    def encoding_size(self):
        pass


def GRLWrapper(EncoderBaseClass):
    def __init__(self, cond_dims, cond_vocabs, *args, **kwargs):
        EncoderBaseClass.__init__(self, *args, **kwargs)

        if len(cond_dims) != len(cond_vocabs):
            raise ValueError("cond_dims & cond_vocabs must be same length")

        encoding_dim, _ = self.encoding_size
        if encoding_dim > 2:
            raise ValueError("GRLRNNEncoder can't regularize 3D summaries")

        # MLPs regularizing on input conditions
        grls = nn.ModuleList()
        _, hid_dim = self.encoding_size  # size of the encoder output
        for cond_vocab, cond_dim in zip(cond_vocabs, cond_dims):
            grls.append(MLP(hid_dim, hid_dim, cond_vocab))

        self.add_module('grls', grls)

    def loss(self, out, conds, test=False):
        grl_loss = []
        for cond, grl in zip(conds, self.grls):
            cond_out = F.log_softmax(grad_reverse(grl(out)), 1)
            grl_loss.append(F.nll_loss(cond_out, cond, size_average=True))

        if not test:
            (sum(grl_loss) / len(self.grls)).backward(retain_graph=True)

        return [l.data[0] for l in grl_loss]

    return type('GRL{}'.format(EncoderBaseClass.__name__),
                (EncoderBaseClass,),
                {'__init__': __init__,
                 'loss': loss,
                 'conditional': property(lambda self: True)})


GRLRNNEncoder = GRLWrapper(RNNEncoder)
GRLCNNEncoder = GRLWrapper(CNNEncoder)


if __name__ == '__main__':
    import os
    from seqmod.misc import Dict
    from seqmod.modules.embedding import Embedding
    text = open(os.path.realpath(__file__)).read().split()
    emb = Embedding.from_dict(Dict().fit(text), 100)
    GRLRNNEncoder([10], [10], emb, 10, 1, 'LSTM', summary='mean')
