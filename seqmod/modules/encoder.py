
import torch
import torch.nn as nn
import torch.nn.functional as F

from seqmod.modules.ff import grad_reverse, MLP, MaxOut


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
            cond_out = F.log_softmax(grl(grad_reverse(out)), 1)
            grl_loss.append(F.nll_loss(cond_out, cond, size_average=True))

        if not test:
            (sum(grl_loss) / len(self.grls)).backward(retain_graph=True)

        return [l.item() for l in grl_loss], cond.size(0)

    return type('GRL{}'.format(EncoderBaseClass.__name__),
                (EncoderBaseClass,),
                {'__init__': __init__,
                 'loss': loss,
                 'conditional': property(lambda self: True)})


class MaxoutWindowEncoder(BaseEncoder):
    """
    Pseudo-CNN encoding on top of embedding features with MaxOut activations
    """
    def __init__(self, embedding, layers, maxouts=2, downproj=None, dropout=0.0):
        self.layers = layers
        self.dropout = dropout
        super(MaxoutWindowEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        inp_dim = embedding.embedding_dim
        # down projection
        if downproj is not None:
            self.downproj = nn.Linear(inp_dim, downproj)
            inp_dim = downproj
        # maxouts
        self.maxouts = nn.ModuleList(
            [MaxOut(inp_dim * 3, inp_dim, maxouts)
             for _ in range(layers)])

        self.inp_dim = inp_dim

    def extract_window(self, inp):
        """
        Parameters:
        -----------
        - inp: (seq_len x batch_size x emb_dim)

        Returns:
        --------
        - output: (seq_len x batch_size x emb_dim * 3)
        """
        return torch.cat([
            F.pad(inp[:-1], (0, 0, 0, 0, 1, 0)),
            inp,
            F.pad(inp[1:],  (0, 0, 0, 0, 0, 1))
        ], dim=2)

    def forward(self, inp, **kwargs):
        """
        Parameters:
        -----------
        - inp (seq_len x batch)
        """
        seq_len, batch = inp.size()

        emb = self.embedding(inp)

        if hasattr(self, 'downproj'):
            emb = self.downproj(emb)

        for maxout in self.maxouts:
            emb = self.extract_window(emb)
            emb = F.dropout(emb, p=self.dropout, training=self.training)
            emb = maxout(emb.view(seq_len * batch, -1)).view(seq_len, batch, -1)

        return torch.cat([emb.mean(0), emb.max(0)[0]], 1)

    @property
    def encoding_size(self):
        return 2, self.inp_dim * 2
