
import torch.nn as nn
import torch.nn.functional as F

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

        return [l.data[0] for l in grl_loss], cond.size(0)

    return type('GRL{}'.format(EncoderBaseClass.__name__),
                (EncoderBaseClass,),
                {'__init__': __init__,
                 'loss': loss,
                 'conditional': property(lambda self: True)})


# legacy imports
from seqmod.modules.rnn_encoder import RNNEncoder
