
import math

import torch.nn as nn
import torch.nn.functional as F

from seqmod.modules.encoder import BaseEncoder
from seqmod.modules import conv_utils


class DCNNEncoder(BaseEncoder):
    """
    Implementation of 'A Convolutional NN for Modelling Sentences'
    https://arxiv.org/pdf/1404.2188.pdf

    Warning: from the paper it seems that the convolution is 2D:
        input matrix DxS is convolved with filter DxM (where D is the
        input embedding dimension, S is the sentence length and M is
        the filter size). Usually, this results in an output of 1x(S-M+1)
        since each filter activation is a sum of matrix-matrix element-wise
        multiplications (plus bias). However, in the paper the convolution
        is describe row-wise (each row in the filter is convolved with each
        input row) resulting in an output of Dx(S-M+1).

    Multi-layer CNN with Dynamic K-max pooling and folding.

    Convolutions between 1d-filters (d x m) and the embedding sentence
    matrix (d x s) are applied in 2-d yielding a matrix (d x (s + m - 1))
    or (d x (s - m + 1) depending on whether wide or narrow
    convolutions are used (the difference being in using padding or not).
    After each convolutional layer, the top k features of the resulting
    feature map are taken row-wise (e.g. the number of top k operations
    is equal to the embedding dimension d) resulting in a subsampling down
    to k. k is dynamically computed by a non-learned function (see ktop).
    """
    def __init__(self, embeddings, dropout=0.0,
                 kernel_sizes=(7, 5), out_channels=(6, 14),
                 ktop=4, folding_factor=2, conv_type='wide', act='tanh',
                 **kwargs):

        if len(kernel_sizes) != len(out_channels):
            raise ValueError("Need same number of feature maps for "
                             "`kernel_sizes` and `out_channels`")

        self.emb_dim = embeddings.embedding_dim

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.ktop = ktop
        self.folding_factor = folding_factor
        self.conv_type = conv_type
        self.act = act
        self.dropout = dropout

        super(DCNNEncoder, self).__init__()

        # Embeddings
        self.embeddings = embeddings

        # CNN
        C_i, H, conv = 1, self.emb_dim, []

        for l, (C_o, W) in enumerate(zip(self.out_channels, self.kernel_sizes)):
            # feature map H gets folded each layer
            H = math.ceil(H / self.folding_factor)
            padding = (0, conv_utils.get_padding(W, self.conv_type))
            conv.append(nn.Conv2d(C_i, C_o, (1, W), padding=padding))
            C_i = C_o
        self.conv = nn.ModuleList(conv)

        self.output_size = C_o * H * self.ktop

    @property
    def encoding_size(self):
        return 2, self.output_size

    def forward(self, inp, lengths=None):
        # Embedding
        if hasattr(self.embeddings, 'is_complex'):
            if lengths is None:
                raise ValueError("ComplexEmbedding requires char-level length info")
            emb, lengths = self.embeddings(inp, lengths)
        else:
            emb = self.embeddings(inp)

        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_in = emb
        for l, conv_layer in enumerate(self.conv):
            # - C_o: number of kernels run in parallel (#feat-maps)
            # - folded_dim: #feats per feature map (gets folded over every layer)
            # - s_i: output length of the conv at current layer

            # (batch x C_o x folded_dim x s_i)
            conv_out = conv_layer(conv_in)
            # (batch x C_o x (folded_dim / 2) x s_i)
            conv_out = conv_utils.folding(conv_out, factor=self.folding_factor)
            L, s = len(self.conv), conv_out.size(3)
            ktop = conv_utils.dynamic_ktop(l+1, L, s, self.ktop)
            # (batch x C_o x (folded_dim / 2) x ktop)
            conv_out = conv_utils.global_kmax_pool(conv_out, ktop)
            conv_out = getattr(F, self.act)(conv_out)
            conv_in = conv_out

        # Apply dropout to the penultimate layer after the last non-linearity
        conv_out = F.dropout(
            conv_out, p=self.dropout, training=self.training)

        return conv_out.view(conv_out.size(0), -1)  # (batch x proj input)


# seqlen, batch, emb = 10, 5, 25
# inp = torch.rand(seqlen, batch, emb)
# inp = inp.transpose(0, 1).transpose(1, 2).unsqueeze(1)  # batch, 1, emb, seqlen

# # conv2d input is (batch, C_in, H, W)
# nn.Conv2d(1, 6, (1, 3))(inp).size()   # batch, 6, 25, 8
# nn.Conv2d(1, 6, (emb, 3))(inp).size()  # batch, 6, 1, 8

# inp = torch.rand(seqlen, batch, emb)
# inp = inp.transpose(0, 1).transpose(1, 2).unsqueeze(2)
# # conv2d input is (batch, H, C_in, W)
# nn.Conv2d(emb, 6, (1, 3), groups=emb)().size()  # batch, 6, 1, 8
