
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.modules import conv_utils


class CNNTextEncoder(nn.Module):
    """
    'Convolutional Neural Networks for Sentence Classification'
    http://www.aclweb.org/anthology/D14-1181

    Parameters:
    -----------
    out_channels: number of channels for all kernels
       This can't vary across filters their output congruency
    kernel_sizes: tuple of int, one for each kernel, i.e. number of kernels
       will equal the length of this argument. In practice, this parameter
       only controls the width of each filter since the height is fixed to
       the dimension of the embeddings to ensure a kernel output height of 1
       (the kernel output width will vary depending on the input seq_len,
       but will as well get max pooled over)
    act: str, activation function after the convolution
    dropout: float
    """
    def __init__(self, embeddings, dropout=0.0, conv_type='wide', ktop=1,
                 out_channels=100, kernel_sizes=(5, 4, 3), act='relu', **kwargs):

        self.emb_dim = embeddings.embedding_dim
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.act = act
        self.dropout = dropout
        self.conv_type = conv_type
        self.ktop = ktop

        super(CNNTextEncoder, self).__init__()

        # Embeddings
        self.embeddings = embeddings

        # CNN
        padding, conv = 0, []
        C_i, C_o, H = 1, out_channels, self.emb_dim
        for W in self.kernel_sizes:
            padding = (0, conv_utils.get_padding(W, self.conv_type))
            conv.append(nn.Conv2d(C_i, C_o, (H, W), padding=padding))
        self.conv = nn.ModuleList(conv)

    @property
    def encoding_size(self):
        return 2, len(self.kernel_sizes) * self.out_channels * self.ktop

    def forward(self, inp, lengths=None):
        # Embedding
        emb = self.embeddings(inp)
        emb = emb.transpose(0, 1)  # (batch x seq_len x emb_dim)
        emb = emb.transpose(1, 2)  # (batch x emb_dim x seq_len)
        emb = emb.unsqueeze(1)     # (batch x 1 x emb_dim x seq_len)

        # CNN
        conv_outs = []
        for conv in self.conv:
            conv_out = conv(emb).squeeze(2)
            # (batch x C_o x seq_len)
            conv_out = getattr(F, self.act)(conv_out)
            # (batch x C_o x ktop): maxpool over seq_len
            conv_out = conv_utils.global_kmax_pool(conv_out, self.ktop, dim=2)
            # conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            batch, C_o, ktop = conv_out.size()
            conv_outs.append(conv_out.view(batch, C_o * ktop))

        conv_outs = torch.cat(conv_outs, dim=1)

        return F.dropout(
            conv_outs, p=self.dropout, training=self.training)
