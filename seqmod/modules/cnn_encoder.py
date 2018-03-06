
import torch
import torch.nn as nn
import torch.nn.functional as F

from seqmod.modules.conv_utils import get_padding
from seqmod.modules.encoder import BaseEncoder


class StackedResidualCNN(nn.Module):
    def __init__(self, inp_dim, kernel_size, layers,
                 conv_type='wide', act='gated', dropout=0.0):
        self.inp_dim = inp_dim
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.act = act
        self.dropout = dropout
        super(StackedResidualCNN, self).__init__()

        padding = (get_padding(kernel_size, conv_type), 0)
        convs = nn.ModuleList()
        for i in range(layers):
            if act == 'gated':
                conv = nn.Conv2d(inp_dim, 2*inp_dim, (kernel_size, 1), padding=padding)
            else:
                conv = nn.Conv2d(inp_dim, inp_dim, (kernel_size, 1), padding=padding)
            convs.append(conv)

        self.convs = convs

    def custom_init(self):
        for conv in self.convs:
            torch.nn.init.xavier_uniform(
                conv.weight, gain=(4 * (1 - self.dropout)) ** 0.5)

    def forward(self, inp, lengths=None):
        for conv in self.convs:
            if self.act == 'gated':
                # dropout on input to conv
                output = conv(F.dropout(inp, p=self.dropout, training=self.training))
                output, gate = output.split(int(output.size(1) / 2), dim=1)
                output = output * F.sigmoid(gate)
                output *= (0.5 ** 0.5)  # scale up activation to preserve variance
            else:
                output = getattr(F, self.act)(inp)

            output = inp + output   # residual connection
            inp = output

        return output


class CNNEncoder(BaseEncoder):
    """
    Stacked CNNEncoder
    """
    def __init__(self, embeddings, hid_dim, kernel_size, layers, **kwargs):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings

        if embeddings.embedding_dim != hid_dim:
            # projection from embedding dim to cnn input dimension
            self.linear = nn.Linear(embeddings.embedding_dim, hid_dim)

        self.conv = StackedResidualCNN(hid_dim, kernel_size, layers, **kwargs)

    def forward(self, inp, lengths=None):
        # (seq_len x batch x hid_dim)
        emb = self.embeddings(inp)
        if hasattr(self, 'linear'):
            emb = self.linear(emb)
        emb = emb.transpose(0, 1)  # (batch x seq_len x hid_dim)
        emb = emb.transpose(1, 2)  # (batch x hid_dim x seq_len)
        emb = emb.unsqueeze(3)     # (batch x hid_dim x seq_len x 1)

        output = self.conv(emb)     # (batch x hid_dim x seq_len x 1)
        output = output.squeeze(3)  # (batch x hid_dim x seq_len)
        # (seq_len x batch x hid_dim)
        output = output.transpose(0, 1).contiguous() \
                       .transpose(0, 2).contiguous()

        return output, None

    @property
    def encoding_size(self):
        return 3, self.conv.inp_dim
