
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import utils as u


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, dec_out, enc_outs, mask=None, **kwargs):
        """
        Parameters:
        -----------
        dec_out: (batch x hid_dim)

        enc_outs: (seq_len x batch x hid_dim (== att_dim))
        """
        # (batch x att x 1)
        dec_att = self.linear_in(dec_out).unsqueeze(2)
        # (batch x seq_len x att_dim) * (batch x att x 1) -> (batch x seq_len)
        weights = torch.bmm(enc_outs.t(), dec_att).squeeze(2)
        weights = self.softmax(weights)
        if mask is not None:
            weights.data.masked_fill_(mask, -math.inf)
        # (batch x 1 x seq_len) * (batch x seq_len x att) -> (batch x att)
        weighted = weights.unsqueeze(1).bmm(enc_outs.t()).squeeze(1)
        # (batch x att_dim * 2)
        combined = torch.cat([weighted, dec_out], 1)
        output = nn.functional.tanh(self.linear_out(combined))
        return output, weights


class BahdanauAttention(nn.Module):
    def __init__(self, att_dim, enc_hid_dim, dec_hid_dim):
        super(BahdanauAttention, self).__init__()
        self.att_dim = att_dim
        self.enc2att = nn.Linear(enc_hid_dim, att_dim, bias=False)
        self.dec2att = nn.Linear(dec_hid_dim, att_dim, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(att_dim))

    def project_enc_outs(self, enc_outs):
        """
        mapping: (seq_len x batch x hid_dim) -> (seq_len x batch x att_dim)

        Parameters:
        -----------
        enc_outs: torch.Tensor (seq_len x batch x hid_dim),
            output of encoder over seq_len input symbols

        Returns:
        --------
        enc_att: torch.Tensor (seq_len x batch x att_dim),
            Projection of encoder output onto attention space
        """
        return torch.cat([self.enc2att(i).unsqueeze(0) for i in enc_outs])

    def forward(self, dec_out, enc_outs, enc_att=None, mask=None, **kwargs):
        """
        Parameters:
        -----------
        dec_out: torch.Tensor (batch x dec_hid_dim)
            Output of decoder at current step

        enc_outs: torch.Tensor (seq_len x batch x enc_hid_dim)
            Output of encoder over the entire sequence

        enc_att: see self.project_enc_outs(self, enc_outs)

        Returns: contexts, weights
        --------
        context: torch.Tensor (batch x hid_dim)
            Matrix of context vectors, which are then combined in the
            computation of the model output at the present timestep

        weights: torch.Tensor (batch x seq_len)
            Attention weights in range [0, 1] for each input term
        """
        enc_att = enc_att or self.project_enc_outs(enc_outs)
        # enc_outputs * weights
        # weights: softmax(E) (seq_len x batch)
        # E: att_v (att_dim) * tanh(dec_att + enc_att) -> (seq_len x batch)
        # tanh(dec_out_att + enc_output_att) -> (seq_len x batch x att_dim)
        seq_len, batch, hid_dim = enc_att.size()
        # project current decoder output onto attention (batch_size x att_dim)
        dec_att = self.dec2att(dec_out)
        # elemwise addition of dec_out over enc_att
        # dec_enc_att: (batch x seq_len x att_dim)
        dec_enc_att = nn.functional.tanh(enc_att + u.tile(dec_att, seq_len))
        # dec_enc_att (seq_len x batch x att_dim) * att_v (att_dim)
        #   -> weights (batch x seq_len)
        weights = F.softmax(u.bmv(dec_enc_att.t(), self.att_v).squeeze(2))
        if mask is not None:
            weights.data.masked_fill_(mask, -math.inf)
        # enc_outs: (seq_len x batch x hid_dim) * weights (batch x seq_len)
        #   -> context: (batch x hid_dim)
        context = weights.unsqueeze(1).bmm(enc_outs.t()).squeeze(1)
        return context, weights
