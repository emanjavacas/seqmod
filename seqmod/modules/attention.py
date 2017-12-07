
import torch
import torch.nn as nn
import torch.nn.functional as F


def DotScorer(dec_out, enc_outs, **kwargs):
    return torch.bmm(enc_outs.transpose(0, 1), dec_out.unsqueeze(2)).squeeze(2)


class GeneralScorer(nn.Module):
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()

        self.W_a = nn.Linear(dim, dim, bias=False)

    def forward(self, dec_out, enc_outs, **kwargs):
        return DotScorer(self.W_a(dec_out), enc_outs)


class BahdanauScorer(nn.Module):
    def __init__(self, hid_dim, att_dim):
        super(BahdanauScorer, self).__init__()
        # params
        self.W_s = nn.Linear(hid_dim, att_dim, bias=False)
        self.W_t = nn.Linear(hid_dim, att_dim, bias=True)
        self.v_a = nn.Parameter(torch.Tensor(att_dim, 1))
        self.v_a.data.uniform_(-0.05, 0.05)

    def project_enc_outs(self, enc_outs):
        """
        mapping: (seq_len x batch x hid_dim) -> (seq_len x batch x att_dim)

        Returns:
        --------
        torch.Tensor (seq_len x batch x att_dim),
            Projection of encoder output onto attention space
        """
        seq_len, batch, hid_dim = enc_outs.size()
        return self.W_s(enc_outs.view(-1, hid_dim)).view(seq_len, batch, -1)

    def forward(self, dec_out, enc_outs, enc_att=None):
        if enc_att is None:
            # (seq_len x batch x att dim)
            enc_att = self.project_enc_outs(enc_outs)
        # (batch x att_dim)
        dec_att = self.W_t(dec_out)
        # (batch x seq_len x att_dim)
        dec_enc_att = F.tanh(enc_att + dec_att[None,:,:])
        # (batch x seq_len x att_dim) * (1 x att_dim x 1) -> (batch x seq_len)
        return (dec_enc_att.transpose(0, 1) @ self.v_a[None,:,:]).squeeze(2)


class Attention(nn.Module):
    def __init__(self, hid_dim, att_dim, scorer='general'):
        super(Attention, self).__init__()

        if hid_dim != att_dim and scorer != 'bahdanau':
            raise ValueError("Global attention requires attention size "
                             "equal to Encoder/Decoder hidden size")

        # Scorer
        if scorer.lower() == 'dot':
            self.scorer = DotScorer
        elif scorer.lower() == 'general':
            self.scorer = GeneralScorer(hid_dim)
        elif scorer.lower() == 'bahdanau':
            self.scorer = BahdanauScorer(hid_dim, att_dim)
        else:
            raise ValueError(
                "scorer must be one of ('dot', 'general', 'bahdanau') "
                "got {}".format(scorer))

        # Output layer (Luong 15. eq (5))
        self.linear_out = nn.Linear(
            hid_dim * 2, hid_dim, bias=scorer.lower() == 'bahdanau')

    def forward(self, dec_out, enc_outs, enc_att=None, mask=None):
        # weights ()
        weights = F.softmax(self.scorer(dec_out, enc_outs, enc_att=enc_att))
        # apply mask if given
        if mask is not None:
            weights.data.masked_fill_(mask, -float('inf'))
        # (eq 7)
        context = weights.unsqueeze(1).bmm(enc_outs.transpose(0, 1)).squeeze(1)
        # (eq 5) linear out combining context and hidden
        context = F.tanh(self.linear_out(torch.cat([context, dec_out], 1)))

        return context, weights
