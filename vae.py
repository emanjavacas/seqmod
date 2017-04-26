
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.custom import word_dropout, StackedRNN
from modules.encoder import Encoder
from modules import utils as u

from misc.beam_search import Beam


class EncoderVAE(Encoder):
    def __init__(self, z_dim, *args, **kwargs):
        super(EncoderVAE, self).__init__(*args, **kwargs)
        # dimension of the hidden output of the encoder
        self.enc_dim = self.hid_dim * self.num_dirs
        self.z_dim = z_dim
        self.Q_mu = nn.Linear(self.enc_dim, self.z_dim)
        self.Q_logvar = nn.Linear(self.enc_dim, self.z_dim)

    def reparametrize(self, mu, logvar):
        """
        z = mu + eps *. sqrt(exp(log(s^2)))

        The second term obtains interpreting logvar as the log-var of z
        and observing that sqrt(exp(s^2)) == exp(s^2/2)
        """
        eps = Variable(logvar.data.new(*logvar.size()).normal_())
        std = logvar.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def forward(self, inp, hidden=None, **kwargs):
        _, hidden = super(EncoderVAE, self).forward(inp, hidden, **kwargs)
        h_t = hidden[0] if self.cell.startswith('LSTM') else hidden
        batch = h_t.size(1)
        h_t = h_t.view(self.num_layers, self.num_dirs, batch, self.hid_dim)
        # only use last layer activations
        # h_t (batch x hid_dim * num_dirs)
        h_t = h_t[-1].t().contiguous().view(batch, -1)
        mu, logvar = self.Q_mu(h_t), self.Q_logvar(h_t)
        return mu, logvar


class DecoderVAE(nn.Module):
    def __init__(self, z_dim, emb_dim, hid_dim, num_layers, cell,
                 dropout=0.0, maxout=0, add_z=False, project_init=False):
        in_dim = emb_dim if not add_z else z_dim + emb_dim
        self.z_dim = z_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.add_z = add_z
        self.project_init = project_init
        assert project_init or (num_layers * hid_dim == z_dim), \
            "Cannot interpret z as initial hidden state. Use project_init."
        super(DecoderVAE, self).__init__()

        # project_init
        if self.project_init:
            self.project_z = nn.Linear(z_dim, self.hid_dim * self.num_layers)

        # rnn
        self.rnn_step = StackedRNN(
            self.num_layers, in_dim, self.hid_dim,
            cell=self.cell, dropout=dropout)

    def init_hidden_for(self, z):
        batch = z.size(0)
        if self.project_init:
            h_0 = self.project_z(z)
            h_0 = h_0.view(batch, self.num_layers, self.hid_dim).t()
        else:                 # rearrange z to match hidden cell shape
            h_0 = z.view(self.num_layers, batch, self.hid_dim)
        if self.cell.startswith('LSTM'):
            c_0 = z.data.new(self.num_layers, batch, self.hid_dim)
            c_0 = Variable(nn.init.xavier_uniform(c_0))
            return h_0, c_0
        else:
            return h_0

    def forward(self, prev, hidden, z=None):
        """
        Parameters:
        -----------
        prev: (batch x emb_dim). Conditioning item at current step.
        hidden: (batch x hid_dim). Hidden state at current step.
        z: None or (batch x z_dim). Latent code for the input sequence.
            If it is provided, it will be used to condition the generation
            of each item in the sequence.
        """
        if self.add_z:
            assert z, "z must be given when add_z is set to True"
            inp = torch.cat([prev, z], 1)
        else:
            inp = prev
        out, hidden = self.rnn_step(inp, hidden)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out, hidden


class SequenceVAE(nn.Module):
    def __init__(self, emb_dim, hid_dim, z_dim, src_dict, num_layers=1,
                 cell='LSTM', bidi=True, dropout=0.0, word_dropout=0.0,
                 project_init=False, add_z=False,
                 tie_weights=False, project_on_tied_weights=False):
        if isinstance(hid_dim, tuple):
            enc_hid_dim, dec_hid_dim = hid_dim
        else:
            enc_hid_dim, dec_hid_dim = hid_dim, hid_dim
        if isinstance(num_layers, tuple):
            enc_num_layers, dec_num_layers = num_layers
        else:
            enc_num_layers, dec_num_layers = num_layers, num_layers
        self.add_z = add_z
        self.src_dict = src_dict
        vocab_size = len(src_dict)
        super(SequenceVAE, self).__init__()

        # word_dropout
        self.word_dropout = word_dropout
        self.target_code = self.src_dict.get_unk()
        self.reserved_codes = (self.src_dict.get_bos(),
                               self.src_dict.get_eos(),
                               self.src_dict.get_pad())

        # embedding layer(s)
        self.embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=self.src_dict.get_pad())

        # encoder
        self.encoder = EncoderVAE(
            z_dim, emb_dim, enc_hid_dim, enc_num_layers,
            cell=cell, bidi=bidi, dropout=dropout)

        # decoder
        self.decoder = DecoderVAE(
            z_dim, emb_dim, dec_hid_dim, dec_num_layers, cell,
            project_init=project_init, add_z=add_z, dropout=dropout)

        # projection
        if tie_weights:
            projection = nn.Linear(emb_dim, vocab_size)
            projection.weight = self.embeddings.weight
            if not project_on_tied_weights:
                assert emb_dim == dec_hid_dim, \
                    "When tying weights, output projection and " + \
                    "embedding layer should have equal size"
                self.out_proj = projection
            else:
                tied_projection = nn.Linear(dec_hid_dim, emb_dim)
                self.out_proj = nn.Sequential(tied_projection, projection)
        else:
            self.out_proj = nn.Linear(dec_hid_dim, vocab_size)

    def init_embeddings(self, weight):
        emb_elements = self.embeddings.weight.data.nelement()
        mismatch_msg = "Expected " + str(emb_elements) + "elements but got %d"
        if isinstance(weight, np.ndarray):
            assert emb_elements == weight.size, mismatch_msg % weight.size
            self.embeddings.weight.data = torch.Tensor(weight)
        elif isinstance(weight, torch.Tensor):
            assert emb_elements == weight.nelement(), \
                mismatch_msg % weight.nelement()
            self.embeddings.weight.data = weight
        elif isinstance(weight, nn.Parameter):
            assert emb_elements == weight.nelement(), \
                mismatch_msg % weight.nelement()
            self.embeddings.weight = weight
        else:
            raise ValueError("Unknown weight type [%s]" % type(weight))

    def project(self, dec_outs):
        """
        Parameters:
        -----------
        dec_outs: (seq_len x batch x hid_dim), decoder output

        Returns: dec_logs (seq_len * batch x vocab_size)
        """
        seq_len, batch, hid_dim = dec_outs.size()
        dec_outs = self.out_proj(dec_outs.view(batch * seq_len, hid_dim))
        dec_logs = F.log_softmax(dec_outs)
        return dec_logs

    def forward(self, src, trg, labels=None):
        """
        Parameters:
        -----------

        inp: (seq_len x batch). Input batch of sentences to be encoded.
            It is assumed that inp has <bos> and <eos> symbols.
        labels: None or (batch x num_labels). To be used by conditional VAEs.

        Returns:
        --------
        preds: (batch x vocab_size * seq_len)
        mu: (batch x z_dim)
        logvar: (batch x z_dim)
        """
        # - encoder
        emb = self.embeddings(src)
        mu, logvar = self.encoder(emb)
        z = self.encoder.reparametrize(mu, logvar)
        # - decoder
        hidden = self.decoder.init_hidden_for(z)
        dec_outs, z_cond = [], z if self.add_z else None
        # apply word dropout on the conditioning targets
        trg = word_dropout(
            trg, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        for emb_t in self.embeddings(trg).chunk(trg.size(0)):
            # rnn
            dec_out, hidden = self.decoder(emb_t.squeeze(0), hidden, z=z_cond)
            dec_outs.append(dec_out)
        dec_outs = torch.stack(dec_outs)
        return self.project(dec_outs), mu, logvar

    def generate(self, inp=None, z_params=None, max_inp_len=2, max_len=20, **kwargs):
        """
        inp: None or (seq_len x 1). Input sequences to be encoded. It will
            be ignored if `z_params` is not None.
        z_params: None or tuple(mu, logvar). If given, decoding will be done
            from the latent code and `inp` will be ignored.
            - mu: (1 x z_dim)
            - logvar: (1 x z_dim)
        """
        assert inp is not None or z_params is not None, \
            "At least one of (inp, z_params) must be given"
        # encoder
        if z_params is None:
            mu, logvar = self.encoder(self.embeddings(inp))
            max_len = len(inp) * max_inp_len
        else:
            mu, logvar = z_params
            max_len = max_len
        # sample from the hidden code
        z = self.encoder.reparametrize(mu, logvar)
        # decoder
        hidden = self.decoder.init_hidden_for(z)
        scores, preds, z_cond = [], [], z if self.add_z else None
        prev_data = mu.data.new([self.src_dict.get_bos()]).long()
        prev = Variable(prev_data, volatile=True).unsqueeze(0)
        for _ in range(max_len):
            prev_emb = self.embeddings(prev).squeeze(0)
            dec_out, hidden = self.decoder(prev_emb, hidden, z=z_cond)
            dec_out = self.project(dec_out.unsqueeze(0))
            score, pred = dec_out.max(1)
            scores.append(score.squeeze().data[0])
            preds.append(pred.squeeze().data[0])
            prev = pred
            if prev.data.eq(self.src_dict.get_eos()).nonzero().nelement() > 0:
                break
        return [sum(scores)], [preds]
