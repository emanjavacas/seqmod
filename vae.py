
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
        self.enc_dim = self.hid_dim * self.num_dirs * self.num_layers
        self.z_dim = z_dim
        self.Q_mu = nn.Linear(self.enc_dim, self.z_dim)
        self.Q_logvar = nn.Linear(self.enc_dim, self.z_dim)

    def reparametrize(self, mu, logvar):
        """
        z = mu + eps *. sqrt(exp(log(s^2)))

        The second term obtains interpreting logvar as the log-var of z
        and observing that sqrt(exp(s^2)) == exp(s^2/2)
        """
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(*std.size()).normal_())
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, inp, hidden=None, **kwargs):
        _, hidden = super(EncoderVAE, self).forward(inp, hidden, **kwargs)
        if self.cell.startswith('LSTM'):
            h_t, c_t = hidden
        else:
            h_t = hidden
        batch = h_t.size(1)
        h_t = h_t.t().view(batch, -1)
        mu, logvar = self.Q_mu(h_t), self.Q_logvar(h_t)
        return mu, logvar


class DecoderVAE(nn.Module):
    def __init__(self, z_dim, emb_dim, hid_dim, num_layers, cell,
                 dropout=0.0, maxout=0, add_prev=True, project_init=True):
        in_dim = emb_dim if not add_prev else hid_dim + emb_dim
        self.z_dim = z_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.add_prev = add_prev
        self.project_init = project_init
        assert project_init or (num_layers * hid_dim != z_dim), \
            "Cannot interpret z as initial hidden state. Use project_z."
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
        else:
            h_0 = z.view(self.num_layers, batch, self.hid_dim)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(z.data.new(self.num_layers, batch, self.hid_dim))
            return h_0, c_0
        else:
            return h_0

    def init_output_for(self, hidden):
        if self.cell.startswith('LSTM'):
            hidden = hidden[0]
        batch = hidden.size(1)
        data = hidden.data.new(batch, self.hid_dim).zero_()
        return Variable(data, requires_grad=False)

    def forward(self, prev, hidden, out=None, z=None):
        """
        Parameters:
        -----------
        prev: (batch x emb_dim). Conditioning item at current step.
        hidden: (batch x hid_dim). Hidden state at current step.
        out: None or (batch x hid_dim). Hidden state at previous step.
            This should be provided for all steps except first
            when `add_prev` is True.
        z: None or (batch x z_dim). Latent code for the input sequence.
            If it is provided, it will be used to condition the generation
            of each item in the sequence.
        """
        if self.add_prev:
            print(prev.size())
            inp = torch.cat([prev, out or self.init_output_for(hidden)])
        else:
            inp = prev
        out, hidden = self.rnn_step(inp, hidden)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out, hidden


class SequenceVAE(nn.Module):
    def __init__(self, num_layers, emb_dim, hid_dim, z_dim, src_dict,
                 cell='LSTM', bidi=True, dropout=0.0, word_dropout=0.0,
                 tie_weights=False, project_on_tied_weights=False,
                 add_prev=True, add_z=False):
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
            add_prev=add_prev, dropout=dropout)

        # projection
        if tie_weights:
            project = nn.Linear(emb_dim, vocab_size)
            project.weight = self.embeddings.weight
            if not project_on_tied_weights:
                assert emb_dim == dec_hid_dim, \
                    "When tying weights, output projection and " + \
                    "embedding layer should have equal size"
                self.project = nn.Sequential(project, nn.LogSoftmax())
            else:
                project_tied = nn.Linear(dec_hid_dim, emb_dim)
                self.project = nn.Sequential(
                    project_tied, project, nn.LogSoftmax())
        else:
            self.project = nn.Sequential(
                nn.Linear(dec_hid_dim, vocab_size),
                nn.LogSoftmax())

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
        # encoder
        src = word_dropout(
            src, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        emb = self.embeddings(src)
        mu, logvar = self.encoder(emb)
        z = self.encoder.reparametrize(mu, logvar)
        # decoder
        hidden = self.decoder.init_hidden_for(z)
        dec_outs, dec_out = [], None
        z_cond = z if self.add_z else None
        for emb_t in self.embeddings(trg).chunk(trg.size(0)):
            dec_out, hidden = self.decoder(
                emb_t.squeeze(0), hidden, out=dec_out, z=z_cond)
            dec_outs.append(dec_out)
        # (batch_size x hid_dim * seq_len)
        batch = src.size(1)
        dec_outs = torch.stack(dec_outs).view(batch, -1)
        return self.project(dec_outs), mu, logvar

    def generate(self, inp=None, z_params=None, max_decode_len=2, **kwargs):
        """
        inp: None or (seq_len x 1). Input sequences to be encoded. It will
            be ignored if `z_params` is not None.
        z_params: None or tuple(mu, logvar). If given, decoding will be done
            from the latent code and `inp` will be ignored.
            - mu: (1 x z_dim)
            - logvar: (1 x z_dim)
        """
        assert inp or z_params, "At least one of (inp, z_params) must be given"
        # encoder
        if z_params is None:
            emb = self.embeddings(inp)
            mu, logvar = self.encoder(inp)
        else:
            mu, logvar = z_params
        # sample from the hidden code
        z_data = inp.data.new(self.z_dim).normal_(mu, logvar.mul(0.5).exp_())
        z = Variable(z_data, volatile=True)
        # decoder
        hidden = self.decoder.init_hidden_for(z)
        dec_outs, dec_out = [], None
        z_cond = z if self.add_z else None
        prev = Variable(inp.data.new([self.src_dict.get_bos()]), volatile=True)
        prev = prev.unsqueeze(0)
        for _ in range(len(inp) * max_decode_len):
            prev_emb = self.embeddings(prev).unsqueeze(0)
            dec_out, hidden = self.decoder(
                prev_emb, hidden, out=dec_out, z=z_cond)
            dec_outs.append(dec_out)
        dec_outs = torch.stack(dec_outs).view(1, -1)
        return self.project(dec_outs)
