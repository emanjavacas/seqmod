
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.custom import word_dropout, StackedRNN
from modules.encoder import Encoder
from modules import utils as u

from misc.beam_search import Beam
from misc.trainer import Trainer


class EncoderVAE(Encoder):
    def __init__(self, z_dim, *args, **kwargs):
        super(EncoderVAE, self).__init__(*args, **kwargs)
        # dimension of the hidden output of the encoder
        self.enc_dim = self.hid_dim * self.num_dirs * self.num_layers
        self.z_dim = z_dim
        self.Q_mu = nn.Linear(self.enc_dim, self.z_dim)
        self.Q_logvar = nn.Linear(self.enc_dim, self.z_dim)

    def encode(self, inp, hidden=None, **kwargs):
        _, hidden = super(EncoderVAE, self)(inp, hidden=hidden, **kwargs)
        if self.cell.startswith('LSTM'):
            h_t, c_t = hidden
        else:
            h_t = hidden
        mu, logvar = self.Q_mu(h_t), self.Q_logvar(h_t)
        return mu, logvar, hidden

    def reparametrize(self, mu, logvar):
        """
        z = mu + eps *. sqrt(exp(log(s^2)))

        The second term obtains interpreting logvar as the log-var of z
        and observing that sqrt(exp(s^2)) == exp(s^2/2)
        """
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(*std.size())).normal_()
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, inp, hidden=None, **kwargs):
        mu, logvar, hidden = self.encode(inp, hidden=hidden)
        z = self.reparametrize(mu, logvar)
        return z, hidden


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
            add_prev=add_prev, dropout=dropout, word_dropout=word_dropout)

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
        out: (batch x vocab_size * seq_len)
        z: (batch x z_dim)
        """
        # encoder
        src = word_dropout(
            src, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        trg_emb = self.embeddings(trg)  # TODO: make this efficient
        emb = self.embeddings(src)
        z, hidden = self.encoder(emb)
        # decoder
        hidden = self.decoder.init_hidden_for(hidden)
        dec_outs, dec_out = [], None
        z_cond = z if self.add_z else None
        for emb_t in trg_emb.chunk(trg_emb.size(0)):
            dec_out, hidden = self.decoder(
                emb_t, hidden, out=dec_out, z=z_cond)
            dec_outs.append(dec_out)
        # (batch_size x hid_dim * seq_len)
        batch = src.size(1)
        dec_outs = torch.stack(dec_outs).view(batch, -1)
        return self.project(dec_outs), z

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
        if z_params is not None:
            mu, logvar = z_params
            z = torch.randn(self.z_dim).mul(logvar.mul(0.5).exp_()).add_(mu)
        else:
            z, _ = self.encoder(inp)
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


def make_vae_criterion(vocab_size, pad):
    # Reconstruction loss
    weight = torch.ones(vocab_size)
    weight[pad] = 0
    log_loss = nn.NLLLoss(weight, size_average=False)

    # KL-divergence loss
    def KL_loss(mu, logvar):
        """
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        element = mu.pow(2).mul_(-1).add_(-logvar.exp_()).add_(1).add_(logvar)
        return torch.sum(element).mul_(-0.5)

    # loss function
    def loss(logs, targets, mu, logvar):
        return log_loss(logs, targets), KL_loss(mu, logvar)

    return loss


def generic_sigmoid(a=1, b=1, c=1):
    return lambda x: a/(1 + b * math.exp(-x * c))


def kl_annealing_schedule(inflection):
    # TODO: figure the params to get the right inflection point
    sigmoid = generic_sigmoid()

    def func(step):
        unshifted = sigmoid(step)
        return (unshifted - 0.5) * 2

    return func


class VAETrainer(Trainer):
    def __init__(self, *args, inflection_point=5000, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)
        self.kl_weight = 0.0    # start at 0
        self.kl_schedule = kl_annealing_schedule(inflection_point)
        self.epoch = 1
        self.size_average = False

    def format_loss(self, loss):
        return math.exp(min(loss, 100))

    def run_batch(self, batch_data, dataset='train', **kwargs):
        valid = dataset != 'train'
        pad, eos = self.model.src_dict.get_pad(), self.model.src_dict.get_eos()
        source, labels = batch_data
        # remove <eos> from decoder targets dealing with different <pad> sizes
        decode_targets = Variable(u.map_index(source[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_targets = source[1:].view(-1)
        # preds: (batch * seq_len x vocab)
        preds, mu, logvar = self.model(source, decode_targets, labels=labels)
        log_loss, kl_loss = self.criterion(preds, loss_targets, mu, logvar)
        if not valid:
            batch_size = source.size(1)
            log_loss.div(batch_size)
            loss = self.kl_weight * kl_loss + log_loss
            loss.backward()
            self.optimizer_step()
        return log_loss, kl_loss

    def num_batch_examples(self, batch_data):
        source, _ = batch_data
        return source[1:].data.ne(self.model.src_dict.get_pad()).sum()

    def on_batch_end(self, batch, loss):
        # reset kl weight
        self.kl_weight = self.kl_schedule(batch * self.epoch)

    def on_epoch_end(self, epoch, *args):
        self.epoch += 1
