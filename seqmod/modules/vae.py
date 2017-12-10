
import math
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import seqmod.utils as u
from seqmod.modules.custom import word_dropout, StackedLSTM, StackedGRU, Highway
from seqmod.misc import Beam
from seqmod.modules.encoder_decoder import Encoder


def generic_sigmoid(a=1, b=1, c=1):
    return lambda x: a / (1 + b * math.exp(-x * c))


def kl_sigmoid_annealing_schedule(inflection, steepness=3):
    b = 10 ** steepness
    return generic_sigmoid(b=b, c=math.log(b) / inflection)


def kl_linear_annealing_schedule(max_steps):
    return lambda x: x / max_steps


# KL-divergence loss
def KL_loss(mu, logvar):
    """
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class EncoderVAE(Encoder):
    def __init__(self, z_dim, *args, summary='mean', **kwargs):
        super(EncoderVAE, self).__init__(*args, **kwargs)
        # dimension of the hidden output of the encoder
        self.summary = summary
        self.z_dim = z_dim
        summaries = ('attention', 'mean', 'mean-concat', 'last')
        enc_dim = self.hid_dim * self.num_dirs

        if self.summary.lower() not in summaries:
            raise ValueError('summary must be one of {}'.format(summaries))

        if self.summary == 'mean-concat':
            enc_dim *= 2

        elif self.summary == 'attention':
            # concatenation of input embedding and hidden activation
            self.attention = nn.Linear(self.in_dim + enc_dim, enc_dim)

        self.Q_mu = nn.Linear(enc_dim, self.z_dim)
        self.Q_logvar = nn.Linear(enc_dim, self.z_dim)

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
        out, _ = super(EncoderVAE, self).forward(inp, hidden, **kwargs)

        if self.summary == 'last':
            context = out[-1]

        if self.summary == 'mean':
            context = out.mean(0)

        elif self.summary == 'mean-concat':
            context = torch.cat([out[:-1].mean(0), out[-1]], 1)

        elif self.summary == 'attention':
            seq_len, batch_size, _ = inp.size()
            # combine across feature dimension and project to hid_dim
            weights = self.attention(
                torch.cat(
                    [inp.view(-1, self.in_dim),
                     out.view(-1, self.hid_dim * self.num_dirs)], 1))
            # apply softmax over the seq_len dimension
            weights = F.softmax(weights.view(seq_len, batch_size, -1), 0)
            # weighted sum of encoder outputs (feature-wise)
            context = (weights * out).sum(0)

        mu, logvar = self.Q_mu(context), self.Q_logvar(context)

        return mu, logvar


class DecoderVAE(nn.Module):
    def __init__(self, z_dim, emb_dim, hid_dim, num_layers=1, cell='LSTM',
                 dropout=0.0, add_z=False, train_init=False):
        self.z_dim = z_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.cell = cell
        self.num_layers = num_layers
        self.dropout = dropout
        self.add_z = add_z
        self.train_init = train_init
        super(DecoderVAE, self).__init__()

        # train initial
        if self.train_init:
            self.h_0 = nn.Parameter(
                torch.Tensor(self.num_layers, 1, self.hid_dim).zero_())

        # add highway projection
        if self.add_z:
            self.z_proj = Highway(z_dim, num_layers=1)  # add option

        # rnn
        in_dim = emb_dim if not add_z else z_dim + emb_dim
        stacked = StackedLSTM if self.cell == 'LSTM' else StackedGRU
        self.rnn_step = stacked(
            self.num_layers, in_dim, self.hid_dim, dropout=dropout)

    def init_hidden_for(self, z):
        batch_size = z.size(0)
        size = (self.num_layers, batch_size, self.hid_dim)

        if self.train_init:
            h_0 = self.h_0.repeat(1, batch_size, 1)
        else:
            h_0 = z.data.new(*size).zero_()
            h_0 = Variable(h_0, volatile=not self.training)

        if self.cell.startswith('LSTM'):
            c_0 = z.data.new(*size).zero_()
            c_0 = Variable(c_0, volatile=not self.training)
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
            assert z is not None, "z must be given when add_z is set to True"
            prev = torch.cat([prev, self.z_proj(z)], 1)

        out, hidden = self.rnn_step(prev, hidden)

        return out, hidden


class SequenceVAE(nn.Module):
    def __init__(self, emb_dim, hid_dim, z_dim, src_dict, num_layers=1,
                 cell='LSTM', bidi=True, dropout=0.0, word_dropout=0.0,
                 train_init=False, add_z=False, tie_weights=False,
                 summary='mean-concat',
                 kl_schedule=kl_sigmoid_annealing_schedule(inflection=5000)):
        self.hid_dim, self.num_layers = hid_dim, num_layers
        self.add_z = add_z
        self.src_dict = src_dict
        self.cell = cell
        vocab_size = len(src_dict)
        super(SequenceVAE, self).__init__()

        # Training stuff
        self.nll_weight = torch.ones(vocab_size)
        self.nll_weight[self.src_dict.get_pad()] = 0
        self.kl_weight = 0.0
        self.kl_schedule = kl_schedule

        # Word_dropout
        self.word_dropout = word_dropout
        self.target_code = self.src_dict.get_unk()
        self.reserved_codes = (self.src_dict.get_bos(),
                               self.src_dict.get_eos(),
                               self.src_dict.get_pad())

        # Embedding layer(s)
        self.embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=self.src_dict.get_pad())

        # Encoder
        self.encoder = EncoderVAE(
            z_dim, emb_dim, hid_dim, num_layers,
            summary=summary, cell=cell, bidi=bidi, dropout=dropout)

        # Decoder
        self.decoder = DecoderVAE(
            z_dim, emb_dim, hid_dim, num_layers=num_layers, cell=cell,
            train_init=train_init, add_z=add_z, dropout=dropout)

        # Projection
        if tie_weights:
            projection = nn.Linear(emb_dim, vocab_size)
            projection.weight = self.embeddings.weight
            if emb_dim != hid_dim:
                logging.warn("When tying weights, output layer and "
                             "embedding layer should have equal size. "
                             "A projection layer will be insterted.")
                tied_projection = nn.Linear(hid_dim, emb_dim)
                self.proj = nn.Sequential(tied_projection, projection)
            else:
                self.proj = projection
        else:
            self.proj = nn.Linear(hid_dim, vocab_size)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_embeddings(self, weight):
        emb_elements = self.embeddings.weight.data.nelement()
        mismatch_msg = "Expected " + str(emb_elements) + "elements but got {}"

        if isinstance(weight, np.ndarray):
            assert emb_elements == weight.size, \
                mismatch_msg.format(weight.size)
            self.embeddings.weight.data = torch.Tensor(weight)

        elif isinstance(weight, torch.Tensor):
            assert emb_elements == weight.nelement(), \
                mismatch_msg.format(weight.nelement())
            self.embeddings.weight.data = weight

        elif isinstance(weight, nn.Parameter):
            assert emb_elements == weight.nelement(), \
                mismatch_msg.format(weight.nelement())
            self.embeddings.weight = weight

        else:
            raise ValueError("Unknown weight type [{}]".format(type(weight)))

    def project(self, dec_outs):
        """
        Parameters:
        -----------
        dec_outs: (seq_len x batch x hid_dim), decoder output

        Returns: dec_logs (seq_len * batch x vocab_size)
        """
        seq_len, batch, hid_dim = dec_outs.size()
        dec_outs = dec_outs.view(batch * seq_len, hid_dim)

        return F.log_softmax(self.proj(dec_outs))

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
        hidden, dec_outs = self.decoder.init_hidden_for(z), []
        # apply word dropout on the conditioning targets
        trg = word_dropout(
            trg, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)

        for emb_t in self.embeddings(trg).chunk(trg.size(0)):
            # rnn
            dec_out, hidden = self.decoder(emb_t.squeeze(0), hidden, z=z)
            dec_outs.append(dec_out)

        dec_outs = torch.stack(dec_outs)

        return dec_outs, mu, logvar

    def loss(self, batch_data, test=False, split=25):
        """
        Compute loss, eventually backpropagate and return losses and batch size
        for speed monitoring
        """
        pad, eos = self.src_dict.get_pad(), self.src_dict.get_eos()
        src, _ = batch_data

        # remove <eos> from decoder targets dealing with different <pad> sizes
        dec_trg = Variable(u.map_index(src[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_trg = src[1:]
        # preds (batch * seq_len x vocab)
        dec_outs, mu, logvar = self(src, dec_trg)

        # compute loss
        weight = self.nll_weight
        if next(self.parameters()).is_cuda:
            weight = weight.cuda()

        shard_data = {'out': dec_outs, 'trg': loss_trg}
        rec_examples = src.data.ne(pad).int().sum()
        rec_loss = 0

        for shard in u.shards(shard_data, size=split, test=test):
            out, trg = shard['out'], shard['trg'].view(-1)
            shard_loss = F.nll_loss(
                self.project(out), trg, weight=weight, size_average=False
            ) / rec_examples       # normalize by number of words
            rec_loss += shard_loss  # accumulate (for report)

            if not test:
                shard_loss.backward(retain_graph=True)

        # Normalize by same number of elements as in reconstruction
        kl_loss = self.kl_weight * (KL_loss(mu, logvar) / rec_examples)
        if not test:
            kl_loss.backward()

        return (rec_loss.data[0], kl_loss.data[0]), rec_examples

    def generate(self, inp=None, z_params=None, beam_width=5, method='beam',
                 max_inp_len=2, max_len=20, **kwargs):
        """
        inp: None or (seq_len x batch_size). Input sequences to be encoded.
            It will be ignored if `z_params` is not None.
        z_params: None or tuple(mu, logvar). If given, decoding will be done
            from the latent code and `inp` will be ignored.
            - mu: (batch_size x z_dim)
            - logvar: (batch_size x z_dim)
        """
        if inp is None and z_params is None:
            raise ValueError("At least one of (inp, z_params) must be given")

        # Encoder
        if z_params is None:
            mu, logvar = self.encoder(self.embeddings(inp))
            max_len = len(inp) * max_inp_len
        else:
            mu, logvar = z_params

        # Sample from the hidden code
        z = self.encoder.reparametrize(mu, logvar)

        if method == 'argmax':
            return self.argmax(z, max_len)

        elif method == 'beam':
            return self.beam(z, max_len, beam_width)

    def beam(self, z, max_len, width):
        # local variables
        eos, bos = self.src_dict.get_eos(), self.src_dict.get_bos()
        batch_size = z.size(0)
        # output variables
        scores, preds = [], []

        for idx in range(batch_size):
            batch_z = z[idx].unsqueeze(0).repeat(width, 1)
            hidden = self.decoder.init_hidden_for(batch_z)
            beam = Beam(width, bos, eos=eos, gpu=self.is_cuda())

            while beam.active and len(beam) < max_len:
                # advance beam
                prev = Variable(beam.get_current_state().unsqueeze(0),
                                volatile=True)
                dec_out, hidden = self.decoder(
                    self.embeddings(prev).squeeze(0), hidden, z=batch_z)
                logprobs = self.project(dec_out.unsqueeze(0))
                beam.advance(logprobs.data)

                # repackage according to source beam
                source_beam = beam.get_source_beam()
                if self.cell.startswith('LSTM'):
                    hidden = (u.swap(hidden[0], 1, source_beam),
                              u.swap(hidden[1], 1, source_beam))
                else:
                    hidden = u.swap(hidden, 1, source_beam)

            batch_scores, batch_preds = beam.decode(n=1)
            scores.append(batch_scores[0]), preds.append(batch_preds[0])

        return scores, preds

    def argmax(self, z, max_len):
        # local variables
        eos, bos = self.src_dict.get_eos(), self.src_dict.get_bos()
        batch = z.size(0)
        # output variables
        scores, preds, mask = 0, [], z.data.new(batch).long() + 1
        # model inputs
        hidden = self.decoder.init_hidden_for(z)
        prev = Variable(z.data.new(batch).zero_().long() + bos, volatile=True)

        for _ in range(max_len):
            prev_emb = self.embeddings(prev).squeeze(0)
            dec_out, hidden = self.decoder(prev_emb, hidden, z=z)
            dec_out = self.project(dec_out.unsqueeze(0))

            score, pred = dec_out.max(1)
            scores += score.squeeze().data
            preds.append(pred.squeeze().data)
            prev = pred

            mask = mask * (pred.squeeze().data[0] != eos)
            if mask.int().sum() == 0:
                break

        return scores.tolist(), torch.stack(preds).transpose(0, 1).tolist()
