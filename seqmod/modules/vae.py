
import torch
import torch.nn as nn
from torch.autograd import Variable

import seqmod.utils as u

from seqmod.modules.ff import Highway
from seqmod.misc import inflection_sigmoid, linear
from seqmod.modules.encoder_decoder import RNNEncoder
from seqmod.modules.decoder import RNNDecoder, State
from seqmod.modules.encoder_decoder import EncoderDecoder, make_embeddings


def kl_sigmoid_annealing_schedule(inflection, steepness=3):
    return inflection_sigmoid(inflection, steepness)


def kl_linear_annealing_schedule(max_steps):
    return linear(max_steps)


def KL_loss(mu, logvar):
    """
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reparametrize(mu, logvar):
    """
    z = mu + eps *. sqrt(exp(log(s^2)))

    The second term obtains interpreting logvar as the log-var of z
    and observing that sqrt(exp(s^2)) == exp(s^2/2)
    """
    eps = Variable(logvar.data.new(*logvar.size()).normal_())
    std = logvar.mul(0.5).exp_()
    return eps.mul(std).add_(mu)


class VAERNNEncoder(RNNEncoder):
    def __init__(self, z_dim, *args, **kwargs):
        super(VAERNNEncoder, self).__init__(*args, **kwargs)
        if self.summary == 'full':
            raise ValueError("VAE can't use full encoder summary")

        self.kl_weight = 0.0

        self.z_dim = z_dim
        _, enc_dim = self.encoding_size

        self.Q_mu = nn.Linear(enc_dim, self.z_dim)
        self.Q_logvar = nn.Linear(enc_dim, self.z_dim)

    def forward(self, inp, hidden=None, **kwargs):
        context, _ = super(VAERNNEncoder, self).forward(inp, hidden, **kwargs)
        mu, logvar = self.Q_mu(context), self.Q_logvar(context)
        return (mu, logvar), None

    def loss(self, enc_outs, _, test=False):
        (mu, logvar) = enc_outs
        num_examples = mu.size(0)

        kl_loss = self.kl_weight * (KL_loss(mu, logvar) / num_examples)

        if not test:
            kl_loss.backward(retain_graph=True)

        return [kl_loss.data[0]], num_examples  # encoder loss is a list


class VAERNNDecoder(RNNDecoder):
    def __init__(self, z_dim, *args, add_z=False, **kwargs):
        super(VAERNNDecoder, self).__init__(*args, **kwargs)
        self.add_z = add_z

        if self.has_attention:
            raise ValueError("Attentional VAE is ill-defined")

        if self.input_feed:
            raise ValueError("Input feeding for VAE is ill-defined")

        if self.add_z:
            # add projection
            self.z_proj = Highway(z_dim, num_layers=2)
            # rebuild rnn
            self.rnn = self.build_rnn(
                self.num_layers,
                self.rnn.in_dim + z_dim, # append z dim to input
                self.hid_dim,
                self.cell,
                self.dropout)

    def init_hidden_for(self, z):
        batch_size = z.size(0)
        size = (self.num_layers, batch_size, self.hid_dim)

        if self.train_init:
            h_0 = self.h_0.repeat(1, batch_size, 1)
        else:
            h_0 = z.data.new(*size).zero_()
            h_0 = Variable(h_0, volatile=not self.training)

        if self.add_init_jitter:
            h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

        if self.cell.startswith('LSTM'):
            return h_0, torch.zeros_like(h_0)
        else:
            return h_0

    def init_state(self, enc_outs, enc_hidden, src_lengths, conds=None, **kwargs):
        mu, logvar = enc_outs
        z = reparametrize(mu, logvar)

        hidden = self.init_hidden_for(z)

        if self.conditional:
            if conds is None:
                raise ValueError("Conditional decoder requires `conds`")
            conds = torch.cat(
                [emb(c) for c, emb in zip(conds, self.cond_embs)], 1)

        return VAEDecoderState(z, hidden, conds=conds)

    def forward(self, inp, state):
        """
        Parameters:
        -----------
        prev: (batch x emb_dim). Conditioning item at current step.
        hidden: (batch x hid_dim). Hidden state at current step.
        z: None or (batch x z_dim). Latent code for the input sequence.
            If it is provided, it will be used to condition the generation
            of each item in the sequence.
        """
        inp = self.embeddings(inp)

        if self.add_z:
            inp = torch.cat([inp, self.z_proj(state.z)], 1)

        if self.conditional:
            inp = torch.cat([inp, *state.conds], 1)

        out, hidden = self.rnn(inp, state.hidden)

        # update state
        state.hidden = hidden

        return out, None


class VAEDecoderState(State):
    def __init__(self, z, hidden, conds=None):
        self.z = z
        self.hidden = hidden
        self.conds = conds

    def expand_along_beam(self, width):
        self.z = self.z.repeat(width, 1)

        if isinstance(self.hidden, tuple):
            hidden = (self.hidden[0].repeat(1, width, 1),
                      self.hidden[1].repeat(1, width, 1))
        else:
            hidden = self.hidden.repeat(1, width, 1)
        self.hidden = hidden

        if self.conds is not None:
            self.conds = self.conds.repeat(width, 1)

    def reorder_beam(self, beam_ids):
        if isinstance(self.hidden, tuple):
            hidden = (u.swap(self.hidden[0], 1, beam_ids),
                      u.swap(self.hidden[1], 1, beam_ids))
        else:
            hidden = u.swap(self.hidden, 1, beam_ids)
        self.hidden = hidden


def make_vae_encoder_decoder(
        z_dim,
        num_layers,
        emb_dim,
        hid_dim,
        src_dict,
        cell='LSTM',
        bidi=True,
        encoder_summary='inner-attention',
        dropout=0.0,
        word_dropout=0.0,
        add_z=True,
        deepout_layers=0,
        deepout_act='ReLU',
        tie_weights=False,
        train_init=False,
        add_init_jitter=False,
        cond_dims=None,
        cond_vocabs=None
):
    if encoder_summary == 'full':
        raise ValueError("VAE encoder can't use full summaries")

    src_embeddings, trg_embeddings = make_embeddings(
        src_dict, None, emb_dim, word_dropout)

    encoder = VAERNNEncoder(z_dim, src_embeddings, hid_dim, num_layers,
                            cell, bidi=bidi, dropout=dropout,
                            summary=encoder_summary,
                            train_init=train_init,
                            add_init_jitter=add_init_jitter)

    decoder = VAERNNDecoder(z_dim, trg_embeddings, hid_dim, num_layers,
                            cell, add_z=add_z, dropout=dropout,
                            deepout_layers=deepout_layers, deepout_act=deepout_act,
                            tie_weights=tie_weights, train_init=train_init,
                            add_init_jitter=add_init_jitter, cond_dims=cond_dims,
                            cond_vocabs=cond_vocabs)

    return EncoderDecoder(encoder, decoder)
