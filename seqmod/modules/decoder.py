
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod.modules.rnn import StackedGRU, StackedLSTM
from seqmod.modules.ff import Highway
from seqmod.modules import attention
from seqmod import utils as u


class BaseDecoder(nn.Module):
    """
    Base abstract class
    """
    def init_state(self, context, enc_outs, **kwargs):
        """
        Parameters:
        -----------
        context: summary output from the decoder
        enc_outs: extra encoder output (e.g. sequential last layer activations
            for the RNN encoder)
        """
        raise NotImplementedError

    def forward(self, inp, enc_out, enc_hidden, lengths, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.LongTensor(seq_len x batch)
        enc_out: summary vector from the encoder
        enc_hidden: hidden state of encoder (needed when initializing RNN
            decoder with last RNN encoder hidden step)
        """
        raise NotImplementedError

    @property
    def conditional(self):
        """
        Whether the current decoder implements takes conditions
        """
        return False

    def build_output(self, hid_dim, deepout_layers, deepout_act, tie_weights):
        """
        Create output projection (from decoder output to softmax)
        """
        output = []

        if deepout_layers > 0:
            output.append(
                Highway(hid_dim, num_layers=deepout_layers,
                        activation=deepout_act))

        emb_dim = self.embeddings.embedding_dim
        vocab_size = self.embeddings.num_embeddings

        if not tie_weights:
            proj = nn.Linear(hid_dim, vocab_size)
        else:
            proj = nn.Linear(emb_dim, vocab_size)
            proj.weight = self.embeddings.weight
            if emb_dim != hid_dim:
                # inp embeddings are (vocab x emb_dim); output is (hid x vocab)
                # if emb_dim != hidden, we insert a projection
                logging.warn("When tying weights, output layer and "
                             "embedding layer should have equal size. "
                             "A projection layer will be insterted.")
                proj = nn.Sequential(nn.Linear(hid_dim, emb_dim), proj)

        output.append(proj)
        output.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*output)

    def project(self, dec_out):
        """
        Run output projection (from the possibly attended output til softmax).
        During training the input for the entire target sequence is processed
        at once for efficiency.
        """
        if dec_out.dim() == 3:  # collapse seq_len and batch_size (training)
            dec_out = dec_out.view(-1, dec_out.size(2))

        return self.proj(dec_out)


class RNNDecoder(BaseDecoder):
    """
    RNNDecoder

    Parameters:
    -----------

    - input_feed: bool, whether to concatenate last attentional vector
        to current rnn input. (See Luong et al. 2015). Mostly useful
        for attentional models.
    """
    def __init__(self, embeddings, hid_dim, num_layers, cell,
                 dropout=0.0, input_feed=False, att_type=None,
                 deepout_layers=0, deepout_act='ReLU', tie_weights=False,
                 train_init=False, add_init_jitter=False, reuse_hidden=True,
                 cond_dims=None, cond_vocabs=None):

        if train_init and reuse_hidden:
            logging.warn("Decoder `train_init` is True therefore "
                         "`reuse_hidden` will be ignored.")

        super(RNNDecoder, self).__init__()
        self.embeddings = embeddings
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.input_feed = input_feed
        self.att_type = att_type
        self.train_init = train_init
        self.add_init_jitter = add_init_jitter
        self.reuse_hidden = reuse_hidden

        in_dim = self.embeddings.embedding_dim
        if input_feed:
            in_dim += hid_dim

        # conditions
        self.has_conditions = False
        if cond_dims is not None:
            self.has_conditions, self.cond_embs = True, nn.ModuleList()

            for cond_dim, cond_vocab in zip(cond_dims, cond_vocabs):
                self.cond_embs.append(nn.Embedding(cond_vocab, cond_dim))
                in_dim += cond_dim

        # rnn layer
        self.rnn = self.build_rnn(num_layers, in_dim, hid_dim, cell, dropout)

        # train init
        if self.train_init:
            init_size = self.num_layers, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*init_size).zero_())

        # attention network (optional)
        if self.att_type is not None and self.att_type.lower() != 'none':
            self.attn = attention.Attention(
                self.hid_dim, self.hid_dim, scorer=self.att_type)
        self.has_attention = hasattr(self, 'attn')

        # output projection
        self.proj = self.build_output(
            hid_dim, deepout_layers, deepout_act, tie_weights)

    def build_rnn(self, num_layers, in_dim, hid_dim, cell, dropout):
        stacked = StackedLSTM if cell == 'LSTM' else StackedGRU
        return stacked(num_layers, in_dim, hid_dim, dropout=dropout)

    @property
    def conditional(self):
        return self.has_conditions

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable to be fed as init hidden step.

        Returns:
        --------
        torch.Tensor(num_layers x batch x hid_dim)
        """
        # unpack
        if self.cell.startswith('LSTM'):
            h_0, _ = enc_hidden
        else:
            h_0 = enc_hidden

        # compute h_0
        if self.train_init:
            h_0 = self.h_0.repeat(1, h_0.size(1), 1)
        else:
            if not self.reuse_hidden:
                h_0 = torch.zeros_like(h_0)

        if self.add_init_jitter:
            h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

        # pack
        if self.cell.startswith('LSTM'):
            return h_0, torch.zeros_like(h_0)
        else:
            return h_0

    def init_output_for(self, hidden):
        """
        Creates a variable to be concatenated with previous target
        embedding as input for the first rnn step. This is used
        for the first decoding step when using the input_feed flag.

        Returns:
        --------
        torch.Tensor(batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            hidden = hidden[0]

        _, batch, hid_dim = hidden.size()

        output = torch.normal(hidden.data.new(batch, hid_dim).zero_(), 0.3)

        return Variable(output, volatile=not self.training)

    def init_state(self, outs, hidden, lengths, conds=None):
        """
        Must be call at the beginning of the decoding

        Parameters:
        -----------

        outs: torch.FloatTensor, summary vector(s) from the Encoder.
        hidden: torch.FloatTensor, previous hidden decoder state.
        conds: (optional) tuple of (batch) with conditions
        """
        hidden = self.init_hidden_for(hidden)

        mask, enc_att = None, None
        if self.has_attention:
            mask = u.make_length_mask(lengths)
            if self.att_type.lower() == 'bahdanau':
                enc_att = self.attn.scorer.project_enc_outs(outs)

        input_feed = None
        if self.input_feed:
            input_feed = self.init_output_for(hidden)

        if self.conditional:
            if conds is None:
                raise ValueError("Conditional decoder requires `conds`")
            conds = torch.cat(
                [emb(c) for c, emb in zip(conds, self.cond_embs)], 1)

        return RNNDecoderState(hidden, outs, mask=mask, enc_att=enc_att,
                               input_feed=input_feed, conds=conds)

    def forward(self, inp, state):
        """
        Parameters:
        -----------

        inp: torch.FloatTensor(batch), Embedding input.
        state: DecoderState, object data persisting throughout the decoding.
        """
        inp = self.embeddings(inp)

        if self.input_feed:
            inp = torch.cat([inp, state.input_feed], 1)

        if self.conditional:
            inp = torch.cat([inp, *state.conds], 1)

        out, hidden = self.rnn(inp, state.hidden)

        weight = None
        if self.has_attention:
            out, weight = self.attn(
                out, state.enc_outs, enc_att=state.enc_att, mask=state.mask)

        # update state
        state.hidden = hidden
        if self.input_feed:
            state.input_feed = out

        return out, weight


class State(object):
    """
    Abstract state class to be implemented by different decoder states.
    It is used to carry over data across subsequent steps of the decoding
    process. For beam search two methods are obligatory.
    """
    def expand_along_beam(self, width):
        raise NotImplementedError

    def reorder_beam(self, beam_ids):
        raise NotImplementedError


class RNNDecoderState(State):
    """
    DecoderState implementation for RNN-based decoders.
    """
    def __init__(self, dec_hidden, enc_outs,
                 input_feed=None, enc_att=None, mask=None, conds=None):
        self.hidden = dec_hidden
        self.enc_outs = enc_outs
        self.input_feed = input_feed
        self.enc_att = enc_att
        self.mask = mask
        self.conds = conds

    def expand_along_beam(self, width):
        """
        Expand state attributes to match the beam width
        """
        if isinstance(self.hidden, tuple):
            hidden = (self.hidden[0].repeat(1, width, 1),
                      self.hidden[1].repeat(1, width, 1))
        else:
            hidden = self.hidden.repeat(1, width, 1)
        self.hidden = hidden
        self.enc_outs = self.enc_outs.repeat(1, width, 1)

        if self.input_feed is not None:
            self.input_feed = self.input_feed.repeat(width, 1)
        if self.enc_att is not None:
            self.enc_att = self.enc_att.repeat(1, width, 1)
        if self.mask is not None:
            self.mask = self.mask.repeat(width, 1)
        if self.conds is not None:
            self.conds = self.conds.repeat(width, 1)

    def reorder_beam(self, beam_ids):
        """
        Reorder state attributes to match the previously decoded beam order
        """
        if self.input_feed is not None:
            self.input_feed = u.swap(self.input_feed, 0, beam_ids)
        if isinstance(self.hidden, tuple):
            hidden = (u.swap(self.hidden[0], 1, beam_ids),
                      u.swap(self.hidden[1], 1, beam_ids))
        else:
            hidden = u.swap(self.hidden, 1, beam_ids)
        self.hidden = hidden
