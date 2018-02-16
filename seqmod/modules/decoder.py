
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod.modules.rnn import StackedGRU, StackedLSTM
from seqmod.modules.ff import OutputSoftmax
from seqmod.modules import attention
from seqmod.modules.torch_utils import make_length_mask, make_dropout_mask, swap


class BaseDecoder(nn.Module):
    """
    Base abstract class
    """
    def init_state(self, context, hidden, **kwargs):
        """
        Parameters:
        -----------
        context: summary output from the decoder
        hidden: encoder hidden state (e.g. for the RNN encoder)
        """
        raise NotImplementedError

    def forward(self, inp, state, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.LongTensor(batch)
        state: DecoderState
        """
        raise NotImplementedError

    def fast_forward(self, inp, state, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.LongTensor(batch)
        state: DecoderState
        """
        raise NotImplementedError

    @property
    def conditional(self):
        """
        Whether the current decoder implements takes conditions
        """
        return False


class RNNDecoder(BaseDecoder):
    """
    RNNDecoder

    Parameters:
    -----------

    - input_feed: bool, whether to concatenate last attentional vector
        to current rnn input. (See Luong et al. 2015). Mostly useful
        for attentional models.
    - context_feed: bool, whether to concatenate the encoding to the
        rnn input to facilitate gradient flow from the encoder to each
        decoder step. (Useful for non-attentive decoders.)
    """
    def __init__(self, embeddings, hid_dim, num_layers, cell, encoding_size,
                 dropout=0.0, variational=False, input_feed=False,
                 context_feed=False,
                 att_type=None, deepout_layers=0, deepout_act='ReLU',
                 tie_weights=False, train_init=False, add_init_jitter=False,
                 reuse_hidden=True, cond_dims=None, cond_vocabs=None):

        if train_init and reuse_hidden:
            logging.warn("Decoder `train_init` is True therefore "
                         "`reuse_hidden` will be ignored.")

        super(RNNDecoder, self).__init__()
        self.embeddings = embeddings
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.variational = variational
        self.input_feed = input_feed
        self.context_feed = context_feed
        self.att_type = att_type
        self.train_init = train_init
        self.add_init_jitter = add_init_jitter
        self.reuse_hidden = reuse_hidden

        self.has_attention = False
        if self.att_type is not None and self.att_type.lower() != 'none':
            self.has_attention = True

        self.is_fast_forward = False
        if not (variational or input_feed):
            self.is_fast_forward = True

        in_dim = self.embeddings.embedding_dim
        if input_feed:
            in_dim += hid_dim
        if context_feed:
            in_dim += encoding_size

        # conditions
        self.has_conditions = False
        if cond_dims is not None:
            self.has_conditions, self.cond_embs = True, nn.ModuleList()

            for cond_dim, cond_vocab in zip(cond_dims, cond_vocabs):
                self.cond_embs.append(nn.Embedding(cond_vocab, cond_dim))
                in_dim += cond_dim

        # rnn layer
        self.rnn = self.build_rnn(
            num_layers, in_dim, hid_dim, cell, dropout=dropout)

        # train init
        self.h_0 = None
        if self.train_init:
            init_size = self.num_layers, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*init_size).zero_())

        # attention network (optional)
        if self.has_attention:
            self.attn = attention.Attention(
                self.hid_dim, self.hid_dim, scorer=self.att_type)

        # output projection
        self.proj = OutputSoftmax(
            hid_dim, self.embeddings.embedding_dim, self.embeddings.num_embeddings,
            tie_weights=tie_weights, dropout=dropout, deepout_layers=deepout_layers,
            deepout_act=deepout_act)

        if tie_weights:
            self.project.tie_embedding_weights(self.embeddings)

    def build_rnn(self, num_layers, in_dim, hid_dim, cell, **kwargs):
        if self.is_fast_forward:
            stacked = nn.LSTM if cell == 'LSTM' else nn.GRU
            return stacked(in_dim, hid_dim, num_layers, **kwargs)
        else:
            stacked = StackedLSTM if cell == 'LSTM' else StackedGRU
            return stacked(num_layers, in_dim, hid_dim, **kwargs)

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

    def init_state(self, context, hidden, lengths, conds=None):
        """
        Must be call at the beginning of the decoding

        Parameters:
        -----------

        context: torch.FloatTensor, summary vector(s) from the Encoder.
        hidden: torch.FloatTensor, previous hidden decoder state.
        conds: (optional) tuple of (batch) with conditions
        """
        hidden = self.init_hidden_for(hidden)

        enc_att, mask = None, None
        if self.has_attention:
            mask = make_length_mask(lengths)
            if self.att_type.lower() == 'bahdanau':
                enc_att = self.attn.scorer.project_enc_outs(context)

        input_feed = None
        if self.input_feed:
            input_feed = self.init_output_for(hidden)

        if self.conditional:
            if conds is None:
                raise ValueError("Conditional decoder requires `conds`")
            conds = torch.cat(
                [emb(c) for c, emb in zip(conds, self.cond_embs)], 1)

        # sample variational mask if needed
        dropout_mask = None
        if hasattr(self, 'variational') and self.variational:
            size = (lengths.size(0), self.hid_dim)
            dropout_mask = make_dropout_mask(context, self.dropout, size)

        return RNNDecoderState(
            hidden, context, mask=mask, enc_att=enc_att,
            input_feed=input_feed, conds=conds, dropout_mask=dropout_mask)

    def project(self, dec_out):  # legacy code
        if dec_out.dim() == 3:
            dec_out = dec_out.view(-1, dec_out.size(2))

        return self.proj(dec_out)

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

        # condition on encoder summary for non-attentive decoders
        if self.context_feed:
            inp = torch.cat([inp, state.context], 1)

        if self.conditional:
            inp = torch.cat([inp, state.conds], 1)

        if self.is_fast_forward:
            # rnn module expects 3D input and doesn't take dropout_mask
            out, hidden = self.rnn(inp.unsqueeze(0), state.hidden)
            out = out.squeeze(0)
        else:
            out, hidden = self.rnn(inp, state.hidden, dropout_mask=state.dropout_mask)

        weight = None
        if self.has_attention:
            out, weight = self.attn(
                out, state.context, enc_att=state.enc_att, mask=state.mask)

        # update state
        state.hidden = hidden
        if self.input_feed:
            state.input_feed = out

        return out, weight

    def fast_forward(self, inp, state):
        """
        Parameters:
        -----------

        inp: torch.FloatTensor(seq_len x batch), Embedding input.
        state: DecoderState, object data persisting throughout the decoding.
        """
        if not self.is_fast_forward:
            raise ValueError("Decoder is not configured for fast_forward")

        inp = self.embeddings(inp)

        if self.input_feed:
            raise ValueError("Fast decoder can't use `input_feed`")

        # condition on encoder summary for non-attentive decoders (2D contexts)
        if self.context_feed:
            # repeat along seq_len dim
            context = state.context.unsqueeze(0).repeat(inp.size(0), 1, 1)
            inp = torch.cat([inp, context], dim=2)

        if self.conditional:
            # repeat along seq_len dim
            conds = state.conds.unsqueeze(0).repeat(inp.size(0), 1, 1)
            inp = torch.cat([inp, conds], dim=2)

        out, hidden = self.rnn(inp, state.hidden)

        weight = None
        if self.has_attention:
            out, weight = self.attn.fast_forward(
                out, state.context, enc_att=state.enc_att, mask=state.mask)

        # update state
        state.hidden = hidden

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

    def split_batches(self):
        raise NotImplementedError


class RNNDecoderState(State):
    """
    DecoderState implementation for RNN-based decoders.
    """
    def __init__(self, hidden, context, input_feed=None, enc_att=None,
                 mask=None, conds=None, dropout_mask=None):
        self.hidden = hidden
        self.context = context
        self.input_feed = input_feed
        self.enc_att = enc_att
        self.mask = mask
        self.conds = conds
        self.dropout_mask = dropout_mask

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
        if self.context.dim() == 2:
            self.context = self.context.repeat(width, 1)
        else:
            self.context = self.context.repeat(1, width, 1)

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
            self.input_feed = swap(self.input_feed, 0, beam_ids)
        if isinstance(self.hidden, tuple):
            hidden = (swap(self.hidden[0], 1, beam_ids),
                      swap(self.hidden[1], 1, beam_ids))
        else:
            hidden = swap(self.hidden, 1, beam_ids)
        self.hidden = hidden

    def split_batches(self):
        """
        After encoding, split the decoder state into single decoder states
        (one per batch), for decoding.
        """
        batch_size = self.context.size(0 if self.context.dim() == 2 else 1)

        if isinstance(self.hidden, tuple):
            hidden = list(zip(self.hidden[0].chunk(batch_size, 1),
                              self.hidden[1].chunk(batch_size, 1)))
        else:
            hidden = self.hidden.chunk(batch_size, 1)

        if self.context.dim() == 2:
            context = self.context.chunk(batch_size, 0)
        else:
            context = self.context.chunk(batch_size, 1)

        input_feed, enc_att, mask, conds = None, None, None, None
        if self.input_feed is not None:
            input_feed = self.input_feed.chunk(batch_size, 0)
        if self.enc_att is not None:
            enc_att = self.enc_att.chunk(batch_size, 1)
        if self.mask is not None:
            mask = self.mask.chunk(batch_size, 0)
        if self.conds is not None:
            conds = self.conds.chunk(batch_size, 0)

        for b in range(batch_size):

            yield RNNDecoderState(
                hidden[b], context[b],
                input_feed=input_feed[b] if input_feed is not None else None,
                enc_att=enc_att[b] if enc_att is not None else None,
                mask=mask[b] if mask is not None else None,
                conds=conds[b] if conds is not None else None)
