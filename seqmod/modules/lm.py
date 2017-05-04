
import math
import functools
import operator
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import seqmod.utils as u
from seqmod.modules.custom import word_dropout, MaxOut
from seqmod.misc.beam_search import Beam


class Decoder(object):
    def __init__(self, model, d, gpu=False):
        self.torch = torch if not gpu else torch.cuda
        self.model = model
        self.d = d
        self.bos, self.eos = self.d.get_bos(), self.d.get_eos()

    def seed(self, seed_text=None, **kwargs):
        """
        Parameters
        -----------
        seed_text: iterable or None, Seed text to initialize the generator.
            It should be an iterable of strings over vocabulary tokens.

        Returns
        ---------
        prev: (1 x 1), first integer token to feed into the generator
        hidden: hidden state to seed the generator, may be None if no
            seed text was passed to the generation function.
        """
        hidden = None
        if seed_text is not None:
            # prediction after last seed input
            seed_text = [self.d.index(i) for i in seed_text]
            if self.bos:
                seed_text = [self.bos] + seed_text
            inp = Variable(self.torch.LongTensor(seed_text).unsqueeze(1))
            # outs (seq_len (* 1) x vocab)
            outs, hidden, _ = self.model(inp, **kwargs)
            # prev_data (1)
            _, prev_data = outs.data[-1].max(0)
        elif self.bos is not None:
            # initialize with <bos>
            prev_data = self.torch.LongTensor([self.bos])
        else:
            # random uniform sample from vocab
            prev_data = (self.torch.rand(1) * self.model.vocab).long()
        prev = Variable(prev_data.unsqueeze(0), volatile=True)
        return prev, hidden

    def argmax(self, seed_text=None, max_seq_len=25, **kwargs):
        prev, hidden = self.seed(seed_text)
        hyp, scores = [], []
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            score, prev = outs.max(1)
            hyp.append(prev.squeeze().data[0])
            scores.append(score.squeeze().data[0])
            if self.eos and prev.data.eq(self.eos).nonzero().nelement() > 0:
                break
        return [math.exp(sum(scores))], [hyp]

    def sample(self, temperature=1, seed_text=None, max_seq_len=25, **kwargs):
        prev, hidden = self.seed(seed_text)
        hyp, scores = [], []
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            prev = outs.div(temperature).exp_().multinomial()
            score = outs.squeeze()[prev.squeeze().data[0]]
            hyp.append(prev.squeeze().data[0])
            scores.append(score.squeeze().data[0])
            if self.eos and prev.data.eq(self.eos).nonzero().nelement() > 0:
                break
        return [functools.reduce(operator.mul, scores)], [hyp]

    def beam(self, width=5, seed_text=None, max_seq_len=25, **kwargs):
        prev, hidden = self.seed(seed_text)
        beam = Beam(width, prev.squeeze().data[0], eos=self.eos)
        while beam.active and len(beam) < max_seq_len:
            prev_data = beam.get_current_state().unsqueeze(0)
            prev = Variable(prev_data, volatile=True)
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            beam.advance(outs.data)
            if self.model.cell.startswith('LSTM'):
                hidden = (u.swap(hidden[0], 1, beam.get_source_beam()),
                          u.swap(hidden[1], 1, beam.get_source_beam()))
            else:
                hidden = u.swap(hidden, 1, beam.get_source_beam())
        scores, hyps = beam.decode(n=width)
        return scores, hyps


class Attention(nn.Module):
    def __init__(self, att_dim, hid_dim, emb_dim):
        self.att_dim = att_dim
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        super(Attention, self).__init__()
        self.hid2att = nn.Linear(hid_dim, att_dim)    # W
        self.emb2att = nn.Linear(emb_dim, att_dim)       # U
        self.att_v = nn.Parameter(torch.Tensor(att_dim))  # b

    def project_emb(self, emb):
        """
        Returns:
        --------
        torch.Tensor: (seq_len x batch_size x att_dim)
        """
        return torch.stack([self.emb2att(i) for i in emb])

    def forward(self, hid, emb, emb_att=None):
        """
        Parameters:
        -----------
        hid: torch.Tensor (batch_size x hid_dim)
            Hidden state at step h_{t-1}
        emb: torch.Tensor (seq_len x batch_size x emb_dim)
            Embeddings of input words up to step t-1

        Returns: context, weights
        --------
        context: torch.Tensor (batch_size x emb_dim)
        weights: torch.Tensor (batch_size x seq_len)
        """
        seq_len, batch_size, _ = emb.size()
        # hid_att: (batch_size x hid_dim)
        hid_att = self.hid2att(hid)
        emb_att = emb_att or self.project_emb(emb)
        # att: (seq_len x batch_size x att_dim)
        att = F.tanh(emb_att + u.tile(hid_att, seq_len))
        # weights: (batch_size x seq_len)
        weights = F.softmax(u.bmv(att.t(), self.att_v).squeeze(2))
        # context: (batch_size x emb_dim)
        context = weights.unsqueeze(1).bmm(emb.t()).squeeze(1)
        return context, weights


class AttentionalProjection(nn.Module):
    def __init__(self, att_dim, hid_dim, emb_dim):
        super(AttentionalProjection, self).__init__()
        self.attn = Attention(att_dim, hid_dim, emb_dim)
        self.hid2hid = nn.Linear(hid_dim, hid_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)

    def forward(self, outs, emb):
        """
        Runs attention for a given input sequence

        Returns: output, weights
        --------
        output: torch.Tensor (seq_len x batch_size x hid_dim)
        weights: list of torch.Tensor(batch_size x 0:t-1) of length seq_len
        """
        emb_att = self.attn.project_emb(emb)
        output, weights = [], []
        for idx, hid in enumerate(outs):
            t = max(0, idx-1)  # use same hid at t=0
            context, weight = self.attn(
                outs[t], emb[:max(1, t)], emb_att=emb_att[:max(1, t)])
            output.append(self.hid2hid(hid) + self.emb2hid(context))
            weights.append(weight)
        return torch.stack(output), weights


class DeepOut(nn.Module):
    """
    DeepOut for Language Model following https://arxiv.org/pdf/1312.6026.pdf

    Parameters:
    ===========
    in_dim: int, input dimension for first layer
    layers: iterable of output dimensions for the hidden layers
    activation: str (ReLU, Tanh, MaxOut), activation after linear layer
    """
    def __init__(self, in_dim, layers, activation, dropout=0.0):
        self.in_dim = in_dim
        self.layers = layers
        self.activation = activation
        self.dropout = dropout

        super(DeepOut, self).__init__()
        self.layers, in_dim = [], self.in_dim
        for idx, out_dim in enumerate(layers):
            if activation == 'MaxOut':
                layer = MaxOut(in_dim, out_dim, 2)
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim), getattr(nn, activation))
            self.add_module('deepout_%d' % idx, layer)
            self.layers.append(layer)
            in_dim = out_dim

    def forward(self, inp):
        for layer in self.layers:
            out = layer(inp)
            out = F.dropout(out, p=self.dropout, training=self.training)
            inp = out
        return out


class LM(nn.Module):
    """
    Vanilla RNN-based language model.

    Parameters:
    ===========
    - vocab: int, vocabulary size.
    - emb_dim: int, embedding size,
        This value has to be equal to hid_dim if tie_weights is True and
        project_on_tied_weights is False, otherwise input and output
        embedding dimensions wouldn't match and weights cannot be tied.
    - hid_dim: int, hidden dimension of the RNN.
    - num_layers: int, number of layers of the RNN.
    - cell: str, one of GRU, LSTM, RNN.
    - bias: bool, whether to include bias in the RNN.
    - dropout: float, amount of dropout to apply in between layers.
    - tie_weights: bool, whether to tie input and output embedding layers.
        In case of unequal emb_dim and hid_dim values a linear project layer
        will be inserted after the RNN to match back to the embedding dim
    - att_dim: int, whether to add an attention module of dimension `att_dim`
        over the prefix. No attention will be added if att_dim is None or 0
    - deepout_layers: int, whether to add deep output after hidden layer and
        before output projection layer. No deep output will be added if
        deepout_layers is 0 or None.
    - deepout_act: str, activation function for the deepout module in camelcase
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='GRU', bias=True, dropout=0.0, word_dropout=0.0,
                 target_code=None, reserved_codes=(),
                 att_dim=None, tie_weights=False,
                 deepout_layers=0, deepout_act='MaxOut'):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.tie_weights = tie_weights
        self.add_attn = att_dim and att_dim > 0
        self.add_deepout = deepout_layers and deepout_layers > 0
        if tie_weights and not self.emb_dim == self.hid_dim:
            logging.warn("When tying weights, output layer and embedding " +
                         "layer should have equal size. A projection layer " +
                         "will be insterted to accomodate for this")
        self.num_layers = num_layers
        self.cell = cell
        self.bias = bias
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        super(LM, self).__init__()

        # word dropout
        self.word_dropout = word_dropout
        self.target_code = target_code
        self.reserved_codes = reserved_codes
        # input embeddings
        self.embeddings = nn.Embedding(vocab, self.emb_dim)
        # rnn
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)
        # attention
        if self.add_attn:
            assert self.cell == 'RNN' or self.cell == 'GRU', \
                "currently only RNN, GRU supports attention"
            assert att_dim is not None, "Need to specify att_dim"
            self.att_dim = att_dim
            self.attn = AttentionalProjection(
                self.att_dim, self.hid_dim, self.emb_dim)
        # deepout
        if self.add_deepout:
            self.deepout = DeepOut(
                in_dim=self.hid_dim,
                layers=tuple([self.hid_dim] * deepout_layers),
                activation=deepout_act,
                dropout=self.dropout)
        # output projection
        if self.tie_weights:
            if self.emb_dim == self.hid_dim:
                self.project = nn.Linear(self.hid_dim, self.vocab)
                self.project.weight = self.embeddings.weight
            else:
                project = nn.Linear(self.emb_dim, self.vocab)
                project.weight = self.embeddings.weight
                self.project = nn.Sequential(
                    nn.Linear(self.hid_dim, self.emb_dim), project)
        else:
            self.project = nn.Linear(self.hid_dim, self.vocab)

    def parameters(self):
        for p in super(LM, self).parameters():
            if p.requires_grad is True:
                yield p

    def n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    def freeze_submodule(self, module, flag=False):
        for p in getattr(self, module).parameters():
            p.requires_grad = flag

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp, hidden=None, schedule=None, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch_size)

        Returns:
        --------
        outs: torch.Tensor (seq_len * batch_size x vocab)
        hidden: see output of RNN, GRU, LSTM in torch.nn
        weights: None or list of weights (batch_size x seq_len),
            It will only be not None if attention is provided.
        """
        inp = word_dropout(
            inp, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        weights = None
        if self.add_attn:
            outs, weights = self.attn(outs, emb)
        seq_len, batch, hid_dim = outs.size()
        outs = outs.view(seq_len * batch, hid_dim)
        if self.add_deepout:
            outs = self.deepout(outs)
        outs = F.log_softmax(self.project(outs))
        return outs, hidden, weights

    def generate(self, d, seed_text=None, max_seq_len=25, gpu=False,
                 method='sample', temperature=1, width=5, **kwargs):
        """
        Generate text using simple argmax decoding

        Returns:
        --------
        scores: list of floats, unnormalized scores, one for each hypothesis
        hyps: list of lists, decoded hypotheses in integer form
        """
        if self.training:
            logging.warn("Generating in training modus!")
        decoder = Decoder(self, d, gpu=gpu)
        if method == 'argmax':
            scores, hyps = decoder.argmax(
                seed_text=seed_text, max_seq_len=max_seq_len)
        elif method == 'sample':
            scores, hyps = decoder.sample(
                temperature=temperature, seed_text=seed_text,
                max_seq_len=max_seq_len)
        elif method == 'beam':
            scores, hyps = decoder.beam(
                width=width, seed_text=seed_text, max_seq_len=max_seq_len)
        else:
            raise ValueError("Wrong decoding method: %s" % method)
        return scores, hyps

    def predict_proba(self, inp, gpu=False, **kwargs):
        if self.training:
            logging.warn("Generating in training modus!")
        inp_vec = Variable(torch.LongTensor([inp]), volatile=True)
        if gpu:
            inp_vec.cuda()
        outs, hidden, _ = self(inp_vec, **kwargs)
        outs = u.select_cols(outs, inp).sum()
        return outs.data[0] / len(inp)


class ForkableLM(LM):
    """
    A LM model that allows to create forks of the current instance with
    frozen Embedding (and eventually RNN) layers for fine tunning the
    non-frozen parameters to particular dataset.
    The parent cannot have the projection layer for tied embeddings,
    since tied layers don't fit in this setup.
    """
    def __init__(self, *args, **kwargs):
        super(ForkableLM, self).__init__(*args, **kwargs)

    def fork_model(self, freeze_rnn=True):
        """
        Creates a child fork from the current model with frozen input
        embeddings (and eventually also frozen RNN).

        Parameters:
        ===========
        - freeze_rnn: optional, whether to also freeze the child RNN layer.
        """
        model = ForkableLM(
            self.vocab, self.emb_dim, self.hid_dim, num_layers=self.num_layers,
            cell=self.cell, dropout=self.dropout, bias=self.bias,
            tie_weights=False, add_deepout=self.add_deepout)
        source_dict, target_dict = self.state_dict(), model.state_dict()
        target_dict['embeddings.weight'] = source_dict()['embeddings.weight']
        for layer, p in source_dict.items():
            if layer.startswith('project') \
               and self.tie_weights \
               and self.hid_dim != self.emb_dim:
                logging.warn(
                    "Warning: Forked model couldn't use projection layer " +
                    "of parent for the initialization of layer [%s]" % layer)
                continue
            else:
                target_dict[layer] = p
        model.load_state_dict(target_dict)
        model.freeze_submodule('embeddings')
        if freeze_rnn:
            model.freeze_submodule('rnn')
        return model


class MultiheadLM(LM):
    """
    A variant LM that has multiple output embeddings (one for each of a
    given number of heads). This allows the model to fine tune different
    output distribution on different datasets.
    """
    def __init__(self, *args, heads=(), **kwargs):
        super(MultiheadLM, self).__init__(*args, **kwargs)
        assert heads, "MultiheadLM requires at least 1 head but got 0"
        import copy
        if self.add_attn:
            attn = self.attn
            del self.attn
            self.attn = {}
        project = self.project
        del self.project
        self.project = {}
        for head in heads:
            project_module = copy.deepcopy(project)
            self.add_module(head, project_module)
            self.project[head] = project_module
            if self.add_attn:
                attn_module = copy.deepcopy(attn)
                self.add_module(head, attn_module)
                self.attn[head] = attn_module

    def forward(self, inp, hidden=None, head=None):
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        weights = None
        if self.add_attn:
            outs, weights = self.attn[head](outs, emb)
        seq_len, batch, hid_dim = outs.size()
        outs = outs.view(seq_len * batch, hid_dim)
        if self.add_deepout:
            outs = self.deepout(outs)
        outs = self.project[head](outs)
        return outs, hidden, weights

    @classmethod
    def from_pretrained_model(cls, that_model, heads):
        """
        Create a multihead model from a pretrained LM initializing all weights
        to the LM states and all heads to the same output projection layer
        weights of the parent.
        """
        assert isinstance(that_model, LM)
        this_model = cls(
            that_model.vocab, that_model.emb_dim, that_model.hid_dim,
            num_layers=that_model.num_layers, cell=that_model.cell,
            bias=that_model.bias, dropout=that_model.dropout, heads=heads,
            att_dim=that_model.att_dim,
            deepout_layers=that_model.deepout_layers,
            deepout_act=that_model.deepout_act)
        this_state_dict = this_model.state_dict()
        for p, w in that_model.state_dict().items():
            if p in this_state_dict:
                this_state_dict[p] = w
            else:               # got project or attn layer
                *_, weight = p.split('.')
                for head in this_model.heads:
                    this_state_dict[head + "." + weight] = w
        this_model.load_state_dict(this_state_dict)
        return this_model
