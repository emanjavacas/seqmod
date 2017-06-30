
import math
import functools
import operator
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod import utils as u
from seqmod.modules.custom import word_dropout, MaxOut
from seqmod.misc.beam_search import Beam


def strip_post_eos(sents, eos):
    out = []
    for sent in sents:
        for idx, item in enumerate(sent):
            if item == eos:
                out.append(sent[:idx+1])
                break
        else:
            out.append(sent)
    return out


def read_batch(m, seed_texts, method, temperature=1., gpu=False):
    assert method in ('sample', 'argmax'), 'method must be sample or argmax'
    hs, cs, prev, scores = [], [], [], []
    for seed_text in seed_texts:
        inp = Variable(torch.LongTensor(seed_text).unsqueeze(1), volatile=True)
        if gpu:
            inp = inp.cuda()
        outs, hidden, _ = m(inp)
        outs = outs[-1]         # pick last step
        if m.cell.startswith('LSTM'):
            h, c = hidden
            hs.append(h), cs.append(c)
        else:
            hs.append(hidden)
        if method == 'sample':
            prev_t = outs.div(temperature).exp_().multinomial()
            scores.append(outs[prev_t.data[0]].cpu().data)
        else:                   # argmax over single step
            score, prev_t = outs.max(0)
            scores.append(score.cpu().data)
        prev.append(prev_t)
    scores, prev = torch.cat(scores), torch.stack(prev, 1)
    if m.cell.startswith('LSTM'):
        return scores, prev, (torch.cat(hs, 1), torch.cat(cs, 1))
    else:
        return scores, prev, torch.cat(hs, 1)


class Decoder(object):
    def __init__(self, model, d, gpu=False):
        self.gpu = gpu
        self.model = model
        self.d = d
        self.bos, self.eos = self.d.get_bos(), self.d.get_eos()

    def _seed(self, seed_texts, batch_size, method, bos, eos, **kwargs):
        """
        Parameters
        -----------
        seed_texts: list (of list (of string)) or None,
            Seed text to initialize the generator. It should be an iterable of
            lists of strings over vocabulary tokens. If seed_texts is given and
            is of length 1, the seed will be broadcasted to batch_size.
        batch_size: int, number of items in the seed batch.
        method: str, one of 'sample' or 'argmax' to use for sampling first item
        bos: bool, whether to prefix the seed with the bos_token. Only used if
            seed_texts is given and the dictionary has a bos_token.
        eos: bool, whether to suffix the seed with the eos_token. Only used if
            seed_texts is given and the dictionary has a eos_token.

        Returns
        ---------
        scores: (batch_size), torch Tensor holding the scores for the first
            sampled item.
        prev: (1 x batch_size), first integer token to feed into the generator
        hidden: hidden state to seed the generator, may be None if no seed text
            was passed to the generation function.
        """
        hidden, scores = None, 0
        if seed_texts is not None:
            seed_texts = [[self.d.index(i) for i in s] for s in seed_texts]
            if bos and self.bos is not None:  # prepend bos to seeds
                seed_texts = [[self.bos] + s for s in seed_texts]
            if eos and self.eos is not None:
                seed_texts = [s + [self.eos] for s in seed_texts]
            scores, prev, hidden = read_batch(
                self.model, seed_texts, method, gpu=self.gpu, **kwargs)
            if len(seed_texts) == 1:  # project over batch if only single seed
                scores = scores.expand(batch_size)
                prev = prev.expand(1, batch_size)
                if self.model.cell.startswith('LSTM'):
                    layers, _, hid_dim = hidden[0].size()
                    hidden = (hidden[0].expand(layers, batch_size, hid_dim),
                              hidden[1].expand(layers, batch_size, hid_dim))
                else:
                    layers, _, hid_dim = hidden.size()
                    hidden = hidden.expand(layers, batch_size, hid_dim)
        elif self.bos is not None:
            # initialize with <bos>
            prev_data = torch.LongTensor([self.bos] * batch_size).unsqueeze(0)
            prev = Variable(prev_data, volatile=True)
        else:
            # random uniform sample from vocab
            prev_data = (torch.rand(batch_size) * self.model.vocab).long()
            prev = Variable(prev_data, volatile=True)
        if self.gpu:
            prev = prev.cuda()
        return scores, prev, hidden

    def argmax(self, seed_texts=None, max_seq_len=25, batch_size=10,
               ignore_eos=False, bos=False, eos=False, **kwargs):
        scores, prev, hidden = self._seed(
            seed_texts, batch_size, 'argmax', bos, eos)
        batch = prev.size(1)
        hyps = [prev.squeeze().data.tolist()]
        finished = np.array([False] * batch)
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            score, prev = outs.max(1)
            hyps.append(prev.data.tolist())
            if self.eos is not None and not ignore_eos:
                mask = (prev.squeeze().data == self.eos).cpu().numpy() == 1
                finished[mask] = True
                if all(finished == True):  # nopep8
                    break
                # 0-mask scores for finished batch examples
                score.data[torch.ByteTensor(finished.tolist())] = 0
            scores += score.data
        return scores.tolist(), list(zip(*hyps))

    def sample(self, temperature=1., seed_texts=None, max_seq_len=25,
               batch_size=10, ignore_eos=False, bos=False, eos=False,
               **kwargs):
        scores, prev, hidden = self._seed(
            seed_texts, batch_size, 'sample', bos, eos,
            temperature=temperature)
        batch = prev.size(1)
        hyps = [prev.squeeze().data.tolist()]
        finished = np.array([False] * batch)
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            prev = outs.div(temperature).exp_().multinomial().t()
            score = u.select_cols(outs.data.cpu(), prev.squeeze().data.cpu())
            hyps.append(prev.squeeze().data.tolist())
            if self.eos is not None and not ignore_eos:
                mask = (prev.squeeze().data == self.eos).cpu().numpy() == 1
                finished[mask] = True
                if all(finished == True):  # nopep8
                    break
                # 0-mask scores for finished batch examples
                score[torch.ByteTensor(finished.tolist())] = 0
            scores += score
        return scores.tolist(), list(zip(*hyps))

    def beam(self, width=5, seed_texts=None, max_seq_len=25, batch_size=1,
             ignore_eos=False, bos=False, eos=False, **kwargs):
        if len(seed_text) > 1 or batch_size > 1:
            raise ValueError(
                "Currently beam search is limited to single item batches")
        prev, hidden = self._seed(seed_texts, batch_size, 'argmax', bos, eos)
        eos = self.eos if not ignore_eos else None
        beam = Beam(width, prev.squeeze().data[0], eos=eos)
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
        seq_len, batch_size, _ = emb.size()
        return self.emb2att(emb.view(seq_len * batch_size, -1)) \
                   .view(seq_len, batch_size, self.att_dim)

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
    - emb_dim: int, embedding size.
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

    def set_dropout(self, dropout):
        for m in self.children():
            if hasattr(m, 'dropout'):
                m.dropout = dropout
            if hasattr(m, 'has_dropout'):
                m.has_dropout = bool(dropout)

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

    def generate(self, d, seed_texts=None, max_seq_len=25, gpu=False,
                 method='sample', temperature=1., width=5, bos=False,
                 eos=False, ignore_eos=False, batch_size=10, **kwargs):
        """
        Generate text using a specified method (argmax, sample, beam)

        Parameters:
        -----------
        d: Dict used during training to fit the vocabulary
        seed_texts: None or list of sentences to use as seed for the generator
        max_seq_len: int, maximum number of symbols to be generated. The output
            might actually be less than this number if the Dict was fitted with
            a <eos> token (in which case generation will end after the first
            generated <eos>)
        gpu: bool, whether to generate on the gpu
        method: str, one of 'sample', 'argmax', 'beam' (check the corresponding
            functions in Decoder for more info)
        temperature: float, temperature for multinomial sampling (only applies
            to method 'sample')
        width: int, beam size width (only applies to the 'beam' method)
        ignore_eos: bool, whether to stop generation after hitting <eos> or not
        batch_size: int, number of parallel generations (only used if
            seed_texts is None)

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
                seed_texts=seed_texts, max_seq_len=max_seq_len,
                batch_size=batch_size, ignore_eos=ignore_eos, bos=bos, eos=eos,
                **kwargs)
        elif method == 'sample':
            scores, hyps = decoder.sample(
                temperature=temperature, seed_texts=seed_texts,
                batch_size=batch_size, max_seq_len=max_seq_len,
                ignore_eos=ignore_eos, bos=bos, eos=eos, **kwargs)
        elif method == 'beam':
            scores, hyps = decoder.beam(
                width=width, seed_texts=seed_texts, max_seq_len=max_seq_len,
                ignore_eos=ignore_eos, bos=bos, eos=eos, **kwargs)
        else:
            raise ValueError("Wrong decoding method: %s" % method)
        if not ignore_eos and d.get_eos() is not None:
            # strip content after <eos> for each batch
            hyps = strip_post_eos(hyps, d.get_eos())
        return [s/len(hyps[idx]) for idx, s in enumerate(scores)], hyps

    def predict_proba(self, inp, gpu=False, **kwargs):
        if self.training:
            logging.warn("Generating in training modus!")
        inp_vec = Variable(torch.LongTensor([inp]), volatile=True)
        if gpu:
            inp_vec.cuda()
        outs, _, _ = self(inp_vec, **kwargs)
        log_probs = u.select_cols(outs[:-1], inp[1:])
        return log_probs.sum().data[0] / len(log_probs)


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
