
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod import utils as u
from seqmod.modules import custom
from seqmod.modules.custom import word_dropout, MaxOut
from seqmod.misc.beam_search import Beam


def strip_post_eos(sents, eos):
    """
    remove suffix of symbols after <eos> in generated sentences
    """
    out = []
    for sent in sents:
        for idx, item in enumerate(sent):
            if item == eos:
                out.append(sent[:idx+1])
                break
        else:
            out.append(sent)
    return out


def read_batch(m, seed_texts, temperature=1., gpu=False, **kwargs):
    """
    Computes the hidden states for a bunch of seeds in iterative fashion.
    This is currently being done so because LM doesn't use padding and
    input seeds might have different sizes (in the future this will have
    to be implemented using pack_padded_sequence which allows for variable
    length inputs)

    Parameters:
    -----------
    seed_texts: list of lists of ints

    Returns:
    --------
    prev: torch.LongTensor (1 x batch_size), sampled symbols in the batch
    hidden: torch.FloatTensor (num_layers x batch_size x hid_dim)
    """
    prev, hs, cs = [], [], []
    for seed_text in seed_texts:
        seed_text, prev_i = seed_text[:-1], seed_text[-1]
        prev.append(prev_i)
        inp = Variable(torch.LongTensor(seed_text).unsqueeze(1), volatile=True)
        if gpu:
            inp = inp.cuda()
        _, hidden, _ = m(inp, **kwargs)
        if m.cell.startswith('LSTM'):
            h, c = hidden
            hs.append(h), cs.append(c)
        else:
            hs.append(hidden)
    prev = torch.LongTensor(prev).unsqueeze(0)
    if m.cell.startswith('LSTM'):
        return prev, (torch.cat(hs, 1), torch.cat(cs, 1))
    else:
        return prev, torch.cat(hs, 1)


class Decoder(object):
    """
    General Decoder class for language models.

    Parameters:
    -----------
    model: LM, fitted LM model to use for generation.
    d: Dict, dictionary fitted on the LM's input vocabulary.
    gpu: bool, whether to run generation on the gpu.
    """
    def __init__(self, model, d, gpu=False):
        self.gpu = gpu
        self.model = model
        self.d = d
        self.bos, self.eos = self.d.get_bos(), self.d.get_eos()

    def _seed(self, seed_texts, batch_size, bos, eos, **kwargs):
        """
        Handles the input to the actual generation method taking into account
        multiple variables. If seed_texts are given, it takes care of reading
        them to initialize the hidden state. In that case the input to the
        generation is a symbol predicted right after reading the batch.
        In case of generation from scratch (seed_texts is None), it will
        feed in the <bos> symbol if it exists, otherwise, it will just randomly
        sample a symbol from the vocabulary.

        If seed_texts isn't given or its length is just one, the generation
        will be done in a batch of size `batch_size` thus generating as many
        outputs.

        Parameters
        -----------
        seed_texts: list (of list (of string)) or None,
            Seed text to initialize the generator. It should be an iterable of
            lists of strings over vocabulary tokens. If seed_texts is given and
            is of length 1, the seed will be broadcasted to batch_size.
        batch_size: int, number of items in the seed batch.
        bos: bool, whether to prefix the seed with the bos_token. Only used if
            seed_texts is given and the dictionary has a bos_token.
        eos: bool, whether to suffix the seed with the eos_token. Only used if
            seed_texts is given and the dictionary has a eos_token.
        kwargs: extra LM.forward parameters

        Returns
        ---------
        prev: (1 x batch_size), first integer token to feed into the generator
        hidden: hidden state to seed the generator, may be None if no seed text
            was passed to the generation function.
        """
        hidden = None
        if seed_texts is not None:
            # read input seed batch
            seed_texts = [[self.d.index(i) for i in s] for s in seed_texts]
            if bos and self.bos is not None:  # prepend bos to seeds
                seed_texts = [[self.bos] + s for s in seed_texts]
            if eos and self.eos is not None:  # append eos to seeds
                seed_texts = [s + [self.eos] for s in seed_texts]
            prev_data, hidden = read_batch(
                self.model, seed_texts, gpu=self.gpu, **kwargs)
            if len(seed_texts) == 1:  # project over batch if only single seed
                prev_data = prev_data.repeat(1, batch_size)
                if self.model.cell.startswith('LSTM'):
                    hidden = (hidden[0].repeat(1, batch_size, 1),
                              hidden[1].repeat(1, batch_size, 1))
                else:
                    hidden = hidden.repeat(1, batch_size, 1)
        elif self.eos is not None:
            # initialize with <eos>
            prev_data = torch.LongTensor([self.eos] * batch_size).unsqueeze(0)
        else:
            # random uniform sample from vocab
            prev_data = (torch.rand(batch_size) * self.model.vocab).long()
            prev_data = prev_data.unsqueeze(0)
        # wrap prev in variable
        prev = Variable(prev_data, volatile=True)
        if self.gpu:
            prev = prev.cuda()
        return prev, hidden

    def argmax(self, seed_texts=None, max_seq_len=25,
               ignore_eos=False, bos=False, eos=False, **kwargs):
        """
        Generate a sequence sampling the element with highest probability
        in the output distribution at each generation step.
        """
        prev, hidden = self._seed(seed_texts, 1, bos, eos)
        hyps, scores = [], 0
        finished = np.array([False])
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            score, prev = outs.max(1)
            score, prev = score.squeeze(), prev.t()
            hyps.append(prev.squeeze().data.tolist())
            if self.eos is not None and not ignore_eos:
                mask = (prev.squeeze().data == self.eos).cpu().numpy() == 1
                finished[mask] = True
                if all(finished == True):  # nopep8
                    break
                # 0-mask scores for finished batch examples
                score.data[torch.ByteTensor(finished.tolist())] = 0
            scores += score.data
        return scores.tolist(), list(zip(*hyps))

    def beam(self, width=5, seed_texts=None, max_seq_len=25,
             ignore_eos=False, bos=False, eos=False, **kwargs):
        """
        Approximation to the highest probability output over the generated
        sequence using beam search.
        """
        prev, hidden = self._seed(seed_texts, 1, bos, eos)
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

    def sample(self, temperature=1., seed_texts=None, max_seq_len=25,
               batch_size=10, ignore_eos=False, bos=False, eos=False,
               **kwargs):
        """
        Generate a sequence multinomially sampling from the output
        distribution at each generation step. The output distribution
        can be tweaked by the input parameter `temperature`.
        """
        prev, hidden = self._seed(
            seed_texts, batch_size, bos, eos, temperature=temperature)
        batch_size = prev.size(1)  # not equal to input if seed_texts
        hyps, scores = [], 0
        finished = np.array([False] * batch_size)
        for _ in range(max_seq_len):
            outs, hidden, _ = self.model(prev, hidden=hidden, **kwargs)
            prev = outs.div_(temperature).exp().multinomial(1).t()
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
    def __init__(self, in_dim, layers, activation, maxouts=2, dropout=0.0):
        self.in_dim = in_dim
        self.layers = layers
        self.activation = activation
        self.dropout = dropout

        super(DeepOut, self).__init__()
        self.layers, in_dim = [], self.in_dim
        for idx, out_dim in enumerate(layers):
            if activation == 'MaxOut':
                layer = MaxOut(in_dim, out_dim, maxouts)
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
    - conds: list of condition-maps for a conditional language model
        in the following form:
            [{'name': str, 'varnum': int, 'emb_dim': int}, ...]
        where 'name' is a screen-name for the conditions, 'varnum' is the
        cardinality of the conditional variable and 'emb_dim' is the
        embedding size for that condition.
        Note that the conditions should be specified in the same order as
        they are passed into the model at run-time.
    - tie_weights: bool, whether to tie input and output embedding layers.
        In case of unequal emb_dim and hid_dim values a linear project layer
        will be inserted after the RNN to match back to the embedding dim
    - att_dim: int, whether to add an attention module of dimension `att_dim`
        over the prefix. No attention will be added if att_dim is None or 0
    - deepout_layers: int, whether to add deep output after hidden layer and
        before output projection layer. No deep output will be added if
        deepout_layers is 0 or None.
    - deepout_act: str, activation function for the deepout module in camelcase
    - maxouts: int, only used if deepout_act is MaxOut (number of parts to use
        to compose the non-linearity function).
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='GRU', bias=True, dropout=0.0, conds=None,
                 word_dropout=0.0, target_code=None, reserved_codes=(),
                 att_dim=None, tie_weights=False, train_init=False,
                 deepout_layers=0, deepout_act='MaxOut', maxouts=2):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        self.cell = cell
        self.bias = bias
        self.train_init = train_init
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        self.add_attn = att_dim and att_dim > 0
        self.deepout_layers = deepout_layers
        self.deepout_act = deepout_act
        self.maxouts = maxouts
        self.add_deepout = deepout_layers and deepout_layers > 0
        if tie_weights and not self.emb_dim == self.hid_dim:
            logging.warn("When tying weights, output layer and embedding " +
                         "layer should have equal size. A projection layer " +
                         "will be insterted to accomodate for this")
        if cell.startswith('RHN'):
            self.num_layers = 1
        self.conds = conds
        super(LM, self).__init__()

        # word dropout
        self.word_dropout = word_dropout
        self.target_code = target_code
        self.reserved_codes = reserved_codes

        # embeddings
        self.embeddings = nn.Embedding(vocab, self.emb_dim)
        rnn_input_size = self.emb_dim
        if self.conds is not None:
            conds = []
            for c in self.conds:
                conds.append(nn.Embedding(c['varnum'], c['emb_dim']))
                rnn_input_size += c['emb_dim']
            self.conds = nn.ModuleList(conds)

        # rnn
        if self.train_init:
            self.h_0 = nn.Parameter(torch.zeros(hid_dim * self.num_layers))
            if cell.startswith('LSTM'):
                self.c_0 = nn.Parameter(torch.zeros(hid_dim * self.num_layers))
        if hasattr(nn, cell):
            cell = getattr(nn, cell)
        else:                   # assume custom cell
            cell = getattr(custom, cell)
        self.rnn = cell(
            rnn_input_size, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)

        # (optional) attention
        if self.add_attn:
            if self.conds is not None:
                raise ValueError("Attention is not supported with conditions")
            if self.cell not in ('RNN', 'GRU'):
                raise ValueError("currently only RNN, GRU supports attention")
            if att_dim is None:
                raise ValueError("Need to specify att_dim")
            self.att_dim = att_dim
            self.attn = AttentionalProjection(
                self.att_dim, self.hid_dim, self.emb_dim)

        # (optional) deepout
        if self.add_deepout:
            self.deepout = DeepOut(
                in_dim=self.hid_dim,
                layers=tuple([self.hid_dim] * deepout_layers),
                activation=deepout_act,
                maxouts=maxouts,
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
        # trainable initial hidden state
        if hasattr(self, 'h_0'):
            h_0 = self.h_0.view(self.num_layers, self.hid_dim)
            h_0 = h_0.unsqueeze(1).repeat(1, batch, 1)
            if hasattr(self, 'c_0'):
                c_0 = self.c_0.view(self.num_layers, self.hid_dim)
                c_0 = c_0.unsqueeze(1).repeat(1, batch, 1)
                return h_0, c_0
            else:
                return h_0
        # non-trainable intial hidden state
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(),
                       requires_grad=False, volatile=not self.training)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(),
                           requires_grad=False, volatile=not self.training)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp, hidden=None, conds=None, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch_size)
        hidden: None or torch.Tensor (num_layers x batch_size x hid_dim)
        conds: None or tuple of torch.Tensor (seq_len x batch_size) of length
            equal to the number of model conditions. `conditions` are required
            in case of a CLM.

        Returns:
        --------
        outs: torch.Tensor (seq_len * batch_size x vocab)
        hidden: see output of RNN, GRU, LSTM in torch.nn
        weights: None or list of weights (batch_size x seq_len),
            It will only be not None if attention is provided.
        """
        if hasattr(self, 'conds') and self.conds is not None and conds is None:
            raise ValueError("Conditional model expects conditions as input")
        inp = word_dropout(
            inp, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        emb = self.embeddings(inp)
        if conds is not None:
            conds = torch.cat(
                [c_emb(inp_c) for c_emb, inp_c in zip(self.conds, conds)],
                2)
            emb = torch.cat([emb, conds], 2)
        if self.has_dropout and not self.cell.startswith('RHN'):
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        hidden = hidden if hidden is not None else self.init_hidden_for(emb)
        outs, hidden = self.rnn(emb, hidden)
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

    def generate(self, d, conds=None, seed_texts=None, max_seq_len=25,
                 gpu=False, method='sample', temperature=1., width=5,
                 bos=False, eos=False, ignore_eos=False, batch_size=10,
                 **kwargs):
        """
        Generate text using a specified method (argmax, sample, beam)

        Parameters:
        -----------
        d: Dict used during training to fit the vocabulary
        conds: list, list of integers specifying the input conditions for
            sampling from a conditional language model. Implies that all output
            elements in the batch will have same conditions
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
        bos: bool, whether to prefix the seed with the bos_token. Only used if
            seed_texts is given and the dictionary has a bos_token.
        eos: bool, whether to suffix the seed with the eos_token. Only used if
            seed_texts is given and the dictionary has a eos_token.
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

        if hasattr(self, 'conds') and self.conds is not None:
            # expand conds to batch
            if conds is None:
                raise ValueError("conds is required for generating with a CLM")
            conds = [torch.LongTensor([c]).repeat(1, batch_size) for c in conds]
            conds = [Variable(c, volatile=True) for c in conds]
            if gpu:
                conds = [c.cuda() for c in conds]

        decoder = Decoder(self, d, gpu=gpu)
        if method == 'argmax':
            scores, hyps = decoder.argmax(
                seed_texts=seed_texts, max_seq_len=max_seq_len, conds=conds,
                batch_size=batch_size, ignore_eos=ignore_eos, bos=bos, eos=eos,
                **kwargs)
        elif method == 'sample':
            scores, hyps = decoder.sample(
                temperature=temperature, seed_texts=seed_texts, conds=conds,
                batch_size=batch_size, max_seq_len=max_seq_len,
                ignore_eos=ignore_eos, bos=bos, eos=eos, **kwargs)
        elif method == 'beam':
            scores, hyps = decoder.beam(
                width=width, seed_texts=seed_texts, max_seq_len=max_seq_len,
                conds=conds, ignore_eos=ignore_eos, bos=bos, eos=eos, **kwargs)
        else:
            raise ValueError("Wrong decoding method: %s" % method)

        if not ignore_eos and d.get_eos() is not None:
            # strip content after <eos> for each batch
            hyps = strip_post_eos(hyps, d.get_eos())

        norm_scores = [s/len(hyps[idx]) for idx, s in enumerate(scores)]
        return norm_scores, hyps

    def predict_proba(self, inp, gpu=False, **kwargs):
        """
        Compute the probability assigned by the model to an input sequence.
        In the future this should use pack_padded_sequence to run prediction
        on multiple input sentences at once.

        Parameters:
        -----------
        inp: list of ints representing the input sequence
        gpu: bool, whether to use the gpu
        kwargs: other model parameters

        Returns:
        --------
        float representing the log probability of the input sequence
        """
        if self.training:
            logging.warn("Generating in training modus!")
        if isinstance(inp, list):
            inp = torch.LongTensor(inp)
        if gpu:
            inp = inp.cuda()
        outs, *_ = self(Variable(inp, volatile=True), **kwargs)
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

    def forward(self, inp, hidden=None, head=None, **kwargs):
        inp = word_dropout(
            inp, self.target_code, p=self.word_dropout,
            reserved_codes=self.reserved_codes, training=self.training)
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        hidden = hidden if hidden is not None else self.init_hidden_for(emb)
        outs, hidden = self.rnn(emb, hidden)
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        weights = None
        if self.add_attn:
            outs, weights = self.attn[head](outs, emb)
        seq_len, batch, hid_dim = outs.size()
        outs = outs.view(seq_len * batch, hid_dim)
        if self.add_deepout:
            outs = self.deepout(outs)
        outs = F.log_softmax(self.project[head](outs))
        return outs, hidden, weights

    @classmethod
    def from_pretrained_model(cls, that, heads):
        """
        Create a multihead model from a pretrained LM initializing all weights
        to the LM states and all heads to the same output projection layer
        weights of the parent.
        """
        assert isinstance(that, LM) and that.conds is None
        this = cls(
            that.vocab, that.emb_dim, that.hid_dim, heads=heads,
            num_layers=that.num_layers, cell=that.cell, bias=that.bias,
            dropout=that.dropout, word_dropout=that.word_dropout,
            target_code=that.target_code, reserved_codes=that.reserved_codes,
            att_dim=that.att_dim, train_init=that.train_init,
            maxouts=that.maxouts, deepout_layers=that.deepout_layers,
            deepout_act=that.deepout_act)
        this_state_dict = this.state_dict()
        for p, w in that.state_dict().items():
            if p in this_state_dict:
                this_state_dict[p] = w
            else:               # got project or attn layer
                *_, weight = p.split('.')
                for head in this.heads:
                    this_state_dict[head + "." + weight] = w
        this.load_state_dict(this_state_dict)
        return this
