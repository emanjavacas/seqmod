
import logging
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.modules.custom import word_dropout, grad_reverse
from seqmod.modules.custom import StackedLSTM, StackedGRU, MaxOut, MLP
from seqmod.modules import attention as attn
from seqmod.misc.beam_search import Beam
from seqmod import utils as u


def detach_vars(data):
    """Split variables from the tree to allow for memory efficient computation
    of softmax losses over a whole sequence."""
    for k, v in data.items():
        if v.requires_grad:
            v = Variable(v.data, requires_grad=True, volatile=False)
        yield k, v


def shards(data, size=25, test=False):
    """
    Generator over variables that will be involved in a costly loss computation
    such as the softmax. It yields dictionaries of the same form as the input,
    where the variables have been splitted in smaller shards and detach from
    the graph. It expects the consumer to back propagate through them in shards
    of given a size. After all shards are consumed, the generator will take
    care of backprop further from the input using the accumulated gradients.
    """
    # Inspired by www.github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Loss.py
    if test:
        yield data
        return

    detached = dict(detach_vars(data))
    splits = ((key, torch.split(v, size)) for key, v in detached.items())
    keys, splits = zip(*splits)

    for split in zip(*splits):
        yield dict(zip(keys, split))  # go and accumulate some loss

    inputs, grads = [], []
    for key, var in detached.items():
        if var.grad is not None:
            inputs.append(data[key]), grads.append(var.grad.data)

    torch.autograd.backward(inputs, grads, retain_graph=True)


class Encoder(nn.Module):
    """
    RNN Encoder that computes a sentence matrix representation
    of the input using an RNN.
    """
    def __init__(self, in_dim, hid_dim, num_layers, cell,
                 dropout=0.0, bidi=True):
        self.in_dim, self.cell = in_dim, cell
        self.bidi, self.num_dirs = bidi, 2 if bidi else 1
        if hid_dim % self.num_dirs != 0:
            raise ValueError("Hidden dimension must be even for BiRNNs")
        self.hid_dim, self.num_layers = hid_dim // self.num_dirs, num_layers
        super(Encoder, self).__init__()
        self.rnn = getattr(nn, cell)(self.in_dim, self.hid_dim,
                                     num_layers=self.num_layers,
                                     dropout=dropout,
                                     bidirectional=self.bidi)

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_dirs * self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, inp, hidden=None):
        """
        Paremeters:
        -----------

        - inp: torch.Tensor (seq_len x batch x emb_dim)

        - hidden: tuple (h_0, c_0)
            h_0: ((num_layers * num_dirs) x batch x hid_dim)
            n_0: ((num_layers * num_dirs) x batch x hid_dim)

        Returns: output, (h_t, c_t)
        --------

        - output: (seq_len x batch x hidden_size * num_directions)
        - h_t: (num_layers x batch x hidden_size * num_directions)
        - c_t: (num_layers x batch x hidden_size * num_directions)
        """
        hidden = hidden if hidden is not None else self.init_hidden_for(inp)
        outs, hidden = self.rnn(inp, hidden)
        if self.bidi:
            # BiRNN encoder outputs (num_layers * 2 x batch x hid_dim)
            # but decoder expects   (num_layers x batch x hid_dim * 2)
            if self.cell.startswith('LSTM'):
                hidden = (u.repackage_bidi(hidden[0]),
                          u.repackage_bidi(hidden[1]))
            else:
                hidden = u.repackage_bidi(hidden)
        return outs, hidden


class Decoder(nn.Module):
    """
    Attentional decoder for the EncoderDecoder architecture.

    Parameters:
    -----------

    - add_prev: bool, whether to append last hidden state.
    """
    def __init__(self, emb_dim, hid_dim, num_layers, cell,
                 att_dim, att_type='Bahdanau', maxout=2, dropout=0.0,
                 add_prev=False, cond_dim=None):
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.cell = cell
        self.add_prev = add_prev
        self.dropout = dropout
        super(Decoder, self).__init__()

        in_dim = emb_dim if not add_prev else hid_dim + emb_dim

        # handle conditions
        if cond_dim is not None:
            in_dim += cond_dim
            # TODO: implement FactorCell

        # rnn layers
        stacked = StackedLSTM if cell == 'LSTM' else StackedGRU
        self.rnn_step = stacked(
            self.num_layers, in_dim, hid_dim, dropout=dropout)

        # attention network
        self.att_type = att_type

        if att_type == 'Bahdanau':
            self.attn = attn.BahdanauAttention(att_dim, hid_dim)
        elif att_type == 'Global':
            if att_dim != hid_dim:
                raise ValueError(
                    "Global attention requires same size Encoder and Decoder")
            self.attn = attn.GlobalAttention(hid_dim)
        else:
            raise ValueError("Unknown attention network [%s]" % att_type)

        # maxout
        self.has_maxout = bool(maxout)
        if self.has_maxout:
            self.maxout = MaxOut(att_dim + emb_dim, att_dim, maxout)

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable at decoding step 0 to be fed as init hidden step.

        Returns (h_0, c_0):
        -------------------

        - h_0: torch.Tensor (num_layers x batch x hid_dim)
        - c_0: torch.Tensor (num_layers x batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_0, _ = enc_hidden
            c_0 = h_0.data.new(*h_0.size()).zero_()
            c_0 = Variable(c_0, requires_grad=False)
            return h_0, c_0
        else:
            h_0 = enc_hidden
            return h_0

    def init_output_for(self, dec_hidden):
        """
        Creates a variable to be concatenated with previous target
        embedding as input for the first rnn step. This is used
        for the first decoding step when using the add_prev flag.

        Parameters:
        -----------

        - hidden: tuple (h_0, c_0)
        - h_0: torch.Tensor (num_layers x batch x hid_dim)
        - c_0: torch.Tensor (num_layers x batch x hid_dim)

        Returns:
        --------

        torch.Tensor (batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            dec_hidden = dec_hidden[0]
        data = dec_hidden.data.new(dec_hidden.size(1), self.hid_dim).zero_()
        return Variable(data, requires_grad=False)

    def forward(self, prev, hidden, enc_outs, conds=None,
                prev_out=None, enc_att=None, mask=None):
        """
        Parameters:
        -----------

        prev: torch.Tensor (batch x emb_dim),
            Previously decoded output.
        hidden: Used to seed the initial hidden state of the decoder.
            h_t: (num_layers x batch x hid_dim)
            c_t: (num_layers x batch x hid_dim)
        enc_outs: torch.Tensor (seq_len x batch x hid_dim),
            Output of the encoder at the last layer for all encoding steps.
        prev_out: torch.Tensor (batch x hid_dim), previous hidden output
        """
        if self.add_prev:
            # include last out as input for the prediction of the next item
            if not isinstance(prev_out, Variable):
                prev_out = self.init_output_for(hidden)
            prev = torch.cat([prev, prev_out], 1)

        # handle conditions
        if conds is not None:
            prev = torch.cat([prev, conds], 1)

        # step
        out, hidden = self.rnn_step(prev, hidden)

        # attention (batch x hid_dim), att_weight (batch x seq_len)
        out, att_weight = self.attn(out, enc_outs, enc_att=enc_att, mask=mask)

        # deep output
        if self.has_maxout:
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.maxout(torch.cat([out, prev], 1))

        return out, hidden, att_weight


class EncoderDecoder(nn.Module):
    """
    Vanilla configurable encoder-decoder architecture

    Parameters:
    -----------

    - num_layers: int,
        Number of layers for both the encoder and the decoder.
    - emb_dim: int, embedding dimension
    - hid_dim: int, Hidden state size for the encoder and the decoder
    - att_dim: int, Hidden state for the attention network.
        Note that it has to be equal to the encoder/decoder hidden
        size when using GlobalAttention.
    - src_dict: Dict, A fitted Dict used to encode the data into integers.
    - trg_dict: Dict, Same as src_dict in case of bilingual training.
    - cell: string, Cell type to use. One of (LSTM, GRU).
    - att_type: string, Attention mechanism to use. One of (Global, Bahdanau).
    - dropout: float
    - word_dropout: float
    - bidi: bool, Whether to use bidirectional.
    - add_prev: bool,
        Whether to feed back the last decoder state as input to
        the decoder for the next step together with the last
        predicted word embedding.
    """
    def __init__(self,
                 num_layers,
                 emb_dim,
                 hid_dim,
                 att_dim,
                 src_dict,
                 trg_dict=None,
                 cell='LSTM',
                 att_type='Global',
                 dropout=0.0,
                 word_dropout=0.0,
                 maxout=0,
                 bidi=True,
                 add_prev=False,
                 tie_weights=False,
                 cond_vocabs=None,
                 cond_dims=None):
        super(EncoderDecoder, self).__init__()
        self.cell = cell
        self.add_prev = add_prev
        self.src_dict = src_dict
        self.trg_dict = trg_dict or src_dict
        src_vocab_size = len(self.src_dict)
        trg_vocab_size = len(self.trg_dict)
        self.bilingual = bool(trg_dict)

        # Word_dropout
        self.word_dropout = word_dropout
        self.target_code = self.src_dict.get_unk()
        self.reserved_codes = (self.src_dict.get_eos(),
                               self.src_dict.get_bos(),
                               self.src_dict.get_pad())

        # NLLLoss weight (downweight loss on pad)
        self.nll_weight = torch.ones(len(self.trg_dict))
        self.nll_weight[self.trg_dict.get_pad()] = 0

        # Embedding layer(s)
        self.src_embeddings = nn.Embedding(
            src_vocab_size, emb_dim, padding_idx=self.src_dict.get_pad())
        if self.bilingual:
            self.trg_embeddings = nn.Embedding(
                trg_vocab_size, emb_dim, padding_idx=self.trg_dict.get_pad())
        else:
            self.trg_embeddings = self.src_embeddings

        # Encoder
        self.encoder = Encoder(
            emb_dim, hid_dim, num_layers,
            cell=cell, bidi=bidi, dropout=dropout)

        # Decoder
        # (handle conditions)
        self.cond_dim, self.cond_embs, self.grls = None, None, None

        if cond_dims is not None:
            if len(cond_dims) != len(cond_vocabs):
                raise ValueError("cond_dims & cond_vocabs must be same length")
            # total cond embedding size
            self.cond_dim = 0
            # allocate parameters
            self.cond_embs, self.grls = nn.ModuleList(), nn.ModuleList()
            for cond_vocab, cond_dim in zip(cond_vocabs, cond_dims):
                # (add conds embeddings)
                self.cond_embs.append(nn.Embedding(cond_vocab, cond_dim))
                self.cond_dim += cond_dim
                # (add GRL)
                self.grls.append(MLP(hid_dim, hid_dim, cond_vocab))

        # (add rnn)
        self.decoder = Decoder(
            emb_dim, hid_dim, num_layers, cell, att_dim,
            dropout=dropout, maxout=maxout, add_prev=add_prev,
            att_type=att_type, cond_dim=self.cond_dim)

        # Output projection
        output_size = trg_vocab_size if self.bilingual else src_vocab_size
        if tie_weights:
            project = nn.Linear(emb_dim, output_size)
            project.weight = self.trg_embeddings.weight
            if emb_dim != hid_dim:
                logging.warn("When tying weights, output layer and " +
                             "embedding layer should have equal size. " +
                             "A projection layer will be insterted.")
                project_tied = nn.Linear(hid_dim, emb_dim)
                self.project = nn.Sequential(
                    project_tied, project, nn.LogSoftmax())
            else:
                self.project = nn.Sequential(project, nn.LogSoftmax())
        else:
            self.project = nn.Sequential(
                nn.Linear(hid_dim, output_size),
                nn.LogSoftmax())

    # General utility functions
    def is_cuda(self):
        "Whether the model is on a gpu. We assume no device sharing."
        return next(self.parameters()).is_cuda

    def parameters(self):
        for p in super(EncoderDecoder, self).parameters():
            if p.requires_grad is True:
                yield p

    def n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    # Initializers
    def init_encoder(self, model, layer_map={'0': '0'}, target_module='rnn'):
        merge_map = {}
        for p in model.state_dict().keys():
            if not p.startswith(target_module):
                continue
            from_layer = ''.join(filter(str.isdigit, p))
            if from_layer not in layer_map:
                continue
            s = p.replace(target_module, 'encoder.rnn') \
                 .replace(from_layer, layer_map[from_layer])
            merge_map[p] = s
        state_dict = u.merge_states(
            self.state_dict(), model.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_decoder(self, model, target_module='rnn', layers=(0,)):
        assert isinstance(model.rnn, type(self.decoder.rnn_step))
        target_rnn = getattr(model, target_module).state_dict().keys()
        source_rnn = self.decoder.rnn_step.state_dict().keys()
        merge_map = {}
        for param in source_rnn:
            try:
                # Decoder has format "LSTMCell_0.weight_ih"
                num, suffix = re.findall(r".*([0-9]+)\.(.*)", param)[0]
                # LM rnn has format "weight_ih_l0"
                target_param = suffix + "_l" + num
                if target_param in target_rnn:
                    merge_map[target_param] = "decoder.rnn_step." + param
            except IndexError:
                continue        # couldn't find target module
        state_dict = u.merge_states(
            self.state_dict(), model.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_embedding(self, model,
                       source_module='src_embeddings',
                       target_module='embeddings'):
        state_dict = u.merge_states(
            model.state_dict(), self.state_dict(),
            {target_module: source_module})
        self.load_state_dict(state_dict)

    def load_embeddings(self, weight, words, target_embs='src', verbose=False):
        """
        Load embeddings from a weight matrix with words `words` as rows.

        Parameters
        -----------
        weight: (vocab x emb_dim)
        words: list of words corresponding to each row in `weight`
        """
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight)
        assert weight.size(1) == self.emb_dim, \
            "Mismatched embedding dim %d for model with dim %d" % \
            (weight.size(1), self.emb_dim)
        target_words = {word: idx for idx, word in enumerate(words)}
        for idx, word in enumerate(self.src_dict.vocab):
            if word not in target_words:
                if verbose:
                    logging.warn("Couldn't find word [%s]" % word)
                continue
            if target_embs == 'src':
                self.src_embeddings.weight.data[idx, :].copy_(
                    weight[target_words[word], :])
            elif target_embs == 'trg':
                self.trg_embeddings.weight.data[idx, :].copy_(
                    weight[target_words[word], :])
            else:
                raise ValueError('target_embs must be `src` or `trg`')

    def init_batch(self, src):
        """
        Constructs a first prev batch for initializing the decoder.
        """
        batch, bos = src.size(1), self.src_dict.get_bos()
        return src.data.new(1, batch).fill_(bos)

    def freeze_submodule(self, module):
        for p in getattr(self, module).parameters():
            p.requires_grad = False

    def forward(self, inp, trg, conds=None):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch), Train data for a single batch.
        trg: torch.Tensor (seq_len x batch), Target output for a single batch.

        Returns: outs, att_weights
        --------
        dec_outs: torch.Tensor (seq_len x batch x hid_dim)
        """
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            # (batch_size x total emb dim)
            conds = torch.cat(conds, 1)

        # encoder
        inp = word_dropout(
            inp, self.target_code, reserved_codes=self.reserved_codes,
            p=self.word_dropout, training=self.training)

        enc_outs, enc_hidden = self.encoder(self.src_embeddings(inp))
        cond_out = []
        if self.cond_dim is not None:
            # use last step as summary vector
            enc_out = grad_reverse(enc_outs[-1])
            for grl in self.grls:
                cond_out.append(F.log_softmax(grl(enc_out)))

        # decoder
        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_outs, dec_out, enc_att = [], None, None

        if self.decoder.att_type == 'Bahdanau':
            # cache encoder att projection for bahdanau
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)

        for prev in trg:
            # (seq_len x batch x emb_dim)
            prev_emb = self.trg_embeddings(prev)
            # (batch x emb_dim)
            prev_emb = prev_emb.squeeze(0)
            dec_out, dec_hidden, att_weight = self.decoder(
                prev_emb, dec_hidden, enc_outs, enc_att=enc_att,
                prev_out=dec_out, conds=conds)
            dec_outs.append(dec_out)

        return torch.stack(dec_outs), tuple(cond_out)

    def loss(self, batch_data, test=False, split=25):
        """
        Return batch-averaged loss and examples processed for speed monitoring
        """
        pad, eos = self.src_dict.get_pad(), self.src_dict.get_eos()
        src, trg = batch_data

        src_conds, trg_conds = None, None
        if self.cond_dim is not None:
            (src, *src_conds), (trg, *trg_conds) = src, trg

        # remove <eos> from decoder targets substituting them with <pad>
        dec_trg = Variable(u.map_index(trg[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_trg = trg[1:]

        # compute model output
        dec_outs, cond_outs = self(src, dec_trg, conds=src_conds)

        # compute cond loss
        cond_loss = []
        if trg_conds is not None:
            cond_loss = [F.nll_loss(pred, target, size_average=True)
                         for pred, target in zip(cond_outs, trg_conds)]

        # compute memory efficient word loss
        weight = self.nll_weight
        if self.is_cuda():
            weight = self.nll_weight.cuda()

        num_examples = trg.data.ne(pad).sum()
        loss = 0
        hid_dim = dec_outs.size(2)
        shard_data = {'out': dec_outs, 'trg': loss_trg}

        for shard in shards(shard_data, size=split, test=test):
            # shard is of shape:
            #  {'out': Tensor(split x batch x hid_dim),
            #   'trg': LongTensor(split x batch)}
            logprobs = self.project(shard['out'].view(-1, hid_dim))
            shard_loss = F.nll_loss(logprobs, shard['trg'].view(-1),
                                    weight=weight, size_average=False)
            loss += shard_loss / num_examples

            if not test:
                # accumulate word gradient
                loss.backward(retain_graph=True)

        if not test:
            # accumulate cond gradient
            if trg_conds is not None:
                sum(cond_loss).backward()

        return (loss.data[0], *[l.data[0] for l in cond_loss]), num_examples

    def translate(self, src, max_decode_len=2, conds=None):
        """
        Translate a single input sequence using greedy decoding.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x batch_size)

        Returns (scores, hyps, atts):
        --------
        scores: (batch_size)
        hyps: (batch_size x seq_len)
        atts: (batch_size x seq_len x source_seq_len)
        """
        eos = self.src_dict.get_eos()
        bos = self.src_dict.get_bos()
        seq_len, batch_size = src.size()

        # output variables
        scores, hyps, atts = 0, [], []

        # Encode
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(emb)

        # Decode
        # (handler conditions)
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            # (batch_size x total emb dim)
            conds = torch.cat(conds, 1)

        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_out, enc_att = None, None
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)

        prev = src.data.new([bos]).expand(batch_size)
        prev = Variable(prev, volatile=True)
        mask = src.data.new(batch_size).zero_().float() + 1

        for _ in range(len(src) * max_decode_len):
            prev = prev.unsqueeze(1)  # add seq_len dim
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, dec_hidden, enc_outs, prev_out=dec_out,
                enc_att=enc_att)
            # (batch x vocab_size)
            logprobs = self.project(dec_out)
            # (batch) argmax over logprobs
            logprobs, prev = logprobs.max(1)
            # accumulate
            scores += logprobs.data.cpu()
            hyps.append(prev.data)
            atts.append(att_weights.data)
            # update mask
            mask = mask * (prev.data != eos).float()

            # terminate if all done
            if mask.sum() == 0:
                break

        hyps = torch.stack(hyps).transpose(0, 1).tolist()
        atts = torch.stack(atts).transpose(0, 1).tolist()

        return scores, hyps, atts

    def translate_beam(self, src, max_decode_len=2, beam_width=5, conds=None):
        """
        Translate a single input sequence using beam search.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x 1)
        """
        eos = self.src_dict.get_eos()
        bos = self.src_dict.get_bos()
        gpu = src.is_cuda

        # Encode
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(emb)
        enc_outs = enc_outs.repeat(1, beam_width, 1)
        if self.cell.startswith('LSTM'):
            enc_hidden = (enc_hidden[0].repeat(1, beam_width, 1),
                          enc_hidden[1].repeat(1, beam_width, 1))
        else:
            enc_hidden = enc_hidden.repeat(1, beam_width, 1)

        # Decode
        # (handler conditions)
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            # (batch_size x total emb dim)
            conds = torch.cat(conds, 1)
            conds = conds.repeat(beam_width, 1)

        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_out, enc_att = None, None
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)

        beam = Beam(beam_width, bos, eos=eos, gpu=gpu)

        while beam.active and len(beam) < len(src) * max_decode_len:
            # (width) -> (1 x width)
            prev = beam.get_current_state().unsqueeze(0)
            prev = Variable(prev, volatile=True)
            prev_emb = self.trg_embeddings(prev).squeeze(0)

            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, dec_hidden, enc_outs, prev_out=dec_out,
                enc_att=enc_att, conds=conds)
            # (width x vocab_size)
            logprobs = self.project(dec_out)
            beam.advance(logprobs.data)

            # repackage according to source beam
            dec_out = u.swap(dec_out, 0, beam.get_source_beam())
            if self.cell.startswith('LSTM'):
                dec_hidden = (u.swap(dec_hidden[0], 1, beam.get_source_beam()),
                              u.swap(dec_hidden[1], 1, beam.get_source_beam()))
            else:
                dec_hidden = u.swap(dec_hidden, 1, beam.get_source_beam())

        scores, hyps = beam.decode(n=beam_width)

        return scores, hyps, None
