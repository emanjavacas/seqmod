
import logging
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.modules.custom import grad_reverse
from seqmod.modules.embedding import word_dropout
from seqmod.modules.custom import StackedLSTM, StackedGRU, MLP, Highway
from seqmod.modules import attention as attn
from seqmod.misc.beam_search import Beam
from seqmod import utils as u


class Encoder(nn.Module):
    """
    RNN Encoder that computes a sentence matrix representation
    of the input using an RNN.
    """
    def __init__(self, in_dim, hid_dim, num_layers, cell,
                 train_init=False, add_init_jitter=True,
                 dropout=0.0, bidi=True):
        self.in_dim = in_dim
        self.cell = cell
        self.bidi = bidi
        self.train_init = train_init
        self.add_init_jitter = add_init_jitter
        self.num_layers = num_layers
        self.num_dirs = 2 if bidi else 1
        self.hid_dim = hid_dim // self.num_dirs

        if hid_dim % self.num_dirs != 0:
            raise ValueError("Hidden dimension must be even for BiRNNs")

        super(Encoder, self).__init__()
        self.rnn = getattr(nn, cell)(self.in_dim, self.hid_dim,
                                     num_layers=self.num_layers,
                                     dropout=dropout, bidirectional=self.bidi)

        if self.train_init:
            train_init_size = self.num_layers * self.num_dirs, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*train_init_size).zero_())

    def init_hidden_for(self, enc_outs):
        batch_size = enc_outs.size(1)
        size = (self.num_dirs * self.num_layers, batch_size, self.hid_dim)

        if self.train_init:
            h_0 = self.h_0.repeat(1, batch_size, 1)
        else:
            h_0 = enc_outs.data.new(*size).zero_()
            h_0 = Variable(h_0, volatile=not self.training)

        if self.add_init_jitter:
            h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

        if self.cell.startswith('LSTM'):
            # compute memory cell
            c_0 = enc_outs.data.new(*size).zero_()
            c_0 = Variable(c_0, volatile=not self.training)
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
        if hidden is None:
            hidden = self.init_hidden_for(inp)

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

    - input_feed: bool, whether to concatenate last attentional vector
        to current rnn input. (See Luong et al. 2015)
    """
    def __init__(self, emb_dim, hid_dim, num_layers, cell,
                 att_dim=None, att_type='general', dropout=0.0,
                 input_feed=False, cond_dim=None):
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.cell = cell
        self.att_dim = att_dim or hid_dim
        self.att_type = att_type
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.input_feed = input_feed
        super(Decoder, self).__init__()

        in_dim = emb_dim if not input_feed else hid_dim + emb_dim

        # handle conditions
        if cond_dim is not None:
            in_dim += cond_dim

        # rnn layers
        stacked = StackedLSTM if cell == 'LSTM' else StackedGRU
        self.rnn_step = stacked(
            self.num_layers, in_dim, self.hid_dim, dropout=self.dropout)

        # attention network (optional)
        if self.att_type and self.att_type.lower() != 'none':
            self.attn = attn.Attention(
                self.hid_dim, self.att_dim, scorer=self.att_type)

        self.has_attention = hasattr(self, 'attn')

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable at decoding step 0 to be fed as init hidden step.
        Returns: torch.Tensor(num_layers x batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            h_0, _ = enc_hidden
            c_0 = h_0.data.new(*h_0.size()).zero_()
            c_0 = Variable(c_0, volatile=not self.training)
            return h_0, c_0
        else:
            return enc_hidden

    def init_output_for(self, hidden):
        """
        Creates a variable to be concatenated with previous target
        embedding as input for the first rnn step. This is used
        for the first decoding step when using the input_feed flag.

        Returns: torch.Tensor(batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            hidden = hidden[0]

        _, batch, hid_dim = hidden.size()

        output = torch.normal(hidden.data.new(batch, hid_dim).zero_(), 0.3)

        return Variable(output, volatile=not self.training)

    def forward(self, inp, hidden, enc_outs, enc_att=None, prev_out=None,
                mask=None, conds=None):
        """
        Parameters:
        -----------

        - inp: torch.Tensor (batch x emb_dim), Previously decoded output.
        - hidden: Used to seed the initial hidden state of the decoder.
            h_t: (num_layers x batch x hid_dim)
            c_t: (num_layers x batch x hid_dim)
        - enc_outs: torch.Tensor (seq_len x batch x hid_dim),
            Output of the encoder at the last layer for all encoding steps.
        - prev_out: torch.Tensor (batch x hid_dim), previous context vector,
            (required for input feeding)

        Returns:
        --------
        - out: torch.Tensor(batch x hid_dim)
        - hidden: torch.Tensor(num_layers x batch_size x hid_dim)
        - weight (optional): torch.Tensor(batch x seq_len)
        """
        weight = None

        if self.input_feed:
            if prev_out is None:
                prev_out = self.init_output_for(hidden)
            inp = torch.cat([inp, prev_out], 1)

        if conds is not None:
            inp = torch.cat([inp, conds], 1)

        out, hidden = self.rnn_step(inp, hidden)

        if self.has_attention:
            out, weight = self.attn(out, enc_outs, enc_att=enc_att, mask=mask)

        return out, hidden, weight


class EncoderDecoder(nn.Module):
    """
    Configurable encoder-decoder architecture

    Parameters:
    -----------

    - num_layers: int, Number of layers for both the encoder and the decoder.
    - emb_dim: int, Embedding dimension.
    - hid_dim: int, Hidden state size for the encoder and the decoder.
    - att_dim: int, Hidden state for the attention network.
    - src_dict: Dict, A fitted Dict used to encode the data into integers.
    - trg_dict: Dict, Same as src_dict in case of bilingual training.
    - cell: string, Cell type to use. One of (LSTM, GRU).
    - att_type: string, Attention mechanism to use. One of (Global, Bahdanau).
    - dropout: float
    - word_dropout: float
    - deepout_layers: int, Whether to use a highway layer before the output
        projection in the decoder.
    - deepout_act: str, Non-linear activation in the deepout layer if given.
    - bidi: bool, Whether to use bidirectional encoder.
    - input_feed: bool,
        Whether to feed back the previous context as input to the decoder
        for the next step together with the last predicted word embedding.
    - tie_weights: bool, Whether to tie embedding input and output weights.
        It wouldn't make much sense in bilingual settings.
    """
    def __init__(self,
                 num_layers,
                 emb_dim,
                 hid_dim,
                 att_dim,
                 src_dict,
                 trg_dict=None,
                 cell='LSTM',
                 att_type='general',
                 dropout=0.0,
                 word_dropout=0.0,
                 deepout_layers=0,
                 deepout_act='ReLU',
                 bidi=True,
                 input_feed=False,
                 train_init=False,
                 tie_weights=False,
                 scheduled_rate=1.,
                 cond_vocabs=None,
                 cond_dims=None):
        super(EncoderDecoder, self).__init__()
        self.cell = cell
        self.emb_dim = emb_dim
        self.input_feed = input_feed
        self.src_dict = src_dict
        self.trg_dict = trg_dict or src_dict
        src_vocab_size = len(self.src_dict)
        trg_vocab_size = len(self.trg_dict)
        self.bilingual = bool(trg_dict)

        # Word_dropout
        self.word_dropout = word_dropout
        self.target_code = self.src_dict.get_unk()
        codes = [self.src_dict.get_eos(),
                 self.src_dict.get_bos(),
                 self.src_dict.get_pad()]
        self.reserved_codes = tuple(code for code in codes if code is not None)

        # NLLLoss weight (downweight loss on pad) & schedule
        self.nll_weight = torch.ones(len(self.trg_dict))
        self.nll_weight[self.trg_dict.get_pad()] = 0
        self.scheduled_rate = scheduled_rate

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
            cell=cell, bidi=bidi, dropout=dropout, train_init=train_init)

        # Conditions (optional)
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

        # Decoder
        self.decoder = Decoder(
            emb_dim, hid_dim, num_layers, cell, att_dim,
            dropout=dropout, input_feed=input_feed,
            att_type=att_type, cond_dim=self.cond_dim)

        # Deepout (optional)
        if deepout_layers > 0:
            self.deepout = Highway(
                hid_dim, num_layers=deepout_layers, activation=deepout_act)

        self.has_deepout = hasattr(self, 'deepout')

        # Output projection
        if tie_weights:
            proj = nn.Linear(emb_dim, trg_vocab_size)
            proj.weight = self.trg_embeddings.weight
            if emb_dim != hid_dim:
                # inp embeddings are (vocab x emb_dim); output is (hid x vocab)
                # if emb_dim != hidden, we insert a projection
                logging.warn("When tying weights, output layer and "
                             "embedding layer should have equal size. "
                             "A projection layer will be insterted.")
                proj = nn.Sequential(nn.Linear(hid_dim, emb_dim), proj)
        else:
            # no tying
            proj = nn.Sequential(nn.Linear(hid_dim, trg_vocab_size))

        self.proj = nn.Sequential(proj, nn.LogSoftmax())

    def project(self, dec_out):
        """
        Run output projection (from the possibly attended output til softmax).
        During training the input for the entire target sequence is processed
        at once for efficiency.
        """
        if dec_out.dim() == 3:  # collapse seq_len and batch_size (training)
            seq_len, batch_size, _ = dec_out.size()
            dec_out = dec_out.view(seq_len * batch_size, -1)

        if self.has_deepout:
            dec_out = self.deepout(dec_out)

        return self.proj(dec_out)

    def is_cuda(self):
        "Whether the model is on a gpu. We assume no device sharing."
        return next(self.parameters()).is_cuda

    def parameters(self, only_trainable=True):
        """
        Return trainable parameters
        """
        for p in super(EncoderDecoder, self).parameters():
            if only_trainable and not p.requires_grad:
                continue
            yield p

    def n_params(self, only_trainable=True):
        """
        Return number of (trainable) parameters
        """
        return sum([p.nelement() for p in self.parameters(only_trainable)])

    def init_encoder(self, model, layer_map={'0': '0'}, target_module='rnn'):
        """
        Use a Language Model to initalize the encoder
        """
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
        """
        Use a Language Model to initalize the decoder
        """
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

    def load_embeddings(self, weight, words, target_embs='src', verbose=False):
        """
        Load embeddings from a weight matrix with words `words` as rows.

        Parameters
        -----------
        - weight: (vocab x emb_dim)
        - words: list of words corresponding to each row in `weight`
        """
        # wrap in tensor
        if isinstance(weight, list):
            weight = torch.Tensor(weight).float()
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float()
        # check embedding size
        assert weight.size(1) == self.emb_dim, \
            "Mismatched embedding dim {} for model with dim {}".format(
                (weight.size(1), self.emb_dim))

        target_module = getattr(self, '{}_embeddings'.format(target_embs))
        target_dict = getattr(self, '{}_dict'.format(target_embs))

        src_idxs, trg_idxs = [], []
        for trg_idx, word in enumerate(words):
            try:
                src_idxs.append(target_dict.s2i[word])
                trg_idxs.append(trg_idx)
            except KeyError:
                pass

        trg_idxs = torch.LongTensor(trg_idxs)
        src_idxs = torch.LongTensor(src_idxs)
        target_module.weight.data[src_idxs] = weight[trg_idxs]

    def freeze_submodule(self, module):
        """
        Makes a submodule untrainable
        """
        for p in getattr(self, module).parameters():
            p.requires_grad = False

    def get_scheduled_step(self, prev, dec_out):
        """
        Resample n inputs to next iteration from the model itself. N is itself
        sampled from a bernoulli independently for each example in the batch
        with weights equal to the model's variable self.scheduled_rate.

        Parameters:
        -----------

        - prev: torch.LongTensor(batch_size)
        - dec_out: torch.Tensor(batch_size x hid_dim)

        Returns: partially resampled input
        --------
        - prev: torch.LongTensor(batch_size)
        """
        keep_mask = torch.bernoulli(
            torch.zeros_like(prev).float() + self.scheduled_rate) == 1

        # return if no sampling is necessary
        if len(keep_mask.nonzero()) == len(prev):
            return prev

        sampled = self.project(Variable(dec_out, volatile=True)).max(1)[1].data

        if keep_mask.nonzero().dim() == 0:  # return all sampled
            return sampled

        keep_mask = keep_mask.nonzero().squeeze(1)
        sampled[keep_mask] = prev[keep_mask]

        return sampled

    def forward(self, inp, trg, conds=None, use_schedule=False):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch), Train data for a single batch.
        trg: torch.Tensor (seq_len x batch), Target output for a single batch.

        Returns: outs, conds
        --------
        outs: torch.Tensor (seq_len x batch x hid_dim)
        weights: tuple or None with as many entries as conditions in the model.
            Each entry is of size (batch x n_classes)
        """
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            # (batch_size x total emb dim)
            conds = torch.cat(conds, 1)

        # Encoder
        inp = word_dropout(
            inp, self.target_code, reserved_codes=self.reserved_codes,
            p=self.word_dropout, training=self.training)

        enc_outs, enc_hidden = self.encoder(self.src_embeddings(inp))

        cond_out = []
        if self.cond_dim is not None:
            # use last step as summary vector
            # enc_out = grad_reverse(enc_outs[-1]) # keep this for experiments
            # use average step as summary vector
            enc_out = grad_reverse(enc_outs.mean(dim=0))
            for grl in self.grls:
                cond_out.append(F.log_softmax(grl(enc_out), 1))

        # Decoder
        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_outs, dec_out, enc_att = [], None, None

        if self.decoder.att_type.lower() == 'bahdanau':
            # cache encoder att projection for bahdanau
            enc_att = self.decoder.attn.scorer.project_enc_outs(enc_outs)

        inp_mask = inp != self.src_dict.get_pad()

        for step, prev in enumerate(trg):
            # schedule
            if use_schedule and step > 0 and self.scheduled_rate < 1.0:
                prev = self.get_scheduled_step(prev.data, dec_out.data)
                prev = Variable(prev, volatile=not self.training)

            # (batch x emb_dim)
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, weight = self.decoder(
                prev_emb, dec_hidden, enc_outs, enc_att=enc_att,
                prev_out=dec_out, conds=conds, mask=inp_mask)
            dec_outs.append(dec_out)

        return torch.stack(dec_outs), tuple(cond_out)

    def loss(self, batch_data, test=False, split=25, use_schedule=False):
        """
        Return batch-averaged loss and examples processed for speed monitoring
        """
        pad, eos = self.src_dict.get_pad(), self.src_dict.get_eos()
        src, trg = batch_data

        src_conds, trg_conds = None, None
        if self.cond_dim is not None:
            (src, *src_conds), (trg, *trg_conds) = src, trg

        # remove <eos> from decoder targets substituting them with <pad>
        # dec_trg = Variable(u.map_index(trg[:-1].data, eos, pad))
        dec_trg = trg[:-1]
        # remove <bos> from loss targets
        loss_trg = trg[1:]

        # compute model output
        dec_outs, cond_outs = self(
            src, dec_trg, conds=src_conds, use_schedule=use_schedule)

        # compute cond loss
        cond_loss = []
        if trg_conds is not None:
            cond_loss = [F.nll_loss(pred, target, size_average=True)
                         for pred, target in zip(cond_outs, trg_conds)]

        # compute memory efficient word loss
        weight = self.nll_weight
        if self.is_cuda():
            weight = self.nll_weight.cuda()

        shard_data = {'out': dec_outs, 'trg': loss_trg}
        num_examples = trg.data.ne(pad).int().sum()
        loss = 0

        for shard in u.shards(shard_data, size=split, test=test):
            out, trg = shard['out'], shard['trg'].view(-1)
            out = self.project(out)
            shard_loss = F.nll_loss(out, trg, weight, size_average=False)
            shard_loss /= num_examples
            loss += shard_loss

            if not test:
                # accumulate word gradient
                shard_loss.backward(retain_graph=True)

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

        # Encoder
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(emb)

        # Conditions (optional)
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            conds = torch.cat(conds, 1)  # (batch_size x total emb dim)

        # Decoder
        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_out, enc_att = None, None
        if self.decoder.att_type.lower() == 'bahdanau':
            enc_att = self.decoder.attn.scorer.project_enc_outs(enc_outs)

        prev = src.data.new([bos]).expand(batch_size)
        prev = Variable(prev, volatile=True)
        mask = src.data.new(batch_size).zero_().float() + 1
        inp_mask = src != self.src_dict.get_pad()

        for _ in range(len(src) * max_decode_len):
            prev = prev.unsqueeze(0)  # add seq_len dim
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, dec_hidden, enc_outs, prev_out=dec_out,
                enc_att=enc_att, mask=inp_mask)
            logprobs = self.project(dec_out)  # (batch x vocab_size)
            logprobs, prev = logprobs.max(1)  # (batch) argmax over logprobs
            # accumulate
            scores += logprobs.data
            hyps.append(prev.data)
            # update mask
            mask = mask * (prev.data != eos).float()

            # terminate if all done
            if mask.sum() == 0:
                break

        hyps = torch.stack(hyps).transpose(0, 1).tolist()

        return scores.cpu().tolist(), hyps, None

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

        # Encoder
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(emb)
        enc_outs = enc_outs.repeat(1, beam_width, 1)
        if self.cell.startswith('LSTM'):
            enc_hidden = (enc_hidden[0].repeat(1, beam_width, 1),
                          enc_hidden[1].repeat(1, beam_width, 1))
        else:
            enc_hidden = enc_hidden.repeat(1, beam_width, 1)

        # Conditions (optional)
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            conds = torch.cat(conds, 1)  # (batch_size x total emb dim)
            conds = conds.repeat(beam_width, 1)

        # Decoder
        dec_hidden = self.decoder.init_hidden_for(enc_hidden)
        dec_out, enc_att = None, None
        if self.decoder.att_type.lower() == 'bahdanau':
            enc_att = self.decoder.attn.scorer.project_enc_outs(enc_outs)
        inp_mask = (src != self.src_dict.get_pad()).repeat(1, beam_width)

        beam = Beam(beam_width, bos, eos=eos, gpu=gpu)

        while beam.active and len(beam) < len(src) * max_decode_len:
            # (width) -> (1 x width)
            prev = beam.get_current_state().unsqueeze(0)
            prev = Variable(prev, volatile=True)
            prev_emb = self.trg_embeddings(prev).squeeze(0)

            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, dec_hidden, enc_outs, prev_out=dec_out,
                enc_att=enc_att, conds=conds)
            logprobs = self.project(dec_out)  # (width x vocab_size)
            beam.advance(logprobs.data)

            # repackage according to source beam
            source_beam = beam.get_source_beam()
            dec_out = u.swap(dec_out, 0, source_beam)
            if self.cell.startswith('LSTM'):
                dec_hidden = (u.swap(dec_hidden[0], 1, source_beam),
                              u.swap(dec_hidden[1], 1, source_beam))
            else:
                dec_hidden = u.swap(dec_hidden, 1, source_beam)

        scores, hyps = beam.decode(n=beam_width)

        return scores, hyps, None
