
import logging
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.modules.embedding import Embedding
from seqmod.modules.custom import grad_reverse
from seqmod.modules.custom import StackedLSTM, StackedGRU, MLP, Highway
from seqmod.modules import attention as attn
from seqmod.misc.beam_search import Beam
from seqmod import utils as u


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

        # NLLLoss weight (downweight loss on pad) & schedule
        self.nll_weight = torch.ones(len(self.trg_dict))
        self.nll_weight[self.trg_dict.get_pad()] = 0
        self.scheduled_rate = scheduled_rate

        # Embedding layer(s)
        self.src_embeddings = Embedding(
            src_vocab_size, emb_dim, d=src_dict, word_dropout=word_dropout)
        if self.bilingual:
            self.trg_embeddings = Embedding(
                trg_vocab_size, emb_dim, d=trg_dict, word_dropout=word_dropout)
        else:
            self.trg_embeddings = self.src_embeddings

        # Encoder
        self.encoder = Encoder(
            emb_dim, hid_dim, num_layers,
            cell=cell, bidi=bidi, dropout=dropout, train_init=train_init)

        # Decoder
        self.decoder = Decoder(
            emb_dim, hid_dim, num_layers, cell, att_dim,
            dropout=dropout, input_feed=input_feed,
            att_type=att_type, cond_dim=self.cond_dim)

        self.proj = self._build_projection(
            self.trg_embeddings, self.decoder.hid_dim,
            deepout_layers, deepout_act)

    def _build_projection(self, embs, hid_dim, deepout_layers, deepout_act):
        output = []

        if deepout_layers > 0:
            highway = Highway(
                hid_dim, num_layers=deepout_layers, activation=deepout_act)
            output.append(highway)

        emb_dim, vocab_size = embs.embedding_size, embs.num_embeddings

        if not tie_weights:
            proj = nn.Linear(hid_dim, vocab_size)
        else:
            proj = nn.Linear(emb_dim, vocab_size)
            proj.weight = embeddings.weight
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
        # Encoder
        enc_outs, enc_hidden = self.encoder(inp)

        # Decoder
        if self.cond_dim is not None:
            if conds is None:
                raise ValueError("Conditional decoder needs conds")
            conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
            # (batch_size x total emb dim)
            conds = torch.cat(conds, 1)

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

        if self.encoder.conditional:
            src_conds, trg_conds = None, None
            (src, *src_conds), (trg, *trg_conds) = src, trg

        # remove <eos> from decoder targets, remove <bos> from loss targets
        dec_trg, loss_trg = trg[:-1], trg[1:]

        # compute model output
        enc_outs, enc_hidden = self.encoder(src)
        enc_loss = self.encoder.loss(enc_outs, src_conds, test=test)
        dec_outs = self.decoder(dec_trg, enc_outs, enc_hidden, conds=trg_conds)

        # compute memory efficient word loss
        weight = self.nll_weight
        if self.is_cuda():
            weight = self.nll_weight.cuda()

        shard_data = {'out': dec_outs, 'trg': loss_trg}
        num_examples = trg.data.ne(pad).int().sum()
        loss = 0

        for shard in u.shards(shard_data, size=split, test=test):
            shard_loss = F.nll_loss(
                self.project(shard['out']), shard['trg'].view(-1),
                weight, size_average=False)
            shard_loss /= num_examples
            loss += shard_loss

            if not test:
                shard_loss.backward(retain_graph=True)

        return (loss.data[0], *enc_loss), num_examples

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
