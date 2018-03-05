
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.misc.beam_search import Beam
from seqmod.modules.encoder import GRLWrapper
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.softmax import SampledSoftmax
from seqmod.modules.decoder import RNNDecoder
from seqmod.modules.embedding import Embedding
from seqmod.modules.torch_utils import flip, shards, select_cols
from seqmod.modules.exposure import scheduled_sampling


class EncoderDecoder(nn.Module):
    """
    Configurable encoder-decoder architecture

    Parameters:
    -----------
    - encoder: BaseEncoder
    - decoder: BaseDecoder
    - exposure_rate: float (0.0, 1.0), initial exposure to model predictions
        during training.
    - reverse: bool, whether to run the decoder in reversed order
    """
    def __init__(self, encoder, decoder, exposure_rate=1., reverse=False):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.exposure_rate = exposure_rate
        self.reverse = reverse

        # NLLLoss weight (downweight loss on pad)
        self.nll_weight = torch.ones(len(self.decoder.embeddings.d))
        self.nll_weight[self.decoder.embeddings.d.get_pad()] = 0

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

    def decoder_loss(self, dec_state, trg, num_examples,
                     test=False, split=25, use_schedule=False):

        # word-level loss weight (weight down padding)
        weight = self.nll_weight
        if self.is_cuda():
            weight = weight.cuda()

        # [<bos> ... <eos> pad pad]
        dec_trg, loss_trg = trg[:-1], trg[1:]

        # should we run fast_forward?
        fast_run = hasattr(self.decoder, 'is_fast_forward')
        fast_run = fast_run and self.decoder.is_fast_forward
        fast_run = fast_run and not use_schedule

        if fast_run:
            dec_outs, _ = self.decoder.fast_forward(dec_trg, dec_state)

        else:
            dec_outs = []
            for step, t in enumerate(dec_trg):
                if use_schedule and step > 0 and self.exposure_rate < 1.0:
                    t = scheduled_sampling(
                        t, dec_outs[-1],
                        self.decoder.project,
                        self.exposure_rate)
                    t = Variable(t, volatile=not self.training)
                out, _ = self.decoder(t, dec_state)
                dec_outs.append(out)
            dec_outs = torch.stack(dec_outs)

        # compute memory efficient decoder loss
        loss, shard_data = 0, {'out': dec_outs, 'trg': loss_trg}

        for shard in shards(shard_data, size=split, test=test):
            out, true = shard['out'], shard['trg'].view(-1)
            if isinstance(self.decoder.project, SampledSoftmax) and self.training:
                out, new_true = self.decoder.project(
                    out, targets=true, normalize=False, reshape=False)
                shard_loss = F.cross_entropy(out, new_true, size_average=False)
            else:
                shard_loss = F.nll_loss(
                    self.decoder.project(out), true, weight, size_average=False)
            shard_loss /= num_examples
            loss += shard_loss

            if not test:
                shard_loss.backward(retain_graph=True)

        return loss.data[0]

    def loss(self, batch_data, test=False, split=25, use_schedule=False):
        """
        Return batch-averaged loss and examples processed for speed monitoring

        Parameters:
        -----------

        - split: int, max targets per binned softmax loss computation
        - use_schedule: bool, whether to use scheduled sampling when computing
            the decoder loss. The rate of sampling is defined by the
            instance variable `exposure_rate`.
        """
        # unpack batch data
        (src, trg), (src_conds, trg_conds) = batch_data, (None, None)
        if self.encoder.conditional:
            (src, *src_conds) = src
        (src, src_lengths) = src
        if self.decoder.conditional:
            (trg, *trg_conds) = trg
        (trg, trg_lengths) = trg

        num_examples = trg_lengths.data.sum()

        # - compute encoder output
        enc_outs, enc_hidden = self.encoder(src, lengths=src_lengths)
        # - compute encoder loss
        enc_losses, _ = self.encoder.loss(enc_outs, src_conds, test=test)

        # - compute decoder loss
        # remove <eos> from decoder targets, remove <bos> from loss targets
        if hasattr(self, 'reverse') and self.reverse:
            # assume right aligned data: [pad pad <bos> ... <eos>]
            trg = flip(trg, 0)

        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, src_lengths, conds=trg_conds)
        dec_loss = self.decoder_loss(
            dec_state, trg, num_examples,
            test=test, split=split, use_schedule=use_schedule)

        return (dec_loss, *enc_losses), num_examples

    def translate(self, src, lengths, conds=None, max_decode_len=2,
                  on_init_state=None, on_step=None, sample=False, tau=1.0):
        """
        Translate a single input sequence using greedy decoding.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x batch_size)

        Returns (scores, hyps, atts):
        --------
        scores: list of floats (batch_size)
        hyps: (list of) list of ints (batch_size x trg_seq_len)
        atts: ((list of) list of) floats (batch_size x trg_seq_len x source_seq_len)
        """
        eos = self.decoder.embeddings.d.get_eos()
        bos = self.decoder.embeddings.d.get_bos()
        if hasattr(self, 'reverse') and self.reverse:
            bos, eos = eos, bos
        seq_len, batch_size = src.size()

        enc_outs, enc_hidden = self.encoder(src, lengths=lengths)
        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, lengths, conds=conds)

        if on_init_state is not None:
            on_init_state(self, dec_state)

        scores, hyps, weights = 0, [], []
        mask = src.data.new(batch_size).zero_().float() + 1
        prev = Variable(src.data.new([bos]).expand(batch_size), volatile=True)

        for _ in range(len(src) * max_decode_len):
            if on_step is not None:
                on_step(self, dec_state)

            out, weight = self.decoder(prev, dec_state)
            # decode
            logprobs = self.decoder.project(out)

            if sample:
                prev = logprobs.div_(tau).exp().multinomial(1).squeeze()
                logprobs = select_cols(logprobs, prev)
            else:
                logprobs, prev = logprobs.max(1)

            # accumulate
            hyps.append(prev.data)
            if self.decoder.has_attention:
                weights.append(weight.data)
            scores += logprobs.data
            # update mask
            mask = mask * (prev.data != eos).float()

            if mask.sum() == 0:
                break

        hyps = torch.stack(hyps).transpose(0, 1).tolist()  # batch first
        scores = scores.tolist()
        if self.decoder.has_attention:
            weights = torch.stack(weights).tolist()

        if hasattr(self, 'reverse') and self.reverse:
            hyps = [hyp[::-1] for hyp in hyps]

        return scores, hyps, weights

    def translate_beam(self, src, lengths, conds=None, beam_width=5,
                       max_decode_len=2, on_init_state=None, on_step=None):
        """
        Translate a single input sequence using beam search.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x batch_size)
        lengths: torch.LongTensor (batch_size)
        conds: (optional) conditions for the decoder
        beam_width: int, width of the beam
        max_decode_len: int, limit to the length of the output sequence
            in terms of the size of the input sequence

        Returns:
        --------
        scores: (batch_size), corresponding to the decoded
            hypotheses in descending order.
        hyps: (list of) list of ints (batch_size x max_seq_len), corresponding to the
            decoded hypotheses in descending order. `max_seq_len` corresponds to the
            size of the longest decoded hypotheses up to `max_decode_len` * the length
            of the input sequence.
        atts: None (WIP)
        """
        eos = self.decoder.embeddings.d.get_eos()
        bos = self.decoder.embeddings.d.get_bos()
        if hasattr(self, 'reverse') and self.reverse:
            bos, eos = eos, bos
        gpu = src.is_cuda

        scores, hyps, weights = [], [], []

        enc_outs, enc_hidden = self.encoder(src, lengths)
        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, lengths, conds=conds)

        # run callback
        if on_init_state is not None:
            on_init_state(self, dec_state)

        for state in dec_state.split_batches():
            # create beam
            state.expand_along_beam(beam_width)
            beam = Beam(beam_width, bos, eos=eos, gpu=gpu)

            while beam.active and len(beam) < len(src) * max_decode_len:
                # run callback
                if on_step is not None:
                    on_step(self, state)

                # advance
                prev = Variable(beam.get_current_state(), volatile=True)
                dec_out, weight = self.decoder(prev, state)
                logprobs = self.decoder.project(dec_out)  # (width x vocab_size)
                beam.advance(logprobs.data)
                state.reorder_beam(beam.get_source_beam())
                # TODO: add attention weight for decoded steps

            bscores, bhyps = beam.decode(n=1)
            bscores, bhyps = bscores[0], bhyps[0]
            if hasattr(self, 'reverse') and self.reverse:
                bhyps = bhyps[::-1]

            scores.append(bscores)
            hyps.append(bhyps)

        return scores, hyps, weights


def make_embeddings(src_dict, trg_dict, emb_dim, word_dropout):
    src_embeddings = Embedding.from_dict(src_dict, emb_dim)
    if trg_dict is not None:
        trg_embeddings = Embedding.from_dict(trg_dict, emb_dim, p=word_dropout)
    else:
        trg_embeddings = Embedding.from_dict(src_dict, emb_dim, p=word_dropout)
        trg_embeddings.weight = src_embeddings.weight

    return src_embeddings, trg_embeddings


def make_rnn_encoder_decoder(
        num_layers,
        emb_dim,
        hid_dim,
        src_dict,
        trg_dict=None,
        cell='LSTM',
        bidi=True,
        encoder_summary='full',
        att_type=None,
        sampled_softmax=False,
        dropout=0.0,
        variational=False,
        input_feed=True,
        context_feed=None,
        word_dropout=0.0,
        deepout_layers=0,
        deepout_act='ReLU',
        tie_weights=False,
        reuse_hidden=True,
        train_init=False,
        add_init_jitter=False,
        cond_dims=None,
        cond_vocabs=None,
        reverse=False
):
    """
    - num_layers: int, Number of layers for both the encoder and the decoder.
    - emb_dim: int, Embedding dimension.
    - hid_dim: int, Hidden state size for the encoder and the decoder.
    - src_dict: Dict, A fitted Dict used to encode the data into integers.
    - trg_dict: Dict, Same as src_dict in case of bilingual training.
    - cell: string, Cell type to use. One of (LSTM, GRU).
    - bidi: bool, Whether to use bidirectional encoder.
    - encoder_summary: How to compute summary vector for the decoder.
    - att_type: string, Attention mechanism to use.
    - dropout: float,
    - variational: bool, whether to do variational dropout on the decoder
    - input_feed: bool,
        Whether to feed back the previous context as input to the decoder
        for the next step together with the last predicted word embedding.
    - word_dropout: float
    - deepout_layers: int, Whether to use a highway layer before the output
        projection in the decoder.
    - deepout_act: str, Non-linear activation in the deepout layer if given.
    - tie_weights: bool, Whether to tie embedding input and output weights.
        It wouldn't make much sense in bilingual settings.
    - reuse_hidden: bool, whether to reuse encoder hidden for initializing the
        decoder.
    - train_init: bool, whether to train the initial hidden state of both
        encoder and decoder.
    - add_init_jitter: bool, whether to add gaussian noise the the initial
        hidden state.
    - cond_dims: tuple of integers with the embedding dimension corresponding
        to each condition.
    - cond_vocabs: tuple of integers with the number of classes for each
        condition in same order as `cond_dims`.
    """
    src_embeddings, trg_embeddings = make_embeddings(
        src_dict, trg_dict, emb_dim, word_dropout)

    if isinstance(num_layers, tuple):
        enc_layers, dec_layers = num_layers
    else:
        enc_layers, dec_layers = num_layers, num_layers

    if reuse_hidden and enc_layers != dec_layers:
        raise ValueError("`reuse_hidden` requires equal number of layers")

    encoder = RNNEncoder(src_embeddings, hid_dim, enc_layers, cell=cell,
                         bidi=bidi, dropout=dropout, summary=encoder_summary,
                         train_init=False, add_init_jitter=False)

    if context_feed is None:
        # only disable it if it explicitely desired (passing False)
        context_feed = att_type is None or att_type.lower() == 'none'

    encoder_dims, encoder_size = encoder.encoding_size

    decoder = RNNDecoder(trg_embeddings, hid_dim, dec_layers, cell, encoder_size,
                         dropout=dropout, variational=variational, input_feed=input_feed,
                         context_feed=context_feed, sampled_softmax=sampled_softmax,
                         att_type=att_type, deepout_layers=deepout_layers,
                         deepout_act=deepout_act,
                         tie_weights=tie_weights, reuse_hidden=reuse_hidden,
                         train_init=train_init, add_init_jitter=add_init_jitter,
                         cond_dims=cond_dims, cond_vocabs=cond_vocabs)

    if decoder.has_attention:
        if encoder_summary != 'full':
            raise ValueError("Attentional decoder needs full encoder summary")
    else:
        if encoder_dims != 2:
            raise ValueError("Attentionless decoder can't work with `full` "
                             "summaries, set `encoder_summary` to a different "
                             "value")

    return EncoderDecoder(encoder, decoder, reverse=reverse)


GRLRNNEncoder = GRLWrapper(RNNEncoder)


def make_grl_rnn_encoder_decoder(
        num_layers,
        emb_dim,
        hid_dim,
        src_dict,
        trg_dict=None,
        cell='LSTM',
        bidi=True,
        encoder_summary='inner-attention',
        dropout=0.0,
        variational=False,
        context_feed=True,
        word_dropout=0.0,
        deepout_layers=0,
        deepout_act='ReLU',
        tie_weights=False,
        train_init=False,
        add_init_jitter=False,
        cond_dims=None,
        cond_vocabs=None,
        conditional_decoder=True,
        reverse=False
):

    if encoder_summary == 'full':
        raise ValueError("GRL encoder can't use full summaries")

    if cond_dims is None or cond_vocabs is None:
        raise ValueError("GRL needs conditions")

    src_embeddings, trg_embeddings = make_embeddings(
        src_dict, trg_dict, emb_dim, word_dropout)

    encoder = GRLRNNEncoder(cond_dims, cond_vocabs,
                            src_embeddings, hid_dim, num_layers, cell,
                            bidi=bidi, dropout=dropout,
                            summary=encoder_summary,
                            train_init=False, add_init_jitter=True)

    _, encoding_size = encoder.encoding_size

    if not conditional_decoder:
        cond_dims, cond_vocabs = None, None

    decoder = RNNDecoder(src_embeddings, hid_dim, num_layers, cell, encoding_size,
                         dropout=dropout, variational=variational,
                         input_feed=False, context_feed=context_feed,
                         deepout_layers=deepout_layers, deepout_act=deepout_act,
                         tie_weights=tie_weights, reuse_hidden=False,
                         train_init=train_init, add_init_jitter=add_init_jitter,
                         cond_dims=cond_dims, cond_vocabs=cond_vocabs)

    return EncoderDecoder(encoder, decoder, reverse=reverse)
