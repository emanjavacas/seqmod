
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.misc.beam_search import Beam
from seqmod.modules.encoder import RNNEncoder, GRLRNNEncoder
from seqmod.modules.decoder import RNNDecoder
from seqmod.modules.embedding import Embedding
from seqmod import utils as u

from seqmod.modules.exposure import schedule_sampling


class EncoderDecoder(nn.Module):
    """
    Configurable encoder-decoder architecture

    Parameters:
    -----------
    - encoder: BaseEncoder
    - decoder: BaseDecoder
    - exposure_rate: float (0.0, 1.0), initial exposure to model predictions
        during training.
    """
    def __init__(self, encoder, decoder, exposure_rate=1., train_data=None):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.exposure_rate = exposure_rate
        self.train_data = train_data

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

    def loss(self, batch_data, test=False, split=25, use_schedule=False):
        """
        Return batch-averaged loss and examples processed for speed monitoring
        """
        # unpack batch data
        (src, trg), (src_conds, trg_conds) = batch_data, (None, None)
        if self.encoder.conditional:
            (src, *src_conds) = src
        (src, src_lengths) = src
        if self.decoder.conditional:
            (trg, *trg_conds) = trg
        (trg, trg_lengths) = trg

        # word-level loss weight (weight down padding)
        weight = self.nll_weight
        if self.is_cuda():
            weight = weight.cuda()

        # remove <eos> from decoder targets, remove <bos> from loss targets
        dec_trg, loss_trg = trg[:-1], trg[1:]

        # compute model output
        enc_outs, enc_hidden = self.encoder(src, lengths=src_lengths)
        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, src_lengths, conds=trg_conds)

        dec_outs = []
        for step, t in enumerate(dec_trg):
            if use_schedule and step > 0 and self.exposure_rate < 1.0:
                t = schedule_sampling(
                    t, dec_outs[-1],
                    self.decoder.project,
                    self.exposure_rate)
                t = Variable(t, volatile=not self.training)
            out, _ = self.decoder(t, dec_state)
            dec_outs.append(out)
        dec_outs = torch.stack(dec_outs)

        # compute loss and backprop
        enc_losses, _ = self.encoder.loss(enc_outs, src_conds, test=test)

        # compute memory efficient word loss
        shard_data = {'out': dec_outs, 'trg': loss_trg}
        num_examples, loss = trg_lengths.data.sum(), 0

        for shard in u.shards(shard_data, size=split, test=test):
            out, true = shard['out'], shard['trg'].view(-1)
            pred = self.decoder.project(out)
            shard_loss = F.nll_loss(pred, true, weight, size_average=False)
            shard_loss /= num_examples
            loss += shard_loss

            if not test:
                shard_loss.backward(retain_graph=True)

        return (loss.data[0], *enc_losses), num_examples

    def translate(self, src, lengths, conds=None, max_decode_len=2):
        """
        Translate a single input sequence using greedy decoding.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x batch_size)

        Returns (scores, hyps, atts):
        --------
        scores: (batch_size)
        hyps: (batch_size x trg_seq_len)
        atts: (batch_size x trg_seq_len x source_seq_len)
        """
        eos = self.decoder.embeddings.d.get_eos()
        bos = self.decoder.embeddings.d.get_bos()
        seq_len, batch_size = src.size()

        enc_outs, enc_hidden = self.encoder(src)
        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, lengths, conds=conds)

        hyps, weights, scores = [], [], 0
        mask = src.data.new(batch_size).zero_().float() + 1
        prev = Variable(src.data.new([bos]).expand(batch_size), volatile=True)

        for _ in range(len(src) * max_decode_len):
            out, weight = self.decoder(prev, dec_state)
            # decode
            logprobs = self.decoder.project(out)
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

        return scores, hyps, weights

    def translate_beam(self, src, lengths, mask=None, conds=None,
                       beam_width=5, max_decode_len=2):
        """
        Translate a single input sequence using beam search.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x 1)
        lengths: torch.LongTensor (1)
        mask: (optional) see Decoder.init_state
        conds: (optional) conditions for the decoder
        """
        eos = self.decoder.embeddings.d.get_eos()
        bos = self.decoder.embeddings.d.get_bos()
        gpu = src.is_cuda

        weights = []

        enc_outs, enc_hidden = self.encoder(src)
        dec_state = self.decoder.init_state(
            enc_outs, enc_hidden, lengths, conds=conds, mask=mask)

        dec_state.expand_along_beam(beam_width)

        beam = Beam(beam_width, bos, eos=eos, gpu=gpu)

        while beam.active and len(beam) < len(src) * max_decode_len:
            # (width) -> (1 x width)
            prev = Variable(beam.get_current_state(), volatile=True)
            dec_out, weight = self.decoder(prev, dec_state)
            logprobs = self.decoder.project(dec_out)  # (width x vocab_size)
            beam.advance(logprobs.data)
            dec_state.reorder_beam(beam.get_source_beam())
            # TODO: add attention weight for decoded steps

        scores, hyps = beam.decode(n=beam_width)

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
        dropout=0.0,
        input_feed=True,
        word_dropout=0.0,
        deepout_layers=0,
        deepout_act='ReLU',
        tie_weights=False,
        reuse_hidden=True,
        train_init=False,
        add_init_jitter=False,
        cond_dims=None,
        cond_vocabs=None,
        train_data=None
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
    - dropout: float
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

    encoder = RNNEncoder(src_embeddings, hid_dim, num_layers, cell=cell,
                         bidi=bidi, dropout=dropout, summary=encoder_summary,
                         train_init=False, add_init_jitter=False)

    decoder = RNNDecoder(trg_embeddings, hid_dim, num_layers, cell=cell,
                         dropout=dropout, input_feed=input_feed,
                         att_type=att_type, deepout_layers=deepout_layers,
                         deepout_act=deepout_act,
                         tie_weights=tie_weights, reuse_hidden=reuse_hidden,
                         train_init=train_init,
                         add_init_jitter=add_init_jitter,
                         cond_dims=cond_dims, cond_vocabs=cond_vocabs)

    if decoder.has_attention and encoder_summary != 'full':
        raise ValueError("Attentional decoder needs full encoder summary")

    return EncoderDecoder(encoder, decoder, train_data=train_data)


def make_grl_rnn_encoder_decoder(
        num_layers,
        emb_dim,
        hid_dim,
        src_dict,
        trg_dict=None,
        cell='LSTM',
        bidi=True,
        encoder_summary='full',
        dropout=0.0,
        input_feed=True,
        word_dropout=0.0,
        deepout_layers=0,
        deepout_act='ReLU',
        tie_weights=False,
        train_init=False,
        add_init_jitter=False,
        cond_dims=None,
        cond_vocabs=None,
        train_data=None
):

    if encoder_summary == 'full':
        raise ValueError("GRL encoder can't use full summaries")

    if cond_dims is None or cond_vocabs is None:
        raise ValueError("GRL needs conditions")

    src_embeddings, trg_embeddings = make_embeddings(
        src_dict, trg_dict, emb_dim, word_dropout)

    encoder = GRLRNNEncoder(src_embeddings, hid_dim, num_layers, cell,
                            cond_dims=cond_dims, cond_vocabs=cond_vocabs,
                            bidi=bidi, dropout=dropout,
                            summary=encoder_summary,
                            train_init=False, add_init_jitter=True)

    decoder = RNNDecoder(trg_embeddings, hid_dim, num_layers, cell=cell,
                         dropout=dropout, input_feed=input_feed,
                         deepout_layers=deepout_layers,
                         deepout_act=deepout_act,
                         tie_weights=tie_weights, reuse_hidden=False,
                         train_init=train_init,
                         add_init_jitter=add_init_jitter,
                         cond_dims=cond_dims, cond_vocabs=cond_vocabs)

    return EncoderDecoder(encoder, decoder, train_data=train_data)
