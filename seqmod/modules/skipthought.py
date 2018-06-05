
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.softmax import FullSoftmax, SampledSoftmax


def run_decoder(decoder, thought, hidden, inp, lengths):
    """
    Single decoder run
    """
    # project thought across length dimension
    inp = torch.cat([inp, thought.unsqueeze(0).repeat(len(inp), 1, 1)], dim=2)

    # sort faster processing
    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()
    if isinstance(hidden, tuple):
        hidden = hidden[0][:,sort,:], hidden[1][:,sort,:]
    else:
        hidden = hidden[:,sort,:]

    # pack
    inp = pack(inp[:, sort], lengths.tolist())

    # run decoder
    output, _ = decoder(inp, hidden)

    # unpack & unsort
    output, _ = unpack(output)
    output = output[:, unsort]

    return output


class SkipthoughtLoss(nn.Module):

    def __init__(self, embeddings, cell, thought_dim, hid_dim,
                 dropout=0.0, mode='prev+post', clone=False, softmax='full'):

        self.mode = mode.lower().split('+')
        if sum([part in ('prev', 'same', 'post') for part in self.mode]) == 0:
            raise ValueError("Needs at least one target sentence but got mode: {}"
                             .format(mode))

        self.hid_dim = hid_dim
        super(SkipthoughtLoss, self).__init__()

        # Embedding
        self.embeddings = embeddings
        inp_size = thought_dim + embeddings.embedding_dim

        # RNN
        prev = getattr(nn, cell)(inp_size, hid_dim) if 'prev' in self.mode else None
        same = getattr(nn, cell)(inp_size, hid_dim) if 'same' in self.mode else None
        post = getattr(nn, cell)(inp_size, hid_dim) if 'post' in self.mode else None
        if clone:
            if prev is None or post is None:
                raise ValueError("Can't clone decoders if `prev` or `post` is missing")
            prev = post
        self._decoders = nn.ModuleList([prev, same, post])

        nll_weight = torch.ones(len(embeddings.d))
        if embeddings.d.get_pad() is not None:
            nll_weight[embeddings.d.get_pad()] = 0
        self.register_buffer('nll_weight', nll_weight)

        if softmax == 'full':
            self.logits = FullSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings)
        elif softmax == 'tied':
            self.logits = FullSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings,
                tie_weights=True)
            self.logits.tie_embedding_weights(embeddings)
        elif softmax == 'sampled':
            self.logits = SampledSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings)
        else:
            raise ValueError("Unknown softmax {}".format(softmax))

    def forward(self, thought, hidden, sents):
        for idx, (sent, rnn) in enumerate(zip(sents, self._decoders)):
            if sent is not None:
                if rnn is None:
                    raise ValueError("Unexpected input at pos {}".format(idx + 1))
    
                (sent, lengths) = sent
                # rearrange targets for loss
                inp, target, lengths = sent[:-1], sent[1:], lengths - 1
                num_examples = lengths.sum().item()
                # run decoder
                inp = self.embeddings(inp)
                output = run_decoder(rnn, thought, hidden, inp, lengths)
    
                yield output, target, num_examples

    def loss(self, thought, hidden, sents, test=False):
        loss, num_examples, report_loss = 0, 0, [0, 0, 0]

        output = self.forward(thought, hidden, sents)
        for idx, (out, trg, examples) in enumerate(output):
            out, trg = out.view(-1, self.hid_dim), trg.view(-1)
            dec_loss = 0

            if isinstance(self.logits, SampledSoftmax) and self.training:
                out, new_trg = self.logits(out, targets=trg, normalize=False)
                dec_loss = F.cross_entropy(out, new_trg, size_average=False)
            else:
                dec_loss = F.cross_entropy(
                    self.logits(out, normalize=False), trg, size_average=False,
                    weight=self.nll_weight)

            dec_loss /= examples
            loss += dec_loss

            # report
            report_loss[idx] = dec_loss.item()
            num_examples += examples

        if not test:
            loss.backward()

        return tuple(report_loss), num_examples


class Skipthought(nn.Module):
    def __init__(self, embeddings, mode, softmax='full', cell='GRU', hid_dim=2400,
                 num_layers=1, bidi=True, summary='last', dropout=0.0):
        super(Skipthought, self).__init__()

        self.encoder = RNNEncoder(embeddings, hid_dim, num_layers, cell,
                                  bidi=bidi, dropout=dropout, summary=summary,
                                  train_init=False, add_init_jitter=False)

        self.decoder = SkipthoughtLoss(
            embeddings, cell, self.encoder.encoding_size[1], hid_dim,
            mode=mode, dropout=dropout, softmax=softmax)

    def loss(self, batch_data, test=False):
        (inp, lengths), sents = batch_data
        thought, hidden = self.encoder(inp, lengths)
        losses, num_examples = self.decoder.loss(thought, hidden, sents)
        return losses, num_examples
