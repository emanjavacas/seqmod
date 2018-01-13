
import numpy as np

from itertools import accumulate

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from seqmod.loaders import EmbeddingLoader
from seqmod.modules.torch_utils import init_hidden_for, pack_sort, split, pad_sequence
from seqmod.modules.conv_utils import get_padding


def word_dropout_mask(X, dropout_rate, reserved_codes=()):
    """
    Computes a binary mask across batch examples based on a
    bernoulli distribution with mean equal to dropout.
    """
    probs = torch.zeros_like(X).float() + dropout_rate
    # zero reserved_codes (avoid dropping reserved symbols)
    if len(reserved_codes) > 0:
        probs[sum((X == x) for x in reserved_codes)] = 0
    # return binary mask
    return torch.bernoulli(probs).byte()


def word_dropout(inp, target, p=0.0, training=True, reserved_codes=()):
    """
    Applies word dropout to an input Variable. Dropout isn't constant
    across batch examples. This is only to be used to drop input symbols
    (i.e. before the embedding layer)

    Parameters:
    -----------
    - inp: torch.LongTensor
    - target: int, code to use as replacement for dropped timesteps
    - dropout: float, dropout rate
    - reserved_codes: tuple of ints, ints in the input that should never
        be dropped
    - training: bool
    """
    if not training or p == 0:
        return inp

    mask = word_dropout_mask(inp.data, p, reserved_codes=reserved_codes)

    return inp.masked_fill(Variable(mask), target)


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for non-recurrent neural
    networks. Implementation based on https://arxiv.org/abs/1706.03762

    Based on: https://github.com/OpenNMT/OpenNMT-py/onmt/modules/Embeddings.py

    Parameters:
    -----------

    - emb_dim: int, embedding size
    - max_len: int (optional), default to 5000
    """

    def __init__(self, emb_dim, max_len=5000):
        self.emb_dim = emb_dim

        # precompute the sin and cos values into pe
        # pe is of shape (max_len x 1 x emb_dim)
        # even and odd entries along the 3th dimensions correspond
        # to sin and cos waves over i (index along the 1st dim)
        # divided by 10000^(2*i/emb_dim)
        pe = torch.arange(0, max_len) \
                  .unsqueeze(1).expand(max_len, emb_dim)
        power = torch.arange(0, emb_dim * 2, 2) / emb_dim
        pe = pe * (1 / torch.pow(10000, power)).expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(1)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)

    def forward(self, emb, pos=None):
        """
        Parameters:
        -----------

        - emb: FloatTensor ((seq_len, ), batch, emb_dim)
        """
        if emb.dim() == 3:
            pos_emb = self.pe[:emb.size(0), :1, :emb.size(2)]
        else:
            if pos is None:
                raise ValueError(
                    "2D input to positional encoding needs `pos` input")

            pos_emb = self.pe[pos, :1, :emb.size(1)]

        return emb + Variable(pos_emb.expand_as(emb), requires_grad=False)


class Embedding(nn.Embedding):
    """
    Custom Embedding class with all bells and whistles as well as
    seqmod Dict integration.
    """

    # ignore this
    _SECRET = 'secret!'

    def __init__(self, *args, _secret=None, **kwargs):
        if _secret != Embedding._SECRET:
            raise ValueError("This class must be instantiated "
                             "with the `from_dict` classmethod")

        super(Embedding, self).__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, d, emb_dim, p=0.0,
                  add_positional=False, max_len=5000,
                  padding_idx=None, **kwargs):
        """
        Instantiate an Embedding module from a fitted Dict.

        Parameters:
        -----------

        d: Dict
        emb_dim: int
        p: float, word dropout rate (reset input symbols to unknowns)
        add_positional: bool, whether to add positional encoding
            information to the embedding activations.
        max_len: int, see PositionalEncoding
        padding_idx: None, will get ignored in favor of the Dict's own pad token
        kwargs: rest nn.Embedding parameters
        """

        if p > 0.0 and d.get_unk() is None:
            raise ValueError("Word dropout requires unknown token")

        inst = cls(len(d), emb_dim, padding_idx=d.get_pad(),
                   _secret=Embedding._SECRET, **kwargs)
        inst.d, inst.p, inst.target_code = d, p, d.get_unk()
        codes = [d.get_eos(), d.get_bos(), d.get_pad()]
        inst.reserved_codes = tuple([c for c in codes if c is not None])

        if add_positional:
            inst.add_positional(max_len)

        return inst

    def add_positional(self, max_len):
        """
        Add elementwise positional embeddings to the output
        """
        positional = PositionalEncoding(self.embedding_dim, max_len=max_len)
        self.add_module('positional', positional)
        self.positional = positional

    def forward(self, inp, pos=None):
        inp = word_dropout(inp, self.target_code, p=self.p,
                           reserved_codes=self.reserved_codes,
                           training=self.training)

        inp = super(Embedding, self).forward(inp)

        if hasattr(self, 'positional'):
            inp = self.positional(inp, pos=pos)

        return inp

    def init_embeddings(self, weight, words):
        """
        Load embeddings from a weight matrix with words `words` as rows.

        Parameters
        -----------
        - weight: (vocab x emb_dim)
        - words: list of word indices corresponding to each row in `weight`
        """
        # wrap in tensor
        if isinstance(weight, list):
            weight = torch.Tensor(weight).float()
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float()
        # check embedding size
        if weight.size(1) != self.embedding_dim:
            raise ValueError("Mismatched embedding dim {} for model "
                             "with dim {}".format(weight.size(1),
                                                  self.embedding_dim))

        self_idxs, other_idxs = [], []
        for other_idx, word in enumerate(words):
            try:
                self_idxs.append(self.d.s2i[word])
                other_idxs.append(other_idx)
            except KeyError:
                pass

        other_idxs = torch.LongTensor(other_idxs)
        self_idxs = torch.LongTensor(self_idxs)
        self.weight.data[self_idxs] = weight[other_idxs]

    def init_embeddings_from_file(self, filepath, mode=None, **kwargs):
        """
        Initialize embeddings from a file with a specified format (mode)

        - filepath: str, path to embeddings file
        - mode: (optional) str, embedding type (glove, fasttext). If not given,
            it will be guessed based on the filename.
        """
        words = self.d.vocab
        weight, words = EmbeddingLoader(filepath, mode).load(words, **kwargs)
        self.init_embeddings(weight, words)


def flatten_batch(inp, lengths, breakpoint_idx):
    # flatten (char_len x batch x dim) into (max_word_len x num_words x dim)
    flattened, word_lenghts = [], []
    # decrease since we are going to use them for indexing
    lengths = lengths.data - 1

    for idx in range(inp.size(1)):
        seq = inp[:, idx]
        breakpoints = (seq.data == breakpoint_idx).nonzero()
        # remove trailing dim added by nonzero()
        breakpoints = breakpoints.squeeze(1).tolist()
        breakpoints.append(lengths[idx])
        # accumulate sequence lengths in terms of words
        word_lenghts.append(len(breakpoints))
        flattened.extend(split(seq, breakpoints))

    return flattened, word_lenghts


class ComplexEmbedding(nn.Module):
    @property
    def is_complex(self):
        return True

    def forward(self, inp, lengths):
        raise NotImplementedError


class RNNEmbedding(ComplexEmbedding):
    """
    Use a RNN at the character level to extract word-level features (aka. word
    embeddings).
    """

    # ignore this
    _SECRET = 'secret!'

    def __init__(self, num_embeddings, embedding_dim, breakpoint_idx,
                 num_layers=1, cell='GRU', bias=True, bidirectional=False,
                 contextual=False, dropout=0.0, _secret=None):

        if _secret != Embedding._SECRET:
            raise ValueError("This class must be instantiated "
                             "with the `from_dict` classmethod")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = int(bidirectional) + 1
        self.cell = cell
        self.breakpoint_idx = breakpoint_idx
        self.contextual = contextual
        super(RNNEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        layers = []
        hidden_size = embedding_dim // self.num_dirs
        input_size = embedding_dim
        for layer in range(num_layers):
            layers.append(getattr(nn, cell)(
                input_size, hidden_size, bidirectional=bidirectional,
                bias=bias, dropout=dropout))
            input_size = hidden_size * self.num_dirs
        self.layers = nn.ModuleList(layers)

    @classmethod
    def from_dict(cls, d, emb_dim, breakpoint_token=' ', **kwargs):
        try:
            breakpoint_idx = d.s2i[breakpoint_token]
        except KeyError:
            raise ValueError("Breakpoint_token `{}` not in vocabulary"
                             .format(breakpoint_token))
        return cls(len(d), emb_dim, breakpoint_idx=breakpoint_idx,
                   _secret=RNNEmbedding._SECRET, **kwargs)

    def _run_rnn(self, inp, lengths):
        hidden = init_hidden_for(
            inp, self.num_dirs, self.num_layers,
            self.embedding_dim // self.num_dirs, self.cell,
            training=self.training)

        rnn_inp = inp
        rnn_inp, unsort = pack_sort(rnn_inp, lengths)

        for layer in self.layers:
            rnn_inp, hidden = layer(rnn_inp, hidden)

        rnn_inp, _ = unpack(rnn_inp)
        rnn_inp = rnn_inp[:, unsort]

        return rnn_inp

    def _forward_with_context(self, inp, lengths):
        # (char_len x batch x dim)
        out = self._run_rnn(self.embedding(inp), lengths.data)
        embs = []

        # iterate over batch
        for i in range(inp.size(1)):
            breakpoints = (inp[:, i] == self.breakpoint_idx).nonzero()
            breakpoints = torch.cat([breakpoints.squeeze(1), lengths[i]-1])
            embs.append(out[:, i][breakpoints])

        embs, lengths = pad_sequence(embs)

        return embs.transpose(0, 1), lengths

    def _forward(self, inp, lengths):
        # flatten batch to sequence-independent words
        flattened, word_lenghts = flatten_batch(inp, lengths, self.breakpoint_idx)
        flattened, char_lenghts = pad_sequence(flattened)
        flattened = flattened.transpose(0, 1)

        # embed characters (max_word_len x num_words x emb_dim)
        flattened = self.embedding(flattened)

        # take last rnn activation as embedding: (num_words x emb_dim)
        char_lenghts = torch.LongTensor(char_lenghts)
        if flattened.is_cuda:
            char_lenghts = char_lenghts.cuda()
        *_, out = self._run_rnn(flattened, char_lenghts)

        # reshape to original sequence: (max_seq_len x batch x emb_dim)
        embs, _ = pad_sequence(split(out, list(accumulate(word_lenghts))))

        return embs.transpose(0, 1), word_lenghts

    def forward(self, inp, lengths):
        """
        Parameters:
        -----------

        - inp: torch.Variable(seq_len x batch_size) where seq_len is given
            in terms of character length.
        - lengths: list of number of real characters (besides pad symbols)
            in the input batch.
        """
        if self.contextual:
            return self._forward_with_context(inp, lengths)
        else:
            return self._forward(inp, lengths)

        # extract hidden outputs at breakpoints along the first dim (seq_len)
        # - warning: currently the challenge is to do this without copying data
        #     into a new variable given that it assumes different lengths per
        #     batch. Perhaps the easiest way is:
        #       - mask out all items not corresponding to breakpoint
        #       - permute to get all embeddings to the beginning of the batch
        #       - crop down to max length
        #     Another way of doing this would be:
        #       - conflate seq_len and batch dims (seq_len * batch, emb_dim)
        #       - select the embeddings corresponding to breakpoints
        #       - split by batch length (determined by the number of breakpoints)
        #       - pad them into a sequence with torch.nn.utils.rnn.pad_sequence


class CNNEmbedding(ComplexEmbedding):
    """
    Use multiple CNN filters at the character level to extract word-level
    features (aka. word embeddings).

    In contrast to the traditional nn.Embedding, the input parameter `embedding_dim`
    is used for the character-level embeddings over which the convolutions
    are applied. In contrast to RNNEmbedding, where the output embeddings do have
    the expected dimensionality, in CNNEmbedding the output embedding_dim
    corresponds to the sum of the output_channels. For convenience, it can be
    retrieved accessing the property self.embedding_dim.
    """

    # ignore this
    _SECRET = 'secret!'

    def __init__(self, num_embeddings, embedding_dim, breakpoint_idx,
                 kernel_sizes=range(1, 7), output_channels=lambda x: x * 25,
                 _secret=None):

        if _secret != Embedding._SECRET:
            raise ValueError("This class must be instantiated "
                             "with the `from_dict` classmethod")

        self.num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.max_kernel = max(kernel_sizes)
        if callable(output_channels):
            self.output_channels = [output_channels(k) for k in kernel_sizes]
        else:
            if not isinstance(output_channels, (tuple, list)):
                raise ValueError("`output_channels` must be tuple or func")
            if not len(output_channels) == len(kernel_sizes):
                raise ValueError(
                    "needs same number of `output_channels` and `kernel_sizes`")
            self.output_channels = output_channels

        self.breakpoint_idx = breakpoint_idx
        super(CNNEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        convs = []
        for W, C_o in zip(self.kernel_sizes, self.output_channels):
            convs.append(
                nn.Conv2d(1, C_o, (self._embedding_dim, W),
                          padding=(0, get_padding(W, mode="wide"))))
        self.convs = nn.ModuleList(convs)

    @property
    def embedding_dim(self):
        return sum(self.output_channels)

    @classmethod
    def from_dict(cls, d, emb_dim, breakpoint_token=' ', **kwargs):
        try:
            breakpoint_idx = d.s2i[breakpoint_token]
        except KeyError:
            raise ValueError("Breakpoint_token `{}` not in vocabulary"
                             .format(breakpoint_token))
        return cls(len(d), emb_dim, breakpoint_idx=breakpoint_idx,
                   _secret=CNNEmbedding._SECRET, **kwargs)

    def forward(self, inp, lengths):
        flattened, word_lenghts = flatten_batch(inp, lengths, self.breakpoint_idx)

        # (num_words x max_word_len)
        flattened, _ = pad_sequence(flattened)
        # (num_words x max_word_len x emb_dim)
        flattened = self.embedding(flattened)

        if flattened.size(1) < self.max_kernel:
            flattened = F.pad(
                # pad second dim to the right if necessary
                flattened, (0, 0, 0, self.max_kernel - flattened.size(1)))

        # (num_words x 1 x emb_dim x max_word_len)
        flattened = flattened.transpose(1, 2).unsqueeze(1)

        embs = []
        for conv in self.convs:
            # (num_words x C_o x max_word_len)
            emb = F.tanh(conv(flattened)).squeeze(2)
            # (num_words x C_o x 1)
            emb = F.max_pool1d(emb, emb.size(2)).squeeze(2)
            embs.append(emb)

        embs = torch.cat(embs, 1)
        # reshape to original sequence: (max_seq_len x batch x emb_dim)
        embs, lengths = pad_sequence(split(embs, list(accumulate(word_lenghts))))

        return embs.transpose(0, 1), lengths


if __name__ == '__main__':
    from seqmod.misc.dataset import Dict, pad_batch
    from seqmod.utils import PAD
    from string import ascii_letters
    import random
    import timeit

    def generate_word(n):
        maxchars = len(ascii_letters) - 1
        return ''.join(ascii_letters[random.randint(0, maxchars)]
                       for _ in range(n))

    def generate_sent(maxlen=10):
        return ' '.join(generate_word(random.randint(1, 15))
                        for _ in range(random.randint(5, maxlen)))

    def create_batches(batch_size, corpus, d):
        corpus = list(d.transform(corpus))
        inps, lengths = [], []
        num_batches = len(corpus) // batch_size
        prev = 0
        for b in range(num_batches):
            to = min((b + 1) * batch_size, len(corpus) - 1)
            inp, length = pad_batch(corpus[prev: to], d.get_pad(), True, False)
            inps.append(Variable(torch.LongTensor(inp)))
            lengths.append(Variable(torch.LongTensor(length)))
            prev = to

        return inps, lengths

    char_text = [generate_sent() for _ in range(100)]
    word_text = [s.split() for s in char_text]
    char_d = Dict(pad_token=PAD).fit(char_text)
    word_d = Dict(pad_token=PAD).fit(word_text)

    char_inps, char_lengths = create_batches(10, char_text, char_d)
    word_inps, word_lengths = create_batches(10, word_text, word_d)
    n_char_inps = sum(sum(l.data.tolist()) for l in char_lengths)
    n_word_inps = sum(sum(l.data.tolist()) for l in word_lengths)

    def make_word_embedding_runner(embedding):
        def runner():
            for word_inp in word_inps:
                embedding(word_inp)

        return runner

    def make_char_embedding_runner(embedding):
        def runner():
            for char_inp, char_length in zip(char_inps, char_lengths):
                embedding(char_inp, char_length)

        return runner

    embedding = Embedding.from_dict(word_d, 100, p=0.2)
    rnn_embedding_context = RNNEmbedding.from_dict(char_d, 100, contextual=True)
    rnn_embedding = RNNEmbedding.from_dict(char_d, 100)
    cnn_embedding = CNNEmbedding.from_dict(char_d, 30)

    emb_types = {
        'embedding': embedding,
        'rnn_embedding_context': rnn_embedding_context,
        'rnn_embedding': rnn_embedding,
        'cnn_embedding': cnn_embedding
    }

    runners = {
        'embedding': make_word_embedding_runner(embedding),
        'rnn_embedding': make_char_embedding_runner(rnn_embedding),
        'rnn_embedding_context': make_char_embedding_runner(rnn_embedding_context),
        'cnn_embedding': make_char_embedding_runner(cnn_embedding)
    }

    for emb_type in emb_types:
        runner = runners[emb_type]
        print("Benchmarking {}".format(emb_type))
        print(" * n parameters: {}".format(
            sum(p.nelement() for p in emb_types[emb_type].parameters())))
        total_inps = n_word_inps if emb_type == 'embedding' else n_char_inps
        print(" * n inputs: {}".format(total_inps))
        print(min(timeit.repeat(runner, number=10, repeat=10)))
        print()

    # Benchmarking embedding
    #  * n parameters: 73500
    #  * n inputs: 745
    # 0.006557075001182966
    #

    # Benchmarking rnn_embedding_context
    #  * n parameters: 66100
    #  * n inputs: 6681
    # 3.367520097999659
    #

    # Benchmarking rnn_embedding
    #  * n parameters: 66100
    #  * n inputs: 6681
    # 3.2836688839997805
    #

    # Benchmarking cnn_embedding
    #  * n parameters: 70425
    #  * n inputs: 6681
    # 1.945244102000288
    #
