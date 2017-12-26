
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from seqmod import utils as u


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


class Embedding(nn.Embedding):
    @classmethod
    def from_dict(cls, d, emb_dim, p=0.0, padding_idx=None, **kwargs):
        if p > 0.0 and d.get_unk() is None:
            raise ValueError("Word dropout requires unknown token")

        inst = cls(len(d), emb_dim, padding_idx=d.get_pad(), **kwargs)
        inst.d, inst.p, inst.target_code = d, p, d.get_unk()
        codes = [d.get_eos(), d.get_bos(), d.get_pad()]
        inst.reserved_codes = tuple([c for c in codes if c is not None])

        return inst

    def forward(self, inp):
        inp = word_dropout(inp, self.target_code, p=self.p,
                           reserved_codes=self.reserved_codes,
                           training=self.training)

        return super(Embedding, self).forward(inp)

    def init_embeddings(self, weight, words, verbose=False):
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
        assert weight.size(1) == self.embedding_dim, \
            "Mismatched embedding dim {} for model with dim {}".format(
                weight.size(1), self.embedding_dim)

        self_idxs, other_idxs = [], []
        for other_idx, word in enumerate(words):
            try:
                self_idxs.append(self.d.s2i[word])
                other_idxs.append(other_idx)
            except KeyError:
                if verbose:
                    print("Couldn't find {} in dictionary".format(word))
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

        if mode is None:
            if 'glove' in os.path.basename(filepath).lower():
                mode = 'glove'
            elif 'fasttext' in os.path.basename(filepath).lower():
                mode = 'fasttext'
            else:
                raise ValueError("Unrecognized embedding type")

        weights, words = u.EmbeddingLoader(filepath, mode).load(words)
        self.init_embeddings(weights, words, **kwargs)


if __name__ == '__main__':
    from seqmod.misc.dataset import Dict
    import inspect
    import collections

    text = []
    for _, func in inspect.getmembers(collections):
        doc = func.__doc__
        if doc is not None:
            text.extend([l.split() for l in doc.split('\n')])

    d = Dict().fit(text)
    emb = Embedding.from_dict(d, 100, p=0.2)
    filepath = 'test/data/glove.test1000.100d.txt'
    emb.init_embeddings_from_file(filepath, 'glove')

    weights, words = u.EmbeddingLoader(filepath, 'glove').load()
    weights = torch.Tensor(weights)
    by_word = dict(zip(words, weights))

    for weight, word in zip(emb.weight.data, emb.d.vocab):
        if word in by_word:
            assert torch.equal(weight, by_word[word])

    inp = Variable(torch.LongTensor(10).random_(emb.num_embeddings))
