
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Cache(object):
    """
    Continuous Cache for Neural Models following: https://arxiv.org/abs/1612.04426

    Parameters:
    -----------
    dim: int, dimensionality of the keys
    size: int, size of the cache
    vocab: int, size of the output vocabulary
    theta: float, flattening parameter (similar to temperature), lower values
        flatten the output scores. for zero theta, the distribution is flattened.

    mode: mixture mode:
        - linear: linear interpolation. `alpha` corresponds to a weight assigned
            to the probability distribution of the cache. Range in (0, 1).
        - global: global normalization. `alpha` corresponds to to a weight to
            increase the importance of the probability distribution of the cache.
            Range usually in (0, 4).
    """
    def __init__(self, dim, size, vocab,
                 theta=1.0, alpha=0.5, mode='linear', gpu=False):

        self.dim = dim
        self.size = size
        self.vocab = vocab
        self.theta = theta
        self.alpha = alpha
        self.mode = mode.lower()
        self.gpu = gpu

        if self.mode not in ('linear', 'global'):
            raise ValueError("Unknown mode \"{}\"".format(mode))

        self.stored = 0         # number of items stored in the cache
        self.current = 0        # index along size that should get written
        self.memkeys = torch.FloatTensor(self.size, 1, self.dim).zero_()
        self.memvals = torch.LongTensor(self.size, 1).zero_()
        self.memkeys = self.memkeys.cuda() if gpu else self.memkeys
        self.memvals = self.memvals.cuda() if gpu else self.memvals

    def reset(self):
        self.stored = 0
        self.current = 0
        self.memkeys.zero_()
        self.memvals.zero_()

    def add(self, keys, vals):
        """
        Parameters:
        -----------

        keys: torch.Tensor(n, batch, dim)
        vals: torch.LongTensor(n, batch)
        """
        if keys.size()[:-1] != vals.size():
            raise ValueError("Wrong key-val dims. Keys: {}, vals: {}".format(
                str(keys.size()), str(vals.size())))

        batch = keys.size(1)

        if self.memkeys.size(1) == 1 and batch > 1:
            # expand along batch dimension
            self.memkeys = self.memkeys.repeat(1, batch, 1)
            self.memvals = self.memvals.repeat(1, batch)

        if self.memkeys.size(1) != batch:
            raise ValueError(
                "Wrong batch dimension. Expected {} but got {} elements".format(
                    self.memkeys.size(1), batch))

        if keys.size(0) > self.size:
            keys, vals = keys[-self.size:], vals[-self.size:]

        limit = min(self.size, self.current + keys.size(0))
        index = torch.arange(self.current, limit).long()
        index = index.cuda() if self.gpu else index
        self.memkeys.index_copy_(0, index, keys[:len(index)])
        self.memvals.index_copy_(0, index, vals[:len(index)])
        self.current = limit % self.size

        if len(index) < len(keys):
            indexed = len(index)
            index = torch.arange(self.current, len(keys) - indexed).long()
            index = index.cuda() if self.gpu else index
            self.memkeys.index_copy_(0, index, keys[indexed:])
            self.memvals.index_copy_(0, index, vals[indexed:])
            self.current = len(keys) - indexed

        self.stored = min(self.size, self.stored + len(keys))

    # TODO: allow for processing multiple queries at once
    # expanding the memcache and memvals along the batch dim
    def query(self, query):
        """
        Return scores for words in the cache given an input query.

        Parameters:
        -----------
        query: torch.Tensor(batch, hid_dim)

        Returns:
        --------
        scores: torch.Tensor(batch, size), output is just the dotproduct
            with the keys in the cache
        vals: torch.LongTensor(batch, size)
        """
        # select full entries
        memkeys, memvals = self.memkeys[:self.stored], self.memvals[:self.stored]
        # dot product => (batch x size)
        scores = (memkeys * query.unsqueeze(0)).sum(2)

        return scores.t(), memvals.t()
