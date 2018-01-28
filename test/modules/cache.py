
import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from seqmod.modules.cache import Cache


class CacheTest(unittest.TestCase):
    def test_add(self):
        def func(keys, current, memkeys, size=40):
            limit = min(size, current + keys.size(0))
            index = torch.arange(current, limit).long()
            memkeys.index_copy_(0, index, keys[:len(index)])
            # memvals.index_copy_(0, index, vals[:len(index)])
            current = limit % size

            if len(index) < len(keys):
                indexed = len(index)
                index = torch.arange(current, len(keys) - indexed).long()
                memkeys.index_copy_(0, index, keys[indexed:])
                # memvals.index_copy_(0, index, vals[indexed:])
                current = len(keys) - indexed

            return current

        keys = torch.rand(6, 1)
        current = 0
        memkeys = torch.zeros(10, 1)
        for i in range(10):
            keys = torch.rand(6, 1)
            newcurrent = func(keys, current, memkeys, size=10)
            limit = min(len(memkeys), current + len(keys))
            equals = memkeys[current: limit] == keys[:limit - current] - 1
            self.assertEqual(
                equals.nonzero().nelement(), 0,
                "Stored items don't agree with input items")

            if limit - current < len(keys):
                equals = memkeys[:newcurrent] == keys[-newcurrent:] - 1
                self.assertEqual(
                    equals.nonzero().nelement(), 0,
                    "Stored items don't agree with input items")
            current = newcurrent

    def test_query(self):
        # params
        size, batch, dim, vocab = 10, 5, 100, 15
        cache = Cache(dim, size, vocab)
        # insert some key-val pairs
        keys = torch.randn(15, batch, dim)
        vals = torch.LongTensor(15, batch).random_(vocab)
        cache.add(keys, vals)
        # get random query
        unnormed, vals = cache.query(torch.rand(batch, dim))
        # normalize output probability
        normed = cache.get_full_probs(F.log_softmax(Variable(unnormed), 1).data, vals)
        # index symbols in the cache
        index = cache.memvals + torch.arange(0, batch) \
                                     .unsqueeze(0) \
                                     .expand_as(cache.memvals) \
                                     .type_as(cache.memvals) * vocab
        # elements in the full output distribution retrieved by the index
        keyed = normed.view(-1).index_select(0, index.t().contiguous().view(-1))

        self.assertEqual(len(np.unique(keyed)), (normed != 0).sum(),
                         "Same number of nonzero entries in the full output dist "
                         "as in the cache")
        # we shouldn't have retrieved any zero elements
        self.assertEqual(keyed.nonzero().nelement(), index.nelement(),
                         "Elements are not in their right position. Probably "
                         "0-value elements were retrieved.")
