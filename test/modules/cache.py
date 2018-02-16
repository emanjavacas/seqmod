
import unittest

import torch

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

    def test_add_sequence(self):
        size, batch, dim, vocab = 10, 5, 100, 15
        cache = Cache(dim, size, vocab)
        # insert some key-val pairs
        keys = torch.randn(15, batch, dim)
        vals = torch.LongTensor(15, batch).random_(vocab)
        for k, v in zip(keys, vals):
            cache.add(k.unsqueeze(0), v.unsqueeze(0))

        for idx, v in enumerate(vals[-size:]):
            checked = False
            for c in cache.memvals:
                if (v == c).sum() == batch:
                    checked = True
                    break
            self.assertTrue(checked, 'row {} was found in the cache'.format(idx))
            checked = False
