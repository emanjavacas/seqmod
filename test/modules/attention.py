
import unittest

import torch


class Sum4DTest(unittest.TestCase):
    def test_sum(self):
        trg_seq_len, src_seq_len, batch, dim = 7, 5, 2, 4
        x = torch.rand(src_seq_len, batch, dim)
        y = torch.rand(trg_seq_len, batch, dim)

        test_output = torch.zeros(src_seq_len, trg_seq_len, batch, dim)
        for i in range(src_seq_len):
            for j in range(trg_seq_len):
                test_output[i, j] = x[i] + y[j]

        batched_output = (
            x.unsqueeze(0)
             .repeat(y.size(0), 1, 1, 1)
             .transpose(0, 1)
            + y.unsqueeze(0))

        self.assertTrue((test_output == batched_output).all())
