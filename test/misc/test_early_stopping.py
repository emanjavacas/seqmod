
import unittest

from seqmod.misc import early_stopping


class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.maxsize = 5
        self.patience = 3
        self.e_s = early_stopping.EarlyStopping(self.maxsize, self.patience)

    def test_queue(self):
        run = [2.5, 3.0, 2.0, 1.5]
        for checkpoint in run:
            self.e_s.add_checkpoint(checkpoint)
        priority, _ = self.e_s.pop()  # EarlyStopping is just a queue
        self.assertEqual(priority, max(run))
        priority, _ = self.e_s.get_min()
        self.assertEqual(priority, min(run))

    def test_check(self):
        run = [3.0, 2.5, 2.0, 1.5, 0.9, 1.0, 1.1, 1.2, 1.0]
        # best is 0.99 at position 4, error at previous to last position
        run_test = False
        for idx, checkpoint in enumerate(run):
            try:
                self.e_s.add_checkpoint(checkpoint, model=idx)
            except early_stopping.EarlyStoppingException as e:
                run_test = True
                message, data = e.args
                self.assertEqual(len(self.e_s.checks), len(run) - 1)
                self.assertEqual(data['model'] + 1, 5)  # 0-index
                self.assertEqual(self.e_s.find_smallest(), 5)
                self.assertEqual(data['smallest'], 0.9)
        self.assertTrue(run_test)

    def test_early_stopping1(self):
        des_run, asc_run = [3.0, 2.5, 2.0, 1.5, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]
        run_test = False
        for checkpoint in des_run + asc_run:
            try:
                self.e_s.add_checkpoint(checkpoint)
            except early_stopping.EarlyStoppingException as e:
                message, data = e.args
                self.assertEqual(data['smallest'], min(des_run))
                self.assertEqual(checkpoint, asc_run[self.patience-1])
                run_test = True
                break
        self.assertTrue(run_test)

    def test_early_stopping2(self):
        run = [3.0, 2.5, 2.0, 1.5, 1.0, 1.1, 1.2, 0.9]
        for checkpoint in run:
            self.e_s.add_checkpoint(checkpoint)
        priority, _ = self.e_s.get_min()
        self.assertEqual(priority, min(run))

    def test_reset(self):
        run = [3, 2.9, 2.8, 2.9, 3.0, 3.1]
        # run resetting (shouldn't throw any exceptions)
        for checkpoint in run:
            self.e_s.add_checkpoint(checkpoint)
        # run without resetting (should error)
        self.e_s = early_stopping.EarlyStopping(
            self.maxsize, self.patience, reset_patience=False)
        run_test = False
        for checkpoint in run:
            try:
                self.e_s.add_checkpoint(checkpoint)
            except early_stopping.EarlyStoppingException as e:
                run_test = True
                message, data = e.args
                self.assertEqual(data['smallest'], min(run))
        self.assertTrue(run_test)
