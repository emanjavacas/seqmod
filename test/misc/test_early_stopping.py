
import unittest

from seqmod.misc import early_stopping


tests = [
    # Add tests here with the expected behaviour:
    # If the exception should be raised, add a field 'should_raise' and
    # set it to True, otherwise False.
    # You can overwrite default params 'maxsize', 'reset_patience' and
    # 'reset_on_emptied'. See `make_es` for further defaults.
    {
        'run': [2.5, 3.0, 2.0, 1.5],
        'should_raise': False
    }, {
        'run': [3.0, 2.5, 2.0, 1.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'patience': 4,
        'should_raise': True
    }, {
        'run': [3.0, 2.5, 2.0, 1.5, 0.9, 1.0, 1.1, 1.2, 1.0],
        'patience': 4,
        'should_raise': True
    }, {
        'run': [3, 4, 5, 2, 3, 4],
        'reset_patience': False,
        'patience': 3,
        'should_raise': True
    }, {
        'run': [5, 4, 3, 2, 3],
        'patience': 1,
        'should_raise': True
    }
]

test_emptying = {
    'run': [2.5, 3.0, 2.0, 1.5],
    'patience': 2,
    'maxsize': 3,
    'should_raise': False,
}

test_queue = {
    'run': [2.5, 3.0, 2.0, 1.5],
    'should_raise': False
}

test_reset = {
    'run': [3, 2.8, 2.9, 3.0, 2.7, 3.1],
    'patience': 3
}


def get_best_idx(run):
    "get position of the smallest checkpoint"
    return sorted(enumerate(run), reverse=False, key=lambda i: i[1])[0][0]


def get_checkpoints(test):
    "get the number of checkpoints that should be run for a test"
    es = make_es(test)
    best = get_best_idx(test['run'])

    if test['should_raise']:
        if es.reset_patience:
            # run until the smallest plus patience (best is 0-index)
            checkpoints = es.patience + best + 1
        else:
            # find the actual number of checkpoints before failing
            checkpoints, fails, best = 0, 0, float('inf')
            for i in test['run']:
                checkpoints += 1
                if i < best:
                    best = i
                else:
                    fails += 1
                if fails >= es.patience:
                    break
    else:
        # run through
        checkpoints = len(test['run'])

    return checkpoints


def make_es(test, patience=5, maxsize=10,
            reset_patience=True, reset_on_emptied=False):
    return early_stopping.EarlyStopping(
        test.get('patience', patience),
        maxsize=test.get('maxsize', maxsize),
        reset_on_emptied=test.get('reset_on_emptied', reset_on_emptied),
        reset_patience=test.get('reset_patience', reset_patience))


class TestEarlyStopping(unittest.TestCase):
    def _test_early_stopping(self, test, test_id):
        es = make_es(test)
        best = get_best_idx(test['run'])
        checkpoints = get_checkpoints(test)

        run_test = False

        for idx, checkpoint in enumerate(test['run']):
            try:
                es.add_checkpoint(checkpoint, model=idx)
            except early_stopping.EarlyStoppingException as e:
                run_test = True
                message, data = e.args
                self.assertEqual(
                    len(es.checks), checkpoints,
                    "check number of registered checkpoints: {}".format(test_id))
                self.assertEqual(
                    data['model'], best,
                    "check best registered id/model: {}".format(test_id))
                # _find_smallest is 1-index
                self.assertEqual(
                    es._find_smallest(), best + 1,
                    "check smallest registered checkpoint id: {}".format(test_id))
                self.assertEqual(
                    data['smallest'], test['run'][best],
                    "check smallest registered checkpoint: {}".format(test_id))

        self.assertEqual(
            run_test, test['should_raise'],
            "check raised: {}".format(test_id))

    def test_early_stopping(self):
        for idx, test in enumerate(tests):
            self._test_early_stopping(test, idx + 1)

    def test_queue(self):
        es = make_es(test_queue)

        for checkpoint in test_queue['run']:
            es.add_checkpoint(checkpoint)

        # EarlyStopping is just a queue
        priority, _ = es.pop()  # popping first item added (also largest check)
        self.assertEqual(priority, max(test_queue['run']))
        priority, _ = es.get_min()
        self.assertEqual(priority, min(test_queue['run']))

    def test_emptying(self):
        es = make_es(test_emptying)

        for checkpoint in test_emptying['run']:
            es.add_checkpoint(checkpoint)

        # EarlyStopping is just a queue
        priority, _ = es.pop()
        # only item in the queue, since it got emptied after adding the 4th check
        self.assertEqual(priority, min(test_emptying['run']))

    def test_reset(self):
        test_reset['should_raise'] = False
        test_reset['reset_patience'] = True
        es = make_es(test_reset)

        # run resetting (shouldn't throw any exceptions)
        for checkpoint in test_reset['run']:
            es.add_checkpoint(checkpoint)

        smallest, _ = es.get_min()
        self.assertEqual(smallest, min(test_reset['run']),
                         'best checkpoint won')
        self.assertEqual(len(es.checks), len(test_reset['run']),
                         'all checkpoints got registered')

        # run without resetting (should error)
        test_reset['should_raise'] = True
        test_reset['reset_patience'] = False
        es = make_es(test_reset)

        run_test = False
        for checkpoint in test_reset['run']:
            try:
                es.add_checkpoint(checkpoint)
            except early_stopping.EarlyStoppingException as e:
                run_test = True
                message, data = e.args
                self.assertEqual(data['smallest'], min(test_reset['run']),
                                 'best checkpoint won')

        self.assertEqual(run_test, test_reset['should_raise'],
                         'should raise')
