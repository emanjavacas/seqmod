
import logging
import numpy as np
from visdom import Visdom


class Logger(object):
    def log(self, event, payload, verbose=True):
        if verbose and hasattr(self, event):
            getattr(self, event)(payload)


class StdLogger(Logger):
    def __init__(self, outputfile=None, level='INFO',
                 msgfmt="%(message)s", datafmt='%m-%d %H:%M'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, level))
        formatter = logging.Formatter(msgfmt, datafmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if outputfile is not None:
            fh = logging.FileHandler(outputfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def epoch_begin(self, payload):
        self.logger.info("Starting epoch [%d]" % payload['epoch'])

    def epoch_end(self, payload):
        tokens_sec = payload["examples"] / payload["duration"]
        self.logger.info(
            "Epoch [%d], train loss: %g, processed: %d tokens/sec" %
            (payload['epoch'], payload["loss"], tokens_sec))

    def validation_end(self, payload):
        self.logger.info("Epoch [%d], valid loss: %g" %
                         (payload['epoch'], payload['loss']))

    def test_begin(self, payload):
        self.logger.info("Testing...")

    def test_end(self, payload):
        self.logger.info("Test loss: %g" % payload["loss"])

    def checkpoint(self, payload):
        tokens_sec = payload["examples"] / payload["duration"]
        self.logger.info("Batch [%d/%d], loss: %g, processed %d tokens/sec" %
                         (payload["batch"], payload["total_batches"],
                          payload["loss"], tokens_sec))

    def info(self, payload):
        self.logger.info("message: %s" % payload["message"])


class VisdomLogger(Logger):
    def __init__(self):
        self.viz = Visdom()
        self.win = None
        self.last = {'train': {'X': 1, 'Y': 1},
                     'valid': {'X': 1, 'Y': 1}}  # start at root
        self.legend = ['train', 'valid']

    def _winline(self, X, Y, **kwargs):
        if self.win is not None:
            self.viz.line(X=X, Y=Y,
                          win=self.win, update='append',
                          opts={'legend': self.legend},
                          **kwargs)
        else:
            self.win = self.viz.line(X=X, Y=Y,
                                     opts={'legend': self.legend},
                                     **kwargs)

    def epoch_end(self, payload):
        valid_X, valid_Y = self.last['valid']['X'], self.last['valid']['Y']
        train_X, train_Y = self.last['train']['X'], self.last['train']['Y']
        loss, epoch = payload['loss'], payload['epoch']
        X = np.column_stack(([train_X, epoch], [valid_X, valid_X]))
        Y = np.column_stack(([train_Y, loss],  [valid_Y, valid_Y]))
        self.last['train']['X'] = epoch
        self.last['train']['Y'] = loss
        self._winline(X, Y)

    def validation_end(self, payload):
        valid_X, valid_Y = self.last['valid']['X'], self.last['valid']['Y']
        train_X, train_Y = self.last['train']['X'], self.last['train']['Y']
        loss, epoch = payload['loss'], payload['epoch']
        X = np.column_stack(([train_X, train_X], [valid_X, epoch]))
        Y = np.column_stack(([train_Y, train_Y], [valid_Y, loss]))
        self.last['valid']['X'] = epoch
        self.last['valid']['Y'] = loss
        self._winline(X, Y)

    def checkpoint(self, payload):
        valid_X, valid_Y = self.last['valid']['X'], self.last['valid']['Y']
        train_X, train_Y = self.last['train']['X'], self.last['train']['Y']
        epoch = payload['epoch'] + payload["batch"] / payload["total_batches"]
        loss = payload['loss']
        X = np.column_stack(([train_X, epoch], [valid_X, valid_X]))
        Y = np.column_stack(([train_Y, loss],  [valid_Y, valid_Y]))
        self.last['train']['X'] = epoch
        self.last['train']['Y'] = loss
        self._winline(X, Y)
