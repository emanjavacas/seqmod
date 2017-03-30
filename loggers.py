
import logging
import numpy as np
from visdom import Visdom


class Logger(object):
    def log(self, event, payload, verbose=True):
        if verbose and hasattr(self, event):
            getattr(self, event)(payload)


class StdLogger(Logger):
    def __init__(self, outputfile=None,
                 level='INFO',
                 msgfmt="[%(asctime)s] %(message)s",
                 datefmt='%m-%d %H:%M:%S'):
        """
        Standard python logger.

        Parameters:
        ===========
        outputfile: str, file to print log to. If None, only a console
            logger will be used.
        level: str, one of 'INFO', 'DEBUG', ... See logging.
        msgfmt: str, message formattter
        datefmt: str, date formatter
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, level))
        formatter = logging.Formatter(msgfmt, datefmt)
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
        self.logger.info(
            "Epoch[%d], batch [%d/%d], loss: %g, processed %d tokens/sec" %
            (payload["epoch"], payload["batch"], payload["total_batches"],
             payload["loss"], tokens_sec))

    def info(self, payload):
        if isinstance(payload, dict):
            payload = payload['message']
        self.logger.info(payload)


class VisdomLogger(Logger):
    def __init__(self, env=None, log_checkpoints=True):
        self.viz = Visdom()
        self.env = env
        self.pane = None
        self.last = {'train': None, 'valid': None}
        self.log_checkpoints = log_checkpoints

    def _update_last(self, epoch, loss, name):
        self.last[name]['X'] = epoch
        self.last[name]['Y'] = loss

    def _line(self, X, Y, name, **kwargs):
        X = np.array([self.last['X'][name], X])
        Y = np.array([self.last['Y'][name], Y])
        if self.pane is None:
            self.pane = self.viz.line(
                X=X, Y=Y, env=self.env, **kwargs)
        else:
            self.viz.updateTrace(
                X=X, Y=Y, name=name,
                win=self.pane, update='append', env=self.env, **kwargs)

    def epoch_end(self, payload):
        if self.log_checkpoints:
            # only use epoch end if checkpoint isn't being used
            return
        loss, epoch = payload['loss'], payload['epoch']
        if self.last['train'] is None:
            self._update_last(epoch, loss, 'train')
            return
        else:
            self._line(X=epoch, Y=loss, name='train')

    def validation_end(self, payload):
        loss, epoch = payload['loss'], payload['epoch']
        if self.last['valid'] is None:
            self._update_last(epoch, loss, 'valid')
            return
        else:
            self._line(X=epoch, Y=loss, name='valid')

    def checkpoint(self, payload):
        if not self.log_checkpoints:
            return
        epoch = payload['epoch'] + payload["batch"] / payload["total_batches"]
        loss = payload['loss']
        if self.last['train'] is None:
            self._update_last(epoch, loss, 'train')
            return
        else:
            self._line(X=epoch, Y=loss, name='train')
