
import os
import logging
import warnings
import numpy as np

try:
    from visdom import Visdom
except ImportError:
    Visdom = None
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def skip_on_import_error(flag, verbose=False):
    msg = 'Omitting call to [{}.{}]'

    def decorator(func):
        def func_wrapper(self, *args, **kwargs):
            if flag:
                return func(self, *args, **kwargs)
            else:
                if verbose:
                    print(msg.format(type(self).__name__, func.__name__))
        return func_wrapper
    return decorator


class Logger(object):
    """
    Abstract logger class. Subclasses of Logger can be passed to the Trainer
    class, which will use Logger's own methods to do logging.
    See Trainer for methods that can be overwritten and their signatures.
    A short summary is provided here:

    - checkpoint(payload : {"epoch": int, "batch": int, "total_batches": int,
                            "examples": int, "duration": float, "loss": dict})
    - on_batch_end(epoch : int, batch : int, loss : int)
    - on_epoch_end(epoch : int, loss : dict, examples : int, duration : float,
                   valid_loss (optional) : None or dict)
    - on_validation_begin(epoch : int)
    - on_validation_end(epoch : int, loss : dict)
    - on_test_begin()
    - on_test_end(loss : dict)
    """
    def log(self, event, payload, verbose=True):
        if verbose and hasattr(self, event):
            getattr(self, event)(payload)

        return


class StdLogger(Logger):
    """
    Standard python logger.

    Parameters
    ----------

    - outputfile: str, file to print log to. If None, only a console
        logger will be used.
    - level: str, one of 'INFO', 'DEBUG', ... See logging.
    - msgfmt: str, message formattter
    - datefmt: str, date formatter
    """
    def __init__(self, outputfile=None, level='INFO',
                 msgfmt="[%(asctime)s] %(message)s", datefmt='%m-%d %H:%M:%S'):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.handlers = []
        self.logger.setLevel(getattr(logging, level))
        formatter = logging.Formatter(msgfmt, datefmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if outputfile is not None:
            if os.path.isdir(outputfile):
                outputfile = os.path.join(outputfile, 'train.log')
            fh = logging.FileHandler(outputfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    @staticmethod
    def loss_str(loss, phase):
        return "; ".join([phase + " {}: {:g}".format(k, v)
                          for (k, v) in loss.items()])

    def epoch_begin(self, payload):
        self.logger.info("Starting epoch [{}]".format(payload['epoch']))

    def epoch_end(self, payload):
        speed = payload["examples"] / payload["duration"]
        loss = StdLogger.loss_str(payload['loss'], 'train')
        self.logger.info("Epoch[{}]; {}; speed: {:g} tokens/sec"
                         .format(payload['epoch'], loss, speed))

    def validation_end(self, payload):
        loss = StdLogger.loss_str(payload['loss'], 'valid')
        self.logger.info("Epoch[{}]; {}".format(payload['epoch'], loss))

    def test_begin(self, payload):
        self.logger.info("Testing...")

    def test_end(self, payload):
        self.logger.info(StdLogger.loss_str(payload['loss'], 'Test'))

    def checkpoint(self, payload):
        e, b, bs = payload['epoch'], payload['batch'], payload['total_batches']
        speed = payload["examples"] / payload["duration"]
        loss = StdLogger.loss_str(payload['loss'], 'train')
        self.logger.info("Epoch[{}]; batch [{}/{}]; {}; speed {:g} tokens/sec"
                         .format(e, b, bs, loss, speed))

    def info(self, payload):
        if isinstance(payload, dict):
            payload = payload['message']
        self.logger.info(payload)


class VisdomLogger(Logger):
    """
    Logger that uses visdom to create learning curves

    Parameters
    ----------

    - env: str, name of the visdom environment
    - log_checkpoints: bool, whether to use checkpoints or epoch averages
        for training loss
    - legend: tuple, names of the different losses that will be plotted.
    """
    def __init__(self,
                 env=None,
                 log_checkpoints=True,
                 losses=('loss', ),
                 phases=('train', 'valid'),
                 server='http://localhost',
                 port=8097,
                 max_y=None,
                 **opts):
        if Visdom is None:
            warnings.warn("Couldn't import visdom: `pip install visdom`")
        else:
            self.viz = Visdom(server=server, port=port, env=env)

        self.legend = ['{}.{}'.format(p, l) for p in phases for l in losses]
        opts.update({'legend': self.legend})
        self.opts = opts
        self.env = env
        self.max_y = max_y
        self.log_checkpoints = log_checkpoints
        self.losses = set(losses)
        self.last = {p: {l: None for l in losses} for p in phases}
        self.pane = self._init_pane()

    @skip_on_import_error(Visdom)
    def _init_pane(self):
        nan = np.array([np.NAN, np.NAN])
        X = np.column_stack([nan] * len(self.legend))
        Y = np.column_stack([nan] * len(self.legend))
        return self.viz.line(
            X=X, Y=Y, env=self.env, opts=self.opts)

    def _update_last(self, epoch, loss, phase, loss_label):
        self.last[phase][loss_label] = {'X': epoch, 'Y': loss}

    def _plot_line(self, X, Y, phase, loss_label):
        name = "{}.{}".format(phase, loss_label)
        X = np.array([self.last[phase][loss_label]['X'], X])
        Y = np.array([self.last[phase][loss_label]['Y'], Y])
        if self.max_y:
            Y = np.clip(Y, Y.min(), self.max_y)
        self.viz.updateTrace(
            X=X, Y=Y, name=name, append=True, win=self.pane, env=self.env)

    def _plot_payload(self, epoch, losses, phase):
        for label, loss in losses.items():
            if label not in self.losses:
                continue
            if self.last[phase][label] is not None:
                self._plot_line(epoch, loss, phase=phase, loss_label=label)
            self._update_last(epoch, loss, phase, label)

    @skip_on_import_error(Visdom)
    def epoch_end(self, payload):
        if self.log_checkpoints:
            # only use epoch end if checkpoint isn't being used
            return

        losses, epoch = payload['loss'], payload['epoch'] + 1
        self._plot_payload(epoch, losses, 'train')

    @skip_on_import_error(Visdom)
    def validation_end(self, payload):
        losses, epoch = payload['loss'], payload['epoch'] + 1
        self._plot_payload(epoch, losses, 'valid')

    @skip_on_import_error(Visdom)
    def checkpoint(self, payload):
        if not self.log_checkpoints:
            return

        epoch = payload['epoch'] + payload["batch"] / payload["total_batches"]
        losses = payload['loss']
        self._plot_payload(epoch, losses, 'train')

    @skip_on_import_error(Visdom)
    def attention(self, payload):
        title = "epoch {epoch}/ batch {batch_num}".format(**payload)
        if 'title' in self.opts:
            title = self.opts['title'] + ": " + title
        self.viz.heatmap(
            X=np.array(payload["att"]),
            env=self.env,
            opts={'rownames': payload["hyp"],
                  'columnnames': payload["target"],
                  'title': title})


class TensorboardLogger(Logger):
    def __init__(self, log_dir=None, comment='',
                 tag='training', log_checkpoints=True):
        if SummaryWriter is None:
            warnings.warn("Couldn't import tensorboardX: "
                          "`pip install tensorboardX`; "
                          "`pip install tensorflow")
        else:
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

        self.tag = tag
        self.log_checkpoints = log_checkpoints

    @skip_on_import_error(SummaryWriter)
    def checkpoint(self, payload):
        if not self.log_checkpoints:
            return

        epoch, batch = payload['epoch'], payload['batch']
        total_batches, loss = payload['total_batches'], payload['loss']
        epoch = epoch + batch / total_batches
        losses = {'train/{}'.format(key): val for key, val in loss.items()}
        self.writer.add_scalars(self.tag, losses, epoch)

    @skip_on_import_error(SummaryWriter)
    def epoch_end(self, payload):
        epoch, loss = payload['epoch'] + 1, payload['loss']
        if self.log_checkpoints:
            return

        losses = {'train/{}'.format(key): val for key, val in loss.items()}
        self.writer.add_scalars(self.tag, losses, epoch)

    @skip_on_import_error(SummaryWriter)
    def validation_end(self, payload):
        epoch, loss = payload['epoch'] + 1, payload['loss']
        losses = {'valid/{}'.format(key): val for key, val in loss.items()}
        self.writer.add_scalars(self.tag, losses, epoch)
