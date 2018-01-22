
import os
import random
import time
import copy
import math
import collections
import yaml
import argparse
from operator import itemgetter
from datetime import datetime

from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm

from seqmod import utils as u
from seqmod.misc.early_stopping import EarlyStoppingException


def ppl(loss):
    return math.exp(min(100, loss))


class LossStatistics(object):
    """
    Accumulator for different losses (for report purposes)

    Parameters:
    ----------

    - losses: str or dict. Loss definition. If string, it will
        be the label for the loss, and defaults for formatting
        will be used. If a dict, it should contain the fields
        `loss`, loss label, and `format`, a format function.
        Losses should be input in same order as output by the
        model.

    - weights: dict of loss->int specifying the weight of the
        referenced loss when calling reduce.
    """
    def __init__(self, *losses, weights=None):
        self.losses = []

        for loss in losses:
            if isinstance(loss, str):
                self.losses.append({'loss': loss, 'format': ppl})
            else:
                if 'format' not in loss:
                    loss['format'] = ppl  # default to ppl
                self.losses.append(loss)

        loss_labels = [loss['loss'] for loss in self.losses]

        if weights is not None:
            if set(weights) != set(loss_labels):
                raise ValueError("Weights requires same items as losses")
            self.weights = weights
        else:
            self.weights = {label: 1 for label in loss_labels}

        self.history, self.examples = [], 0

    def init(self):
        """
        Return a new copy of the current loss.
        """
        return LossStatistics(*self.losses, weights=self.weights)

    def reset(self):
        """
        Reset history
        """
        self.history, self.examples = [], 0

    def add(self, loss, num_examples):
        """
        Accumulate average batch loss
        """
        if not isinstance(loss, collections.Iterable):
            loss = [loss]

        if len(loss) != len(self.losses):
            raise ValueError(
                "Got {len(loss)} losses but needs {}".format(len(self.losses)))

        self.history.append(tuple(loss))
        self.examples += num_examples

    def reduce(self):
        """
        Return a float summary of all the losses using user defined weights.
        Losses are first averaged over the history.
        """
        losses = self.pack()
        return sum(self.weights[loss] * losses[loss] for loss in losses)

    def pack(self):
        """
        Compute average losses over batches.

        Returns a dictionary mapping from loss label to average loss
        """
        losses, packed = list(zip(*self.history)), []
        for formatter, loss in zip(self.losses, losses):
            packed.append(formatter['format'](sum(loss) / len(self.history)))

        return dict(zip([l['loss'] for l in self.losses], packed))


class Checkpoint(object):
    """
    Utility class for storing intermediate models

    Parameters:
    -----------
    topdir: top directory to store models
    subdir: name for the subdirectory to store the current model.
        Will be appended with a timestamp to disambiguate runs.
    buffer_size: max number of best models to keep in disk.
    ext: model file extension.
    """
    def __init__(self, subdir, topdir='./models', buffer_size=1, ext='torch'):
        self.topdir = topdir
        self.subdir = subdir
        self.subdir += '-{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.buffer_size = buffer_size
        self.buf = []
        self.ext = ext
        self.is_setup = False

    def _joinpath(self, *path):
        return os.path.join(self.topdir, self.subdir, *path)

    def _modelname(self, loss):
        return self._joinpath('model-{:.4f}.{}'.format(loss, self.ext))

    def setup(self, args=None, d=None):
        if self.is_setup:
            return

        if not os.path.isdir(os.path.join(self.topdir, self.subdir)):
            os.makedirs(os.path.join(self.topdir, self.subdir))

        if args is not None:
            if isinstance(args, argparse.Namespace):
                args = vars(args)
            with open(self._joinpath('params.yml'), 'w') as f:
                yaml.dump(args, f, default_flow_style=False)

        if d is not None:
            u.save_model(d, self._joinpath('dict'), mode=self.ext)

        self.is_setup = True

        return self

    def save(self, model, loss):
        if len(self.buf) == self.buffer_size:
            if loss < max(self.buf, key=itemgetter(1))[1]:
                worstmodel, worstloss = self.buf.pop()
                os.remove(worstmodel)
            else:
                return

        modelname = u.save_model(model, self._modelname(loss), mode=self.ext)
        self.buf.append((modelname, loss))
        self.buf.sort(key=itemgetter(1))

        return self


class Trainer(object):
    def __init__(self, model, datasets, optimizer, scheduler=None, checkpoint=None,
                 early_stopping=None, test_name='test', valid_name='valid',
                 max_norm=None, losses=('loss',), weights=None, verbose=True):
        """
        Parameter:
        ----------

        - max_norm: float or None, restrict the norm of the gradient to this
            value before each SGD update.
        - losses: tuple of str, a tuple specifying the names of the
            losses return by run_batch (typically this is useful to tell
            apart the different losses in a complex loss function)
        - weights: dict or None, if given the losses will be reduce to a single
            value by a weighted sum using this parameter.
        """
        # attributes
        self.model = model
        self.datasets = datasets   # is a dict with at least a 'train' entry
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.loss = LossStatistics(*losses, weights=weights)
        self.max_norm = max_norm
        # config
        self.verbose = verbose
        # containers
        self.loggers = []
        self.hooks = []
        self.batch_state = {}  # instance var to share state across batches
        self.last_batch_order = None
        self.batch_run = 0
        # properties
        self.test_name = test_name
        self.valid_name = valid_name

    # logging
    def add_loggers(self, *loggers):
        for logger in loggers:
            self.loggers.append(logger)

    def log(self, event, payload):
        for logger in self.loggers:
            logger.log(event, payload, verbose=self.verbose)

    # hooks
    def add_hook(self, hook, hooks_per_epoch=None, num_checkpoints=None):
        """
        Add a trainer hook that gets executed after a number of checkpoints.
        The number of times a hook gets executed per epoch can be specified
        in two different ways using either `hooks_per_epoch` or `num_checkpoints`.
        In either case the basic unit used is a checkpoint, which is a check done
        every given number of batches (the specific number is user-defined and
        corresponds to the argument `checkpoint` passed to the `train` method`).

        When using hooks_per_epoch, a heuristic is used to make sure that the
        hook is evaluated exactly so many times during an epoch in evenly spaced
        intervals (see the implementation of `run_hooks` for more info).

        When using `num_checkpoints`, the hook is evaluated every so many
        checkpoints.

        Note that the number of times the hook is evaluated during an epoch is
        only constant when using `hooks_per_epoch` (for `num_checkpoints` it will
        vary depending on the size of the batch).

        Only one of the two options can be specified.

        Parameters:
        -----------
        hook: fn(trainer, epoch, batch_num, checkpoint)
        """
        if hooks_per_epoch is not None and num_checkpoints is not None:
            raise ValueError("Only one of `hooks_per_epoch` or "
                             "`num_checkpoints` can be passed to ``add_hook``")
        if hooks_per_epoch is None and num_checkpoints is None:
            raise ValueError("Either `num_checkpoints` or `hooks_per_epoch` "
                             "must be passed to ``add_hook``")
        hook = {'hook': hook}
        if hooks_per_epoch is not None:
            hook['hooks_per_epoch'] = hooks_per_epoch
        else:
            hook['num_checkpoints'] = num_checkpoints

        self.hooks.append(hook)

    def run_hooks(self, epoch, batch_num, checkpoint):
        for hook in self.hooks:
            num_checkpoints = batch_num // checkpoint
            if 'hooks_per_epoch' in hook:
                # get repetition frequency
                batches = len(self.datasets['train'])
                rep = max(1, batches // (checkpoint * hook['hooks_per_epoch']))
                if num_checkpoints % rep == 0:
                    hook['hook'](self, epoch, batch_num, num_checkpoints)
            else:  # assume num_checkpoints
                if num_checkpoints % hook['num_checkpoints'] == 0:
                    hook['hook'](self, epoch, batch_num, num_checkpoints)

    # callbacks
    def on_batch_end(self, epoch, batch, loss):
        # reset hidden, and things like that
        pass

    def on_epoch_begin(self, epoch):
        self.log("epoch_begin", {"epoch": epoch})

    def on_epoch_end(self, epoch, loss, examples, duration, valid_loss=None):
        self.log("epoch_end", {"epoch": epoch,
                               "loss": loss.pack(),
                               "examples": examples,
                               "duration": duration})

    def on_validation_begin(self, epoch):
        self.log("validation_begin", {"epoch": epoch})

    def on_validation_end(self, epoch, loss):
        self.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
        if self.early_stopping is not None:
            self.early_stopping.add_checkpoint(
                loss.reduce(), copy.deepcopy(self.model))
        if self.checkpoint is not None:
            self.checkpoint.save(self.model, loss.reduce())

    def on_test_begin(self, epoch):
        self.log("test_begin", {"epoch": epoch})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": loss.pack()})

    # optimizer
    def optimizer_step(self):
        "Runs an optimizing step"
        if self.max_norm is not None:
            clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()

    def scheduler_step(self, epoch, valid_loss):
        "Updates learning rate with provided scheduler after epoch validation"
        if self.scheduler is None:
            return

        old_lrs = [p['lr'] for p in self.optimizer.param_groups]

        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if valid_loss is None:
                self.log("info", "Skipped scheduler: missing loss")
            else:
                self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()  # on new epoch

        if hasattr(self.scheduler, 'verbose') and self.scheduler.verbose:
            new_lrs = [p['lr'] for p in self.optimizer.param_groups]
            update = 'Updated lr: '
            for old, new in zip(old_lrs, new_lrs):
                update += '{} -> {}; '.format(old, new)

            self.log("info", update)

    def validate_model(self, test=False, model=None, **kwargs):
        """
        Run validation over the test or the validation set.

        Parameters:
        -----------

        - test: bool (optional), whether to use the test set instead of the
            validation set. If no test set was provided to Trainer an
            exception is raised.
        - model: nn.Module (optional), whether to use a different model
            (e.g. best model resulting from early stopping)
        - kwargs: extra arguments passed to model.loss
        """
        if test and self.test_name not in self.datasets:
            raise ValueError("Can not validate on test set, "
                             "no test set available.")

        dataset = self.datasets[self.test_name if test else self.valid_name]
        model, loss = model or self.model, self.loss.init()

        for batch_num in range(len(dataset)):
            batch = dataset[batch_num]
            batch_loss, batch_examples = model.loss(batch, test=True, **kwargs)
            loss.add(u.unwrap_variables(batch_loss), batch_examples)
            del batch_loss

        return loss

    def get_batch_order(self, shuffle, num_batches):
        "Get batch order for an undefined number of batches"
        batch_order = list(range(len(self.datasets['train'])))
        if shuffle:
            random.shuffle(batch_order)

        if self.last_batch_order is not None:
            batch_order = self.last_batch_order

        while num_batches > len(batch_order):
            extra_order = list(range(len(self.datasets['train'])))
            if shuffle:
                random.shuffle(extra_order)
            batch_order += extra_order

        self.last_batch_order = batch_order[num_batches:]

        return batch_order[:num_batches]

    def get_epoch_batch_order(self, shuffle):
        "Get batch order for a single epoch"
        batch_order = list(range(len(self.datasets['train'])))
        if shuffle:
            random.shuffle(batch_order)
        return batch_order

    def run_inner_loop(self, epoch, checkpoint, batch_order, **kwargs):
        "General train loop for a single run"
        # compute batch order
        run_loss, check_loss = self.loss.init(), self.loss.init()
        start = time.time()

        for batch_num, batch in enumerate(batch_order):

            self.optimizer.zero_grad()
            batch_data = self.datasets['train'][batch]
            batch_loss, batch_examples = self.model.loss(batch_data, **kwargs)

            if batch_loss is None:  # to skip a batch loss might return None
                continue

            self.optimizer_step()

            batch_loss = u.unwrap_variables(batch_loss)
            run_loss.add(batch_loss, batch_examples)
            check_loss.add(batch_loss, batch_examples)

            self.on_batch_end(epoch, batch_num, run_loss)

            # checkpoint
            if checkpoint and batch_num > 0 and batch_num % checkpoint == 0:
                self.model.eval()
                self.log('checkpoint', {
                    'epoch': epoch,
                    'batch': batch_num,
                    'total_batches': len(batch_order),
                    'examples': check_loss.examples,
                    'duration': time.time() - start,
                    'loss': check_loss.pack()})
                self.run_hooks(epoch, batch_num, checkpoint)
                self.model.train()
                check_loss.reset()
                start = time.time()

        return run_loss

    def run_outer_loop(self, checkpoint, epochs=None, num_batches=None,
                       shuffle=True, run_test=True, **kwargs):
        "General train loop for many runs"

        best_model, valid_loss, test_loss = None, None, None
        start = time.time()

        # check run mode (training for epochs or number of batches)
        batch_first = epochs is None
        if batch_first:
            epochs = 1

        try:
            for e in range(epochs):
                epoch_start = time.time()
                # train
                self.model.train()

                if batch_first:
                    batch_order = self.get_batch_order(shuffle, num_batches)
                    e = self.batch_run   # report number of runs as epoch
                    self.batch_run += 1  # increase number of runs
                else:
                    batch_order = self.get_epoch_batch_order(shuffle)
                    self.on_epoch_begin(e)

                run_loss = self.run_inner_loop(
                    e, checkpoint, batch_order, **kwargs)

                run_time = time.time() - epoch_start
                self.on_epoch_end(e, run_loss, run_loss.examples, run_time)

                # valid
                if self.valid_name in self.datasets:
                    self.model.eval()
                    self.on_validation_begin(e)
                    valid_loss = self.validate_model(**kwargs)
                    self.on_validation_end(e, valid_loss)
                    self.model.train()

                if valid_loss is not None:
                    valid_loss = valid_loss.reduce()

                # scheduler after valid
                self.scheduler_step(e, valid_loss)

        except EarlyStoppingException as e:
            message, data = e.args
            best_model, valid_loss = data['model'], data['smallest']
            self.log("info", message)

        except KeyboardInterrupt:
            self.log("info", "Training interrupted")

        self.log("info", "Trained for [{:.3f} secs]".format(time.time()-start))

        if best_model is None:  # use current model
            best_model = copy.deepcopy(self.model)

        # test
        if run_test and self.test_name in self.datasets:
            self.model.eval()
            self.on_test_begin(self.batch_run)
            test_loss = self.validate_model(test=True, model=best_model, **kwargs)
            self.on_test_end(test_loss)
            test_loss = test_loss.reduce()

        return (best_model.cpu(), valid_loss), test_loss

    def train_batches(self, num_batches, checkpoint,
                      shuffle=False, run_test=False, **kwargs):
        """
        Run training on a given number of batches. `num_batches` might be
        larger than the actual total number of batches in the dataset, in
        which case the trainer will loop over the dataset in cycles.
        A second call to this method will resume training where it was left
        in the previous call (unless the training was interrupted or stopped
        via early stopping).
        Shuffling will be done on a dataset basis (i.e. if the first call to
        `train_batches` didn't consume the full dataset, the second call will
        resume using the remaining batches from the previously shuffled run),
        so as to ensure statistical consistency.

        By default, no testing is done at the end, this can be changed through
        the flag run_test.

        `on_epoch_begin` and `on_epoch_end` are reused in this case, but epoch
        will refer to the current partial run (which will only coincide with
        an actual epoch if `num_batches` equals dataset length).

        Parameters:
        -----------

        - num_batches: int
        - checkpoint: int, log a checkpoint and hooks every x batches
        - run_test: bool, whether to run testing after the number of batches
        - kwargs: rest loss kwargs

        Returns (best_model, valid_loss), test_loss
        -------

        - best_model: nn.Module, deep copy of the best model during training.
            If no early stopping was provided, the best model will be the
            current model after training.
        - valid_loss: float or None, best validation loss. If not early
            stopping is provided, best
            loss will be the last validation loss after training.
        - test_loss: float or None
        """
        return self.run_outer_loop(
            checkpoint, num_batches=num_batches, shuffle=shuffle,
            run_test=run_test, **kwargs)

    def train(self, epochs, checkpoint, shuffle=False, **kwargs):
        """
        Parameters:
        -----------

        - epochs: int
        - checkpoint: int, log a checkpoint and hooks every x batches

        Returns (best_model, valid_loss), test_loss
        -------

        - best_model: nn.Module, deep copy of the best model during training.
            If no early stopping was provided, the best model will be the
            current model after training.
        - valid_loss: LossStatistics or None, best validation loss.
            If not early stopping is provided, best loss will be the last
            validation loss after training.
        - test_loss: LossStatistics, test loss
        """
        return self.run_outer_loop(
            checkpoint, epochs=epochs, shuffle=shuffle,
            run_test=True, **kwargs)
