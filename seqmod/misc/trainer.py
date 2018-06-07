
import os
import random
from time import time
import copy
import math
import collections
import yaml
import argparse
import logging
from operator import itemgetter
from datetime import datetime

import torch
from torch.optim import lr_scheduler

from seqmod import utils as u
from seqmod.misc.early_stopping import EarlyStoppingException
from .git import GitInfo


def ppl(loss):
    try:
        return math.exp(min(100, loss))
    except ValueError:
        return math.nan


def bpc(loss):
    try:
        return math.log2(math.e) * loss
    except ValueError:
        return math.nan

def identity(loss):
    return loss


def get_formatter(loss):
    if loss.lower() == 'ppl':
        return ppl
    elif loss.lower() == 'bpc':
        return bpc
    else:
        return identity


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
                self.losses.append({'loss': loss, 'format': get_formatter(loss)})
            else:
                # check loss name from loss['loss']
                if 'loss' not in loss:
                    raise ValueError("Loss specification missing `loss` entry")
                # check loss['format']: either function or string designating func
                if 'format' not in loss:
                    loss['format'] = loss['loss']
                if type(loss['format']) is str:
                    loss['format'] = get_formatter(loss['format'])
                self.losses.append(loss)

        loss_labels = [loss['loss'] for loss in self.losses]

        if weights is not None:
            if set(weights) != set(loss_labels):
                raise ValueError("Weights requires same number of items as losses")
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
            raise ValueError("Got {} losses but needs {}"
                             .format(len(loss), len(self.losses)))

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
    mode: (default 'nbest') one of 'nbest' or 'nlast'
    keep: max number of best models to keep in disk.
    ext: model file extension.
    """
    def __init__(self, subdir, topdir='./models', mode='nbest', keep=1, ext='torch'):
        self.topdir = topdir
        self.subdir = subdir
        self.subdir += '-{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.mode = mode
        self.keep = keep
        self.buf_best = []
        self.buf_last = []
        self.ext = ext
        self.is_setup = False

        if self.mode not in ('nbest', 'nlast'):
            raise ValueError("Not a mode: {}".format(self.mode))

    def get_modelname(self, index):
        return self.checkpoint_path('model-{}'.format(index))

    def checkpoint_path(self, *path):
        return os.path.join(self.topdir, self.subdir, *path)

    def setup(self, args=None, d=None):
        """
        Initialize the checkpoint register.

        - args: Namespace or dictionary of params associated with the network
        - d: Dict (optional) to store together with the model
        """
        if self.is_setup:
            return self

        if not os.path.isdir(os.path.join(self.topdir, self.subdir)):
            os.makedirs(os.path.join(self.topdir, self.subdir))

        if args is not None:
            if isinstance(args, argparse.Namespace):
                args = vars(args)
            # add git info
            git_info = GitInfo(self.topdir)
            commit, branch = git_info.get_commit(), git_info.get_branch()
            args['git-commit'] = commit
            args['git-branch'] = branch
            from seqmod import __commit__
            args['seqmod-git-commit'] = __commit__
            # dump
            with open(self.checkpoint_path('params.yml'), 'w') as f:
                yaml.dump(args, f, default_flow_style=False)

        if d is not None:
            u.save_model(d, self.checkpoint_path('dict'), mode=self.ext)

        self.is_setup = True

        return self

    def save(self, model, loss=None):
        """
        Dispatch method
        """
        if self.mode == 'nbest':
            if loss is None:
                raise ValueError("`nbest` requires loss")
            return self.save_nbest(model, loss)
        elif self.mode == 'nlast':
            return self.save_nlast(model)

    def save_nlast(self, model):
        """
        Only keep track of n last models regardless loss
        """
        if not self.is_setup:
            raise ValueError("Checkpoint not setup yet")

        if len(self.buf_last) == self.keep:
            oldestm, _ = self.buf_last[-1]
            try:
                os.remove(oldestm)
            except FileNotFoundError:
                logging.warn("Couldn't find model [{}]".format(oldestm))
            self.buf_last.pop()

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        modelname = u.save_model(model, self.get_modelname(timestamp), mode=self.ext)
        self.buf_last.append((modelname, timestamp))
        self.buf_last.sort(key=itemgetter(1), reverse=True)

        return self

    def save_nbest(self, model, loss):
        """
        Save model according to current state and some validation loss
        """
        if not self.is_setup:
            raise ValueError("Checkpoint not setup yet")

        def format_loss(loss): return '{:.4f}'.format(loss)

        if len(self.buf_best) == self.keep:
            losses = [format_loss(l) for _, l in self.buf_best]
            (worstm, worstl) = self.buf_best[-1]
            if loss < worstl and format_loss(loss) not in losses:  # avoid duplicates
                try:
                    os.remove(worstm)
                except FileNotFoundError:
                    logging.warn("Couldn't find model [{}]".format(worstm))
                    print(self.buf_best, worstm, loss, worstl)
                self.buf_best.pop()
            else:
                return

        modelname = u.save_model(
            model, self.get_modelname(format_loss(loss)), mode=self.ext)
        self.buf_best.append((modelname, loss))
        self.buf_best.sort(key=itemgetter(1))

        return self

    def remove(self):
        """
        Remove entire register
        """
        if not self.is_setup:
            raise ValueError("Checkpoint not setup yet")

        import shutil
        shutil.rmtree(os.path.join(self.topdir, self.subdir))


class Trainer(object):
    def __init__(self, model, datasets, optimizer, scheduler=None, checkpoint=None,
                 early_stopping=None, max_norm=None, losses=('loss',), weights=None,
                 verbose=True):
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
        self.last_batch_order = None

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
            # check if train is given and has length
            try:
                len(self.datasets.get('train', None))
            except TypeError:
                raise ValueError("Cannot configure hook on `hooks_per_epoch` since "
                                 "`train` dataset doesn't have length. Use "
                                 "`num_checkpoints` instead")
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
                rep = max(1, batches // (checkpoint * hook['hooks_per_epoch']) - 1)
                if num_checkpoints % rep == 0:
                    hook['hook'](self, epoch, batch_num, num_checkpoints)
            elif 'num_checkpoints' in hook:
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
                # need some extra free GPU memory to duplicate the model before
                # moving to cpu and storing it in the EarlyStopping object
                loss.reduce(), copy.deepcopy(self.model).cpu())
        if self.checkpoint is not None:
            self.checkpoint.save(self.model, loss.reduce())

    def on_test_begin(self):
        self.log("test_begin", {})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": loss.pack()})

    # optimizer
    def optimizer_step(self):
        "Runs an optimizing step"
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
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
        if test and 'test' not in self.datasets:
            raise ValueError("Can not validate on test set, "
                             "no test set available.")

        dataset = self.datasets['test' if test else 'valid']
        model, loss = model or self.model, self.loss.init()

        for batch in dataset:
            batch_loss, batch_examples = model.loss(batch, test=True, **kwargs)
            loss.add(batch_loss, batch_examples)

        return loss

    def _get_batch_mode_batch_order(self, shuffle, num_batches):
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

    def get_batch_order(self, shuffle, num_batches=None):
        "Get batch order for a single epoch"
        if num_batches is None:
            batch_order = list(range(len(self.datasets['train'])))
            if shuffle:
                random.shuffle(batch_order)
            return batch_order

        else:
            return self._get_batch_mode_batch_order(shuffle, num_batches)

    def run_checkpoint(self, epoch, b, checkpoint, duration, total_batches, loss):
        "Run checkpoint when needed"
        # log
        self.log('checkpoint', {
            'epoch': epoch,
            'batch': b,
            'total_batches': total_batches,
            'examples': loss.examples,
            'duration': duration,
            'loss': loss.pack()})
        # run hooks
        self.model.eval()
        with torch.no_grad():
            self.run_hooks(epoch, b, checkpoint)
        self.model.train()

    def run_inner_loop(self, epoch, checkpoint, batch_order, **kwargs):
        """
        General train loop for a single run
        """
        run_loss, check_loss = self.loss.init(), self.loss.init()
        start, total_batches = time(), len(batch_order)

        for b, batch in enumerate(batch_order):
            # optimize
            self.optimizer.zero_grad()
            batch_data = self.datasets['train'][batch]
            batch_loss, batch_examples = self.model.loss(batch_data, **kwargs)
            if batch_loss is None:  # to skip a batch loss might return None
                continue
            self.optimizer_step()
            run_loss.add(batch_loss, batch_examples)
            check_loss.add(batch_loss, batch_examples)
            self.on_batch_end(epoch, b, run_loss)

            # checkpoint
            if checkpoint and b > 0 and b % checkpoint == 0:
                self.run_checkpoint(
                    epoch, b, checkpoint, time()-start, total_batches, check_loss)

                check_loss.reset()
                start = time()

        return run_loss

    def run_inner_generator_loop(self, epoch, checkpoint, generator, **kwargs):
        """
        Custom inner loop for generator datasets
        """
        run_loss, check_loss = self.loss.init(), self.loss.init()
        start, total_batches = time(), '~'

        for b, batch in enumerate(generator):
            # optimize
            self.optimizer.zero_grad()
            batch_loss, batch_examples = self.model.loss(batch, **kwargs)
            if batch_loss is None:
                continue
            self.optimizer_step()
            run_loss.add(batch_loss, batch_examples)
            check_loss.add(batch_loss, batch_examples)
            self.on_batch_end(epoch, b, run_loss)

            # checkpoint
            if checkpoint and b > 0 and b % checkpoint == 0:
                self.run_checkpoint(
                    epoch, b, checkpoint, time()-start, total_batches, check_loss)

                check_loss.reset()
                start = time()

        return run_loss

    def run_outer_loop(self, checkpoint, epochs=None, num_batches=None, generator=None,
                       shuffle=True, run_test=True, **kwargs):
        """
        General train loop for multiple runs/epochs
        """
        best_model, valid_loss, test_loss = None, None, None
        start = time()

        try:
            for e in range(epochs):

                # run epoch
                self.on_epoch_begin(e)
                epoch_start = time()
                self.model.train()

                if generator is not None:
                    run_loss = self.run_inner_generator_loop(
                        e, checkpoint, generator(), **kwargs)
                else:
                    batch_order = self.get_batch_order(shuffle, num_batches=num_batches)
                    run_loss = self.run_inner_loop(e, checkpoint, batch_order, **kwargs)

                self.on_epoch_end(e, run_loss, run_loss.examples, time() - epoch_start)

                # valid
                if 'valid' in self.datasets:
                    self.model.eval()
                    self.on_validation_begin(e)
                    with torch.no_grad():
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

        self.log("info", "Trained for [{:.3f} secs]".format(time() - start))

        # prepare best model
        self.model.cpu()        # free gpu
        best_model = best_model or self.model

        if run_test and 'test' in self.datasets:
            best_model.eval()
            self.on_test_begin()
            best_model = best_model.to(device=self.datasets['test'].device)
            with torch.no_grad():
                test_loss = self.validate_model(test=True, model=best_model, **kwargs)
            self.on_test_end(test_loss)
            test_loss = test_loss.reduce()

        if self.checkpoint is not None:
            if not u.prompt('Do you want to keep intermediate results? (yes/no)'):
                self.checkpoint.remove()

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
            checkpoint, epochs=1, num_batches=num_batches, shuffle=shuffle,
            run_test=run_test, **kwargs)

    def train(self, epochs, checkpoint, shuffle=False, run_test=True, **kwargs):
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
            run_test=run_test, **kwargs)

    def train_generator(self, epochs, generator, checkpoint, run_test=True, **kwargs):
        """
        Train over a generator for memory efficient

        Parameters:
        -----------

        - epochs: int
        - generator: func that returns a generator over training examples
        """
        return self.run_outer_loop(
            checkpoint, epochs=epochs, generator=generator,
            run_test=run_test, **kwargs)
