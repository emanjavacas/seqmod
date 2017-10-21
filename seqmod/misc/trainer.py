
import time
import copy
import math
import numpy as np
from torch.autograd import Variable

from seqmod import utils as u
from seqmod.misc.early_stopping import EarlyStoppingException


# Utility functions (repackage_hidden, memory effective loss, etc.)
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# Base Trainer class
class Trainer(object):
    def __init__(self, model, datasets, criterion, optimizer, scheduler=None,
                 early_stopping=None, test_name='test', valid_name='valid',
                 loss_labels=('loss',), size_average=True, verbose=True):
        """
        Parameter:
        ----------

        - loss_labels: tuple of str, a tuple specifying the names of the
            losses return by run_batch (typically this is useful to tell
            apart the different losses in a complex loss function)
        - size_average: bool,
            whether the loss is already averaged over examples.
            See `size_average` in the torch.nn criterion functions.
        """
        # attributes
        self.model = model
        self.datasets = datasets   # is a dict with at least a 'train' entry
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.loss_labels = loss_labels
        self.epoch = 0          # safe global epoch
        # config
        self.verbose = verbose
        self.size_average = size_average
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
    def add_hook(self, hook, hooks_per_epoch=None, clear=False):
        """
        Add a trainer hook that gets executed after a number of checkpoints.
        The number of times a hook gets executed per epoch can be specified

        Parameters:
        -----------
        hook: fn(trainer, epoch, batch_num, checkpoint)
        """
        self.hooks = [] if clear else self.hooks
        self.hooks.append({'hook': hook, 'hooks_per_epoch': hooks_per_epoch})

    def run_hooks(self, epoch, batch_num, checkpoint):
        batches = len(self.datasets['train'])
        for hook in self.hooks:
            _num_checkpoints = batch_num // checkpoint  # checkpoints in epoch
            if hook['hooks_per_epoch'] is not None:
                _execute_every = max(
                    1, batches // (checkpoint * hook['hooks_per_epoch']))
            else:
                _execute_every = 1
            if _execute_every > 0 and _num_checkpoints % _execute_every == 0:
                hook['hook'](self, epoch, batch_num, _num_checkpoints)

    # callbacks
    def on_batch_end(self, batch, batch_loss):
        # reset hidden, and things like that
        pass

    def on_epoch_begin(self, epoch):
        self.log("epoch_begin", {"epoch": epoch})

    def on_epoch_end(self, epoch, loss, examples, duration, valid_loss=None):
        self.log("epoch_end", {"epoch": epoch,
                               "loss": dict(zip(self.loss_labels, loss)),
                               "examples": examples,
                               "duration": duration})

    def on_validation_end(self, epoch, loss):
        self.log("validation_end", {"epoch": epoch,
                                    "loss": dict(zip(self.loss_labels, loss))})
        if self.early_stopping is not None:
            self.early_stopping.add_checkpoint(
                self.merge_loss(loss), copy.deepcopy(self.model))

    def on_test_begin(self, epoch):
        self.log("test_begin", {"epoch": epoch})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": dict(zip(self.loss_labels, loss))})

    # optimizer
    def zero_grad(self):
        "Reset accumulated gradients for the optimizer"
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()

    def optimizer_step(self, val_loss=None):
        "Runs an optimizing step"
        self.optimizer.step()

        if self.scheduler is not None:
            if val_loss is None:
                self.log("info", "Omitting scheduler, because of missing loss")
            else:
                self.scheduler.step(val_loss)

    # loss
    def init_loss(self):
        "Function defining the shape of the loss before training"
        return tuple([0] * len(self.loss_labels))

    def format_loss(self, loss):
        "Eventually transform loss into something meaningful (e.g. ppl)"
        return loss

    def reweight_loss(self, loss, num_examples):
        "Reweight the loss to account for all instances in batch (deaveraging)"
        weight = (num_examples if self.size_average else 1)
        return tuple([l * weight for l in loss])

    def update_loss(self, acc_loss, loss):
        """
        Updates loss across batches given an accumulated loss `acc_loss`
        and the current new loss `loss`
        """
        return tuple([acc + new for (acc, new) in zip(acc_loss, loss)])

    def average_loss(self, epoch_loss, num_epoch_examples):
        "Computes average loss per instance after epoch"
        return tuple([l / num_epoch_examples for l in epoch_loss])

    def merge_loss(self, loss):
        "Combine in case of complex loss"
        return sum(loss)

    # training code
    def num_batch_examples(self, batch_data):
        """
        By default consider all target elements in batch.
        """
        source, target = batch_data
        return target.nelement()

    def validate_model(self, test=False, **kwargs):
        loss, num_examples = self.init_loss(), 0
        dataset = self.datasets[self.test_name if test else self.valid_name]
        for batch_num in range(len(dataset)):
            batch = dataset[batch_num]
            batch_examples = self.num_batch_examples(batch)
            num_examples += batch_examples
            batch_loss = self.run_batch(
                batch, dataset=self.valid_name, **kwargs)
            batch_loss = self.reweight_loss(batch_loss, batch_examples)
            loss = self.update_loss(loss, batch_loss)
        return self.format_loss(self.average_loss(loss, num_examples))

    def run_batch(self, batch_data, dataset='train', **kwargs):
        """
        Compute batch loss and (eventually) run optimizations on the model.
        It should return the loss as a tuple of floats. It can return None
        if the batch has to be skipped.
        """
        source, targets = batch_data
        outs = self.model(source)
        loss = self.criterion(outs, targets.view(-1))
        if dataset == 'train':
            self.merge_loss(loss).backward()
            self.optimizer_step()
        return (loss.data[0], )

    def _get_batch_order(self, shuffle, num_batches=None):
        "Get batch indices in case of batch level training"
        batch_order = list(range(len(self.datasets['train'])))
        if shuffle:
            batch_order = np.random.permutation(batch_order)
        if num_batches is not None:
            if self.last_batch_order is not None:
                batch_order = self.last_batch_order
            while num_batches > len(batch_order):
                extra_order = list(range(len(self.datasets['train'])))
                if shuffle:
                    extra_order = np.random.permutation(extra_order)
                batch_order += extra_order
            self.last_batch_order = batch_order[num_batches:]
            return batch_order[:num_batches]
        else:
            return batch_order

    def _train(self, epoch, checkpoint, batch_order, **kwargs):
        "General train loop"
        # compute batch order
        dataset = self.datasets['train']
        run_loss, check_loss = self.init_loss(), self.init_loss()
        run_examples, check_examples, start = 0, 0, time.time()

        for batch_num, batch in enumerate(batch_order):
            self.zero_grad()
            batch_data = dataset[batch]
            loss = self.run_batch(batch_data, dataset='train', **kwargs)
            if loss is None:  # to skip a batch run_batch might return None
                continue
            self.on_batch_end(batch_num, self.format_loss(loss))
            # report
            num_examples = self.num_batch_examples(batch_data)
            # for reporting purposes we need the total loss per batch,
            # but it can be just the average loss depending on `size_average`
            batch_loss = self.reweight_loss(loss, num_examples)
            run_loss = self.update_loss(run_loss, batch_loss)
            check_loss = self.update_loss(check_loss, batch_loss)
            run_examples += num_examples
            check_examples += num_examples
            # checkpoint
            if checkpoint and batch_num > 0 and batch_num % checkpoint == 0:
                format_loss = self.average_loss(check_loss, check_examples)
                format_loss = self.format_loss(format_loss)
                self.model.eval()
                self.log('checkpoint', {
                    'epoch': epoch,
                    'batch': batch_num,
                    'total_batches': len(batch_order),
                    'examples': check_examples,
                    'duration': time.time() - start,
                    'loss': dict(zip(self.loss_labels, format_loss))})
                self.run_hooks(epoch, batch_num, checkpoint)
                self.model.train()
                check_loss = self.init_loss()
                check_examples = 0
                start = time.time()
        return run_loss, run_examples

    def train_batches(self, num_batches, checkpoint, shuffle=False, gpu=False,
                      run_test=False, **kwargs):
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
        - gpu: bool
        - run_test: bool, whether to run testing after the number of batches

        Returns (best_model, valid_loss), test_loss
        -------

        - best_model: nn.Module, deep copy of the best model during training.
            If no early stopping was provided, the best model will be the
            current model after training.
        - valid_loss: float or None, best validation loss aggregated according
            to the function merge_loss. If not early stopping is provided, best
            loss will be the last validation loss after training.
        - test_loss: float or None, test loss aggregated as per merge_loss
        """
        start = time.time()
        best_model, valid_loss, test_loss = None, None, None
        try:
            # train
            self.on_epoch_begin(self.batch_run)
            batch_order = self._get_batch_order(shuffle, num_batches)
            run_loss, run_examples = self._train(
                self.batch_run, checkpoint, batch_order, **kwargs)
            self.batch_run += 1  # increase number of runs
            run_loss = self.format_loss(
                self.average_loss(run_loss, run_examples))
            run_time = time.time() - start
            self.on_epoch_end(self.batch_run, run_loss, run_examples, run_time)
            # valid
            if self.valid_name in self.datasets:
                self.model.eval()
                valid_loss = self.validate_model(**kwargs)
                self.on_validation_end(self.batch_run, valid_loss)
                self.model.train()
            if valid_loss is not None:  # merge after callback
                valid_loss = self.merge_loss(valid_loss)
        except EarlyStoppingException as e:
            message, data = e.args
            best_model, valid_loss = data['model'], data['smallest']
            self.log("info", message)
        except KeyboardInterrupt:
            self.log("info", "Training interrupted")
        self.log("info", "Trained for [{:.3f} sec]".format(time.time()-start))
        # test
        if run_test and self.test_name in self.datasets:
            self.model.eval()
            self.on_test_begin(self.batch_run)
            test_loss = self.validate_model(test=True, **kwargs)
            self.on_test_end(test_loss)
            test_loss = self.merge_loss(test_loss)  # merge after callback
        best_model = best_model or copy.deepcopy(self.model)
        return (best_model.cpu(), valid_loss), test_loss

    def train(self, epochs, checkpoint, shuffle=False, gpu=False, **kwargs):
        """
        Parameters:
        -----------

        - epochs: int
        - checkpoint: int, log a checkpoint and hooks every x batches
        - gpu: bool

        Returns (best_model, valid_loss), test_loss
        -------

        - best_model: nn.Module, deep copy of the best model during training.
            If no early stopping was provided, the best model will be the
            current model after training.
        - valid_loss: float or None, best validation loss aggregated according
            to the function merge_loss. If not early stopping is provided, best
            loss will be the last validation loss after training.
        - test_loss: float or None, test loss aggregated as per merge_loss
        """
        start = time.time()
        best_model, valid_loss, test_loss = None, None, None
        for e in range(1, epochs + 1):
            self.epoch, start_epoch = e, time.time()
            self.model.train()
            try:
                # train
                self.on_epoch_begin(e)
                batch_order = self._get_batch_order(shuffle)
                epoch_loss, epoch_examples = self._train(
                    e, checkpoint, batch_order, **kwargs)
                epoch_loss = self.format_loss(
                    self.average_loss(epoch_loss, epoch_examples))
                epoch_time = time.time() - start_epoch
                self.on_epoch_end(e, epoch_loss, epoch_examples, epoch_time)
                # valid
                if self.valid_name in self.datasets:
                    self.model.eval()
                    valid_loss = self.validate_model(**kwargs)
                    self.on_validation_end(e, valid_loss)
                    self.model.train()
                if valid_loss is not None:  # merge after callback
                    valid_loss = self.merge_loss(valid_loss)
            except EarlyStoppingException as ex:
                message, data = ex.args
                best_model, valid_loss = data['model'], data['smallest']
                self.log("info", message)
                break
            except KeyboardInterrupt:
                self.log("info", "Training interrupted")
                break
        self.log("info", "Trained for [{:.3f} sec]".format(time.time()-start))
        # test
        if self.test_name in self.datasets:
            self.model.eval()
            self.on_test_begin(e)
            test_loss = self.validate_model(test=True, **kwargs)
            self.on_test_end(test_loss)
            test_loss = self.merge_loss(test_loss)  # merge after callback
        best_model = best_model or copy.deepcopy(self.model)
        return (best_model.cpu(), valid_loss), test_loss


class LMTrainer(Trainer):
    """
    General LMTrainer for standard LMs
    """
    def __init__(self, *args, reset_hidden=False, **kwargs):
        super(LMTrainer, self).__init__(*args, **kwargs)
        self.reset_hidden = reset_hidden

    def format_loss(self, loss):
        return tuple(math.exp(min(l, 100)) for l in loss)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        # compute loss
        source, targets = batch_data
        hidden = self.batch_state.get('hidden', None)
        output, hidden, _ = self.model(source, hidden=hidden)
        # detach hidden from graph
        self.batch_state['hidden'] = repackage_hidden(hidden)
        loss = self.criterion(output, targets.view(-1))
        # optimize
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return (loss.data[0], )

    def on_batch_end(self, batch, loss):
        if self.reset_hidden:
            if isinstance(self.batch_state['hidden'], tuple):
                for h in self.batch_state['hidden']:
                    h.data.zero_()
            else:
                self.batch_state['hidden'].data.zero_()

    def num_batch_examples(self, batch_data):
        src, trg, *_ = batch_data
        return trg.nelement()


class CyclicLMTrainer(LMTrainer):
    """
    Trainer to be used with a multiheaded LM and a CyclicBlockDataset
    that iterates over batches from different source datasets and finetunes
    a different output distribution per source dataset.
    """
    def run_batch(self, batch_data, dataset='train', subset=None):
        source, targets, head = batch_data
        if subset is not None and subset != head:
            # if subset is given, skip all other subsets
            return
        hidden = self.batch_state.get('hidden', {}).get(head, None)
        output, hidden, _ = self.model(source, hidden=hidden, head=head)
        if 'hidden' not in self.batch_state:
            self.batch_state['hidden'] = {}
        # dettach hidden from graph
        self.batch_state['hidden'][head] = repackage_hidden(hidden)
        loss = self.criterion(output, targets.view(-1))
        # optimize
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return (loss.data[0], )

    def on_batch_end(self, batch, loss):
        if self.reset_hidden:
            for v in self.batch_state['hidden'].values():
                if isinstance(v, tuple):  # lstm
                    for h in v:
                        h.data.zero_()
                else:
                    v.data.zero_()


class CLMTrainer(LMTrainer):
    """
    Trainer for the conditional language model
    """
    def run_batch(self, batch_data, dataset='train', **kwargs):
        (src, *conds), (trg, *_) = batch_data
        hidden = self.batch_state.get('hidden', None)
        output, hidden, _ = self.model(src, hidden=hidden, conds=conds)
        self.batch_state['hidden'] = repackage_hidden(hidden)
        loss = self.criterion(output, trg.view(-1))
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return (loss.data[0], )

    def num_batch_examples(self, batch_data):
        (src, *_), _ = batch_data
        return src.nelement()


class EncoderDecoderTrainer(Trainer):
    """
    Trainer for a general Encoder-Decoder model
    """
    def __init__(self, *args, **kwargs):
        super(EncoderDecoderTrainer, self).__init__(*args, **kwargs)
        # sum loss over batch samples instead of average
        self.size_average = False

    def format_loss(self, loss):
        return tuple(math.exp(min(l, 100)) for l in loss)

    def run_batch(self, batch_data, dataset='train', split=52, **kwargs):
        valid, loss = dataset != 'train', self.init_loss()
        pad, eos = self.model.src_dict.get_pad(), self.model.src_dict.get_eos()
        source, targets = batch_data
        # remove <eos> from decoder targets substituting them with <pad>
        decode_targets = Variable(u.map_index(targets[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_targets = targets[1:]
        # compute model output
        outs = self.model(source[1:], decode_targets)
        # dettach outs from computational graph
        det_outs = Variable(outs.data, requires_grad=not valid, volatile=valid)
        for out, trg in zip(det_outs.split(split), loss_targets.split(split)):
            # (seq_len x batch x hid_dim) -> (seq_len * batch x hid_dim)
            out = out.view(-1, out.size(2))
            pred = self.model.project(out)
            loss = self.update_loss(loss, self.criterion(pred, trg.view(-1)))
        if not valid:
            batch = outs.size(1)
            for l in loss:
                l.div(batch).backward()
            grad = None if det_outs.grad is None else det_outs.grad.data
            outs.backward(grad)
            self.optimizer_step()
        return tuple(l.data[0] for l in loss)

    def num_batch_examples(self, batch_data):
        _, targets = batch_data
        return targets.data.ne(self.model.src_dict.get_pad()).sum()
