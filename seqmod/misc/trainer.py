
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

from seqmod.misc.dataset import CyclicBlockDataset
from seqmod.misc.early_stopping import EarlyStopping, EarlyStoppingException

from seqmod.modules import utils as u


# Utility functions (repackage_hidden, memory effective loss, etc.)
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# Base Trainer class
class Trainer(object):
    def __init__(self, model, datasets, criterion, optimizer,
                 test_name='test', valid_name='valid', loss_labels=('loss',),
                 size_average=True, verbose=True, loggers=None, hooks=None):
        """
        Parameter:
        ==========
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
        self.optimizer = optimizer  # might be a dict
        self.loss_labels = loss_labels
        self.epoch = 0          # safe global epoch
        # config
        self.verbose = verbose
        self.size_average = size_average
        # containers
        self.hooks = []
        if hooks is not None:
            assert isinstance(hooks, list), "hooks must be list"
            for hook in hooks:
                if isinstance(hook, dict):
                    num_checkpoints = hook.get('num_checkpoints', 1)
                    hook = hook['hook']
                else:
                    num_checkpoints = 1
                self.add_hook(hook, num_checkpoints=num_checkpoints)
        self.loggers = loggers or []
        self.batch_state = {}  # instance var to share state across batches
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
    def add_hook(self, hook, num_checkpoints=1, clear=False):
        if clear:
            self.hooks = []
        self.hooks.append({'hook': hook, 'num_checkpoints': num_checkpoints})

    def run_hooks(self, epoch, batch_num, checkpoint):
        for hook in self.hooks:
            num_checkpoints = batch_num // checkpoint
            if hook['num_checkpoints'] > 0 and \
               num_checkpoints % hook['num_checkpoints'] == 0:
                hook['hook'](self, epoch, batch_num, num_checkpoints)

    # callbacks
    def on_batch_end(self, batch, batch_loss):
        # reset hidden, and things like that
        pass

    def on_epoch_begin(self, epoch):
        self.log("epoch_begin", {"epoch": epoch})

    def on_epoch_end(self, epoch, loss, num_examples, duration):
        # SGD lr update
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.items():
                if hasattr(opt, "maybe_update_lr"):
                    opt.maybe_update_lr(epoch, loss)
        else:
            if hasattr(self.optimizer, "maybe_update_lr"):
                self.optimizer.maybe_update_lr(epoch, loss)

        self.log("epoch_end", {"epoch": epoch,
                               "loss": dict(zip(self.loss_labels, loss)),
                               "examples": num_examples,
                               "duration": duration})

    def on_validation_end(self, epoch, loss):
        self.log("validation_end", {"epoch": epoch,
                                    "loss": dict(zip(self.loss_labels, loss))})

    def on_test_begin(self, epoch):
        self.log("test_begin", {"epoch": epoch})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": dict(zip(self.loss_labels, loss))})

    def format_loss(self, loss):
        return loss

    # optimizer
    def zero_grad(self):
        """
        Reset accumulated gradients for the optimizer
        """
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Runs an optimizing step.
        """
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.step()
        else:
            self.optimizer.step()

    # loss
    def init_loss(self):
        """
        Function defining the shape of the loss before training.
        """
        return tuple([0] * len(self.loss_labels))

    def reweight_loss(self, loss, num_examples):
        """
        Reweight the loss to account for all instances in batch (deaveraging).
        """
        weight = (num_examples if self.size_average else 1)
        return tuple([l * weight for l in loss])

    def update_loss(self, acc_loss, loss):
        """
        Updates loss across batches given an accumulated loss `acc_loss`
        and the current new loss `loss`
        """
        return tuple([acc + new for (acc, new) in zip(acc_loss, loss)])

    def average_loss(self, epoch_loss, num_epoch_examples):
        """
        Computes average loss per instance after epoch.
        """
        return tuple([l / num_epoch_examples for l in epoch_loss])

    def merge_loss(self, loss):
        """
        Combine in case of complex loss.
        """
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

    def train_epoch(self, epoch, checkpoint, shuffle, **kwargs):
        # compute batch order
        dataset = self.datasets['train']
        batch_order = range(len(dataset))
        if shuffle:
            batch_order = np.random.permutation(batch_order)
        start = time.time()
        epoch_loss, check_loss = self.init_loss(), self.init_loss()
        epoch_examples, check_examples = 0, 0
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
            epoch_loss = self.update_loss(epoch_loss, batch_loss)
            check_loss = self.update_loss(check_loss, batch_loss)
            epoch_examples += num_examples
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
        return epoch_loss, epoch_examples

    def train(self, epochs, checkpoint, shuffle=False, gpu=False, **kwargs):
        """
        Parameters:
        ===========
        - epochs: int
        - checkpoint: int, log a checkpoint and hooks every x batches
        - gpu: bool
        """
        start = time.time()
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_epoch = time.time()
            self.model.train()
            self.on_epoch_begin(epoch)
            try:
                # train
                epoch_loss, epoch_examples = self.train_epoch(
                    epoch, checkpoint, shuffle, **kwargs)
                epoch_loss = self.format_loss(
                    self.average_loss(epoch_loss, epoch_examples))
                epoch_time = time.time() - start_epoch
                # valid
                if self.valid_name in self.datasets:
                    self.model.eval()
                    valid_loss = self.validate_model(**kwargs)
                    self.on_validation_end(epoch, valid_loss)
                    self.model.train()
                self.on_epoch_end(
                    epoch, epoch_loss, epoch_examples, epoch_time)
            except EarlyStoppingException as e:
                message, _ = e.args
                self.log("info", message)
                break
            except KeyboardInterrupt:
                self.log("info", "Training interrupted")
                break
        self.log("info", "Trained for [%.3f sec]" % (time.time() - start))
        # test
        if self.test_name in self.datasets:
            self.model.eval()
            self.on_test_begin(epoch)
            test_loss = self.validate_model(test=True, **kwargs)
            self.on_test_end(test_loss)


class LMTrainer(Trainer):
    def format_loss(self, loss):
        """
        Turn loss into perplexity.
        """
        return tuple(math.exp(min(l, 100)) for l in loss)

    def run_batch(self, batch_data, dataset='train', subset=None, **kwargs):
        # get dataset
        data = self.datasets[dataset]
        # compute loss
        if isinstance(data, CyclicBlockDataset):
            source, targets, head = batch_data
            if subset is not None and subset != head:
                # if subset is given, skip all other subsets
                return          # skip batch
            hidden = self.batch_state.get('hidden', {}).get(head, None)
            output, hidden, *_ = self.model(source, hidden=hidden, head=head)
            if 'hidden' not in self.batch_state:
                self.batch_state['hidden'] = {}
            # dettach hidden from graph
            self.batch_state['hidden'][head] = repackage_hidden(hidden)
        else:
            source, targets = batch_data
            hidden = self.batch_state.get('hidden', None)
            output, hidden, *_ = self.model(source, hidden=hidden)
            # detach hidden from graph
            self.batch_state['hidden'] = repackage_hidden(hidden)
        loss = self.criterion(output, targets.view(-1))
        # optimize
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return (loss.data[0], )

    def on_batch_end(self, batch, loss):
        if hasattr(self, 'reset_hidden'):
            if isinstance(next(self.datasets.values()), CyclicBlockDataset):
                for v in self.batch_state['hidden'].values():
                    v.zero_()
            else:
                self.batch_state['hidden'].zero_()

    def num_batch_examples(self, batch_data):
        src, trg, *_ = batch_data
        return len(trg)


class EncoderDecoderTrainer(Trainer):
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
            loss[0].div(batch).backward()
            grad = None if det_outs.grad is None else det_outs.grad.data
            outs.backward(grad)
            self.optimizer_step()
        return tuple(l.data[0] for l in loss)

    def num_batch_examples(self, batch_data):
        _, targets = batch_data
        return targets.data.ne(self.model.src_dict.get_pad()).sum()
