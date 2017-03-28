
import time
import math
import torch
import numpy as np
from torch.autograd import Variable
from dataset import CyclicBlockDataset
from early_stopping import EarlyStopping, EarlyStoppingException


# Utility functions (repackage_hidden, memory effective loss, etc.)
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# Loggers
class Logger(object):
    @staticmethod
    def epoch_begin(payload):
        print("Starting epoch [%d]" % payload['epoch'])

    @staticmethod
    def epoch_end(payload):
        tokens_sec = payload["examples"] / payload["duration"]
        print("Epoch [%d], train loss: %g, processed: %d tokens/sec" %
              (payload['epoch'], payload["loss"], tokens_sec))

    @staticmethod
    def validation_end(payload):
        print("Epoch [%d], valid loss: %g" %
              (payload['epoch'], payload['loss']))

    @staticmethod
    def test_begin(payload):
        print("Testing...")

    @staticmethod
    def test_end(payload):
        print("Test loss: %g" % payload["loss"])

    @staticmethod
    def checkpoint(payload):
        tokens_sec = payload["examples"] / payload["duration"]
        print("Batch [%d/%d], loss: %g, processed %d tokens/sec" %
              (payload["batch"], payload["total_batches"],
               payload["loss"], tokens_sec))

    @staticmethod
    def info(payload):
        print("INFO: %s" % payload["message"])

    def log(self, event, payload, verbose=True):
        if verbose and hasattr(self, event):
            getattr(self, event)(payload)


# Base Trainer class
class Trainer(object):
    def __init__(self, model, datasets, criterion, optimizer,
                 test_name='test', valid_name='valid',
                 size_average=True, verbose=True):
        """
        Parameter:
        ==========
        - size_average: bool,
            whether the loss is already averaged over examples.
            See `size_average` in the torch.nn criterion functions.
        """
        # attributes
        self.model = model
        self.datasets = datasets   # is a dict with at least a 'train' entry
        self.criterion = criterion  # might be a dict
        self.optimizer = optimizer  # might be a dict
        # config
        self.verbose = verbose
        self.size_average = size_average
        # containers
        self.hooks = []
        self.loggers = []
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
    def add_hook(self, hook, num_checkpoints=1):
        self.hooks.append({'hook': hook, 'num_checkpoints': num_checkpoints})

    def run_hooks(self, batch_num, checkpoint):
        for hook in self.hooks:
            num_checkpoints = batch_num // checkpoint
            if num_checkpoints % hook['num_checkpoints'] == 0:
                hook['hook'](self, batch_num, num_checkpoints)

    # callbacks
    def on_batch_end(self, batch, batch_loss):
        # reset hidden, and things like that
        pass

    def on_epoch_begin(self, epoch):
        self.log("epoch_begin", {"epoch": epoch})

    def on_epoch_end(self, epoch, loss, num_examples, duration):
        self.log("epoch_end", {"epoch": epoch,
                               "loss": loss,
                               "examples": num_examples,
                               "duration": duration})

    def on_validation_end(self, epoch, loss):
        self.log("validation_end", {"epoch": epoch, "loss": loss})

    def on_test_begin(self, epoch):
        self.log("test_begin", {"epoch": epoch})

    def on_test_end(self, loss):
        self.log("test_end", {"loss": loss})

    def format_loss(self, loss):
        return loss

    # optimizer
    def zero_grad(self):
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if isinstance(self.optimizer, dict):
            for opt in self.optimizer.values():
                opt.step()
        else:
            self.optimizer.step()

    # training code
    def num_batch_examples(self, batch_data):
        """
        By default consider all elements in batch tensor.
        """
        return batch_data.n_element()

    def validate_model(self, test=False, **kwargs):
        self.model.eval()
        loss, num_examples = 0, 0
        dataset = self.datasets[self.test_name if test else self.valid_name]
        for batch_num in range(len(dataset)):
            batch = dataset[batch_num]
            batch_examples = self.num_batch_examples(batch)
            num_examples += batch_examples
            batch_loss = self.run_batch(
                batch, dataset=self.valid_name, **kwargs)
            loss += batch_loss * (batch_examples if self.size_average else 1)
        self.model.train()
        return self.format_loss(loss.data[0] / num_examples)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        """
        Method in charge of computing batch loss and (eventually)
        running optimizations on the model. It should return the
        loss in torch tensor form.
        """
        source, targets = batch_data
        outs = self.model(source, targets)
        loss = self.criterion(outs, targets)
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return loss

    def train_epoch(self, epoch, checkpoint, shuffle, **kwargs):
        # compute batch order
        dataset = self.datasets['train']
        batch_order = range(len(dataset))
        if shuffle:
            batch_order = np.random.permutation(batch_order)
        start = time.time()
        epoch_loss, check_loss, epoch_examples, check_examples = 0, 0, 0, 0
        for batch_num, batch in enumerate(batch_order):
            self.zero_grad()
            # TODO: loss might be complex (perhaps use np.array?)
            batch_data = dataset[batch]
            loss = self.run_batch(batch_data, dataset='train', **kwargs)
            if loss is None:  # to skip a batch run_batch might return None
                continue
            self.on_batch_end(batch, self.format_loss(loss.data[0]))
            # report
            num_examples = self.num_batch_examples(batch_data)
            # depending on loss being averaged. See `size_average`.
            batch_loss = \
                loss.data[0] * (num_examples if self.size_average else 1)
            epoch_loss += batch_loss
            check_loss += batch_loss
            epoch_examples += num_examples
            check_examples += num_examples
            # checkpoint
            if checkpoint and batch_num > 0 and batch_num % checkpoint == 0:
                self.log('checkpoint', {
                    'batch': batch_num,
                    'total_batches': len(batch_order),
                    'examples': check_examples,
                    'duration': time.time() - start,
                    'loss': self.format_loss(check_loss / check_examples)})
                self.run_hooks(batch_num, checkpoint)
                check_loss, check_examples, start = 0, 0, time.time()
        return epoch_loss, epoch_examples

    def train(self, epochs, checkpoint, shuffle=False, gpu=False, **kwargs):
        """
        Parameters:
        ===========
        - epochs: int
        - checkpoint: int, log a checkpoint and hooks every x batches
        - gpu: bool
        """
        for epoch in range(1, epochs + 1):
            start = time.time()
            self.on_epoch_begin(epoch)
            try:  # TODO: EarlyStopping might come from train or valid
                # train
                self.model.train()
                epoch_loss, epoch_examples = self.train_epoch(
                    epoch, checkpoint, shuffle, **kwargs)
                epoch_loss = self.format_loss(epoch_loss / epoch_examples)
                epoch_time = time.time() - start
                # valid
                if self.valid_name in self.datasets:
                    valid_loss = self.validate_model(**kwargs)
                    self.on_validation_end(epoch, valid_loss)
                self.on_epoch_end(
                    epoch, epoch_loss, epoch_examples, epoch_time)
            except EarlyStoppingException as e:
                message, _ = e.args
                self.log("info", {"message": message})
                break
            except KeyboardInterrupt:
                self.log("info", {"message": "Training interrupted"})
                break
        # test
        if self.test_name in self.datasets:
            self.on_test_begin(epoch)
            test_loss = self.validate_model(test=True, **kwargs)
            self.on_test_end(test_loss)


class LMTrainer(Trainer):
    def format_loss(self, loss):
        """
        Turn loss into perplexity.
        """
        return math.exp(min(loss, 100))

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
            output, hidden = self.model(source, hidden=hidden, head=head)
            # dettach hidden from graph
            if 'hidden' not in self.batch_state:
                self.batch_state['hidden'] = {}
            self.batch_state['hidden'][head] = repackage_hidden(hidden)
        else:
            source, targets = batch_data
            hidden = self.batch_state.get('hidden', None)
            output, hidden = self.model(source, hidden=hidden)
            # detach hidden from graph
            self.batch_state['hidden'] = repackage_hidden(hidden)
        loss = self.criterion(output, targets)
        # optimize
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return loss

    def on_batch_end(self, batch, loss):
        if hasattr(self, 'reset_hidden'):
            if isinstance(next(self.datasets.values()), CyclicBlockDataset):
                for v in self.batch_state['hidden'].values():
                    v.zero_()
            else:
                self.batch_state['hidden'].zero_()

    def num_batch_examples(self, batch_data):
        src, trg, *_ = batch_data
        return len(src)


class EncoderDecoderTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(EncoderDecoderTrainer, self).__init__(*args, **kwargs)
        self.size_average = False

    def format_loss(self, loss):
        return math.exp(min(loss, 100))

    def run_batch(self, batch_data, dataset='train', split_batch=52, **kwargs):
        evaluation = dataset != 'train'
        pad, eos = self.model.src_dict[u.PAD], self.model.src_dict[u.EOS]
        loss = 0
        source, targets = batch_data
        # remove <eos> from decoder targets substituting them with <pad>
        decode_targets = Variable(u.map_index(targets[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_targets = targets[1:]
        # remove <bos> from source and compute model output
        outs = self.model(source[1:], decode_targets)
        # dettach outs from computational graph
        det_outs = Variable(
            outs.data, requires_grad=not evaluation, volatile=evaluation)
        for out, trg in zip(
                det_outs.split(split_batch), loss_targets.split(split_batch)):
            loss += self.criterion(
                self.model.project(out.view(-1, out.size(2))), trg.view(-1))
        if not evaluation:
            loss.div(outs.size(1)).backward()
            grad = None if det_outs.grad is None else det_outs.grad.data
            outs.backward(grad)
            self.optimizer_step()
        return loss

    def num_batch_examples(self, batch_data):
        _, targets = batch_data
        return targets.data.ne(self.model.src_dict[u.PAD]).sum()
