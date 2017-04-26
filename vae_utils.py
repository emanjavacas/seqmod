
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod.modules import utils as u

from seqmod.misc.beam_search import Beam
from seqmod.misc.trainer import Trainer


class vae_criterion(object):
    def __init__(self, vocab_size, pad):
        # Reconstruction loss
        weight = torch.ones(vocab_size)
        weight[pad] = 0
        self.log_loss = nn.NLLLoss(weight, size_average=False)

    # KL-divergence loss
    def KL_loss(self, mu, logvar):
        """
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.sum(element).mul_(-0.5)

    def cuda(self):
        self.log_loss.cuda()

    def __call__(self, logs, targets, mu, logvar):
        return self.log_loss(logs, targets), self.KL_loss(mu, logvar)


def kl_weight_hook(trainer, epoch, batch, num_checkpoints):
    trainer.log("info", "kl weight: [%g]" % trainer.kl_weight)


def make_generate_hook(target="This is just a tweet and not much more ...", n=5):

    def hook(trainer, epoch, batch, num_checkpoints):
        d = trainer.datasets['train'].d['src']
        inp = torch.LongTensor([d.index(i) for i in target.split()])
        inp = Variable(inp, volatile=True).unsqueeze(1)
        z_params = trainer.model.encode(inp)
        for hyp_num in range(1, n + 1):
            score, hyp = trainer.model.generate(z_params=z_params)
            trainer.log("info", u.format_hyp(score[0], hyp[0], hyp_num, d))

    return hook


def generic_sigmoid(a=1, b=1, c=1):
    return lambda x: a / (1 + b * math.exp(-x * c))


def kl_annealing_schedule(inflection, steepness=4):
    b = 10 ** steepness
    return generic_sigmoid(b=b, c=math.log(b) / inflection)


class VAETrainer(Trainer):
    def __init__(self, *args, inflection_point=5000, **kwargs):
        super(VAETrainer, self).__init__(*args, **kwargs)
        self.size_average = False
        self.loss_labels = ('rec', 'kl')
        self.kl_weight = 0.0    # start at 0
        self.kl_schedule = kl_annealing_schedule(inflection_point)

    def format_loss(self, losses):
        return tuple(math.exp(min(loss, 100)) for loss in losses)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        valid = dataset != 'train'
        pad, eos = self.model.src_dict.get_pad(), self.model.src_dict.get_eos()
        source, labels = batch_data
        # remove <eos> from decoder targets dealing with different <pad> sizes
        decode_targets = Variable(u.map_index(source[:-1].data, eos, pad))
        # remove <bos> from loss targets
        loss_targets = source[1:].view(-1)
        # preds: (batch * seq_len x vocab)
        preds, mu, logvar = self.model(source[1:], decode_targets, labels=labels)
        # compute loss
        log_loss, kl_loss = self.criterion(preds, loss_targets, mu, logvar)
        if not valid:
            batch = source.size(1)
            loss = (log_loss + (self.kl_weight * kl_loss)).div(batch)
            loss.backward()
            self.optimizer_step()
        return log_loss.data[0], kl_loss.data[0]

    def num_batch_examples(self, batch_data):
        source, _ = batch_data
        return source[1:].data.ne(self.model.src_dict.get_pad()).sum()

    def on_batch_end(self, batch, loss):
        # reset kl weight
        total_batches = len(self.datasets['train'])
        self.kl_weight = self.kl_schedule(batch + total_batches * self.epoch)
