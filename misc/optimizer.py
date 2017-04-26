
import math
import torch.optim as optim
from torch.nn.utils import clip_grad_norm


def root(x, nth=1):
    """
    warning!: this is only exact for p values below 3.
    """
    return math.pow(x, 1/nth)


class Optimizer(object):
    def __init__(self, params, method, lr=1., max_norm=5., weight_decay=0,
                 lr_decay=1, start_decay_at=None, decay_every=1, on_lr_update=None):
        self.params = list(params)
        self.method = method
        self.lr = lr
        self.max_norm = max_norm if max_norm > 0 else None
        self.weight_decay = weight_decay
        # lr decay for vanilla SGD
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.decay_every = decay_every
        self.on_lr_update = on_lr_update
        # attributes
        self.last_loss = None
        self.start_decay = False
        self.optim = getattr(optim, self.method)(
            self.params, lr=self.lr, weight_decay=self.weight_decay)

    def set_params(self, params):
        self.params = list(params)
        self.optim = getattr(optim, self.method)(self.params, lr=self.lr)

    def step(self, norm_type=2):
        """
        Run an update eventually clipping the gradients
        """
        if self.max_norm is not None:
            clip_grad_norm(self.params, self.max_norm, norm_type=norm_type)
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def maybe_update_lr(self, epoch, loss):
        """
        Decay learning rate if validation perplexity does not improve
        or we hit the start_decay_at limit
        """
        last_lr = self.lr
        if self.method == 'SGD':
            if self.start_decay_at and epoch >= self.start_decay_at:
                self.start_decay = True
            if self.last_loss is not None and loss > self.last_loss:
                self.start_decay = True
            if self.start_decay and epoch % self.decay_every == 0:
                new_lr = self.lr * self.lr_decay
                if self.on_lr_update is not None:
                    self.on_lr_update(self.lr, new_lr)
                self.lr = new_lr
                self.optim = getattr(optim, self.method)(
                    self.params, lr=self.lr, weight_decay=self.weight_decay)
            self.last_loss = loss
            return last_lr, self.lr
