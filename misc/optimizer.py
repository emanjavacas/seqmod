
import math
import torch.optim as optim
from torch.nn.utils import clip_grad_norm


def root(x, nth=1):
    """
    warning!: this is only exact for p values below 3.
    """
    return math.pow(x, 1/nth)


class Optimizer(object):
    def __init__(self, params, method, lr=1., max_norm=5.,
                 weight_decay=0, lr_decay=1, start_decay_at=None):
        self.params = list(params)
        self.method = method
        self.lr = lr
        self.max_norm = max_norm if max_norm > 0 else None
        self.weight_decay = weight_decay
        # lr decay for vanilla SGD
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        # attributes
        self.last_ppl = None
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

    def maybe_update_lr(self, epoch, ppl):
        """
        Decay learning rate if validation perplexity does not improve
        or we hit the start_decay_at limit
        """
        last_lr = self.lr
        if self.method == 'SGD':
            if self.start_decay_at and epoch >= self.start_decay_at:
                self.start_decay = True
            if self.last_ppl is not None and ppl > self.last_ppl:
                self.start_decay = True
            if self.start_decay:
                self.lr = self.lr * self.lr_decay
            self.last_ppl = ppl
            self.optim = getattr(optim, self.method)(
                self.params, lr=self.lr, weight_decay=self.weight_decay)
            return last_lr, self.lr
