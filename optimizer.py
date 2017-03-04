import math
import torch.optim as optim


def root(x, nth=1):
    """
    warning!: this is only exact for p values below 3.
    """
    return math.pow(x, 1/nth)


class Optimizer(object):
    def __init__(self, params, method, lr=1., threshold=5.,
                 lr_decay=1, start_decay_at=None):
        self.params = list(params)
        self.lr = lr
        self.threshold = threshold
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.method = method
        self.last_ppl = None
        self.start_decay = False
        self.optim = getattr(optim, self.method)(self.params, lr=self.lr)

    def LP(self, p=1):
        """
        Computes the l_p norm over gradients l_p norms
        """
        norms = sum(math.pow(pm.grad.data.norm(p=p), p) for pm in self.params)
        return root(norms, nth=p)

    def L2(self):
        return self.LP(p=2)

    def L1(self):
        return self.LP(p=1)

    def clip_gradients(self, norm):
        """
        Clips gradients down whenever the total model gradient norm
        exceeds a given threshold
        """
        grad_norm = getattr(self, norm)()
        if grad_norm > self.threshold:
            shrinkage = self.threshold / grad_norm
            for param in self.params:
                param.grad.data.mul_(shrinkage)

    def step(self, norm='L2'):
        """
        Run an update eventually clipping the gradients
        """
        self.clip_gradients(norm)
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
            self.optim = getattr(optim, self.method)(self.params, lr=self.lr)
            return last_lr, self.lr
