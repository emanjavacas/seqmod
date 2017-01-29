import math
import torch.optim as optim


class Optimizer(object):
    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1,
                 start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.optimizer = getattr(optim, self.method)(self.params, lr=self.lr)

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        shrinkage = self.max_grad_norm / grad_norm

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    def update_learning_rate(self, ppl, epoch):
        """
        Decay learning rate if validation perplexity does not improve
        or we hit the start_decay_at limit
        """
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        self.optimizer = getattr(optim, self.method)(self.params, lr=self.lr)
