"""
Inspired by https://github.com/zygmuntz/hyperband/
Copyright (c) 2017, Zygmunt Zając, Enrique Manjavacas
"""

import numpy as np

from math import log, ceil
from time import time, ctime


class Hyperband(object):
    """
    Parameters:
    -----------

    get_params: fn() => sampled param space
    try_params: fn(n_iters, params) => {'loss': float, 'early_stop': bool}

    max_iter: int, total maximum of iterations where an iteration is internally
        defined by try_params.
    eta: number, downsampling rate
    """
    def __init__(self, manager, max_iter=81, eta=3):
        self.manager = manager
        self.max_iter = max_iter  # maximum iterations per configuration
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []       # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    def run(self):
        for s in reversed(range(self.s_max + 1)):
            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)
            # n random configurations
            self.manager.sample_n(n)

            for i in range(s + 1):
                # Run each config `n_iters` & keep best (n_configs/eta) configs
                n_configs = n * self.eta ** (-i)
                n_iters = r * self.eta ** (i)
                # report
                print("\n{} configs x {:.1f} iters".format(n_configs, n_iters))
                # run each remaining config
                for m, data in self.manager:
                    self.counter += 1
                    start_time = time()
                    # report run
                    msg = "\n{} | {} | lowest loss so far: {:.4f} (run {})\n"
                    print(msg.format(self.counter, ctime(),
                                     self.best_loss, self.best_counter))
                    # run
                    result = m(n_iters)
                    # keep track of the best result so far
                    if result['loss'] < self.best_loss:
                        self.best_loss = result['loss']
                        self.best_counter = self.counter

                    # register result
                    result['counter'] = self.counter
                    result['seconds'] = int(round(time() - start_time))
                    result['iterations'] = n_iters
                    self.results.append(result)
                    # add result to model metadata
                    data['runs'].append(result)
                # prune
                self.manager.prune_early_stopped()
                self.manager.prune_topk(int(n_configs / self.eta))

        return self.results
