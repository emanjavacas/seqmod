"""
Taken from https://github.com/zygmuntz/hyperband/
Copyright (c) 2017, Zygmunt Zając
"""

import numpy as np

from random import random
from math import log, ceil
from time import time, ctime


class Hyperband:
    """
    Parameters:
    -----------

    get_params: fn() => sampled param space
    try_params: fn(n_iters, params) => {'loss': float, 'early_stop': bool}

    max_iter: int, total maximum of iterations where an iteration is internally
        defined by try_params.
    eta: number, downsampling rate
    """
    def __init__(self, get_params, try_params, max_iter=81, eta=3):
        self.get_params = get_params
        self.try_params = try_params
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
            T = [self.get_params() for i in range(n)]

            for i in range(s + 1):
                # Run each config `n_iters` & keep best (n_configs/eta) configs
                n_configs = n * self.eta ** (-i)
                n_iters = r * self.eta ** (i)

                print("\n*** {} configurations x {:.1f} iterations each"
                      .format(n_configs, n_iters))

                val_losses, early_stops = [], []

                for t in T:
                    # TODO: cache previously run config
                    self.counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n"
                          .format(self.counter, ctime(),
                                  self.best_loss,
                                  self.best_counter))
                    # record time
                    start_time = time()
                    # run
                    result = self.try_params(n_iters, t)
                    # record loss
                    loss = result['loss']
                    val_losses.append(loss)
                    # record early_stop
                    early_stops.append(result.get('early_stop', False))

                    # keep track of the best result so far (for display only)
                    # could do it by checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    # register result
                    result['counter'] = self.counter
                    result['seconds'] = int(round(time() - start_time))
                    result['params'] = t
                    result['iterations'] = n_iters

                    self.results.append(result)

                # select
                indices = np.argsort(val_losses)
                # filter out early stops
                T = [T[i] for i in indices if not early_stops[i]]
                # select a number of best configurations for the next loop
                T = T[0:int(n_configs / self.eta)]

        return self.results
