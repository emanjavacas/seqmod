"""
Inspired by https://github.com/zygmuntz/hyperband/
Copyright (c) 2017, Zygmunt Zając, Enrique Manjavacas
"""

import numpy as np

from math import log, ceil
from time import time, ctime


class Manager(object):
    """
    Parameters:
    -----------
    param_sampler: callable() -> dict (sampled param space)

    create_runner: callable(params) -> callable(n_iters) -> dict
        result {'loss': float, 'early_stop': bool}
    """
    def __init__(self, param_sampler, create_runner):
        self.param_sampler = param_sampler
        self.runner_builder = create_runner
        self.models = []

    def sample_n(self, n):
        for _ in range(n):
            params = self.param_sampler()
            model = self.runner_builder(params)
            self.models.append((model, {'params': params, 'runs': []}))

    def prune_early_stopped(self):
        self.models = [(m, data) for (m, data) in self.models
                       if not data.get('early_stop', False)]

    def prune_topk(self, k):
        sorted_models = sorted(self.models,
                               key=lambda m: m[1]['runs'][-1]['loss'])

        self.models = sorted_models[:k]


class Hyperband(object):
    """
    Parameters:
    -----------
    param_sampler: callable() -> dict (sampled param space)

    create_runner: callable(params) -> callable(n_iters) -> dict
        result {'loss': float, 'early_stop': bool}

    max_iter: int, total maximum of iterations where an iteration is internally
        defined by try_params.

    eta: number, downsampling rate
    """
    def __init__(self, param_sampler, create_runner, max_iter=81, eta=3):
        self.manager = Manager(param_sampler, create_runner)
        self.max_iter = max_iter  # maximum iterations per configuration
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []       # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = None
        self.best_model = None

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
                for m, data in self.manager.models:
                    self.counter += 1
                    start_time = time()
                    # report run
                    msg = "\n{} | {} | lowest loss so far: {:.4f} (run {})\n"
                    print(msg.format(self.counter, ctime(),
                                     self.best_loss, self.best_counter))
                    # run
                    result = m(n_iters)
                    # get loss
                    loss = result['loss'] or np.inf
                    # keep track of the best result so far
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                        self.best_model = m

                    # register result
                    result['counter'] = self.counter
                    result['seconds'] = int(round(time() - start_time))
                    result['iterations'] = n_iters
                    self.results.append({'params': data['params'], **result})
                    # add result to model metadata
                    data['runs'].append(result)
                # prune
                self.manager.prune_early_stopped()
                self.manager.prune_topk(int(n_configs / self.eta))

        return self.results
