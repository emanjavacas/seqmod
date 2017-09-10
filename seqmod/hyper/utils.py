
import math
from hyperopt import hp
from hyperopt.pyll.stochastic import sample


def make_sampler(space):
    """
    Transforms a space in format:
        {'param': ['hyperopt function name', type, hp_args...]}
    into a proper hyperopt space, and returns a closure that knows
    how to sample from the original space with proper conversion
    to the predefined input type
    """
    hp_space = {k: getattr(hp, hp_fn)(k, *tuple(args))
                for k, (hp_fn, _, *args) in space.items()}

    def fn():
        sampled = sample(hp_space)
        for k in space:
            param_type = space[k][1]
            sampled[k] = param_type(sampled[k])
        return sampled

    return fn


def random_search_generator(space, n_iter=math.inf):
    """
    space = {'lr': ['loguniform', float, math.log(0.00001), math.log(1)],
             'hid_dim': ['loguniform', int, math.log(10), math.log(1000)],
             'emb_dim': ['choice', int, list(range(20, 60)]}
    generator = random_search_generator(space)
    next(generator)
    >>> {'emb_dim': 46, 'hid_dim': 33, 'lr': 0.04909280719702653}
    """
    sampler = make_sampler(space)

    counter = 0
    while counter < n_iter:
        yield sampler()
        counter += 1


class ModelManager(object):
    """
    Parameters:
    -----------
    param_sampler: fn() -> dict (sampled param space)
    model_builder: fn(params) -> fn(n_iters) -> dict
        result {'loss': float, 'early_stop': bool}
    """
    def __init__(self, param_sampler, model_builder):
        self.param_sampler = param_sampler
        self.model_builder = model_builder
        self.models = []

    def sample_n(self, n):
        for _ in range(n):
            params = self.param_sampler()
            self.models.append(
                (self.model_builder(params), {'params': params, 'runs': []})
            )

    def prune_early_stopped(self):
        self.models = [(m, data) for (m, data) in self.models
                       if not data.get('early_stop', False)]

    def prune_topk(self, k):
        sorted_models = sorted(self.models, lambda m: m[1]['runs'][-1]['loss'])
        self.models = sorted_models[:k]
