
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
