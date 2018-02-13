
"""
Schedule functions to tweak parameters.
Most functions are only defined for positive x values.
"""

import math


def generic_sigmoid(a=1, b=1, c=1, inverse=False):
    """
    - a: upper asymptote
    - b: y intercept
    - c: steepness
    """
    if inverse:
        return lambda x: a - (a / (1 + b * math.exp(-x * c)))
    else:
        return lambda x: a / (1 + b * math.exp(-x * c))


def inflection_sigmoid(inflection, steepness, a=1, inverse=False):
    """
    A particular sigmoid as a function of inflection point
    (x value for which the second derivative is 0) and the
    steepness of the curve. Typical smooth steepness values are below 4.
    To invert it, just substract the output value from 1.

    - a: max value (target value, if inverse is False or initial otherwise)
    """
    b = 10 ** steepness
    c = math.log(b) / inflection
    return generic_sigmoid(a=a, b=b, c=c, inverse=inverse)


def linear(convergence):
    """
- convergence: value of x at which y reaches 1
    """
    return lambda x: x * (1 / convergence)


def inverse_linear(convergence):
    """
    y = c * x + 1

    convergence * c = -1

    - convergence: value of x at which y reaches 0
    """
    return lambda x: x * (-1 / convergence)


def exponential(k=1.5, y_intercept=0):
    """
    - k: value above 1 that determines the increase rate. Values above 1.15
        give already a very steep increase.
    - y_intercept: value of y at which x is equal to 0
    """
    return lambda x: k ** x + (y_intercept - 1)


def inverse_exponential(k=0.95, y_intercept=1):
    """
    - k: value below 1 that determines the decrease rate. Values below 0.85
        give already a very steep decrease.
    - y_intercept: value of y at which x is equal to 0
    """
    return lambda x: k ** x - (1 - y_intercept)


def inverse_cosine(a_min, a_max, total):
    """
    A schedules that goes from a_max to a_min over a predefined number of steps
    `total`, following a cosine slope.

    https://arxiv.org/pdf/1801.06146.pdf
    """
    return lambda t: a_min + 1/2 * ((a_max-a_min)*(1+math.cos((t/total) * math.pi)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def squash(x, min_from, max_from, min_to, max_to):
        term = (x - min_from) / (max_from - min_from)
        return ((max_to - min_to) * term) + min_to

    nrows, ncols = 5, 3

    for i in range(nrows * ncols):
        ax = plt.subplot(nrows, ncols, i + 1)

        # test sigmoid
        r = list(range(0, 1000, 10))
        min_i, max_i = 0.5, 5
        i = squash(i, 0, (nrows * ncols) - 1, min_i, max_i)
        plt.plot(r, [1-inflection_sigmoid(500, i+1)(j) for j in r], axes=ax)

        # # test exponential
        # r = list(range(0, 100))
        # # test range
        # min_i, max_i = 1.001, 1.15
        # # squash to test range
        # i = squash(i, 0, (nrows * ncols) - 1, min_i, max_i)
        # plt.plot(r, [exponential(k=i, y_intercept=100)(j) for j in r])

    plt.show()
