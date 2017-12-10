
"""
Schedule functions to tweak parameters.
Most functions are only defined for positive x values.
"""

import math


def generic_sigmoid(a=1, b=1, c=1):
    """
    - a: upper asymptote
    - b: y intercept
    - c: steepness
    """
    return lambda x: a / (1 + b * math.exp(-x * c))


def inflection_sigmoid(inflection, steepness):
    """
    A particular sigmoid as a function of inflection point
    (x value for which the second derivative is 0) and the
    steepness of the curve.
    """
    b = 10 ** steepness
    c = math.log(b) / inflection
    return generic_sigmoid(a=1, b=b, c=c)


def inverse_sigmoid(k):
    """
    A particular sigmoid tha decreases from 1 to 0 at k's pace 

    - k: decrease rate
    """
    return lambda k: k / (k + math.exp(x / k))


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
    - k: value below 0 that determines the decrease rate
    - y_intercept: value of y at x equal to 0
    """
    return lambda x: k^x + (y_intercept - 1)


def inverse_exponential(k=0.5, y_intercept=1):
    """
    - k: value below 0 that determines the decrease rate
    - y_intercept: value of y at x equal to 0
    """
    return lambda x: k^x - (1 - y_intercept)
