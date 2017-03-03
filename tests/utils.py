import string
import os
from random import randint, random

import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, text, \
    sampled_from

_dir = os.path.dirname(__file__)

printable = text(string.printable)


def primes():
    with open(os.path.join(_dir, 'data/primes.dat')) as src:
        p = list(map(int, src))
    return sampled_from(p)


# strategy to test substring functions
@composite
def substring_pair(draw):
    t = draw(printable)  # text to search in
    s = draw(printable)

    # with probability of 0.5 s in t
    if random() > 0.5:
        n = randint(0, len(t))  # len of substring
        start = randint(0, len(t) - n)  # start os occurrence
        return t, t[start:start + n]
    return t, s


def square_matrices(elements=floats(), N=integers(min_value=1, max_value=20)):
    return N.flatmap(lambda n: arrays(np.float, (n, n), elements))
