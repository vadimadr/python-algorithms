import string
from random import random, randint

from hypothesis.strategies import text, composite

printable = text(string.printable)


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
