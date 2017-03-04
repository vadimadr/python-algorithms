import gc
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
from scipy.optimize import curve_fit


class Benchmark(ABC):
    @abstractmethod
    def setup(self, n):
        """Generate data of size n"""
        pass

    @abstractmethod
    def run(self):
        """Run algorithm"""
        pass

    def time(self, n, n_times=1, same_data=True):
        t = []
        self.setup(n)
        for i in range(n_times):
            gcold = gc.isenabled()
            gc.disable()
            try:
                t0 = perf_counter()
                self.run()
                t1 = perf_counter()
            finally:
                if gcold:
                    gc.enable()

            t.append(t1 - t0)
            if not same_data:
                self.setup(n)
            return min(t)


def time_range(bench, n0, n1, num=50, **kwargs):
    """Returns (n, t) for n in [n0, n1) """
    ns = np.linspace(n0, n1, num, dtype=int)
    ys = []
    for n in ns:
        ys.append(bench.time(n, **kwargs))
    return np.array((ns, ys))


def time_dist(bench, n, num=100, **kwargs):
    """Distribution of t for n"""
    ys = []

    for i in range(num):
        ys.append(bench.time(n, **kwargs))
    return np.array(ys)


def fit_theta(ns, ys, th):
    """fit (n, t) to c * Theta(n) """
    popt, pcov = curve_fit(th, ns, ys, 1)
    c = popt[0]
    ys0 = th(ns, c)
    err = ((ys - ys0) ** 2).sum() / len(ns)
    return c, err
