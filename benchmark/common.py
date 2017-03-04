import gc
from abc import ABC, abstractmethod
from operator import itemgetter
from time import perf_counter

import numpy as np
from matplotlib import pyplot
from scipy.optimize import curve_fit, root, brentq


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
    ns = np.array(ns)
    popt, pcov = curve_fit(th, ns, ys, 1)
    c = popt[0]
    ys0 = th(ns, c)
    err = ((ys - ys0) ** 2).sum()
    return c, err


T_ = {
    'lin': lambda n, c: c * n,
    'n^2': lambda n, c: c * n ** 2,
    'n^3': lambda n, c: c * n ** 3,
    'n*logn': lambda n, c: c * n * np.log(n),
    'n^2*logn': lambda n, c: c * (n**2) * np.log(n),
    'n*(logn)^2': lambda n, c: c * n * (np.log(n) **2),
}


def guess_theta(ns, ys):
    """Returns Theta(n) that best fits to (n, t)"""
    t = {}
    for k in T_:
        c, err = fit_theta(ns, ys, T_[k])
        t[k] = (c, err)
    best = min(t, key=lambda k: t[k][1])
    return best, t


def auto_bench(bench, tlim, ns_init=None, num=50, n0=1):
    """Returns (n, t) with n distributed uniformely within time budget

    Parameters
    ----------
    tlim : float
        approximate runtime of this function
    num : int
        number of points
    """

    # find good n0
    if ns_init is None:
        ns = []
        ys = []
        while True:
            py0 = time_dist(bench, n0)
            m = py0.mean()
            sd = py0.std()

            if m > 0.001 and sd * 2 < m:
                break

            n0 += 1
            ns.append(n0)
            ys.append(m)
    else:
        n0 = ns_init[-1]
        ns = list(ns_init)
        ys = [bench.time(n) for n in ns]

    print("n0: %d" % n0)
    t = 0

    def F(d, th, n0, k, c, t):
        return th(np.arange(1, k + 1) * d + n0, c).sum() - t

    for i in range(num):
        trem = max(0, tlim - t)

        best, ds = guess_theta(ns, ys)
        th = T_[best]
        c = ds[best][1]

        k = num - 1
        d = root(F, 0, args=(th, n0, k, c, trem)).x
        n0 += int(d)

        print("guess %d: %s %d %f" % (i, best, n0, c))

        ns.append(n0)
        time = bench.time(n0)
        t += time
        ys.append(time)

    return np.array((ns, ys))


def theta_est(th):
    best, ts = th
    c = ts[best][0]
    return lambda n: T_[best](n, c)


def visualize_range_bench(ns, ys, label=None):
    pyplot.scatter(ns, ys, alpha=.7, label=label)

    best, ts = guess_theta(ns, ys)
    est = theta_est((best, ts))

    pyplot.plot(ns, est(ns), '--')
    pyplot.xlabel('n')
    pyplot.ylabel('time')

    estimate_ = best, ts[best][1], ts[best][0]
    print("Estimated Time complexity: %s (c = %f) RSS: %f" % (estimate_))
