def newton_raphson_root(f, y, x0, df=None, maxiter=100, tol=1e-6):
    """Solve f(x) = y using using the Newton-Raphson or secant method.

    Parameters
    ----------
    x0 : float
        initial guess
    df : function
        derivative of f. If it is None (default), then the secant method is
        used.
    tol : float
        allowed tolerance
    """
    x = x0
    xprev = 1.
    i = 1
    err = abs(f(x) - y)
    while err > tol and i < maxiter:

        if df is not None:
            d = df(x)
        else:
            # secant method
            d = (f(x) - f(xprev)) / (x - xprev)
            xprev = x

        x = x - (f(x) - y) / d
        err = abs(f(x) - y)
        i += 1
    return x


def ternary_search(f, l, r, min_=True, maxiter=100, tol=1e-6):
    """Optimize f(x) using the ternary search

    f(x) is unimodal (only one optima) in a [l, r]
    """
    i = 0
    while r - l > tol and i < maxiter:
        # split (r - l) in 3 parts
        a = (l * 2 + r) / 3  # l + 1/3 * (r - l)
        b = (l + 2 * r) / 3  # l + 2/3 * (r - l)

        # f(x) either increasing or min on [a,b]
        if f(a) < f(b) and min_:
            # [b, r] is no longer of interest
            r = b
        # decreasing or max
        elif f(a) >= f(b) and not min_:
            r = b
        else:
            # [a, l] is no longer of interest
            l = a
        i += 1
    return (l + r) / 2