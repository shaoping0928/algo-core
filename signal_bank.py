"""Per-node operators along the input axis, shared by the validation scripts.

    id           x(t)                                  the intensity itself
    d1           x(t) - x(t-1)                         a differentiator
    d2           x(t) - 2x(t-1) + x(t-2)               second difference
    leak_beta    (1-beta)u(t-1) + beta*x(t)            an exponential decayer
    unleak_beta  (x(t) - (1-beta)x(t-1)) / beta        its exact inverse

Every output is a signal source.  Operators act within a column, so a bank applied to an
(T, N) intensity matrix never mixes nodes.
"""
import numpy as np

BETAS = (0.5, 0.2, 0.1, 0.05, 0.02)


def bank(X, betas=BETAS, means=()):
    X = np.asarray(X, dtype=float)
    T = len(X)
    out = {"id": X.copy()}
    d = X.copy()
    for order in (1, 2):
        prev = np.vstack([X[:1], d[:-1]])
        d = d - prev
        out["d%d" % order] = d
    for w in means:
        m = np.empty_like(X)
        run = np.zeros(X.shape[1])
        for t in range(T):
            run += X[t]
            if t >= w:
                run -= X[t - w]
            m[t] = run / min(t + 1, w)
        out["ma%d" % w] = m
    for b in betas:
        u = np.zeros_like(X)
        for t in range(T):
            u[t] = (1.0 - b) * u[t - 1] + b * X[t] if t else b * X[0]
        out["leak%g" % b] = u
        un = np.empty_like(X)
        un[0] = X[0] / b
        un[1:] = (X[1:] - (1.0 - b) * X[:-1]) / b
        out["unleak%g" % b] = un
    return out


def fit_residual(src, tgt, holdout=0.3):
    """max |target - (a*src + c)| on the held-out part"""
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.stack([np.ones(n), src], 1)
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    return float(np.abs(A[cut:] @ w - tgt[cut:]).max())


def make_world(T=4000, N=3, beta0=0.1, seed=0):
    rng = np.random.RandomState(seed)
    s = rng.randint(0, 2, size=(T, N)).astype(float)
    x = np.zeros((T, N))
    for t in range(T):
        x[t] = (1.0 - beta0) * x[t - 1] + beta0 * s[t] if t else beta0 * s[0]
    return s, x, beta0


def noise_gain(beta):
    """how much the inverse of a leak at `beta` amplifies white noise"""
    return float(np.sqrt(1.0 + (1.0 - beta) ** 2) / beta)
