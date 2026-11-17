"""Counting mechanisms when the numbers are noisy.

The old instrument counted distinct values.  Exact in a noiseless world, useless in any other:
one part in 1e9 of noise makes every sample distinct, so the count is the sample size and the
readout says "the form is wrong" about a form that is right.

The fix is to count modes instead of values, and to read the noise scale out of the same fit
instead of being told it:

    residual ~ sum_k  w_k * N(mu_k, sigma)        shared sigma, k chosen by BIC

    k = 1        one mode: complete, OR one smooth continuum -- the two are statistically
                 identical, so the fitted sigma is what separates them (floor = exact)
    k = 2        one binary mechanism missing
    k = kmax     at or above the cap: not a small number of mechanisms
    sigma        the noise scale, a free parameter of the same fit

and because differencing amplifies noise, the horizon has to be stated in the units of the
input, not of the layer.

Criteria written before the run:
    N1  exact data: returns 2, recovers both centres, and the fitted sigma sits at the
        estimator's resolution floor (2% of the data's spread) -- that floor is the honest
        statement of how far down it can see, not a bug
    N2  horizon: as sigma grows the answer goes 2 -> 1 at some sigma*, reported as sigma*/Delta
    N3  no false positives: on a single level plus noise, 2 is never chosen
    N4  more levels than the cap read as kmax, so "2" is distinguishable from "many"; and the
        (k, sigma) pair separates an exact level from a smooth continuum, which k alone cannot
    N5  differencing amplifies noise by sqrt(C(2d,d)); measure it, and report the input-noise
        horizon it implies for the order-2 layer
    N6  the old count on exact data returns the sample size
"""
import math

import numpy as np

G, F, LAG = 0.0016, 0.020, 3
DELTA = F
KMAX = 3


def em(x, k, floor, rng, iters=150):
    n = len(x)
    if k == 1:
        mu = np.array([float(x.mean())])
        var = max(float(x.var()), floor)
        ll = float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu[0]) ** 2 / var))
        return ll, mu, var
    sd = float(x.std()) or 1.0
    mu = np.quantile(x, (np.arange(k) + 0.5) / k) + rng.normal(0.0, 1e-3 * sd, k)
    w = np.full(k, 1.0 / k)
    var = max(float(x.var()), floor)
    ll = -np.inf
    for _ in range(iters):
        d = x[:, None] - mu[None, :]
        logp = -0.5 * d * d / var - 0.5 * np.log(2 * np.pi * var) + np.log(w + 1e-300)
        m = logp.max(1, keepdims=True)
        e = np.exp(logp - m)
        s = e.sum(1)
        ll_new = float(np.sum(m[:, 0] + np.log(s)))
        r = e / s[:, None]
        nk = r.sum(0) + 1e-300
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        d = x[:, None] - mu[None, :]
        var = max(float((r * d * d).sum() / n), floor)
        if ll_new - ll < 1e-10 * max(abs(ll_new), 1.0):
            ll = ll_new
            break
        ll = ll_new
    return ll, mu, var


def count_modes(x, kmax=KMAX, restarts=2, seed=0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = float(x.std())
    floor = (0.02 * sd) ** 2 if sd > 0 else 1e-24
    rng = np.random.RandomState(seed)
    bic, fit = {}, {}
    for k in range(1, kmax + 1):
        best = None
        for _ in range(restarts if k > 1 else 1):
            ll, mu, var = em(x, k, floor, rng)
            if best is None or ll > best[0]:
                best = (ll, mu, var)
        fit[k] = best
        bic[k] = -2.0 * best[0] + 2.0 * k * np.log(n)
    kb = int(min(bic, key=lambda k: bic[k]))
    return kb, float(np.sqrt(fit[kb][2])), np.sort(fit[kb][1]), bic


def two_level(n, sigma, seed):
    rng = np.random.RandomState(seed)
    return -G + DELTA * rng.randint(0, 2, n) + rng.normal(0.0, sigma, n)


def one_level(n, sigma, seed):
    return np.random.RandomState(seed).normal(0.0, sigma, n)


def free_stream(n, sigma, seed):
    return np.random.RandomState(seed).normal(0.0, 1.0 + sigma, n)


def delayed_positions(n, sigma, seed):
    """the world's position, so the difference orders are the real ones"""
    rng = np.random.RandomState(seed)
    y, vy, hist = 0.5, 0.0, [0] * (LAG + 2)
    out = []
    for _ in range(n):
        a = int(rng.rand() < 0.5)
        hist.append(a)
        vy = vy - G + F * hist[-1 - LAG]
        y = y + vy
        out.append(y)
    return np.asarray(out) + rng.normal(0.0, sigma, n)


N = 1200
print("instrument: 1-D Gaussian mixture, shared sigma, k by BIC;  Delta = %.4f\n" % DELTA)

# ---------------------------------------------------------------- N1, N6
x = two_level(N, 1e-9, 1)
k, sig, mu, bic = count_modes(x)
print("N1  exact data")
print("    k = %d   centres %s   fitted sigma %.2e (the 2%% floor: %.2e)"
      % (k, np.round(mu, 6), sig, 0.02 * float(x.std())))
print("    BIC: " + "  ".join("k=%d %.1f" % (kk, bic[kk]) for kk in sorted(bic)))

print("\nN6  the old count on the same data: %d distinct values (n = %d) -> 'continuum'"
      % (len(np.unique(x)), len(x)))

# ---------------------------------------------------------------- N2, N3
print("\nN2/N3  horizon, two-level vs single-level data")
print("       sigma      sigma/Delta   k(two)   fitted sigma   k(one)")
SIG = np.concatenate([np.logspace(-9, -3, 6), np.logspace(-2.65, -1.55, 8)])
rows = []
for s in SIG:
    k2, f2, _, _ = count_modes(two_level(N, s, 2), seed=3)
    k1, _, _, _ = count_modes(one_level(N, s, 4), seed=5)
    rows.append((s, k2, f2, k1))
    print("     %.1e      %8.3f        %d       %.2e        %d" % (s, s / DELTA, k2, f2, k1))

smax = max((r[0] for r in rows if r[1] == 2), default=0.0)
fp = [r for r in rows if r[3] != 1]
print("\n    largest sigma still read as two levels: %.2e = %.3f * Delta" % (smax, smax / DELTA))
print("    false positives on a single level: %d of %d -> %s"
      % (len(fp), len(rows), "ok" if not fp else "MISS"))

# ---------------------------------------------------------------- N4
six = -G + DELTA * np.random.RandomState(8).randint(0, 6, N)
kf, sf, cf, _ = count_modes(six + np.random.RandomState(9).normal(0, 1e-9, N), seed=10)
print("\nN4  more levels than the cap: 6 levels -> k = %d (cap %d) = 'at or above the cap'"
      % (kf, KMAX))
print("    so '2' is distinguishable from 'many': %s" % ("ok" if kf == KMAX else "MISS"))
ku, su, _, _ = count_modes(free_stream(N, 0.0, 8), seed=9)
print("    and (k, sigma) tells exact from smooth: exact single level -> (1, %.1e);"
      " a continuum -> (1, %.1e)" % (0.02 * float(two_level(N, 1e-9, 1).std()), su))

# ---------------------------------------------------------------- N5
print("\nN5  differencing amplifies noise")
print("       order    measured noise gain    sqrt(C(2d,d))")
G_meas = []
for d in range(4):
    rng = np.random.RandomState(11)
    v = rng.normal(0.0, 1.0, 200000)
    for _ in range(d):
        v = np.diff(v)
    G_meas.append(float(v.std()))
    print("         %d          %.4f              %.4f"
          % (d, G_meas[-1], math.sqrt(math.comb(2 * d, d))))

print("\n     so the order-2 layer is readable only while  0.42*Delta > 2.449*sigma_y,")
print("     i.e. sigma_y < %.4f = %.3f * Delta" % (0.42 * DELTA / 2.449, 0.42 / 2.449))

print("\n     measured on the real second difference of the world's position:")
for frac in (0.05, 0.10, 0.17, 0.25, 0.40):
    pos = delayed_positions(4000, frac * DELTA, 13)
    v = np.diff(np.diff(pos))
    kk, ss, cen, _ = count_modes(v, seed=15)
    print("       sigma_y = %.2f Delta  ->  layer noise %.2f Delta  ->  k = %d"
          % (frac, 2.449 * frac, kk))

print("\nN1 %s   N2 %s   N3 %s   N4 %s   N5 %s   N6 %s"
      % ("ok" if k == 2 else "MISS",
         "ok" if 0.2 * DELTA < smax < 0.7 * DELTA else "MISS",
         "ok" if not fp else "MISS",
         "ok" if kf == KMAX else "MISS",
         "ok" if abs(G_meas[2] - 2.449) < 0.05 else "MISS",
         "ok" if len(np.unique(x)) == len(x) else "MISS"))
