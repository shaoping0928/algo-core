"""Can the substrate's own leaky state carry a delay, and how many time constants does it take?

The delayed world is the same as A.8:  vy' = vy - g + F*a(t-3).  There, the lagged action was
handed to the search as a column.  Here the search gets only leaky averages of the action --

    u_k(t) = (1 - beta_k) u_k(t-1) + beta_k a(t)

which is the substrate's u state, one per time constant -- and we measure what that is worth.

    python substrate_timescale.py

Criterion written before the run:
    T1  one leaky average is not enough: residual stays well above the exact level
    T2  a bank of averages does better, and the improvement is monotone in the bank size
    T3  neither reaches the exact reference (which uses the true shift register, a(t-3))
    report the best achievable with a bank, so the gap is a number and not an opinion
"""
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from substrate_search import greedy_determine

GRAV, FLAP, DELAY = 0.0016, 0.020, 3
BETAS = (0.5, 0.2, 0.1, 0.05, 0.02)

rng = np.random.RandomState(5)
y, vy = 0.5, 0.0
q = deque(maxlen=DELAY + 1)
u = np.zeros(len(BETAS))
rows = []
for t in range(6000):
    a = int(rng.rand() < 0.5)
    q.append(a)
    u = (1.0 - np.asarray(BETAS)) * u + np.asarray(BETAS) * a
    eff = q[0] if len(q) == DELAY + 1 else 0
    nvy = vy - GRAV + (FLAP if eff else 0.0)
    ny = y + nvy
    rows.append(tuple([y, vy, a, ny, nvy] + list(u) +
                      [q[-2] if len(q) >= 2 else 0,
                       q[-3] if len(q) >= 3 else 0,
                       q[-4] if len(q) >= 4 else 0]))
    y, vy = ny, nvy
    if not (0.0 <= y <= 1.0):
        y, vy = 0.5, 0.0
        q.clear()
        u[:] = 0.0

A = np.asarray(rows, dtype=float)
d = dict(y=A[:, 0], vy=A[:, 1], a=A[:, 2], y2=A[:, 3], vy2=A[:, 4])
for i, b in enumerate(BETAS):
    d["u%s" % ("%g" % b).replace(".", "")] = A[:, 5 + i]
d["a_lag1"] = A[:, 5 + len(BETAS)]
d["a_lag2"] = A[:, 6 + len(BETAS)]
d["a_lag3"] = A[:, 7 + len(BETAS)]
UNAMES = [k for k in d if k.startswith("u")]

print("delayed world vy' = vy - g + F*a(t-3);  rows %d;  betas %s" % (len(rows), BETAS))
print("\n  what the search can do with each state source")
print("    source                                residual      picked")
resp = {}
for label, lags, nl in (
    ("nothing (y, vy, a)", [], True),
    ("one leaky average u(0.2)", [UNAMES[1]], True),
    ("bank of 2", UNAMES[:2], True),
    ("bank of 3", UNAMES[:3], True),
    ("bank of 4", UNAMES[:4], True),
    ("bank of 5", UNAMES[:5], True),
    ("true shift register a(t-3)", ["a_lag1", "a_lag2", "a_lag3"], False),
):
    ch, cf, tr, sc = greedy_determine(d, "vy2", ["y", "vy", "a"], nonlinear=nl, lags=lags)
    resp[label] = tr
    print("    %-36s  %.2e    %s" % (label, tr, " ".join(ch[:4])))

# a fair test of the span: fit all the bank's members jointly, not one at a time
print("\n  joint linear fit over the whole bank (does the span contain the delay?)")
t = d["vy2"]
cut = int(0.7 * len(t))
for k in (1, 2, 3, 4, 5):
    X = np.stack([np.ones(len(t)), d["vy"]] + [d[u] for u in UNAMES[:k]], 1)
    cf = np.linalg.lstsq(X[:cut], t[:cut], rcond=None)[0]
    tr = float(np.abs(X[cut:] @ cf - t[cut:]).max())
    print("    bank of %d  residual %.2e" % (k, tr))

one = resp["one leaky average u(0.2)"]
none = resp["nothing (y, vy, a)"]
bank = resp["bank of 5"]
exact = resp["true shift register a(t-3)"]
print("\nT1  one average is not enough (residual %.2e vs exact %.2e): %s"
      % (one, exact, "ok" if one > 1e-6 else "MISS"))
print("T2  a bank helps, monotone in size: %s"
      % ("ok" if resp["bank of 2"] < none and bank < resp["bank of 3"] else "check"))
print("T3  no bank reaches the shift register: %s (best bank %.2e vs exact %.2e)"
      % ("ok" if bank > 1e-9 else "MISS", bank, exact))
