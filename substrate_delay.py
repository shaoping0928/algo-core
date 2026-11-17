"""Delayed quantities as candidate columns: the action that acted three frames ago.

A world where the action takes effect three steps later:

    y'  = y + vy
    vy' = vy - g + F * a(t-3)

With columns (y, vy, a) no basis can predict vy', because the current action is not what moved
the velocity.  Offering the lagged action as a candidate column does not make the search
smarter -- it gives it the variable the mechanism actually uses.  That is the point of the
substrate's three timescales: u, M and r exist so that a delayed quantity is *available*.

    python substrate_delay.py

Criterion written before the run:
    D1  with (y, vy, a) the best basis sits at a definite non-zero residual (the action term)
    D2  adding a(t-3) makes vy' exact, and the recovered coefficients are 1.0 on vy and F on
        the lagged action
    D3  the search picks a(t-3), not a(t-1) or a(t-2)
"""
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from substrate_search import greedy_determine

GRAV, FLAP, DELAY = 0.0016, 0.020, 3

rng = np.random.RandomState(5)
y, vy = 0.5, 0.0
q = deque(maxlen=DELAY + 1)       # holds a(t-3) .. a(t)
rows = []
for t in range(6000):
    a = int(rng.rand() < 0.5)
    q.append(a)
    eff = q[0] if len(q) == DELAY + 1 else 0
    nvy = vy - GRAV + (FLAP if eff else 0.0)
    ny = y + nvy
    rows.append((y, vy, a, ny, nvy,
                 q[-2] if len(q) >= 2 else 0,
                 q[-3] if len(q) >= 3 else 0,
                 q[-4] if len(q) >= 4 else 0))
    y, vy = ny, nvy
    if not (0.0 <= y <= 1.0):
        y, vy = 0.5, 0.0
        q.clear()

A = np.asarray(rows, dtype=float)
d = dict(y=A[:, 0], vy=A[:, 1], a=A[:, 2], y2=A[:, 3], vy2=A[:, 4],
         a_lag1=A[:, 5], a_lag2=A[:, 6], a_lag3=A[:, 7])
print("delayed world: vy' = vy - g + F*a(t-%d)   rows %d" % (DELAY, len(rows)))

print("\n  vy2 from (y, vy, a)          -- the current action is not what moved it")
ch, cf, tr, sc = greedy_determine(d, "vy2", ["y", "vy", "a"], nonlinear=True)
print("    residual %.2e   %s" % (tr, " ".join(ch)))

print("\n  vy2 from (y, vy, a) + lagged actions")
ch2, cf2, tr2, sc2 = greedy_determine(d, "vy2", ["y", "vy", "a"],
                                      nonlinear=True, lags=["a_lag1", "a_lag2", "a_lag3"])
terms = "%.6f" % cf2[0] + " " + " ".join("%+.6f*%s" % (c, n) for c, n in zip(cf2[1:], ch2))
print("    residual %.2e   %s" % (tr2, terms))

picked_lag = [n for n in ch2 if n.startswith("a_lag")]
print("\nD1  best basis without lags is non-exact      : %s (%.2e)"
      % ("ok" if tr > 1e-6 else "MISS", tr))
print("D2  with lags it is exact                     : %s (%.2e)"
      % ("ok" if tr2 <= 1e-12 else "MISS", tr2))
print("D3  the lag it picked is a_lag3               : %s (%s)"
      % ("ok" if picked_lag[:1] == ["a_lag3"] else "MISS", picked_lag))
