"""Does the substrate's memory make an ambiguous observation determinative?

The claim under test: memory is a decaying trace that, on re-access, resolves something the
present input cannot.  The world is built so that only that function can help:

    t = 0        a cue arrives, one of two modes
    t = 1..k     the observation is the same constant for both modes
    outcome      determined by the mode

So no test on the present observation can predict the outcome: the observation carries zero
information about it.  A decaying trace of the cue does, for as long as it lasts.

    python substrate_recognition.py

Criteria written before the run:
    R1  the outcome is NOT determined by the present observation alone
    R2  it IS determined once the recalled mode is in the key
    R3  recall accuracy holds to the trace's time constant and falls afterwards, which is what
        makes this a capacity and not a coincidence
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from substrate_rules import RuleBook

BETA = float(os.environ.get("REC_BETA", "0.05"))     # node trace time constant ~ 1/beta


def episode(mode, k, rng):
    """one cue, then k ambiguous steps; returns the row seen at the last step"""
    u = np.zeros(2)
    cue = np.zeros(2)
    cue[mode] = 1.0
    u = (1.0 - BETA) * u + BETA * cue
    for _ in range(k):
        u = (1.0 - BETA) * u + BETA * np.array([0.5, 0.5])   # same for both modes
    recalled = 1 if u[1] > u[0] else 0
    return dict(o0=0.5, o1=0.5, recalled=float(recalled),
                mode=float(mode), dead=float(mode))


rng = np.random.RandomState(5)
K = int(os.environ.get("REC_K", "10"))

print("ambiguous world: one cue at t=0, then %d steps identical for both modes" % K)
print("node trace beta %.2f -> time constant %.0f steps" % (BETA, 1.0 / BETA))
rows = [episode(int(rng.randint(2)), K, rng) for _ in range(3000)]

book = RuleBook()
book.add_determinacy("dead_by_obs", lambda r: r["dead"], ["o0", "o1"])
book.add_determinacy("dead_by_obs_mem", lambda r: r["dead"], ["o0", "o1", "recalled"])
for r in rows:
    book.observe(r)
book.scan()
for d in book.determinants:
    print("  %s" % d)

print("\n  recall accuracy as the silent gap grows")
print("    k     accuracy    trace magnitude")
for k in (1, 2, 4, 8, 16, 32, 64):
    hit, mag = 0, 0.0
    n = 400
    r2 = np.random.RandomState(11)
    for _ in range(n):
        m = int(r2.randint(2))
        u = np.zeros(2)
        cue = np.zeros(2)
        cue[m] = 1.0
        u = (1.0 - BETA) * u + BETA * cue
        for _ in range(k):
            u = (1.0 - BETA) * u + BETA * np.array([0.5, 0.5])
        hit += int((1 if u[1] > u[0] else 0) == m)
        mag += float(np.abs(u[1] - u[0]))
    print("    %2d    %.3f       %.4f" % (k, hit / n, mag / n))

print("\n  capacity is not accuracy: the sign of a trace survives, its magnitude does not")
print("    sigma        k=4    k=8   k=16   k=32   k=64")
for sig in (0.0, 1e-4, 1e-3, 1e-2):
    row = []
    for k in (4, 8, 16, 32, 64):
        hit = 0
        n = 300
        r3 = np.random.RandomState(13)
        for _ in range(n):
            m = int(r3.randint(2))
            u = np.zeros(2)
            cue = np.zeros(2)
            cue[m] = 1.0
            u = (1.0 - BETA) * u + BETA * cue
            for _ in range(k):
                u = (1.0 - BETA) * u + BETA * np.array([0.5, 0.5])
                if sig:
                    u = u + r3.normal(0, sig, size=2)
            hit += int((1 if u[1] > u[0] else 0) == m)
        row.append(hit / n)
    print("    %.0e   " % sig + " ".join("%5.2f" % x for x in row))

d_obs = next(d for d in book.determinants if d.name == "dead_by_obs")
d_mem = next(d for d in book.determinants if d.name == "dead_by_obs_mem")
print("\nR1  outcome not determined by the present observation: %s (spread %.3f)"
      % ("ok" if not d_obs.exact else "MISS", d_obs.spread))
print("R2  determined once the recalled mode is in the key: %s (spread %.1e)"
      % ("ok" if d_mem.exact else "MISS", d_mem.spread))
