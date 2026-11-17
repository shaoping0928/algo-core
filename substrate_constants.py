"""An atomic operator that proposes features instead of testing the ones I name.

The rule layer (substrate_rules) tests candidates the caller supplies.  That is still me
choosing the feature.  This operator does the choosing: it searches linear combinations and
reports the ones that are *determined* -- i.e. a variable that is an exact linear function of
the others, or a combination that is constant.

    names = ["y2", "y", "vy", "a"]
    find_constants(rows, names)
    ->  y2 = y + vy - 0.0016 + 0.0200*a      residual 1e-16

The search is over subsets, each variable in turn taken as the target and the rest as
predictors, with a constant term.  A hit is a rule: the target does not depend on anything
outside the ones it used, and the coefficients are what the mechanism is.

    python substrate_constants.py

Criterion: it must recover both transitions with no help beyond the column names, and must
report no spurious combination on a shuffled version of the same data.
"""
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
g = {"__name__": "w", "__file__": os.path.join(HERE, "flappy.py")}
src = open(os.path.join(HERE, "flappy.py"), encoding="utf-8").read().split("t0 = time.time()")[0]
old = sys.argv
sys.argv = ["x", "1"]
try:
    exec(compile(src, "flappy.py", "exec"), g)
finally:
    sys.argv = old

Env = g["Flappy"]


def find_constants(rows, names, max_predictors=4, tol=1e-9):
    """For each target, find the smallest set of predictors that determines it linearly.

    Returns a list of dicts: {target, predictors, coeffs, residual, scale}.
    A result counts when the largest residual is below tol * scale.
    """
    found = []
    data = {n: np.array([float(r[n]) for r in rows]) for n in names}
    for target in names:
        t = data[target]
        scale = max(1.0, float(np.abs(t).max()))
        others = [n for n in names if n != target]
        for k in range(1, min(len(others), max_predictors) + 1):
            best = None
            for combo in combinations(others, k):
                X = np.stack([np.ones(len(t))] + [data[n] for n in combo], 1)
                c = np.linalg.lstsq(X, t, rcond=None)[0]
                resid = float(np.abs(X @ c - t).max())
                if resid <= tol * scale and (best is None or resid < best[2]):
                    best = (combo, c, resid)
            if best is not None:
                found.append(dict(target=target, predictors=best[0], coeffs=best[1],
                                  residual=best[2], scale=scale))
                break
    return found


rng = np.random.RandomState(5)
rows = []
for ep in range(300):
    env = Env(3000 + ep)
    while not env.dead:
        y, vy = env.y, env.vy
        a = int(rng.rand() < 0.5)
        env.step(a)
        rows.append(dict(y=y, vy=vy, a=a, y2=env.y, vy2=env.vy))
print("rows: %d" % len(rows))

print("\nsearch over (y2, y, vy, a)")
for r in find_constants(rows, ["y2", "y", "vy", "a"]):
    terms = "%.6f" % r["coeffs"][0]
    for n, c in zip(r["predictors"], r["coeffs"][1:]):
        terms += " %+.6f*%s" % (c, n)
    print("  %-3s = %-46s residual %.1e" % (r["target"], terms, r["residual"]))

print("\nsearch over (vy2, vy, a)")
for r in find_constants(rows, ["vy2", "vy", "a"]):
    terms = "%.6f" % r["coeffs"][0]
    for n, c in zip(r["predictors"], r["coeffs"][1:]):
        terms += " %+.6f*%s" % (c, n)
    print("  %-3s = %-46s residual %.1e" % (r["target"], terms, r["residual"]))

print("\ncontrol: the same columns with y2 shuffled (no dependence should survive)")
shuffled = [dict(r) for r in rows]
ys = [r["y2"] for r in shuffled]
rng2 = np.random.RandomState(7)
rng2.shuffle(ys)
for r, v in zip(shuffled, ys):
    r["y2"] = v
hits = find_constants(shuffled, ["y2", "y", "vy", "a"])
print("  %d combination(s) found (must be 0 or must not involve y2)" % len(hits))
for r in hits:
    print("    %-3s <- %s  residual %.1e"
          % (r["target"], ",".join(r["predictors"]), r["residual"]))
