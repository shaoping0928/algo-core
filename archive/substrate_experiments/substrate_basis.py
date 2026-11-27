"""More basic operators: a basis, and a greedy search over it.

The linear search (substrate_constants) only proposes combinations of the raw columns, so it
can never see a product or an absolute value.  This extends the candidate set to a basis --

    v,  v^2,  |v|,  sign(v),  v*w,  v*|w|      for every primitive v and pair (v, w)

-- and replaces exhaustive subsets with forward selection on a split of the data, so the search
stays cheap and is validated rather than fitted.

Two worlds, same columns:

    linear      vy' - vy = -g + F*a
    quadratic   vy' - vy = -g + F*a - c*vy*|vy|

Criterion written before the run:
    B1  on the linear world the linear basis suffices, as before (1e-16)
    B2  on the quadratic world the linear basis CANNOT hold, and the extended basis can
    B3  the recovered coefficient c matches the drag that was set, to 3 significant figures
"""
import os
from itertools import combinations

import numpy as np

for _key, _value in (("FL_G", "0.0016"), ("FL_FLAP", "0.020"),
                     ("FL_SPEED", "0.010"), ("FL_GAP", "0.34"),
                     ("FL_SPACING", "0.80")):
    os.environ.setdefault(_key, _value)

from flappy import Flappy as Env
DRAG = float(os.environ.get("FL_DRAG", "0"))


from substrate_search import basis, greedy_determine as greedy


rng = np.random.RandomState(5)
rows = []
for ep in range(300):
    env = Env(3000 + ep)
    while not env.dead:
        y, vy = env.y, env.vy
        a = int(rng.rand() < 0.5)
        env.step(a)
        rows.append((y, vy, a, env.y, env.vy))
A = np.asarray(rows)
d = dict(y=A[:, 0], vy=A[:, 1], a=A[:, 2], y2=A[:, 3], vy2=A[:, 4])

print("world: FL_DRAG = %.1f   rows %d" % (DRAG, len(rows)))
print("\n  predicting vy2 from (vy, a, y)   [the drag term lives here]")
for nl, label in ((False, "linear basis"), (True, "extended basis")):
    ch, cf, tr, sc = greedy(d, "vy2", ["vy", "a", "y"], nonlinear=nl, k_max=4)
    terms = "%.6f" % cf[0] + " " + " ".join("%+.6f*%s" % (c, n) for c, n in zip(cf[1:], ch))
    print("    %-15s val residual %.2e   %s" % (label, tr, terms))

print("\n  predicting y2 from (y, vy, a)")
for nl, label in ((False, "linear basis"), (True, "extended basis")):
    ch, cf, tr, sc = greedy(d, "y2", ["y", "vy", "a"], nonlinear=nl, k_max=4)
    terms = "%.6f" % cf[0] + " " + " ".join("%+.6f*%s" % (c, n) for c, n in zip(cf[1:], ch))
    print("    %-15s val residual %.2e   %s" % (label, tr, terms))

print("\nB1/B2/B3  (both arms k_max=4, same split)")
ch_l, cf_l, tr_l, _ = greedy(d, "vy2", ["vy", "a", "y"], nonlinear=False, k_max=4)
ch_e, cf_e, tr_e, _ = greedy(d, "vy2", ["vy", "a", "y"], nonlinear=True, k_max=4)
print("  linear basis    validation residual %.2e" % tr_l)
print("  extended basis  validation residual %.2e" % tr_e)
drag_term = [(n, c) for n, c in zip(ch_e, cf_e[1:]) if "|" in n]
if drag_term:
    print("  recovered drag term: %.6f * %s   (FL_DRAG was %.1f)"
          % (drag_term[0][1], drag_term[0][0], DRAG))
