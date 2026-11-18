"""The rule layer on the substrate's own trace: rules recovered, controls rejected.

Two books, two questions, both from `substrate_rules.RuleBook`:

    invariance book   trained on band-limited data (y in [0.35, 0.65]) so that the
                      extrapolation test later asks about heights and velocities never seen
    determinacy book  trained on all data, because deaths do not occur inside the band
                      (the band is where the bird is safe, which is why it is the band)

    invariant   dy = y' - y     allowed (vy, a)      vary (y)
    invariant   dv = vy' - vy   allowed (a)          vary (y, vy)
    control     dy              allowed (a)          vary (y)        must be rejected
    determinant dead            by (near, outside)                  must be exact
    determinant dead            by (outside)                        must be rejected
    determinant dead            by (oob)                            must be exact

    python rule_mainline.py
"""
import os
import numpy as np

for _key, _value in (("FL_G", "0.0016"), ("FL_FLAP", "0.020"),
                     ("FL_SPEED", "0.010"), ("FL_GAP", "0.34"),
                     ("FL_SPACING", "0.80")):
    os.environ.setdefault(_key, _value)

from flappy import BIRD_X, FLAP, GAP, GRAVITY, Flappy as Env
from substrate_rules import RuleBook

GRAV, WIN = GRAVITY, 0.06
LO, HI = 0.35, 0.65

rng = np.random.RandomState(5)
all_rows, band_rows = [], []
for ep in range(400):
    env = Env(2000 + ep)
    while not env.dead:
        y, vy = env.y, env.vy
        a = int(rng.rand() < 0.5)
        env.step(a)
        p = env.pipes[0]
        row = dict(y=y, vy=vy, a=a, y2=env.y, vy2=env.vy,
                   near=1.0 if abs(p[0] - BIRD_X) < WIN else 0.0,
                   outside=1.0 if abs(p[1] - env.y) > GAP / 2 else 0.0,
                   oob=1.0 if (env.y < 0.0 or env.y > 1.0) else 0.0,
                   dead=1.0 if env.dead else 0.0)
        all_rows.append(row)
        if LO <= y <= HI:
            band_rows.append(row)

inv = RuleBook()
inv.add_invariance("dy", lambda r: r["y2"] - r["y"], ["vy", "a"], ["y"])
inv.add_invariance("dv", lambda r: r["vy2"] - r["vy"], ["a"], ["y", "vy"])
inv.add_invariance("dy_no_vy", lambda r: r["y2"] - r["y"], ["a"], ["y"])
for r in band_rows:
    inv.observe(r)

det = RuleBook()
det.add_determinacy("dead_pipe", lambda r: r["dead"], ["near", "outside"])
det.add_determinacy("dead_band", lambda r: r["dead"], ["outside"])
det.add_determinacy("dead_world", lambda r: r["dead"], ["oob"])
det.add_determinacy("dead_full", lambda r: r["dead"], ["near", "outside", "oob"])
for r in all_rows:
    det.observe(r)

inv.scan()
det.scan()
print("trace: %d rows, %d of them inside the band y in [%.2f, %.2f]"
      % (len(all_rows), len(band_rows), LO, HI))
print("\ninvariance nodes (band-limited training)")
print("\n".join(str(i) for i in inv.invariants))
print("\ndeterminacy nodes (all data)")
print("\n".join(str(d) for d in det.determinants))

dy_inv = next(i for i in inv.invariants if i.name == "dy")
print("\nreading the rule off the invariant and extrapolating")
print("  %s" % dy_inv)
worst = 0.0
for yy in (0.05, 0.20, 0.80, 0.95):
    for vv in (-0.04, 0.0, 0.04):
        for aa in (0, 1):
            true_y = yy + vv - GRAV + (FLAP if aa else 0.0)
            pred_y = yy + dy_inv.predict({"vy": vv, "a": aa})
            worst = max(worst, abs(pred_y - true_y))
print("  worst extrapolation error at unseen y and vy: %.2e" % worst)

bad = next(i for i in inv.invariants if i.name == "dy_no_vy")
dead_band = next(d for d in det.determinants if d.name == "dead_band")
print("\ncontrols")
print("  dy without vy among its allowed variables: holds=%s (ratio %.3f)   must be False"
      % (bad.holds, bad.ratio))
print("  death determined by 'outside the band' alone: exact=%s (spread %.3f)  must be False"
      % (dead_band.exact, dead_band.spread))
print("\nsummary: %d invariant(s) hold, %d determinant(s) exact, %d/2 controls rejected"
      % (len(inv.holding()), len(det.exact_determinants()),
         int(not bad.holds) + int(not dead_band.exact)))
