"""The rule layer inside the substrate's own loop: a sixth signal and a sixth action.

`live_core.Live` already monitors four signals (rank, held, train-held gap, spread) and has
three actions (gain, route, prune).  With `rule_specs` it gains:

    signal   residual of the holding rules, measured on the rows the task feeds in
    action   scan: test the candidate invariants, and keep the ones that hold

The two things that have to be true for this to be an integration rather than an addition:

    R1  after one scan the residual is at machine precision (the rules are exact, not fitted)
    R2  the substrate's existing readout is bit-identical with and without the rule layer

    python rule_loop.py [rows]

The task the substrate is reading is the ridge demo from live_core; the rows fed to the rule
layer come from Flappy, so the rule layer is doing real work on a real trace while the readout
does its own thing.
"""
import os
import sys

import numpy as np

for _key, _value in (("FL_G", "0.0016"), ("FL_FLAP", "0.020"),
                     ("FL_SPEED", "0.010"), ("FL_GAP", "0.34"),
                     ("FL_SPACING", "0.80")):
    os.environ.setdefault(_key, _value)

from flappy import Flappy as Env
from live_core import Live, make

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

SPECS = [
    dict(name="dy", fn=lambda r: r["y2"] - r["y"], allowed=("vy", "a"), vary=("y",)),
    dict(name="dv", fn=lambda r: r["vy2"] - r["vy"], allowed=("a",), vary=("y", "vy")),
]

rng_data = np.random.RandomState(5)
rows = []
for ep in range(500):
    env = Env(3000 + ep)
    while not env.dead and len(rows) < ROWS:
        y, vy = env.y, env.vy
        a = int(rng_data.rand() < 0.5)
        env.step(a)
        rows.append(dict(y=y, vy=vy, a=a, y2=env.y, vy2=env.vy))

rng_a = np.random.RandomState(21)
rng_b = np.random.RandomState(21)
on = Live(rng_a, monitor=True, rule_specs=SPECS, rule_budget=3)
off = Live(rng_b, monitor=True)

print("feeding %d dynamics rows while the readout runs its own task" % len(rows))
h_on, h_off = [], []
for i, row in enumerate(rows):
    tr, te = make(rng_a, [0])
    tr2, te2 = make(rng_b, [0])
    h_on.append(on.step(tr, te, row=row))
    h_off.append(off.step(tr2, te2))
    if (i + 1) % 400 == 0:
        s = on.rule_signal()
        print("  row %3d   held %d   residual %.3e   scans %d"
              % (i + 1, s["held"], s["residual"], s["scans"]))

s = on.rule_signal()
print("\nafter %d rows: %d rule(s) holding, residual %.3e, %d scan(s)"
      % (len(rows), s["held"], s["residual"], s["scans"]))
for inv in on.rules.invariants:
    print("  %s" % inv)

same = h_on == h_off
print("\nR1  residual after the scan <= 1e-12 : %s (%.2e)"
      % ("ok" if s["residual"] <= 1e-12 else "MISS", s["residual"]))
print("R2  readout bit-identical with/without the rule layer: %s (%d vs %d steps)"
      % ("ok" if same else "MISS", len(h_on), len(h_off)))
from collections import Counter
print("\nactions with the rule layer:    %s" % dict(Counter(on.actions)))
print("actions without:                %s" % dict(Counter(off.actions)))
