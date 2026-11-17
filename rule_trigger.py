"""Instrument for the rule layer's three trigger settings.

The rule layer's trigger has three numbers that were set by hand: how many rows before a scan
may fire, what residual counts as "not explained yet", and how many scans are allowed.  This
measures what they should be, the same way §85-§99 measured the memory instruments.

Two quantities matter:

    the resolution of the invariance test   at what row count does a true invariant start to
                                            hold, and does a false one ever hold
    the cost of a trigger setting           scans used, scans wasted (a scan that finds
                                            nothing), and rows until the first success

    python rule_trigger.py

The stream is fixed (3000 rows of Flappy transitions) so every setting sees exactly the same
data; the only thing that changes is when the layer is allowed to look.
"""
import os
import sys

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

from substrate_rules import RuleBook

Env = g["Flappy"]
NROWS = 3000
rng = np.random.RandomState(5)
rows = []
for ep in range(800):
    env = Env(3000 + ep)
    while not env.dead and len(rows) < NROWS:
        y, vy = env.y, env.vy
        a = int(rng.rand() < 0.5)
        env.step(a)
        rows.append(dict(y=y, vy=vy, a=a, y2=env.y, vy2=env.vy))
print("stream: %d rows" % len(rows))


def book_for(subset):
    b = RuleBook()
    b.add_invariance("dy", lambda r: r["y2"] - r["y"], ["vy", "a"], ["y"])
    b.add_invariance("dv", lambda r: r["vy2"] - r["vy"], ["a"], ["y", "vy"])
    b.add_invariance("dy_no_vy", lambda r: r["y2"] - r["y"], ["a"], ["y"])   # false candidate
    for r in subset:
        b.observe(r)
    b.scan()
    return b


print("\n  resolution of the invariance test (same stream, growing prefix)")
print("    rows    dy holds   ratio      dv holds   ratio      false candidate holds")
first = None
for n in (50, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000):
    b = book_for(rows[:n])
    inv = {i.name: i for i in b.invariants}
    if first is None and inv["dy"].holds and inv["dv"].holds:
        first = n
    print("    %4d      %-8s  %.1e   %-8s  %.1e   %s"
          % (n, inv["dy"].holds, inv["dy"].ratio, inv["dv"].holds, inv["dv"].ratio,
             inv["dy_no_vy"].holds))
print("\n  first row count at which both true invariants hold: %s" % first)
print("  the false candidate never holds: %s"
      % all(not book_for(rows[:n]).invariants[2].holds
            for n in (400, 800, 1200, 2000, 3000)))

print("\n  cost of a trigger setting (scan on the same stream, 5 check points)")
print("    rule_min  budget   scans   wasted   rows_to_first_hold   held_at_end")
for rmin in (200, 400, 800, 1200, 2000):
    for budget in (1, 3):
        scans = wasted = 0
        hold_at = None
        b = None
        for i in range(len(rows)):
            if i + 1 < rmin:
                continue
            if (i + 1) % rmin == 0 and scans < budget:
                scans += 1
                b = book_for(rows[:i + 1])
                held = len([x for x in b.invariants if x.holds])
                if held >= 2:
                    if hold_at is None:
                        hold_at = i + 1
                else:
                    wasted += 1
        held_end = len([x for x in b.invariants if x.holds]) if b else 0
        print("    %7d  %6d   %5d   %6d   %-18s   %d"
              % (rmin, budget, min(scans, budget), wasted,
                 str(hold_at) if hold_at else "never", held_end))
