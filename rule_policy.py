"""From the recovered rules to a policy, with no hand-set coefficients.

The rule layer recovers, from band-limited data and with no form supplied:

    dy = y' - y = c0 + c1*vy + c2*a
    dv = vy' - vy = d0 + d1*a

Everything the planner does is then a computation on those six numbers.  Roll the rules
forward H steps (the action on the first step, gliding after), compare the two end heights to
the gap centre, take the closer one.  Because the roll-out is affine, the comparison collapses
to a linear threshold -- and the threshold's coefficients are whatever the rules say they are.

    python rule_policy.py

Reports the recovered coefficients, the implied threshold, and the score of the derived
policy, against the hand-written rule (dy - 1.2*vy10 > 0.1 -> 1200) and the planner it came
from (1133 with future-pipe targeting, 783 with current-pipe targeting).

Criterion written before the run:
    P1  the derived policy reaches >= 1100 with future-pipe targeting
    P2  the implied anticipation coefficient comes out of the rules at 1.2 +- 0.05, i.e. it
        was never typed in
"""
import os
import numpy as np

for _key, _value in (("FL_G", "0.0016"), ("FL_FLAP", "0.020"),
                     ("FL_SPEED", "0.010"), ("FL_GAP", "0.34"),
                     ("FL_SPACING", "0.80")):
    os.environ.setdefault(_key, _value)

from flappy import BIRD_X, FLAP, GRAVITY, SPEED, Flappy as Env
from substrate_rules import RuleBook

GRAV = GRAVITY
H = 12
LO, HI = 0.35, 0.65

# ---------------------------------------------------------------- recover the rules
rng = np.random.RandomState(5)
book = RuleBook()
book.add_invariance("dy", lambda r: r["y2"] - r["y"], ["vy", "a"], ["y"])
book.add_invariance("dv", lambda r: r["vy2"] - r["vy"], ["a"], ["y", "vy"])
for ep in range(600):
    env = Env(3000 + ep)
    while not env.dead:
        y, vy = env.y, env.vy
        a = int(rng.rand() < 0.5)
        env.step(a)
        row = dict(y=y, vy=vy, a=a, y2=env.y, vy2=env.vy)
        if LO <= y <= HI:
            book.observe(row)
book.scan()
dy_inv = next(i for i in book.invariants if i.name == "dy")
dv_inv = next(i for i in book.invariants if i.name == "dv")
assert dy_inv.holds and dy_inv.coeffs is not None, "the dy invariant did not hold"
assert dv_inv.holds and dv_inv.coeffs is not None, "the dv invariant did not hold"

C0, CVY, CA = (float(x) for x in dy_inv.coeffs)
D0, DA = (float(x) for x in dv_inv.coeffs)
print("rules recovered from %d band-limited rows" % len(book))
print("  dy = %.6f %+.6f*vy %+.6f*a" % (C0, CVY, CA))
print("  dv = %.6f %+.6f*a" % (D0, DA))


def step(y, vy, a):
    return y + C0 + CVY * vy + CA * a, vy + D0 + DA * a


def rollout(y, vy, a_first, h=H):
    y, vy = step(y, vy, a_first)
    for _ in range(h - 1):
        y, vy = step(y, vy, 0)
    return y


def derive_threshold():
    """the two roll-outs differ by a constant; the comparison is a line in (y, vy, gc)"""
    y0, v0 = 0.5, 0.0
    e0 = rollout(y0, v0, 0)
    e1 = rollout(y0, v0, 1)
    # flap iff e0 is below the midpoint of the two ends
    # e0(y, vy) = y + a*vy + b  ->  flap iff y + a*vy + b < gc - (e1 - e0)/2
    a = (rollout(0.5, 1.0, 0) - rollout(0.5, 0.0, 0))      # d e0 / d vy
    b = e0 - a * v0 - y0
    # flap  <=>  e0 < gc - (e1 - e0)/2  <=>  dy - a*vy > b + (e1 - e0)/2
    return a, b + (e1 - e0) / 2.0


SLOPE, OFFSET = derive_threshold()
print("\nimplied threshold from the rules alone:  flap iff (gc - y) + %.6f*vy > %.6f"
      % (-SLOPE, OFFSET))
print("  in the observation's units (obs[1] = vy*10):  dy %+.6f*vy10 > %.6f"
      % (-SLOPE * 0.1, OFFSET))


def policy(env, target):
    if target == "current":
        gc = env.pipes[0][1]
    else:
        best, gc = 1e9, env.pipes[0][1]
        for px, gy in env.pipes:
            d = abs(px - BIRD_X - H * SPEED)
            if d < best:
                best, gc = d, gy
    e0, e1 = rollout(env.y, env.vy, 0), rollout(env.y, env.vy, 1)
    return int(abs(e1 - gc) < abs(e0 - gc))


def score(target, seeds=range(100, 120)):
    out = []
    for s in seeds:
        env = Env(s)
        while not env.dead:
            env.step(policy(env, target))
        out.append(env.t)
    return float(np.mean(out))


def hand_rule(seeds=range(100, 120)):
    out = []
    for s in seeds:
        env = Env(s)
        while not env.dead:
            o = env.obs()
            env.step(int(o[3] - 1.2 * o[1] > 0.1))
        out.append(env.t)
    return float(np.mean(out))


print("\n  policy                                        frames")
print("  derived from the rules, current-pipe target   %.1f" % score("current"))
print("  derived from the rules, future-pipe target    %.1f" % score("future"))
print("  hand-written rule (1.2 and 0.1 typed in)      %.1f" % hand_rule())

ant = abs(-SLOPE * 0.1)
print("\nP1  derived policy >= 1100 with future-pipe targeting: %s (%.1f)"
      % ("ok" if score("future") >= 1100 else "MISS", score("future")))
print("P2  implied anticipation coefficient |%.4f| (must be 1.2 +- 0.05): %s"
      % (ant, "ok" if abs(ant - 1.2) <= 0.05 else "MISS"))
