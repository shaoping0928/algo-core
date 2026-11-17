"""A prediction layer, a reality layer, and the difference as the learning signal.

The previous step handed the mechanism over by hand: the drift was a median, the lag an argmax
over a candidate list, the magnitude a max of the residual.  Nothing was learned except the
numbers, and even those were computed with the answer visible.

This is the same knowledge with the answer hidden:

    prediction layer   emits  d_hat(t) = b + sum_k h_k * e(t-k)      before seeing d(t)
    reality layer      supplies d(t), the observed change
    the difference     err = d(t) - d_hat(t)                          trains b and h, online

    h starts at zero.  Nothing says which lag is the right one, or how big the consequence is.
    A bank of gains, one per lag, all live at once, competes; the one that reduces the error is
    the one that grows.  The lag is then read out as the non-zero gain, not supplied.

What is supplied is capacity only (how many lags the bank can hold), never knowledge.

Criteria written before the run:
    P1  after learning, exactly one gain is non-zero and it sits at the true lag
    P2  its value matches the consequence size to 1% (no magnitude was supplied)
    P3  the drift converges to the true drift to 1%, and the unexplained error falls from the
        drift-only level to the learning-noise level
    P4  on a decoy world where the event has NO delayed consequence, no gain grows: the layer
        does not invent an association
    P5  with the error channel closed (no learning signal) nothing is learned -- which is what
        makes the difference the source rather than a by-product
"""
import numpy as np

G, F, LAG = 0.0016, 0.020, 3
K = 8                     # capacity: how many lags the bank can hold (not knowledge)
T = 200000


from predict_layer import LayerPair  # noqa: E402  (the layer lives in one place)


def run_world(delta_fn, event_fn, learn=True, eta0=0.05):
    lp = LayerPair(learn=learn, eta0=eta0)
    rng = np.random.RandomState(11)
    trace = []
    for t in range(T):
        ev = event_fn(rng, t)
        d = delta_fn(ev)                    # the world: delta depends on events in the past
        lp.observe(d, ev)
        if t in (1000, 10000, 100000, T - 1):
            trace.append((t, lp.h.copy(), lp.b, lp.err))
    return lp, trace


def print_learned(name, lp, trace, truth_lag, truth_F):
    print("\n%s" % name)
    print("  step      argmax|h|   h[argmax]   |h| max elsewhere   drift b        mean |err|")
    for t, h, b, err in trace:
        j = int(np.argmax(np.abs(h)))
        rest = float(np.max(np.delete(np.abs(h), j))) if len(h) > 1 else 0.0
        print("  %6d       %d        %+9.6f      %9.2e      %+9.6f     %.2e"
              % (t, j + 1, h[j], rest, b, err))
    j = int(np.argmax(np.abs(lp.h)))
    rest = float(np.max(np.delete(np.abs(lp.h), j)))
    off = ("%.2f%%" % (100 * abs(lp.h[j] - truth_F) / truth_F)) if truth_F else "n/a"
    print("  final: lag %d (truth %d), gain %+.6f (truth %.6f, %s off), "
          "second largest |h| %.1e, b %+.6f (truth %+.6f)"
          % (j + 1, truth_lag, lp.h[j], truth_F, off,
             rest, lp.b, -G))
    return j, lp.h[j], rest, lp.b


# ---------------------------------------------------------------- the delayed-action world
def mk_delayed(lag, mag):
    buf = []

    def event_fn(rng, t):
        buf.append(float(rng.rand() < 0.5))
        return buf[-1]

    def delta_fn(ev):
        fired = buf[-1 - lag] if len(buf) > lag else 0.0
        return -G + mag * fired

    return delta_fn, event_fn


d_fn, e_fn = mk_delayed(LAG, F)
lp1, tr1 = run_world(d_fn, e_fn)
j1, g1, rest1, b1 = print_learned("learned on the delayed world (nothing supplied but capacity)",
                                  lp1, tr1, LAG, F)

# ---------------------------------------------------------------- the external-cue world
def mk_cue(lag, mag):
    buf = []

    def event_fn(rng, t):
        buf.append(float(rng.rand() < 0.35))
        return buf[-1]

    def delta_fn(ev):
        fired = buf[-1 - lag] if len(buf) > lag else 0.0
        return -G + mag * fired

    return delta_fn, event_fn


d_fn, e_fn = mk_cue(LAG, F)
lp2, tr2 = run_world(d_fn, e_fn)
j2, g2, rest2, b2 = print_learned("learned on a world where the cue is external and rarer",
                                  lp2, tr2, LAG, F)

# ---------------------------------------------------------------- decoy: no consequence
d_fn, e_fn = mk_delayed(LAG, 0.0)
lp3, tr3 = run_world(d_fn, e_fn)
j3, g3, rest3, b3 = print_learned("decoy: the event has no delayed consequence",
                                  lp3, tr3, LAG, 0.0)

# ---------------------------------------------------------------- error channel closed
d_fn, e_fn = mk_delayed(LAG, F)
lp4, tr4 = run_world(d_fn, e_fn, learn=False)
print("\nerror channel closed (the difference is never used)")
print("  |h| max %.2e   drift b %+.6f   mean |err| %.2e   (drift-only level is about F/2)"
      % (float(np.max(np.abs(lp4.h))), lp4.b, lp4.err))

print("\nP1  one gain, at the true lag (3)            : %s (argmax %s, second |h| %.1e)"
      % ("ok" if j1 + 1 == LAG and rest1 < 1e-3 else "MISS", j1 + 1, rest1))
print("P2  gain matches the consequence to 1%%       : %s (%.6f vs %.6f)"
      % ("ok" if abs(g1 - F) / F < 0.01 else "MISS", g1, F))
print("P3  drift and unexplained error              : %s (b %+.6f vs %+.6f, err %.1e)"
      % ("ok" if abs(b1 + G) < 0.01 * G and lp1.err < 1e-3 else "MISS", b1, -G, lp1.err))
print("P4  the decoy grows no gain                  : %s (|h| max %.1e)"
      % ("ok" if float(np.max(np.abs(lp3.h))) < 1e-3 else "MISS", float(np.max(np.abs(lp3.h)))))
print("P5  with the channel closed nothing is learned: %s (|h| max %.1e, err %.2e)"
      % ("ok" if float(np.max(np.abs(lp4.h))) < 1e-9 and lp4.err > 1e-3 else "MISS",
         float(np.max(np.abs(lp4.h))), lp4.err))
print("    (the same world, learning on, ends at mean |err| %.1e -- about %.0fx smaller)"
      % (lp1.err, lp4.err / max(lp1.err, 1e-12)))


# ---------------------------------------------------------------- the payoff: learn, then act
from latent_world import SEEDS, Schedule, latent, make_policy  # noqa: E402

r_learned = dict(G0=lp1.b, CVY=1.0, F=float(lp1.h[j1]))   # everything here came from the error
pol_l = make_policy(r_learned, j1 + 1, 14)
scores = []
for s in SEEDS:
    env = latent[j1 + 1](s)
    act, target = pol_l
    sch = Schedule(j1 + 1, r_learned["F"])
    t, prev = 0, 0
    while not env.dead:
        sch.drop_before(t)
        a = act(env.y, env.vy, target(env), sch.upcoming(t), prev)
        sch.issue(t, a)
        prev = a
        env.step(a)
        t += 1
    scores.append(env.t)

print("\npayoff: the closed loop driven only by what was learned")
print("  learned lag %d, learned gain %.9f, learned drift %.9f"
      % (j1 + 1, r_learned["F"], lp1.b))
print("  score %.1f   (hand-supplied truth 4886.9, no memory 16.5)" % np.mean(scores))
