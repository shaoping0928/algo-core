"""Does it notice that its own actions are delayed -- while playing?

A.14 learned the lag from the error channel, but on a stream of *supplied* random events.  The
question here is the closed-loop one: the agent acts, the world answers three frames later, and
nothing tells it the lag.  Does the gain at the right lag grow anyway?

The prediction layer is the same one (`predict_layer.LayerPair`); the only change is where the
event stream comes from.  A closed-loop policy is a function of its state, so its actions are
not exogenous -- the worry was that a nearly periodic policy would make the lagged regressors
collinear and split the gain.  That worry is tested here rather than asserted.

    python substrate_selfdelay.py

Criteria written before the run:
    S1  the open loop identifies the lag (control, on a stream that does not die)
    S2  the closed loop identifies it too, or the report says how it fails and why: the design
        matrix's condition number, the number of informative flaps, and the survival are all
        printed so the answer can be attributed
    S3  exploration is reported as a trade: it adds exogenous variation, and on this task it
        also kills the bird early, which is a real cost and is measured
    S4  a constant action stream cannot identify it at all: gain and drift trade off, so no
        single lag is singled out
"""
import numpy as np

from latent_world import SEEDS, latent, make_policy, recover
from predict_layer import LayerPair

LAG = 3
K = 8
G, F = 0.0016, 0.020
MIN_GAIN = 0.5 * F          # a gain this large cannot be a transient


def synthetic_open(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    lp = LayerPair(k_max=K)
    acts, first = [], None
    for t in range(n):
        a = int(rng.rand() < 0.5)
        acts.append(a)
        fired = acts[t - LAG] if t >= LAG else 0
        lp.observe(-G + F * fired, float(a))
        if first is None and lp.identified(LAG, MIN_GAIN):
            first = lp.t
    return lp, np.asarray(acts, float), first, n


def closed_loop(seed, eps=0.0, rng=None):
    pol = make_policy(R, LAG, 14)
    act, target = pol
    env = latent[LAG](seed)
    lp = LayerPair(k_max=K)
    acts, first, prev = [], None, 0
    while not env.dead:
        a = act(env.y, env.vy, target(env), env.pending(), prev)
        if eps and rng.rand() < eps:
            a = 1 - a
        prev = a
        vy0 = env.vy
        env.step(a)
        lp.observe(env.vy - vy0, float(a))
        acts.append(a)
        if first is None and lp.identified(LAG, MIN_GAIN):
            first = lp.t
    return lp, np.asarray(acts, float), first, env.t


def design(acts):
    """[1, a(t-1) .. a(t-K)]: can the lags be told apart at all?"""
    n = len(acts)
    if n <= K + 2:
        return float("inf")
    rows = [np.concatenate([[1.0], acts[t - 1:t - K - 1:-1]]) for t in range(K + 1, n)]
    sv = np.linalg.svd(np.asarray(rows), compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-300))


R = recover(LAG)
print("world: every flap lands %d frames later;  rule %s" % (LAG, R["form"]))
print("predictor: 8 lags, gains start at 0, nothing says which lag is right")
print("identify = the true lag holds >%.3f and all others are below 25%% of it\n" % MIN_GAIN)

arms = [
    ("open loop (supplied random)", lambda s: synthetic_open(20000, 1000 + s)),
    ("closed loop (own policy)", lambda s: closed_loop(s)),
    ("closed loop + 2% exploration",
     lambda s: closed_loop(s, 0.02, np.random.RandomState(2000 + s))),
]
print("  arm                            identify    found    survival     flap     flaps until")
print("                                  (step)     (of 8)    (steps)     rate      identify")
res = {}
for name, fn in arms:
    out = [fn(s) for s in SEEDS[:8]]
    got = [o for o in out if o[2] is not None]
    med = int(np.median([o[2] for o in got])) if got else None
    flaps_until = np.mean([o[1][:o[2]].sum() for o in got]) if got else float("nan")
    res[name] = (med, len(got), out)
    print("  %-28s   %8s    %3d/8    %8.0f    %5.2f    %10.0f"
          % (name, med if med else "never", len(got), np.mean([o[3] for o in out]),
             np.mean([o[1].mean() for o in out]), flaps_until))

print("\n  final gain profile, lag 1..8 (one run each)")
for name, _ in arms:
    lp = res[name][1] and res[name][2][0][0]
    print("    %-28s " % name + " ".join("%+.4f" % g for _, g in lp.lag_profile()))

print("\n  the design matrix, and why the arms differ")
print("    arm                            condition number")
for name, _ in arms:
    print("    %-28s %8.1f" % (name, np.median([design(o[1]) for o in res[name][2]])))

print("\nS4  a constant action stream (a = 1, 20000 steps)")
lp = LayerPair(k_max=K)
for t in range(20000):
    lp.observe(-G + F, 1.0)
print("    drift b %+.6f   gains " % lp.b + " ".join("%+.4f" % g for _, g in lp.lag_profile()))
print("    b + sum(h) = %.6f = -g + F: the split is arbitrary, no lag is singled out"
      % (lp.b + lp.h.sum()))

m_open = res[arms[0][0]][0]
m_closed = res[arms[1][0]][0]
m_expl = res[arms[2][0]][0]
print("\nS1  open loop identifies                         : %s (step %s, %d/8)"
      % ("ok" if m_open else "MISS", m_open, res[arms[0][0]][1]))
print("S2  closed loop identifies from its own actions  : %s (step %s, %d/8)"
      % ("ok" if m_closed else "MISS", m_closed, res[arms[1][0]][1]))
print("S3  exploration: informative, but costs survival  : %s (step %s, %d/8, survival %.0f vs %.0f)"
      % ("ok" if m_expl else "died before it could", m_expl, res[arms[2][0]][1],
         np.mean([o[3] for o in res[arms[2][0]][2]]),
         np.mean([o[3] for o in res[arms[1][0]][2]])))
