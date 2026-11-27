"""What the infrastructure owes the learner: a sufficient state.

We do not write policies here.  The question is what the state handed to a learner must contain
for any policy to be learnable at all -- i.e. whether the state is sufficient:

    transition      is the next state a function of the current state and the action?
    task            is "what happens next" a function of the state?

Both are queryable with instruments already built, and neither requires choosing an action.

Criteria written before the run:
    T1  the transition is NOT determined by (y, vy, a): the delayed impulse is missing, so the
        residual sits at a definite non-zero value.  With the pending queue in the state it is
        exact (<= 1e-12).
    T2  "dies on the next frame" is NOT determined by (y, vy, pending): a constructed state pair
        with the same (y, vy, pending) and different pipe geometry must give different outcomes.
    T3  the operator bank is not needed for sufficiency: adding it leaves the transition
        residual at zero.  It is a compression/noise tool, not a requirement.
"""
import numpy as np

from latent_world import BIRD_X, latent
from signal_bank import bank

LAG = 3
DRIFT, FLAP, SPEED, GAP = -0.0016, 0.020, 0.010, 0.34


def collect(n_ep=400):
    rows = []
    for ep in range(n_ep):
        env = latent[LAG](5000 + ep)
        rng = np.random.RandomState(6000 + ep)
        while not env.dead:
            pend = env.pending()                       # a(t-3) .. a(t-1)
            y, vy = env.y, env.vy
            a = int(rng.rand() < 0.5)
            dist = env.pipes[0][0] - BIRD_X
            gap = env.pipes[0][1] - env.y
            env.step(a)
            rows.append(dict(y=y, vy=vy, a=float(a), dist=dist, gap=gap,
                             vy2=env.vy, dead=float(env.dead),
                             p1=float(pend[0]), p2=float(pend[1]), p3=float(pend[2])))
    return rows


def resid(cols, tgt, holdout=0.3):
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in cols])
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    d = A[cut:] @ w - tgt[cut:]
    return float(np.sqrt(np.mean(d * d)))


def recall(cols, tgt, holdout=0.3):
    """death recall: deaths are 6.5% of rows, so accuracy would only report the majority class"""
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in cols])
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    p = A[cut:] @ w > 0.5
    t = tgt[cut:] > 0.5
    return float(p[t].mean()) if t.any() else float("nan")


def death(y, vy, p0, dist, gap):
    """the world's next-frame outcome, written from the state alone"""
    vy2 = vy + DRIFT + FLAP * p0
    y2 = y + vy2
    dist2 = dist - SPEED
    gc = y + gap
    return bool((abs(y2 - gc) > GAP / 2 and abs(dist2) < 0.06) or y2 < 0 or y2 > 1)


rs = collect()
c = {k: np.asarray([r[k] for r in rs]) for k in rs[0]}
state = [c["y"], c["vy"], c["p1"], c["p2"], c["p3"]]
print("%d rows from the delayed world (every flap lands %d frames later)\n" % (len(rs), LAG))

print("T1  transition: predict vy' from the state")
print("    (y, vy, a)                        residual %.2e" % resid([c["y"], c["vy"], c["a"]], c["vy2"]))
print("    (y, vy) + pending queue           residual %.2e" % resid(state, c["vy2"]))
B = bank(np.stack([c["y"], c["vy"]], 1), means=(9,))
extra = [B[k][:, 0] for k in ("d1", "d2", "leak0.1", "unleak0.1", "ma9")]
r_ops = resid(state + extra, c["vy2"])
print("    (y, vy) + pending + 5 operators   residual %.2e" % r_ops)

print("\nT2  task: is 'dies next frame' determined by (y, vy, pending)?")
print("    two states with identical (y, vy, pending), different pipe geometry:")
print("      A  y .50  vy 0  pending (0,0,0)  pipe 0.50 away, gap 0.00 -> death %s"
      % death(0.5, 0.0, 0.0, 0.50, 0.00))
print("      B  y .50  vy 0  pending (0,0,0)  pipe at the bird, gap 0.30 -> death %s"
      % death(0.5, 0.0, 0.0, 0.00, 0.30))
print("    so the distance and the gap are part of the task state, not decoration")
print("    (a linear probe cannot represent this conjunction: death recall %.2f without"
      % recall(state, c["dead"]))
print("     them, %.2f with them -- 'a relation holds' is not 'a regression finds it', as in A.20)"
      % recall(state + [c["dist"], c["gap"]], c["dead"]))

r_state = resid(state, c["vy2"])
r_a = resid([c["y"], c["vy"], c["a"]], c["vy2"])
print("\nT1  (y,vy,a) %.2e -> not Markov;  + pending %.2e -> Markov: %s"
      % (r_a, r_state, "ok" if r_a > 1e-3 and r_state <= 1e-12 else "MISS"))
print("T2  a constructed pair with equal (y,vy,pending) and different outcomes: %s"
      % ("ok" if death(0.5, 0.0, 0.0, 0.5, 0.0) != death(0.5, 0.0, 0.0, 0.0, 0.30) else "MISS"))
print("T3  operators change the residual by %.1e: %s"
      % (abs(r_state - r_ops), "ok" if abs(r_state - r_ops) < 1e-12 else "check"))
