"""The difference between adjacent layers is itself a signal -- and it can be counted.

The claim: many signals are learned from the difference between layers -- the memory layer of
adjacent times, and reality against prediction.

That is measurable, and the measure is not a magnitude.  Take the residual of a layer and count
how many distinct values it takes:

    1 value      the layer is complete: nothing left to explain
    2 values     exactly one binary mechanism is missing
    k values     k-1 mechanisms, or a higher-order structure
    a continuum  the FORM is wrong -- no amount of fitting will make it discrete

So the same number both selects the layer to learn at and says when to stop.  This measures it
on one trajectory, for the temporal stack and for the prediction channel.

Criteria written before the run:
    X1  the velocity channel is not primitive: it IS the adjacent-time difference of position
        (d y == vy one step later, exactly)
    X2  the number of distinct values falls to a unique minimum at one difference order, and it
        is the same layer for y (order 2) and for vy (order 1) -- shifted by one, which is the
        same statement as X1
    X3  at that layer the values are exactly two, one binary column explains them completely,
        and the two coefficients come out as the drift and the consequence
    X4  a channel with no events is constant at order 1: one value, nothing to learn
    X5  a wrong form reads as a continuum, and a 1e-4 coefficient error has a tiny RMS but the
        same verdict -- the count is strictly stricter than the magnitude
"""
import numpy as np

from latent_world import BIRD_X, latent

LAG = 3


def collect(n_ep=300):
    eps = []
    for ep in range(n_ep):
        env = latent[LAG](4000 + ep)
        rng = np.random.RandomState(7000 + ep)
        seq, popped = [], []
        while not env.dead:
            a = int(rng.rand() < 0.5)
            px = env.pipes[0][0]
            seq.append((env.y, env.vy, px - BIRD_X, float(a)))
            popped.append(0)
            env.step(a)
            if env.pipes[0][0] > px:          # a pipe was recycled this step
                popped[-1] = 1
        if len(seq) > 12:
            eps.append((np.asarray(seq), np.asarray(popped)))
    return eps


def card(vals, tol=1e-12):
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) == 0:
        return 0
    return int(1 + np.sum(np.diff(v) > tol))


def dth(x, d):
    out = np.asarray(x, dtype=float)
    for _ in range(d):
        out = np.diff(out)
    return out


eps = collect()
nrows = sum(len(s) for s, _ in eps)
print("trajectory: %d episodes, %d rows (random actions, lag %d)\n" % (len(eps), nrows, LAG))

# ---------------------------------------------------------------- X1
worst = 0.0
for seq, _ in eps:
    d = np.diff(seq[:, 0])
    worst = max(worst, float(np.max(np.abs(d - seq[1:, 1]))))
print("X1  does  d y  equal  vy  one step later?   max |dy - vy(t+1)| = %.2e -> %s"
      % (worst, "ok" if worst < 1e-12 else "MISS"))
print("    the velocity channel is the adjacent-time difference of position, not a primitive\n")

# ---------------------------------------------------------------- X2, X3
print("X2/X3  distinct values of the order-d difference (no centring)")
print("       order     y (position)     vy (velocity)     dx (pipe, inside a segment)")
table = {}
for d in range(0, 5):
    cells = []
    for idx in (0, 1, 2):
        cs = []
        for seq, popped in eps:
            if idx == 2 and d >= 1:
                keep = np.ones(len(seq), bool)
                for i in range(len(seq)):
                    if popped[i] or (i > 0 and popped[i - 1]):
                        keep[i] = False
                v = dth(seq[keep, idx], d)
            else:
                v = dth(seq[:, idx], d)
            if len(v):
                cs.append(card(v))
        cells.append(int(np.median(cs)) if cs else 0)
    for name, c in zip(("y", "vy", "dx"), cells):
        table.setdefault(name, []).append(c)
    print("         %d      %10d       %10d        %10d" % (d, cells[0], cells[1], cells[2]))

oy = int(np.argmin(table["y"]))
ov = int(np.argmin(table["vy"]))
print("\n    minimum for y  at order %d   (%d values)" % (oy, table["y"][oy]))
print("    minimum for vy at order %d   (%d values)  -> same layer shifted by one: %s"
      % (ov, table["vy"][ov], "ok" if oy == ov + 1 else "MISS"))

# the two levels, and the one column that explains them
V, E = [], []
for seq, _ in eps:
    v = dth(seq[:, 0], oy)                      # v[j] = d^oy y at row index j
    if len(v) < 8:
        continue
    a = seq[:, 3]
    fired = np.array([a[j + oy - 1 - LAG] if j + oy - 1 - LAG >= 0 else 0.0
                      for j in range(len(v))])
    V.append(v)
    E.append(fired)
V = np.concatenate(V)
E = np.concatenate(E)
X = np.stack([np.ones_like(V), E], 1)
w = np.linalg.lstsq(X, V, rcond=None)[0]
left = V - X @ w
levels = np.unique(np.round(V, 10))
print("\nX3  at order %d the values are %s" % (oy, levels))
print("    one binary column explains them:  b = %.6f (drift), gain = %.6f (consequence)"
      % (w[0], w[1]))
print("    what is left: %d value(s), max |left| %.2e -> %s"
      % (card(np.round(left, 10)), float(np.max(np.abs(left))),
         "ok" if card(np.round(left, 10)) == 1 else "MISS"))

# ---------------------------------------------------------------- X4
consts = []
for seq, popped in eps:
    keep = np.ones(len(seq), bool)
    for i in range(len(seq)):
        if popped[i] or (i > 0 and popped[i - 1]):
            keep[i] = False
    d = np.diff(seq[keep, 2])
    if len(d):
        consts.append(d)
consts = np.concatenate(consts)
print("\nX4  a channel with no events: dx inside a segment, order 1")
print("    %d distinct value(s): %s   (|world speed| %.3f)"
      % (card(np.round(consts, 10)), np.unique(np.round(consts, 10)), 0.010))

# ---------------------------------------------------------------- X5
y1 = np.concatenate([s[:-1, 0] for s, _ in eps])
y2 = np.concatenate([s[1:, 0] for s, _ in eps])
vy1 = np.concatenate([s[:-1, 1] for s, _ in eps])
vy2 = np.concatenate([s[1:, 1] for s, _ in eps])

Xa = np.stack([np.ones_like(y1), y1], 1)
wa = np.linalg.lstsq(Xa, y2, rcond=None)[0]
ra = y2 - Xa @ wa
rb = y2 - (y1 + 1.0001 * vy2)          # the right form, one coefficient off by 1e-4
rc = y2 - (y1 + vy2)                   # the right form
print("\nX5  the count is stricter than the magnitude")
for name, r in (("y' = b + c*y            (velocity missing)", ra),
                ("y' = y + 1.0001*vy'      (form right, gain 1e-4 off)", rb),
                ("y' = y + vy'             (complete)", rc)):
    print("    %-48s rms %.2e   values %6d" % (name, float(np.sqrt(np.mean(r ** 2))),
                                                card(np.round(r, 12))))

print("\nX1 %s   X2 %s   X3 %s   X4 %s   X5 %s"
      % ("ok" if worst < 1e-12 else "MISS",
         "ok" if oy == ov + 1 else "MISS",
         "ok" if len(levels) == 2 and card(np.round(left, 10)) == 1 else "MISS",
         "ok" if card(np.round(consts, 10)) == 1 else "MISS",
         "ok" if card(np.round(ra, 12)) > 100 and card(np.round(rb, 12)) > 100
         and card(np.round(rc, 12)) == 1 else "MISS"))
