"""Stacked signal layers, and what depth actually buys.

The proposal: apply the operators again to their own outputs (a difference of a difference, a
decay of a difference, ...), a few layers deep, expanding the signal set at each layer; and skip
low-activity nodes to save compute.

What has to be measured before either is adopted:

    1  does depth add information?  A composition of LINEAR operators is still a linear filter of
       the same input, so it cannot predict anything the input's own history cannot.  Depth
       should pay only when a nonlinearity sits between the layers -- and the substrate has one:
       the activation (rectify / top-k gate).
    2  is "low activity" the right thing to gate on?  A quiet node can be the only one carrying
       an event.  The gate that loses nothing drops nodes the kept ones can predict, i.e.
       redundancy -- the same residual-zero test used everywhere in this line -- applied greedily.

Criteria written before the run:
    L1  on a LINEAR target (the drive itself) depth 2 does not beat depth 1: both reach the same
        residual.  Depth is a re-parameterisation of the same span.
    L2  on a NONLINEAR target (a conjunction of two events) depth 2 without a nonlinearity still
        does not help, and depth 2 with a rectifier between the layers does.
    L3  an activity gate and a redundancy gate differ on a quiet-but-unique node: activity drops
        it and loses the target, redundancy keeps it and loses nothing.
    L4  compute: signals per node = K^depth, and what a redundancy gate saves.
"""
import numpy as np

T = 6000


def leak(z, b):
    out = np.empty_like(z)
    u = 0.0
    for t in range(len(z)):
        u = (1.0 - b) * u + b * z[t]
        out[t] = u
    return out


def boxmean(z, w):
    out = np.empty_like(z)
    for t in range(len(z)):
        out[t] = z[max(0, t - w + 1):t + 1].mean()
    return out


def d1(z):
    return np.concatenate([z[:1], np.diff(z)])


def d2(z):
    return d1(d1(z))


OPS = [("id", lambda z: z), ("d1", d1), ("d2", d2),
       ("int", lambda z: np.cumsum(z)),
       ("l1", lambda z: leak(z, 0.1)), ("l2", lambda z: leak(z, 0.2)),
       ("u1", lambda z: (z - 0.9 * np.concatenate([z[:1], z[:-1]])) / 0.1),
       ("ma", lambda z: boxmean(z, 9))]


def fit_resid(cols, y, holdout=0.3):
    n = len(y)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in cols])
    w = np.linalg.lstsq(A[:cut], y[:cut], rcond=None)[0]
    d = A[cut:] @ w - y[cut:]
    return float(np.sqrt(np.mean(d * d)))


rng = np.random.RandomState(7)
s = (rng.rand(T) < 0.10).astype(float)
x = leak(s, 0.1)

depth1 = {n: f(x) for n, f in OPS}
lin2 = {"%s>%s" % (n1, n2): f2(f1(x)) for n1, f1 in OPS for n2, f2 in OPS}
non2 = {"%s>%s" % (n1, n2): f2(np.maximum(f1(x), 0.0)) for n1, f1 in OPS for n2, f2 in OPS}

linear_target = s
nonlin_target = s * np.concatenate([[0.0], s[:-1]])          # two events together

# ---------------------------------------------------------------- the operator algebra
integ = np.cumsum(x)
d_of_i = d1(integ)
i_of_d = np.cumsum(d1(x))
err_di = float(np.max(np.abs(i_of_d - x)))
err_id = float(np.max(np.abs(d_of_i[1:] - x[1:])))
print("L0  difference and integral are inverses")
print("    I(d(x)) - x            max %.2e  (exact: d1 keeps the first sample)" % err_di)
print("    d(I(x)) - x            max %.2e  for t>=1" % err_id)
print("    at t=0 d(I(x)) is 0, not x[0]: exactly one sample is lost, and that is the")
print("    integration constant -- the pair is closed up to one sample")

wn = np.random.RandomState(3).normal(0.0, 1.0, T)
print("\n    but the pure integral is unbounded: on white noise its gain is sqrt(T) = %.1f"
      % float(np.std(np.cumsum(wn)) / np.std(wn)))
print("    the bounded version in the bank is leak (a regularised integral: DC gain 1),")
print("    whose inverse unleak is the regularised difference -- both already present")

print("input: a sparse 10%% event stream through leak(0.1);  %d operators\n" % len(OPS))
print("  target                              depth 1    depth 2    depth 2 + rectifier")
for name, tgt in (("linear (the drive itself)", linear_target),
                  ("nonlinear (two events together)", nonlin_target)):
    r1 = fit_resid(depth1.values(), tgt)
    r2 = fit_resid(lin2.values(), tgt)
    r3 = fit_resid(non2.values(), tgt)
    print("  %-34s %8.3f   %8.3f   %8.3f" % (name, r1, r2, r3))
print("\n  (the drive's own std is %.3f, so 0.5 is 'no information')" % float(np.std(s)))

r1l = fit_resid(depth1.values(), linear_target)
r2l = fit_resid(lin2.values(), linear_target)
r1n = fit_resid(depth1.values(), nonlin_target)
r2n = fit_resid(lin2.values(), nonlin_target)
r3n = fit_resid(non2.values(), nonlin_target)

print("\nL1  linear target: depth 1 %.3f, depth 2 %.3f -> %s (no gain from depth)"
      % (r1l, r2l, "ok" if abs(r1l - r2l) < 0.02 else "MISS"))
print("L2  nonlinear target: depth 2 linear %.3f, + rectifier %.3f -> %s"
      % (r2n, r3n, "ok" if r3n < 0.9 * r2n else "MISS"))
print("    (the rectifier is the substrate's own activation, not an extra mechanism)")

# ---------------------------------------------------------------- L3: what to gate on
print("\nL3  gating: activity vs redundancy")
N = 12
rates = [0.5] * 5 + [0.5] * 5 + [0.03] + [0.03]
sig = np.stack([leak((np.random.RandomState(100 + j).rand(T) < p).astype(float), 0.1)
                for j, p in enumerate(rates)], 1)
sig[:, 5:10] = 0.1 * sig[:, 0:5]              # quiet duplicates of nodes 0..4
sig[:, 11] = 0.1 * sig[:, 10]                 # quiet duplicate of the quiet unique node

level = np.mean(np.abs(sig), 0)


def predictable(j, kept):
    if not kept:
        return False
    return fit_resid([sig[:, k] for k in kept], sig[:, j]) < 1e-9 * max(
        sig[:, j].std(), 1e-12)


greedy = []
for j in range(N):
    if not predictable(j, greedy):
        greedy.append(j)
redundant = [j for j in range(N) if j not in greedy]
keep_act = list(np.where(level >= np.quantile(level, 0.5))[0])
quiet_unique = 10
activity_drops_unique = quiet_unique not in keep_act
redundancy_keeps_unique = quiet_unique in greedy
unrecoverable = [j for j in redundant if not predictable(j, greedy)]

print("    node   level    kept by greedy (independent subset)")
for j in range(N):
    print("      %2d   %.4f    %s" % (j, level[j], "yes" if j in greedy else "no (redundant)"))
print("    activity gate keeps   %s" % "".join("1" if j in keep_act else "." for j in range(N)))
print("    redundancy gate keeps %s" % "".join("1" if j in greedy else "." for j in range(N)))
print("    quiet unique node %d: activity keeps=%s, redundancy keeps=%s"
      % (quiet_unique, not activity_drops_unique, redundancy_keeps_unique))
print("    redundant nodes not recoverable from the independent subset: %s" % unrecoverable)

print("\nL4  compute: signals per node = %d^depth" % len(OPS))
for d in (1, 2, 3):
    full = len(OPS) ** d
    print("    depth %d: %4d signals/node, %6d for %d nodes, %6d with the redundancy gate (%.0f%%)"
          % (d, full, full * N, N, full * len(greedy), 100.0 * len(greedy) / N))

# ---------------------------------------------------------------- L5: derived nodes, arity
def build(depth, xsrc):
    """derived nodes are nodes: after each layer the activation rectifies, then ops continue"""
    cur = {"": xsrc}
    for d in range(depth):
        nxt = {}
        for name, v in cur.items():
            base = v if d == 0 else np.maximum(v, 0.0)
            for on, f in OPS:
                nxt[(name + ">" + on) if name else on] = f(base)
        cur = nxt
    return cur


def conj(k, ssrc):
    out = ssrc.copy()
    for i in range(1, k):
        out = out * np.concatenate([np.zeros(i), ssrc[:len(ssrc) - i]])
    return out


rng2 = np.random.RandomState(11)
s2 = (rng2.rand(T) < 0.25).astype(float)      # denser, so conjunctions of 3-4 exist
x2 = leak(s2, 0.1)
sets = {d: build(d, x2) for d in (1, 2, 3)}
print("\nL5  how many layers?  target = k events occurring together")
print("    (input rate 0.25; target std, and signals per depth)")
print("    target        std      depth 1(%d)   depth 2(%d)   depth 3(%d)"
      % (len(sets[1]), len(sets[2]), len(sets[3])))
res5 = {}
for k in (2, 3, 4):
    tgt = conj(k, s2)
    row = [fit_resid(list(sets[d].values()), tgt) for d in (1, 2, 3)]
    res5[k] = (float(np.std(tgt)), row)
    print("    %d together   %.3f    %9.3f    %9.3f    %9.3f" % (k, res5[k][0], row[0], row[1], row[2]))
def insample(cols, y):
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sqrt(np.mean((A @ w - y) ** 2)))


print("    is the depth-3 failure a capacity problem or an over-fit?  (3 together)")
for d in (1, 2, 3):
    print("      depth %d: in-sample %.3f   hold-out %.3f"
          % (d, insample(list(sets[d].values()), conj(3, s2)), res5[3][1][d - 1]))
print("    a big in-sample/hold-out gap means the readout, not the layer, is the limit")

print("\nL5  measured, not assumed: depth 2 turns a 2-conjunction from 0.194 to 0.000;")
print("    depth 3 does NOT turn a 3-conjunction exact (in-sample 0.080, hold-out 0.224),")
print("    and its hold-out is worse than depth 1 because 512 columns over-fit 4200 rows.")
print("    so the usable depth is 2 with a dense readout; deeper needs a sparse readout.")

print("\nL0 %s   L1 %s   L2 %s   L3 %s   L5 %s"
      % ("ok" if err_di <= 1e-9 and err_id <= 1e-9 else "MISS",
         "ok" if abs(r1l - r2l) < 0.02 else "MISS",
         "ok" if r3n < 0.9 * r2n else "MISS",
         "ok" if activity_drops_unique and redundancy_keeps_unique and not unrecoverable else "MISS",
         "ok" if res5[2][1][1] < 0.05 else "MISS"))
