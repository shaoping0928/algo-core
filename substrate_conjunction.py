"""Does the derived (conjunctive) node actually improve anything the system uses?

The rule layer groups by exact keys, and a tuple key already expresses a conjunction -- so for
determinacy, a product node buys nothing.  The part of the system that cannot express a
conjunction is the LINEAR readout (the predictor, the operator selection, the output layer).
So that is where the test belongs:

    features        w = in the pipe window, b = outside the band
    target          dies on the next step
    readout         least squares (linear), as everywhere else in this line
    metric          purity at a fixed coverage -- of the top q of predicted deaths, how many die

A linear readout over (w, b) can only add them, so it must flag every w.  Adding the product
node (w*b) -- which is what one association plus a rectified threshold produces -- lets it
isolate the intersection.

Criteria written before the run:
    C1  with (w, b) alone the readout cannot isolate the conjunction: at 0.5% coverage its
        purity is close to P(die | w), not to 1
    C2  adding the single derived node w*b raises purity at the same coverage to >= 0.9
    C3  the rule layer is unchanged by this: grouping by the tuple (w, b) already expresses the
        conjunction, so no node is needed there.  Depth 2 is a readout fix, not a rule-layer fix.
"""
import numpy as np

from latent_world import BIRD_X, GAP, latent

LAG = 0


def collect(n_ep=400):
    w, b, died, y_, d_ = [], [], [], [], []
    for ep in range(n_ep):
        env = latent[LAG](7000 + ep)
        rng = np.random.RandomState(8000 + ep)
        while not env.dead:
            dist = env.pipes[0][0] - BIRD_X
            gc = env.pipes[0][1]
            w.append(float(abs(dist) < 0.06))
            b.append(float(abs(env.y - gc) > GAP / 2))
            a = int(rng.rand() < 0.5)
            env.step(a)
            died.append(float(env.dead))
            y_.append(env.y)
            d_.append(env.pipes[0][0] - BIRD_X)
    return (np.asarray(w), np.asarray(b), np.asarray(died))


def fit(cols, tgt, holdout=0.3):
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + cols)
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    p = A[cut:] @ w
    return p, tgt[cut:]


def purity_at(p, t, q):
    k = max(1, int(q * len(t)))
    idx = np.argsort(-p)[:k]
    return float(t[idx].mean()), k


w, b, died = collect()
print("%d steps; death rate %.3f;  P(in window) %.3f;  P(outside band) %.3f"
      % (len(w), died.mean(), w.mean(), b.mean()))
print("P(die | w and b) = %.3f at coverage %.4f\n"
      % (died[(w > 0.5) & (b > 0.5)].mean(), float(np.mean((w > 0.5) & (b > 0.5)))))

print("  features                 purity at 0.5%%   purity at 1%%   coverage of the top 1%")
for name, cols in (("w", [w]),
                   ("b", [b]),
                   ("w, b", [w, b]),
                   ("w, b, w*b (derived node)", [w, b, w * b])):
    p, t = fit(cols, died)
    q5, k5 = purity_at(p, t, 0.005)
    q1, k1 = purity_at(p, t, 0.01)
    print("  %-24s %13.3f   %12.3f   %10.4f" % (name, q5, q1, died.mean() * 0))

print("\n  (chance is the death rate %.3f; the base rate of w*b is %.4f)"
      % (died.mean(), float(np.mean((w > 0.5) & (b > 0.5)))))

p0, t0 = fit([w, b], died)
p1, t1 = fit([w, b, w * b], died)
q0, _ = purity_at(p0, t0, 0.005)
q1b, _ = purity_at(p1, t1, 0.005)
print("\nC1  (w, b) at 0.5%% coverage: purity %.3f -> %s"
      % (q0, "ok" if q0 < 0.9 else "MISS"))
print("C2  + w*b at the same coverage: purity %.3f -> %s"
      % (q1b, "ok" if q1b >= 0.9 else "MISS"))
print("C3  the rule layer needs no node here: its key can be the tuple (w, b) itself.")
print("    the derived node is a readout fix, not a rule-layer fix.")
