"""Signal sources: per-node operators along the input axis.

The substrate has one leaky state (u) with one time constant and one difference (the edge
residue r).  This makes the operators explicit and gives every node a bank of them:

    id           x(t)                                  the intensity itself
    d1           x(t) - x(t-1)                         a differentiator
    d2           x(t) - 2x(t-1) + x(t-2)               second difference
    leak_beta    (1-beta)u(t-1) + beta*x(t)            an exponential decayer
    unleak_beta  (x(t) - (1-beta)x(t-1)) / beta        its exact inverse

Every output is a signal source: a column the learning layers may use.  None of them is
selected here; selection is by the residual (A.15/A.16), and that is what gets measured.

Criteria written before the run:
    S1  a leaky observation does not reveal its drive: the identity source leaves a definite
        residual
    S2  the bank contains an exact inverse (unleak at the right beta): residual <= 1e-12
    S3  selection by residual picks that operator and only it, and the runner-up is a definite
        number rather than a near miss
    S4  the operators are per node: three independently driven nodes are recovered column by
        column, and no node's sources predict another node's target
    S5  differentiating a leak amplifies noise by sqrt(1+(1-beta)^2)/beta; measure it, and
        report the observation noise at which the recovery stops being usable
"""
import numpy as np

from signal_bank import bank, fit_residual, make_world


s, x, beta0 = make_world()
T, N = x.shape
print("world: %d steps, %d independently driven nodes;  drive s 0/1, observation x = leak(s, beta=%.2f)\n"
      % (T, N, beta0))

Bx = bank(x)                 # sources derived from the leaky observation
Bs = bank(s)                 # sources derived from the drive
print("bank: %d sources per node  (%s ... )"
      % (len(Bx), ", ".join(sorted(Bx)[:5])))
print("      %d columns per source, operators never mix columns\n" % N)

# ---------------------------------------------------------------- S1, S2, S3
print("S1/S2/S3  recover the drive from the leaky observation (node 0)")
ranked = sorted(((fit_residual(Bx[k][:, 0], s[:, 0]), k) for k in Bx))
for r, k in ranked[:5]:
    print("    %-12s residual %.2e" % (k, r))
print("    ... %d sources in total" % len(ranked))
best_r, best_k = ranked[0]
second_r, second_k = ranked[1]
print("    the identity alone: %.2e -> S1 %s" % (dict((k, r) for r, k in ranked)["id"],
                                                "ok" if dict((k, r) for r, k in ranked)["id"] > 1e-3 else "MISS"))
print("    selected %s (residual %.2e) -> S2 %s" % (best_k, best_r, "ok" if best_r <= 1e-12 else "MISS"))
print("    runner-up %s (residual %.2e, %.0fx larger) -> S3 %s"
      % (second_k, second_r, second_r / max(best_r, 1e-300),
         "ok" if second_k != best_k and second_r > 1e-3 else "MISS"))

print("\n    the same question in the other direction: predict the leak from the drive (node 0)")
ranked2 = sorted(((fit_residual(Bs[k][:, 0], x[:, 0]), k) for k in Bs))
for r, k in ranked2[:3]:
    print("    %-12s residual %.2e" % (k, r))
kind = "the decayer" if ranked2[0][1].startswith("leak") else "something else"
print("    -> picked %s, i.e. %s, not the differentiator" % (ranked2[0][1], kind))

# ---------------------------------------------------------------- S4: per node
print("\nS4  per node: recover each column from its own sources, and from the other nodes")
own, cross = [], []
for j in range(N):
    own.append(fit_residual(Bx[best_k][:, j], s[:, j]))
    for i in range(N):
        if i != j:
            cross.append(fit_residual(Bx[best_k][:, i], s[:, j]))
print("    own column  max residual %.2e" % max(own))
print("    other nodes max residual %.2e  (what a mixed signal would look like)" % max(cross))
print("    -> S4 %s" % ("ok" if max(own) <= 1e-12 and max(cross) > 1e-2 else "MISS"))

# ---------------------------------------------------------------- S5: noise gain
print("\nS5  differentiating a leak amplifies noise")
print("    sigma      recovery error / sigma      predicted sqrt(1+(1-b)^2)/b")
for sig in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
    rng = np.random.RandomState(5)
    xn = x[:, 0] + rng.normal(0.0, sig, T) if sig else x[:, 0]
    rec = (xn[1:] - (1.0 - beta0) * xn[:-1]) / beta0
    truth = s[1:, 0]
    err = float(np.std(rec - truth))
    shown = "exact" if sig == 0.0 else "%8.3f" % (err / sig)
    print("    %.0e        %s                  %8.3f"
          % (sig, shown, np.sqrt(1 + (1 - beta0) ** 2) / beta0))

print("\n    so the drive (0/1, separated by 0.5) survives while 13.4*sigma << 0.5,")
print("    i.e. sigma << %.3f; past that the inverse of the decay is noise, not signal" % (0.5 / 13.4))

print("\nS1 %s   S2 %s   S3 %s   S4 %s"
      % ("ok" if dict((k, r) for r, k in ranked)["id"] > 1e-3 else "MISS",
         "ok" if best_r <= 1e-12 else "MISS",
         "ok" if second_k != best_k and second_r > 1e-3 else "MISS",
         "ok" if max(own) <= 1e-12 and max(cross) > 1e-2 else "MISS"))
