"""What the signal bank does when the observation is noisy.

The inverse of a leak divides by beta, so it amplifies noise by sqrt(1+(1-beta)^2)/beta
(13.5x at beta=0.1).  Two things follow, and both are measurable:

    1  at what noise does the *correct* operator stop being the best choice?
    2  when it stops, what is chosen instead -- the identity (giving up), or a wrong but
       quieter operator (a structural error bought with stability)?

Everything is measured in RMS against the no-information floor: predicting the mean of the
target gives RMS = std(target), and an operator is informative only while it beats that.

Two closed-form predictions were attempted and rejected before the numbers were reported:
a Pythagorean bias (+) gain*sigma gave 4/10 agreement, and an errors-in-variables saturation
gave 0/10, because the structural residual at sigma=0 and the noise are coupled through the
fitted coefficient.  So this reports the measured ladder and one structural claim that the
measurement can check.

Criteria written before the run:
    N1  the selected operator starts at the true inverse and steps to a coarser one as noise
        grows -- a ladder, not a cliff
    N2  the trade is monotone: as sigma grows the chosen operator has a smaller noise gain and
        a larger structural bias (measured at sigma=0).  That is the bias-variance statement,
        and it is checkable without a full model
    N3  the true operator stops winning at a sigma of the same order as floor/gain, and the
        numbers are reported side by side
    N4  the selection never reports "unknown": past the crossing the argmin still answers, and
        only rms/floor and the margin say it means nothing
"""
import numpy as np

from signal_bank import bank, make_world, noise_gain

SIG = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 8e-2, 1.2e-1)


def fit(src, tgt, holdout=0.3):
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.stack([np.ones(n), src], 1)
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    d = A[cut:] @ w - tgt[cut:]
    return float(np.sqrt(np.mean(d * d)))


def gain_of(name):
    if name == "id":
        return 1.0
    if name == "d1":
        return float(np.sqrt(2.0))
    if name == "d2":
        return float(np.sqrt(6.0))
    if name.startswith("leak"):
        b = float(name[4:])
        return float(np.sqrt(b / (2.0 - b)))
    return noise_gain(float(name[6:]))


s, x, beta0 = make_world(T=6000, N=3)
V = float(np.var(s))
floor = float(np.sqrt(V))
B0 = bank(x)
bias0 = {k: float(np.mean([fit(B0[k][:, j], s[:, j]) for j in range(3)])) for k in B0}
gains = {k: gain_of(k) for k in B0}
true_src = "unleak%g" % beta0

print("world: 6000 steps, 3 nodes, x = leak(s, beta=%.2f) + noise" % beta0)
print("no-information floor: RMS %.3f\n" % floor)
print("  operator      noise gain    bias at sigma=0")
for k in ("id", "d1", "leak0.1", "unleak0.05", "unleak0.1", "unleak0.2", "unleak0.5"):
    print("    %-10s   %8.2f      %8.3f" % (k, gains[k], bias0[k]))

print("\n  sigma    chosen      gain   bias0   rms/floor   runner-up   margin")
rows = []
for sig in SIG:
    rng = np.random.RandomState(11)
    xn = x + (rng.normal(0.0, sig, x.shape) if sig else 0.0)
    B = bank(xn)
    res = {k: float(np.mean([fit(B[k][:, j], s[:, j]) for j in range(3)])) for k in B}
    rank = sorted(res.items(), key=lambda kv: kv[1])
    best, r1 = rank[0]
    second, r2 = rank[1]
    rows.append((sig, best, r1, second, r2, r2 / max(r1, 1e-12), r1 / floor))
    print("  %.0e  %-10s %6.2f  %6.3f    %7.2f    %-10s %6.1f"
          % (sig, best, gains[best], bias0[best], r1 / floor, second, r2 / max(r1, 1e-12)))

chosen_g = [gains[r[1]] for r in rows]
chosen_b = [bias0[r[1]] for r in rows]
last_right = max((r[0] for r in rows if r[1] == true_src), default=None)
first_wrong = next((r for r in rows if r[1] != true_src), None)
pred_scale = floor / gains[true_src]

print("\nN1  the ladder")
print("    " + " -> ".join("%s@%.0e" % (r[1], r[0]) for r in rows))

print("\nN2  the trade is monotone")
print("    chosen gain:   " + " ".join("%.1f" % g for g in chosen_g))
print("    chosen bias0:  " + " ".join("%.3f" % b for b in chosen_b))
print("    gain non-increasing: %s   bias non-decreasing: %s"
      % (all(a >= b - 1e-9 for a, b in zip(chosen_g, chosen_g[1:])),
         all(a <= b + 1e-9 for a, b in zip(chosen_b, chosen_b[1:]))))

print("\nN3  the correct operator last wins at sigma = %.0e; first wrong choice %s at %.0e"
      % (last_right, first_wrong[1], first_wrong[0]) if first_wrong else
      "\nN3  the correct operator wins everywhere in the sweep")
print("    the scale to compare it with: floor/gain = %.3f / %.2f = %.3f"
      % (floor, gains[true_src], pred_scale))

print("\nN4  the selection never reports 'unknown'")
for thr in (0.5, 0.8, 0.9):
    hit = next((r for r in rows if r[6] >= thr), None)
    print("    rms/floor reaches %.1f at sigma = %s"
          % (thr, ("%.0e" % hit[0]) if hit else "> 1.2e-1"))
print("    last row: margin %.2fx, best rms %.0f%% of the floor"
      % (rows[-1][5], 100 * rows[-1][6]))
print("    the argmin is still answering -- it has nothing left to say")
