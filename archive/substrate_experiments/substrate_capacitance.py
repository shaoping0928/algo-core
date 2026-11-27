"""Do we need a per-node capacitance, or is a bank of them enough?

u is an RC low-pass with one time constant 1/beta.  A real circuit lets every element have its
own.  The question is whether the substrate needs that, or whether the pair (h, u) plus a linear
readout can cover a range of time constants.

Analytically it cannot be exact away from beta: writing a*h + b*u gives

    a + b*beta/(1 - (1-beta)z^-1)

whose pole is fixed at 1-beta.  So a target with time constant 1/beta_w is reproduced exactly at
beta_w = beta and nowhere else -- the zero can bend the shape, not move the pole.

The measurable question is therefore not "is it exact" but "how far is it", and whether a bank of
betas fixes it.  If the bank's points are the only exact ones, then per-node capacitance is
equivalent to a bank, and the resolution of the capacitance is the spacing of the bank.

Criteria written before the run:
    K1  one beta: the residual is zero at the matching time constant and a definite non-zero
        value everywhere else
    K2  a five-beta bank: zero at each bank point, and much smaller than one beta in between
    K3  the residual between bank points does not vanish: the capacitance is quantised, and its
        spacing is what sets the error
"""
import numpy as np

T = 20000
B0 = 0.1
BANK = (0.5, 0.2, 0.1, 0.05, 0.02)


def leak(z, b):
    out = np.empty_like(z)
    u = 0.0
    for t in range(len(z)):
        u = (1.0 - b) * u + b * z[t]
        out[t] = u
    return out


def fit_resid(cols, y, holdout=0.3):
    n = len(y)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in cols])
    w = np.linalg.lstsq(A[:cut], y[:cut], rcond=None)[0]
    d = A[cut:] @ w - y[cut:]
    return float(np.sqrt(np.mean(d * d)))


rng = np.random.RandomState(5)
drive = rng.normal(0.0, 1.0, T)
u0 = leak(drive, B0)
U = [leak(drive, b) for b in BANK]

print("observation h = a white drive;  one capacitance beta=%.2f, or a bank of five\n" % B0)
print("  beta_w   tau_w   single beta (rms/std)   five-beta bank   bank has this one")
rows = []
for bw in (0.5, 0.35, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.035, 0.02, 0.01):
    tgt = leak(drive, bw)
    sd = float(np.std(tgt))
    r1 = fit_resid([drive, u0], tgt) / sd
    r2 = fit_resid([drive] + U, tgt) / sd
    rows.append((bw, r1, r2))
    print("   %.2f    %5.1f      %12.2e        %10.2e      %s"
          % (bw, 1 / bw, r1, r2, "yes" if any(abs(bw - b) < 1e-12 for b in BANK) else ""))

r1s = [r[1] for r in rows]
r2s = [r[2] for r in rows]
exact1 = [r[0] for r in rows if r[1] < 1e-9]
exact2 = [r[0] for r in rows if r[2] < 1e-9]
print("\nK1  exact with one beta: %s  (residual elsewhere up to %.1e)"
      % (exact1, max(r1s)))
print("K2  exact with the bank: %s" % exact2)
print("    worst between bank points: one beta %.1e, bank %.1e (%.0fx better)"
      % (max(r1s), max(r2s), max(r1s) / max(r2s)))

mid = [r for r in rows if r[0] not in BANK]
print("\nK3  between bank points the bank is not exact: worst %.1e" % max(r[2] for r in mid))
print("    so the capacitance is quantised.  the bank is the resolution, not a per-node value.")
print("\nK1 %s   K2 %s   K3 %s"
      % ("ok" if exact1 == [B0] else "check",
         "ok" if set(exact2) == set(BANK) else "check",
         "ok" if max(r[2] for r in mid) > 1e-6 else "check"))
