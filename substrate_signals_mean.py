"""The mean operator, and the asymmetry between its inverse and the exponential's.

The bank had id, d1, d2, leak (exponential decay) and unleak.  It had no box mean.  Adding one
is easy; what is worth measuring is that the two means behave very differently when you try to
undo them:

    exponential   u(t) = (1-b)u(t-1) + b*x(t)
                  x(t) = (u(t) - (1-b)u(t-1)) / b            one step of memory
    box           m(t) = (1/w) sum_{j<w} x(t-j)
                  x(t) = w*(m(t) - m(t-1)) + x(t-w)          needs x from w steps ago

So the exponential mean is invertible inside the operator family and the box mean is not: its
inverse is recursive in the target, which is the same "retention without addressing" boundary
A.9 measured, now on a concrete operator.

Criteria written before the run:
    M1  the box mean can be *produced* exactly: from the drive, the source ma_w0 reproduces the
        mean to 1e-12, and selection picks it
    M2  it cannot be *inverted* inside the family: without the target delayed by w, the best
        combination leaves a definite residual; with that column it is exact
    M3  at equal effective lag the box and exponential means reject noise equally well, and the
        measured gains match 1/sqrt(w) and sqrt(b/(2-b))
    M4  adding the means to the bank does not change an existing result: the leak-world
        selection at sigma=0 is the same with and without them (new sources are opt-in)
"""
import numpy as np

from signal_bank import bank, make_world

W = 9
BETA = 2.0 / (W + 1)          # the exponential with the same effective lag as ma_9


def fit(cols, tgt, holdout=0.3):
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.column_stack([np.ones(n)] + list(cols))
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    d = A[cut:] @ w - tgt[cut:]
    return float(np.sqrt(np.mean(d * d)))


def box_mean(s, w):
    out = np.empty(len(s))
    for t in range(len(s)):
        out[t] = s[max(0, t - w + 1):t + 1].mean()
    return out


s, x, beta0 = make_world(T=6000, N=3)
drive = s[:, 0]
ma = box_mean(drive, W)

print("box mean of window %d, and an exponential with the same effective lag (beta = %.3f)\n"
      % (W, BETA))

# ---------------------------------------------------------------- M1
B = bank(drive.reshape(-1, 1), means=(3, 5, 9, 17))
res = {k: fit([B[k][:, 0]], ma) for k in B}
rank = sorted(res.items(), key=lambda kv: kv[1])
best_k, best_r = rank[0]
print("M1  produce the mean from the drive")
for k, r in rank[:4]:
    print("    %-10s residual %.2e" % (k, r))
print("    selected %s -> %s"
      % (best_k, "ok" if best_k == "ma%d" % W and best_r <= 1e-12 else "MISS"))

# ---------------------------------------------------------------- M2
print("\nM2  invert the mean: recover the drive from it")
prev_ma = np.concatenate([[ma[0]], ma[:-1]])
lagged = np.concatenate([[0.0] * W, drive[:-W]])          # the target w steps ago
r2 = fit([ma, prev_ma], drive)
# the exact algebraic inverse: s(t) = w*(m(t) - m(t-1)) + s(t-w)
pred = W * (ma - prev_ma) + lagged
r_exact = float(np.abs(pred[W:] - drive[W:]).max())
r3 = fit([ma, prev_ma, lagged], drive)
print("    [m(t), m(t-1)]                     residual %.2e" % r2)
print("    w*(m(t)-m(t-1)) + x(t-%d), verbatim  residual %.2e" % (W, r_exact))
print("    same three columns, least squares    residual %.2e  (ill-conditioned:"
      " the columns are nearly collinear)" % r3)
print("    so the inverse is recursive in the target: %s"
      % ("ok" if r2 > 1e-3 and r_exact <= 1e-12 else "MISS"))
print("    the exponential's inverse needs only its own previous value (unleak)")

# ---------------------------------------------------------------- M3
print("\nM3  equal effective lag, equal noise rejection")
rng = np.random.RandomState(3)
noise = rng.normal(0.0, 1.0, len(drive))


def ema(z, b):
    out = np.empty_like(z)
    run = 0.0
    for t in range(len(z)):
        run = (1.0 - b) * run + b * z[t]
        out[t] = run
    return out


g_ma = float(box_mean(noise, W)[W:].std())
g_em = float(ema(noise, BETA)[W:].std())
print("    ma%d      measured gain %.3f   predicted 1/sqrt(w) = %.3f   effective lag %.1f"
      % (W, g_ma, 1 / np.sqrt(W), (W - 1) / 2))
print("    leak%.2f  measured gain %.3f   predicted sqrt(b/(2-b)) = %.3f   effective lag %.1f"
      % (BETA, g_em, np.sqrt(BETA / (2 - BETA)), (1 - BETA) / BETA))

# ---------------------------------------------------------------- M4
print("\nM4  opt-in: the leak world's selection at sigma=0, with and without means")
sx = x[:, 0]
without = {k: fit([v[:, 0]], s[:, 0]) for k, v in bank(sx.reshape(-1, 1)).items()}
withm = {k: fit([v[:, 0]], s[:, 0]) for k, v in bank(sx.reshape(-1, 1), means=(3, 5, 9, 17)).items()}
b0 = min(without, key=without.get)
b1 = min(withm, key=withm.get)
print("    without means: %s (%.2e)     with means: %s (%.2e)   same: %s"
      % (b0, without[b0], b1, withm[b1], "ok" if b0 == b1 else "MISS"))
