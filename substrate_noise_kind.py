"""Where the noise comes from, and why its spectrum decides the answer.

Every number in A.16 and A.19 came from noise this project injected: iid Gaussian, added to one
channel.  The system itself is deterministic, so those are sensitivity measurements, not
estimates of any real noise source.

That matters because a differentiator is a high-pass filter.  Against white noise it amplifies
by sqrt(1+(1-b)^2)/b (13.5x at b=0.1); against correlated noise it amplifies less, and in the
limit of a random-walk offset it does not amplify at all.  So the horizon measured against white
noise is a worst case, and the same operator would survive a drift far better.

This measures it: same world, same marginal sigma, only the correlation changes.

    n(t) = rho*n(t-1) + e(t),   e scaled so that std(n) = sigma

    predicted gain of unleash_b = sqrt(1 + (1-b)^2 - 2(1-b)rho) / b

Criteria written before the run:
    K1  at rho=0 the result reproduces A.19 (white noise is the control)
    K2  measured gain falls with rho and matches the closed form
    K3  at rho=1 (a drifting offset) the operator is almost transparent: gain ~ 1
    K4  the selection survives at fixed sigma once the noise is correlated -- the same
        observation error, a different spectrum, a different verdict
"""
import numpy as np

from signal_bank import bank, make_world

BETA = 0.1
SIG = 1e-3


def ar1(rng, n, rho, sigma):
    e = rng.normal(0.0, 1.0, n) * sigma * np.sqrt(max(1.0 - rho * rho, 0.0))
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + e[t]
    return out


def rms(src, tgt, holdout=0.3):
    n = len(tgt)
    cut = int((1.0 - holdout) * n)
    A = np.stack([np.ones(n), src], 1)
    w = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    d = A[cut:] @ w - tgt[cut:]
    return float(np.sqrt(np.mean(d * d)))


s, x, beta0 = make_world(T=6000, N=3)
drive = s[:, 0]
obs = x[:, 0]
clean = bank(obs.reshape(-1, 1))
clean_res = {k: rms(clean[k][:, 0], drive) for k in clean}
true_src = "unleak%g" % beta0
floor = float(np.std(drive))
print("world: x = leak(s, %.2f), marginal noise sigma = %.0e, floor %.2f" % (beta0, SIG, floor))
print("clean: %s residual %.2e\n" % (true_src, clean_res[true_src]))

print("  rho    predicted gain   measured rms   rms/sigma   selected        margin")
rows = []
for rho in (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99):
    rng = np.random.RandomState(21)
    xn = obs + ar1(rng, len(obs), rho, SIG)
    B = bank(xn.reshape(-1, 1))
    res = {k: rms(B[k][:, 0], drive) for k in B}
    rank = sorted(res.items(), key=lambda kv: kv[1])
    gain = np.sqrt(1.0 + (1.0 - BETA) ** 2 - 2.0 * (1.0 - BETA) * rho) / BETA
    rows.append((rho, gain, res[true_src], rank[0][1], rank[0][0], rank[1][1] / rank[0][1]))
    print("  %4.2f     %8.2f       %8.2e    %8.2f   %-10s  %8.1f"
          % (rho, gain, res[true_src], res[true_src] / SIG, rank[0][0],
             rank[1][1] / max(rank[0][1], 1e-12)))

ratio = [r[2] / SIG for r in rows]
pred = [r[1] for r in rows]
print("\nK1  rho=0 reproduces A.19: rms/sigma %.2f (A.19 measured 13.5)" % ratio[0])
print("K2  measured vs closed form: " + "  ".join("%.1f/%.1f" % (a, b) for a, b in zip(ratio, pred)))
print("K3  correlation removes the amplification: %.1f -> %.1f  (factor %.1f),"
      % (ratio[0], ratio[-1], ratio[0] / ratio[-1]))
print("    and the closed form's limit as rho->1 is 1.0 (the response to a level, not to noise)")
print("K4  selection at fixed sigma: %s at every rho: %s"
      % (true_src, all(r[4] == true_src for r in rows)))
print("\n    a note on rho = 1 exactly: there the innovations vanish (a constant offset),")
print("    and a constant is absorbed by the intercept, so the measured residual is 0 for a")
print("    reason that has nothing to do with the operator.")
