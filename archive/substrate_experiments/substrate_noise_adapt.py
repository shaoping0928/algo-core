"""Does it adapt to noise, or only follow it passively?

A.19 showed the *choice* of operator shifts with the noise level under a fixed criterion.  That
is passive adaptation: nothing measures sigma, nothing acts on it.  This asks whether the choice
tracks a change mid-stream, holds steady, and comes back.

    arm A  noise switches 0 -> 5e-2 -> 0, world unchanged
    arm B  the world's leak beta switches 0.1 -> 0.3 -> 0.1, no noise, coarse bank
           (0.3 is not in the bank: the nearest member is 0.2)
    arm C  the same switch with 0.3 added to the bank

Arm B is the resolution test: if adaptation lands on 0.2 in B and on 0.3 in C, then the limit of
adaptation is the bank's spacing, not the data.

Criteria written before the run:
    R1  arm C tracks the change to the true member and returns: adaptation works when the
        target is representable
    R2  the lag is set by the window, not by a fixed time constant: report it against W
    R3  arm B lands on the nearest member (0.2), so adaptation resolution == bank resolution
    R4  arm A is stable rather than wobbling, and its fixed point is the coarse inverse: under
        noise it adapts confidently to a wrong structure
"""
import numpy as np

from signal_bank import bank

T = 18000
W = 2000
STEP = 100
BOUNDS = (6000, 12000)
FINE = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)


def rms_in(src, tgt):
    A = np.stack([np.ones(len(tgt)), src], 1)
    w = np.linalg.lstsq(A, tgt, rcond=None)[0]
    d = A @ w - tgt
    return float(np.sqrt(np.mean(d * d)))


def piecewise_leak(s, betas, bounds):
    x = np.zeros(len(s))
    u = 0.0
    for t in range(len(s)):
        b = betas[0] if t < bounds[0] else betas[1] if t < bounds[1] else betas[2]
        u = (1.0 - b) * u + b * s[t]
        x[t] = u
    return x


def select(drive, obs, betas=None):
    B = bank(obs.reshape(-1, 1), betas=betas) if betas else bank(obs.reshape(-1, 1))
    return B


def timeline(drive, obs, betas=None):
    B = select(drive, obs, betas)
    out = []
    for t0 in range(W, len(obs) + 1, STEP):
        sl = slice(t0 - W, t0)
        res = {k: rms_in(B[k][sl, 0], drive[sl]) for k in B}
        out.append((t0, min(res, key=res.get)))
    return out


def segment_choice(drive, obs, lo, hi, betas=None):
    B = select(drive, obs, betas)
    sl = slice(lo, hi)
    res = {k: rms_in(B[k][sl, 0], drive[sl]) for k in B}
    return min(res, key=res.get)


def flips(tl, lo, hi):
    seq = [c for t, c in tl if lo <= t < hi]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b), max(len(seq) - 1, 1)


def lag_of(tl, switch_t, want):
    for t, c in tl:
        if t >= switch_t and c == want:
            return t - switch_t
    return None


rng = np.random.RandomState(31)
drive = rng.randint(0, 2, T).astype(float)

sig = np.zeros(T)
sig[BOUNDS[0]:BOUNDS[1]] = 5e-2
obsA = piecewise_leak(drive, (0.1, 0.1, 0.1), BOUNDS) + rng.normal(0.0, 1.0, T) * sig
obsB = piecewise_leak(drive, (0.1, 0.3, 0.1), BOUNDS)

print("window %d, step %d, %d steps; switches at %d and %d\n" % (W, STEP, T, BOUNDS[0], BOUNDS[1]))

print("arm A  noise 0 -> 5e-2 -> 0, world unchanged")
tlA = timeline(drive, obsA)
stA = [segment_choice(drive, obsA, 0, BOUNDS[0]), segment_choice(drive, obsA, *BOUNDS),
       segment_choice(drive, obsA, BOUNDS[1], T)]
print("        steady choice per segment: " + " / ".join(stA))
for name, (lo, hi) in (("clean before", (0, BOUNDS[0])), ("noisy", BOUNDS),
                       ("clean after", (BOUNDS[1], T))):
    f, n = flips(tlA, lo, hi)
    print("        %-13s flips %2d / %3d windows (%.0f%%)" % (name, f, n, 100.0 * f / n))
print("        lag back to the clean choice: %s steps" % lag_of(tlA, BOUNDS[1], stA[2]))

print("\narm B  leak beta 0.1 -> 0.3 -> 0.1, coarse bank (no 0.3)")
tlB = timeline(drive, obsB)
stB = [segment_choice(drive, obsB, 0, BOUNDS[0]), segment_choice(drive, obsB, *BOUNDS),
       segment_choice(drive, obsB, BOUNDS[1], T)]
print("        steady choice per segment: " + " / ".join(stB))
print("        lag in: %s   lag out: %s"
      % (lag_of(tlB, BOUNDS[0], stB[1]), lag_of(tlB, BOUNDS[1], stB[2])))

print("\narm C  same, with 0.3 added to the bank")
tlC = timeline(drive, obsB, betas=FINE)
stC = [segment_choice(drive, obsB, 0, BOUNDS[0], FINE),
       segment_choice(drive, obsB, *BOUNDS, FINE),
       segment_choice(drive, obsB, BOUNDS[1], T, FINE)]
print("        steady choice per segment: " + " / ".join(stC))
print("        lag in: %s   lag out: %s"
      % (lag_of(tlC, BOUNDS[0], stC[1]), lag_of(tlC, BOUNDS[1], stC[2])))

print("\nR1  arm C tracks to the true member and returns : %s (%s)"
      % ("ok" if stC == ["unleak0.1", "unleak0.3", "unleak0.1"] else "MISS", " / ".join(stC)))
print("R2  lag is set by the window                   : %s (%s/%s vs W=%d)"
      % ("ok" if (lag_of(tlC, BOUNDS[0], stC[1]) or 10 ** 9) <= W else "MISS",
         lag_of(tlC, BOUNDS[0], stC[1]), lag_of(tlC, BOUNDS[1], stC[2]), W))
print("R3  coarse bank lands on the nearest member    : %s (%s)"
      % ("ok" if stB[1] == "unleak0.2" else "MISS", stB[1]))
fA, nA = flips(tlA, *BOUNDS)
print("R4  arm A stable, but on a coarse inverse      : %s (%.0f%% flips, fixed point %s)"
      % ("ok" if fA / nA < 0.1 and stA[1] != "unleak0.1" else "MISS", 100.0 * fA / nA, stA[1]))
