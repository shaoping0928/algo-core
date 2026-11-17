"""Where does the required precision come from?

(b) showed a bang-bang policy derived from the rules needs almost exact pending actions: 2%
error already costs 95% of the score.  That looked like a statement about the memory, but the
noisy arm showed it is a statement about the *plan*: one pending bit moves the predicted
crossing point by F*(H-lag), which at H=12 is 0.18 against a gap of 0.34.  So the quantity
that should predict the tolerance is

    r = F * (H - 1) / GAP        one impulse integrated over the lookahead, in gap units

and a deadband does not help, because the error is larger than any deadband that leaves the
task solvable (that was measured: delta up to 0.02 changed nothing).

This scans the lookahead and the pending error rate together, and places the u-bank decode on
the resulting frontier.

Criteria written before the run:
    M0  control: at eps=0 the H=8 and H=10 arms reproduce the (b) sweep (254 and 1187)
    M1  there is a region of smooth decay (dynamic range), not a cliff: for at least the short
        arms the score must pass through intermediate values as eps grows
    M2  the tolerance (eps at which the score halves) increases as H decreases, and r orders
        the arms the same way
    M3  the decoded arm is placed on the frontier: report the H at which its own error rate
        (~0.3) is something the controller tolerates
"""
import numpy as np

from latent_world import (GAP, SEEDS, fit_decoder, make_policy, recover, run)

LAG = 3
HORS = (4, 6, 8, 10, 12, 14)
EPS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35)
NSEED = 10

r = recover(LAG)
print("LAG=%d  recovered vy' = %s   (val %.2e)" % (LAG, r["form"], r["res"]))
print("gap %.2f   one impulse integrated over a lookahead H displaces the end point by"
      " F*(H-1)\n" % GAP)

print("  score vs pending-bit error rate (true pending, bits flipped), %d seeds" % NSEED)
print("    H     r     " + "".join("%8s" % ("eps%.2f" % e) for e in EPS)
      + "   decoded   decoded/exact")
table = {}
for hor in HORS:
    pol = make_policy(r, LAG, hor)
    row, acts = [], []
    for eps in EPS:
        rng = np.random.RandomState(77)
        sc = []
        for s in SEEDS[:NSEED]:
            t, a, _ = run(LAG, s, pol, "exact" if eps == 0.0 else "noisy", eps=eps, rng=rng)
            sc.append(t)
            if eps == 0.0:
                acts.extend(a)
        row.append(float(np.mean(sc)))
    dec, rep = fit_decoder(acts, LAG)
    decs, derr = [], []
    for s in SEEDS[:NSEED]:
        t, _, e = run(LAG, s, pol, "decoded", dec)
        decs.append(t)
        derr.append(e)
    table[hor] = (row, float(np.mean(decs)), float(np.mean(derr)), rep)
    rr = r["F"] * (hor - 1) / GAP
    print("   %2d  %.3f  " % (hor, rr) + "".join("%8.0f" % x for x in row)
          + "   %7.0f      %.3f" % (np.mean(decs), np.mean(decs) / row[0]))

print("\nM0  control: H=8 at eps=0 is 254 and H=10 is 1187 in the (b) sweep")
print("    here:    H=8 %0.0f,  H=10 %0.0f" % (table[8][0][0], table[10][0][0]))

print("\nM2  tolerance: eps at which the score falls to half of its eps=0 value")
print("     H      r     eps_half   (interpolated)")
for hor in HORS:
    row = table[hor][0]
    half = 0.5 * row[0]
    e50 = None
    for i in range(1, len(EPS)):
        if row[i] <= half <= row[i - 1] and row[i - 1] > row[i]:
            f = (row[i - 1] - half) / (row[i - 1] - row[i])
            e50 = EPS[i - 1] + f * (EPS[i] - EPS[i - 1])
            break
    print("    %2d   %.3f     %s" % (hor, r["F"] * (hor - 1) / GAP,
                                     "> %.2f" % EPS[-1] if e50 is None else "%.3f" % e50))

print("\nM3  the decoded tap (its own error rate is about %.2f) placed on the frontier"
      % np.mean([table[h][2] for h in HORS]))
for hor in HORS:
    row, decs, derr, rep = table[hor]
    print("    H=%2d  exact %6.0f  decoded %6.0f  decoded/exact %.3f  (decode error %.2f)"
          % (hor, row[0], decs, decs / row[0], derr))
best_h = max(HORS, key=lambda h: table[h][0][0] * min(1.0, table[h][1] / table[h][0][0]))
print("\n    best exact/decode compromise by exact*decoded score: H=%d (exact %.0f, decoded %.0f)"
      % (best_h, table[best_h][0][0], table[best_h][1]))
