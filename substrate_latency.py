"""Is a 70% tap good enough?  Latency as the task that actually needs addressing.

A.9 showed the leaky bank cannot address a tap exactly; the tap measurement showed it still
carries about 70% of one.  Exactness is not the question a controller asks, so this builds the
world where the tap is load-bearing:

    Flappy with an L-frame action latency.  A flap issued now lands L frames from now, so a
    planner has to know which flaps are still in flight.  The arms differ only in where that
    knowledge comes from:

        exact     a shift register of the last L issued actions
        decoded   the u bank, linearly decoded, thresholded at the observed flap rate
        none      the pending slots are assumed empty
        noisy     the true pending, with each bit flipped independently (tolerance probe)

Criteria written before the run:
    L0  at LAG=0 with the same frame cap (1200) the pipeline reproduces rule_policy's 1133.1
        exactly -- control; without it nothing below counts
    L1  every transition is recovered exactly from its true lagged column (<=1e-12), and the
        printed form is the columns actually selected, not a template
    L2  the horizon is swept per lag, so a low score is not mistaken for a fact about latency
    L3  the decode is scored by balanced accuracy against the observed flap rate (0.5 is the
        wrong chance level for a sparse action stream)
    L4  the decoded arm is reported as a fraction of the exact arm, AND the controller's own
        tolerance is measured (noisy arm), so a gap can be attributed to the memory or to the
        controller instead of being asserted
"""
import numpy as np

from latent_world import CAP, SEEDS, fit_decoder, make_policy, recover, run

R = {lag: recover(lag) for lag in range(4)}
for lag in sorted(R):
    print("LAG=%d  recovered: vy' = %s   (val %.2e)" % (lag, R[lag]["form"], R[lag]["res"]))
print("\nL1  every transition exact: %s"
      % ("ok" if all(R[k]["res"] <= 1e-12 for k in R) else "MISS"))

HORS = (8, 10, 12, 14, 16, 20)
print("\n  horizon sweep, exact arm (10 seeds) -- is a low score a fact about latency?")
print("    LAG  " + "".join("%8d" % h for h in HORS))
best = {}
for lag in range(4):
    row = []
    for hor in HORS:
        pol = make_policy(R[lag], lag, hor)
        row.append(np.mean([run(lag, s, pol, "exact")[0] for s in SEEDS[:10]]))
    best[lag] = HORS[int(np.argmax(row))]
    print("     %d   " % lag + "".join("%8.0f" % x for x in row) + "   best H=%d" % best[lag])

nof = {}
for lag in range(4):
    pol = make_policy(R[lag], lag, best[lag])
    nof[lag] = np.mean([run(lag, s, pol, "exact", flap=False)[0] for s in SEEDS])

print("\n  arms at the best H for each lag (20 seeds)")
print("    LAG   H   exact   no-flap    none   decoded   decoded/exact   pending errors")
tab = {}
for lag in range(4):
    hor = best[lag]
    pol = make_policy(R[lag], lag, hor)
    sc_e, acts = [], []
    for s in SEEDS:
        t, a, _ = run(lag, s, pol, "exact")
        sc_e.append(t)
        acts.extend(a)
    sc_n = [run(lag, s, pol, "none")[0] for s in SEEDS]
    dec, rep = fit_decoder(acts, lag)
    sc_d, err = [], []
    for s in SEEDS:
        t, _, e = run(lag, s, pol, "decoded", dec)
        sc_d.append(t)
        err.append(e)
    tab[lag] = (np.mean(sc_e), np.mean(sc_n), np.mean(sc_d), np.mean(acts), rep, np.mean(err))
    print("     %d  %3d  %6.1f   %6.1f  %6.1f    %6.1f        %.3f          %.3f"
          % (lag, hor, np.mean(sc_e), nof[lag], np.mean(sc_n), np.mean(sc_d),
             np.mean(sc_d) / np.mean(sc_e), np.mean(err)))

print("\n  L3  decode, balanced accuracy (chance 0.5; the raw base rate is the always-zero score)")
for lag in range(4):
    if tab[lag][4]:
        print("      LAG=%d  " % lag + "  ".join("lag%d bal %.2f (acc %.2f, base %.2f)"
                                               % (k, d, a, b) for k, a, b, d, _r in tab[lag][4]))

pol0 = make_policy(R[0], 0, best[0])
me0 = np.mean([run(0, s, pol0, "exact", cap=1200)[0] for s in SEEDS])
print("\nL0  LAG=0 control at rule_policy's frame cap (1200): %.1f  (rule_policy 1133.1) -> %s"
      % (me0, "ok" if 0.9 * 1133 <= me0 <= 1.1 * 1133 else "MISS"))
print("    with the cap raised to %d the same policy reaches %.1f -- 1133 was a floor"
      % (CAP, tab[0][0]))

print("\n  L4  the controller's own tolerance to wrong pending bits (true pending, bits flipped)")
print("      LAG=3, eps = 0.00 .. 0.30, 12 seeds")
pol3 = make_policy(R[3], 3, best[3])
for eps in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
    rng = np.random.RandomState(77)
    sc = [run(3, s, pol3, "noisy", eps=eps, rng=rng)[0] for s in SEEDS[:12]]
    print("      eps %.2f   mean %7.1f   frac of exact %.3f"
          % (eps, np.mean(sc), np.mean(sc) / tab[3][0]))

print("\n      same, with a hysteresis band on the decision (delta)")
for dl in (0.0, 0.002, 0.005, 0.010, 0.020):
    polh = make_policy(R[3], 3, best[3], delta=dl)
    sc_e = [run(3, s, polh, "exact")[0] for s in SEEDS[:12]]
    ach = []
    for s in SEEDS:
        ach.extend(run(3, s, polh, "exact")[1])
    dech, rep = fit_decoder(ach, 3)
    sc_d = [run(3, s, polh, "decoded", dech)[0] for s in SEEDS[:12]]
    print("      delta %.3f   exact %7.1f   decoded %7.1f   decoded/exact %.3f"
          % (dl, np.mean(sc_e), np.mean(sc_d), np.mean(sc_d) / np.mean(sc_e)))
