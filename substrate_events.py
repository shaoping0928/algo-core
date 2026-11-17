"""Differences, not states: associate a difference with the difference it causes later.

The framing under test: "no flap -> it keeps descending; flap (a difference) -> the height
rises a few frames later (a difference); the system should learn these and propose hypotheses."

That is a different object from a state->state transition.  If the world is one constant drift
plus sparse events, then the residual of the drift has exactly two values, and the whole task
is to associate the issuing of an event with the consequence that arrives later.  Once the
association is learned, the record you need is not the history of inputs (O(lag) bits, needs a
shift register, decays and has to be decoded) but the set of outstanding consequences
(O(#in flight), exact, and it never has to be perceived because it was committed).

Criteria written before the run:
    E1  the residual of "constant drift" takes exactly two values: 0, and one magnitude F.
        Nothing else -- no small spread, no third value.
    E2  the lag is found by a determinacy test on that residual, not supplied: over k=0..6 only
        one k makes the within-group spread vanish, every other k gives a definite non-zero
        spread.
    E3  with a schedule of outstanding consequences the delayed Flappy scores what the true
        queue scores (so the 0.84 decode was never necessary), the schedule never disagrees
        with the world, and it holds ~flap_rate*lag entries rather than lag.
    E4  the same works when the cue is external and can only be remembered: a detected and
        scheduled cue predicts exactly, while the leaky bank (A.9) does not.
"""
import numpy as np

G, F, LAG = 0.0016, 0.020, 3


# ------------------------------------------------------------------ E1, E2: the residual
def delayed_world(n=8000, seed=5):
    rng = np.random.RandomState(seed)
    y, vy = 0.5, 0.0
    rows = []
    for t in range(n):
        rec = dict(y=y, vy=vy, a=float(int(rng.rand() < 0.5)))
        hist = [r["a"] for r in rows[-8:]][::-1]          # a(t-1), a(t-2), ...
        for k in range(1, 9):
            rec["a_lag%d" % k] = float(hist[k - 1]) if len(hist) >= k else 0.0
        fired = rec["a_lag%d" % LAG] if t >= LAG else 0.0
        nvy = vy - G + F * fired
        ny = y + nvy
        rec["y2"], rec["vy2"] = ny, nvy
        rows.append(rec)
        y, vy = ny, nvy
        if not (0.0 <= y <= 1.0):
            y, vy = 0.5, 0.0
    return rows


rows = delayed_world()
dvy = np.array([r["vy2"] - r["vy"] for r in rows])
drift = float(np.median(dvy))                 # the part that repeats: the constant drift
res = dvy - drift

near0 = int(np.sum(np.abs(res) < 1e-12))
nearF = int(np.sum(np.abs(res - F) < 1e-12))
other = len(res) - near0 - nearF
print("E1  the world as drift + events")
print("    drift (median of dvy)      %.12f" % drift)
print("    residual values            ~0 : %4d      ~%.3f : %4d      anything else : %d"
      % (near0, F, nearF, other))
print("    the residual is the event stream: %s\n"
      % ("ok" if other == 0 and nearF > 0 else "MISS"))

print("E2  which lag?  determinacy on the residual, k = 0..6")
print("      k   rows(a=1)   spread inside the groups   verdict")
found = []
for k in range(0, 7):
    key = "a" if k == 0 else "a_lag%d" % k
    g1 = res[[i for i, r in enumerate(rows) if r[key] > 0.5]]
    g0 = res[[i for i, r in enumerate(rows) if r[key] <= 0.5]]
    spread = max(float(np.ptp(g1)) if len(g1) else 0.0,
                 float(np.ptp(g0)) if len(g0) else 0.0)
    if spread <= 1e-12:
        found.append(k)
    print("      %d   %6d      %.3e                %s"
          % (k, len(g1), spread, "determined" if spread <= 1e-12 else ""))
print("    E2: exactly one lag is determined: %s (found %s, truth %d)\n"
      % ("ok" if found == [LAG] else "MISS", found, LAG))


# ------------------------------------------------------------------ E3: the schedule
from latent_world import SEEDS, Schedule, latent, make_policy, recover  # noqa: E402


r3 = recover(LAG)
pol = make_policy(r3, LAG, 14)
print("E3  closed loop on the delayed Flappy, pending state committed instead of decoded")
print("    rule: %s" % r3["form"])

sc, held, bad, total = [], [], 0, 0
for s in SEEDS:
    env = latent[LAG](s)
    act, target = pol
    sch = Schedule(LAG, r3["F"])
    issued = []
    t, prev = 0, 0
    while not env.dead:
        sch.drop_before(t)                      # already fired
        model = 1 if t in sch.due else 0        # what the schedule says fires right now
        truth = issued[t - LAG] if t >= LAG else 0
        bad += int(model != truth)
        total += 1
        pend = sch.upcoming(t)                  # fires at t .. t+lag-1
        a = act(env.y, env.vy, target(env), pend, prev)
        sch.issue(t, a)                         # committed now, fires at t+lag
        prev = a
        issued.append(a)
        env.step(a)
        t += 1
        held.append(sch.held())
    sc.append(env.t)

print("    score with the schedule %.1f   (shift register 4886.9, no memory 16.5, decoded 185.9)"
      % np.mean(sc))
print("    entries held, mean %.2f   (a %d-slot register holds %d; the register is 1x3,"
      " the schedule is only non-empty while something is in flight)"
      % (np.mean(held), LAG, LAG))
print("    schedule vs world disagreements: %d / %d steps -> %s"
      % (bad, total, "ok" if bad == 0 else "MISS"))


# ------------------------------------------------------------------ E4: an external cue
def cue_world(n=8000, seed=9):
    rng = np.random.RandomState(seed)
    y, vy = 0.5, 0.0
    cue_hist, rows = [], []
    for t in range(n):
        c = int(rng.rand() < 0.35)
        cue_hist.append(c)
        rec = dict(y=y, vy=vy, cue=float(c))
        for k in range(1, 5):
            rec["c_lag%d" % k] = float(cue_hist[-1 - k]) if len(cue_hist) > k else 0.0
        fired = cue_hist[-1 - LAG] if len(cue_hist) > LAG else 0
        nvy = vy - G + F * fired
        ny = y + nvy
        rec["y2"], rec["vy2"] = ny, nvy
        rows.append(rec)
        y, vy = ny, nvy
        if not (0.0 <= y <= 1.0):
            y, vy = 0.5, 0.0
    return rows


crows = cue_world()
cdvy = np.array([r["vy2"] - r["vy"] for r in crows])
cdrift = float(np.median(cdvy))
cres = cdvy - cdrift
BETAS = np.array([0.5, 0.2, 0.1, 0.05, 0.02])
u = np.zeros(len(BETAS))
U = []
for r in crows:
    u = (1.0 - BETAS) * u + BETAS * r["cue"]
    U.append(u.copy())
U = np.asarray(U)
cut = int(0.7 * len(cdvy))

print("\nE4  a cue that is only observable, and only pays off LAG frames later")

cfound = None
for k in range(1, 5):
    g1 = cres[[i for i, r in enumerate(crows) if r["c_lag%d" % k] > 0.5]]
    g0 = cres[[i for i, r in enumerate(crows) if r["c_lag%d" % k] <= 0.5]]
    spread = max(float(np.ptp(g1)) if len(g1) else 0.0, float(np.ptp(g0)) if len(g0) else 0.0)
    if spread <= 1e-12:
        cfound = k
print("    lag found by determinacy on the residual: %s (truth %d)" % (cfound, LAG))

F_hat = float(np.max(np.abs(cres)))
pred = np.array([cdrift + F_hat * (1.0 if (i >= cfound and crows[i - cfound]["cue"] > 0.5)
                                  else 0.0) for i in range(len(crows))])
res_sched = float(np.abs(pred - cdvy).max())

X = np.concatenate([np.ones((len(crows), 1)),
                    np.asarray([[r["y"], r["vy"]] for r in crows]), U], 1)
w = np.linalg.lstsq(X[:cut], cdvy[:cut], rcond=None)[0]
res_leaky = float(np.abs(X[cut:] @ w - cdvy[cut:]).max())

X2 = np.concatenate([np.ones((len(crows), 1)),
                     np.asarray([[r["y"], r["vy"], r["cue"]] for r in crows])], 1)
w2 = np.linalg.lstsq(X2[:cut], cdvy[:cut], rcond=None)[0]
res_now = float(np.abs(X2[cut:] @ w2 - cdvy[cut:]).max())

print("    detected + scheduled              residual %.2e" % res_sched)
print("    the same cue through 5 leaky traces  residual %.2e" % res_leaky)
print("    only the current cue value           residual %.2e" % res_now)

print("\nE1 %s   E2 %s   E3 %s   E4 %s"
      % ("ok" if other == 0 else "MISS",
         "ok" if found == [LAG] else "MISS",
         "ok" if bad == 0 else "MISS",
         "ok" if (cfound == LAG and res_sched <= 1e-12 and res_leaky > 1e-6) else "MISS"))
