"""Delayed recall: an instrument whose gate holds by construction.

Every world I built had to be checked for degeneracy, and each check failed for a
different design detail.  This task needs no such check: a symbol is shown once,
then the input is silent for d steps, then the system must report the symbol.  A
memoryless system sees zeros at the report step, so it cannot do better than
chance.  Memory is required by the definition of the task, not by my tuning.

    episode:  [ s ] [ 0 ] x d [ report ]
    symbols:  8, uniform
    error:    only at the report step, e = onehot(s) - softmax(Wo h)
    learning: delta rule on Wo and on the recurrent weights (local, signed)
    measure:  report accuracy vs d      -> d_max is a capacity, comparable across
              mechanisms

Arms (the name is the contract -- see `learn_w` / `learn_o` below):
    frozen        no plasticity at all (the gate: must stay at chance for all d)
    readout       only Wo learns
    readout+umem  only Wo learns, decoded from the activation buffer u
    all           Wo and the recurrent weights learn
    all+residue   both learn, plus per-edge transition residue (facilitation)
    all+assoc     both learn, plus sparse one-shot associative memory (F role)
    all+umem      both learn, decoded from u
    all+typed     both learn, with the E/I/F population split (needs OMNI_TYPES=1)
    all+typed+residue  as above, facilitation restricted to E edges
    all+split     both learn, plus node splitting (a failing node's incoming
                  edges are copied with perturbation into a retired slot)

    python recall_world.py [episodes_per_d] [units] [split 0|1]
    env: OMNI_UMEM_BETA OMNI_UMEM_NOISE OMNI_RES_ALPHA/BETA/GAIN
         OMNI_TYPES OMNI_KAPPA OMNI_FBOOST OMNI_ETA_F OMNI_DECAY_F
         OMNI_ARMS (comma list, to run a subset)
"""
import math
import os
import sys
import time

import numpy as np

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SPLIT = (sys.argv[3] if len(sys.argv) > 3 else "0") == "1"
K = 16
TOP_A = 32
S = int(os.environ.get("OMNI_SYMS", "8"))
D_LIST = [1, 2, 4, 8, 16, 32]
ETA = 0.03
GAMMA = float(os.environ.get("OMNI_TRACE_GAMMA", "0.9"))
ETA_T = float(os.environ.get("OMNI_TRACE_ETA", "0.05"))
BETA = float(os.environ.get("OMNI_UMEM_BETA", "0.1"))   # 1/beta = memory time constant
UNOISE = float(os.environ.get("OMNI_UMEM_NOISE", "0.0"))  # a real memory has noise
# Per-edge residue: a transition that was just used stays easier to take.  This
# lives on the edge, not on the node, and it biases selection rather than
# supplying a value to the readout.
RES_ALPHA = float(os.environ.get("OMNI_RES_ALPHA", "0.3"))    # facilitation per use
RES_BETA = float(os.environ.get("OMNI_RES_BETA", "0.02"))     # decay per step
RES_GAIN = float(os.environ.get("OMNI_RES_GAIN", "2.0"))      # boost of a fresh edge
# Built-in node types.  These are not learned and not chosen per task: the
# population is partitioned once, by index, into
#     E  excitatory    residue (facilitation) acts only on E sources
#     I  inhibitory    their activation feeds one divisive pool on all drives
#     F  fitting       their readout columns and incoming edges learn faster
# which gives the substrate three functional roles without a layer structure.
TYPES = os.environ.get("OMNI_TYPES", "0") == "1"
KAPPA = float(os.environ.get("OMNI_KAPPA", "1.0"))            # inhibition strength
FBOOST = float(os.environ.get("OMNI_FBOOST", "1.0"))          # extra lr for F
# Fast associative memory (the F role, second version).  One-shot Hebbian outer
# products on the existing sparse wiring: M_ik <- M_ik + eta_f * h_i * h_src(ik).
# During silence the recalled pattern is fed back as drive, so the association
# completes the pattern instead of only scaling it.
ETA_F = float(os.environ.get("OMNI_ETA_F", "0.5"))
DECAY_F = float(os.environ.get("OMNI_DECAY_F", "0.02"))
# Hard top-A makes every below-cut unit exactly zero, so no memory can restart a
# pattern once it falls out of the set.  SOFT > 0 keeps those units at a small
# fraction of their value instead of killing them.
SOFT = float(os.environ.get("OMNI_SOFT", "0.0"))
# Vector error (default) versus a scalar reward with a baseline.  The brain mostly
# gets the scalar; we mostly get the vector, and the difference is measurable.
SCALAR = os.environ.get("OMNI_FEEDBACK", "vector") == "scalar"
ETA_S = float(os.environ.get("OMNI_ETA_S", "0.5"))
HNOISE = float(os.environ.get("OMNI_HNOISE", "0.0"))   # noise on the drive
FRAC_E, FRAC_I = 0.70, 0.20                                   # rest is F
SPLIT_EVERY = 400
SEED = int(os.environ.get("OMNI_SEED", "21"))

rng = np.random.RandomState(SEED)
Pin = rng.normal(0, 1.0, size=(N, S)).astype(np.float32)
src = rng.randint(0, N, size=(N, K)).astype(np.int32)
w0 = rng.normal(0, 0.05, size=(N, K)).astype(np.float32)
Wo0 = np.zeros((S, N), dtype=np.float32)
ALIVE0 = np.zeros(N, dtype=bool)
ALIVE0[:int(0.85 * N)] = True
n_split = [0]
# 0 = E, 1 = I, 2 = F
NT = np.zeros(N, dtype=np.int8)
NT[int(FRAC_E * N):int((FRAC_E + FRAC_I) * N)] = 1
NT[int((FRAC_E + FRAC_I) * N):] = 2
N_I = max(1, int(np.sum(NT == 1)))
EMASK = (NT[src] == 0)                      # residue is an E-edge property
FB = np.where(NT == 2, 1.0 + FBOOST, 1.0).astype(np.float32)


def sparsify(v, k, floor=0.0):
    thr = np.partition(np.abs(v), -k)[-k]
    keep = np.where(np.abs(v) >= thr, 1.0, floor)
    return (v * keep).astype(np.float32)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def split(st, alive, erru, tr, u, res, M, h, rng):
    w, src_, Wo = st["w"], st["src"], st["Wo"]
    live = np.where(alive)[0]
    dead = np.where(~alive)[0]
    if not len(dead):
        weakest = live[np.argmin(np.abs(Wo[:, live]).sum(axis=0))]
        alive[weakest] = False
        Wo[:, weakest] = 0.0
        dead = np.array([weakest])
        live = np.where(alive)[0]
    j = dead[0]
    i = live[np.argmax(erru[live])]

    # A recycled slot must not carry traces or associations from its former identity.
    old_source_edges = src_ == j
    tr[old_source_edges] = 0.0
    res[old_source_edges] = 0.0
    M[old_source_edges] = 0.0
    tr[j] = 0.0
    res[j] = 0.0
    M[j] = 0.0
    u[j] = 0.0
    h[j] = 0.0

    src_[j] = src_[i]
    w[j] = w[i] + rng.normal(0, 0.02, size=K).astype(np.float32)
    Wo[:, j] = 0.0
    alive[j] = True
    erru[j] = erru[i]
    n_split[0] += 1


def run(arm, d, episodes, probe=None):
    st = dict(w=w0.copy(), src=src.copy(), Wo=Wo0.copy(), h=np.zeros(N, np.float32))
    alive = ALIVE0.copy()
    erru = np.zeros(N, dtype=np.float32)
    split_rng = np.random.RandomState(SEED + 104729 * d)
    tr = np.zeros((N, K), dtype=np.float32)   # eligibility trace of input activity
    u = np.zeros(N, dtype=np.float32)         # activation intensity (memory carrier)
    res = np.zeros((N, K), dtype=np.float32)  # per-edge transition residue
    M = np.zeros((N, K), dtype=np.float32)    # fast associative memory (F role)
    use_res = "residue" in arm
    use_trace = arm.endswith("trace") or "rtrace" in arm
    use_umem = "umem" in arm
    use_typed = TYPES and "typed" in arm
    use_assoc = "assoc" in arm
    w, src_, Wo, h = st["w"], st["src"], st["Wo"], st["h"]
    # Every arm at the same d must see the same symbol sequence, otherwise the
    # arm-to-arm differences include the sampling noise of the stimuli and the
    # comparison is not paired.  (Measured: two identical arms differed by 5.5pt
    # at d=4 purely from stream position.)
    r = np.random.RandomState(SEED + 7919 * d)
    # Arm names are a contract, so classify them explicitly rather than by
    # substring: a coarsened rule here silently turns the readout control into a
    # full-plasticity arm, which is how a control stops being a control.
    learn_w = arm.startswith("all")
    learn_o = arm != "frozen"
    correct = 0
    keep = [0.0]
    pool_acc = [0.0, 0]        # for the inhibition-magnitude check
    base = [0.5]                # reward baseline
    scalar_eps = [0]
    nz_acc = [0.0, 0]          # fraction of units that are non-zero
    t = 0
    next_split = SPLIT_EVERY
    for ep in range(episodes):
        s = r.randint(0, S)
        hist = []          # per-episode activity, for the backward trace
        if use_umem:
            u[:] = 0.0     # episode boundary: otherwise the previous symbol's
                           # trace leaks in and short delays are penalised
        if use_assoc:
            M[:] = 0.0
        # one stimulus step
        for step in range(d + 2):
            if step == 0:
                o = np.zeros(S, np.float32)
                o[s] = 1.0
            else:
                o = np.zeros(S, np.float32)
            # at the report step, train and score
            if step == d + 1:
                hh = u if use_umem else h
                if probe is not None:
                    b = probe.setdefault(s, [])
                    if len(b) < 20:
                        b.append(hh.copy())
                z0 = Wo @ hh
                p = softmax(z0)
                correct += int(np.argmax(p) == s)
                if learn_o:
                    e = o * 0.0
                    e[s] = 1.0
                    e = e - p
                    if SCALAR:
                        # scalar feedback instead of a vector error: the system
                        # learns only "better or worse than expected", which is
                        # the shape of a dopamine / climbing-fibre signal.
                        act = r.choice(S, p=p)
                        rew = 1.0 if act == s else 0.0
                        adv = rew - base[0]
                        base[0] = 0.95 * base[0] + 0.05 * rew
                        Wo += ETA_S * adv * np.outer(np.eye(S, dtype=np.float32)[act] - p, hh)
                        scalar_eps[0] += 1
                    else:
                        Wo += ETA * np.outer(e, hh) * (FB[None, :] if use_typed else 1.0)
                    if learn_w:
                        g = Wo.T @ (-e)
                        lr = FB[:, None] if use_typed else 1.0
                        w += ETA * lr * h[src_] * g[:, None] - 1e-5 * w
                        np.clip(w, -0.5, 0.5, out=w)
                        if use_trace:
                            # credit the connections that were active while the
                            # symbol was being held, not only at the report step
                            w += ETA_T * tr * g[:, None]
                            np.clip(w, -0.5, 0.5, out=w)
                        if "rtrace" in arm and hist:
                            # backward tracking: walk the episode from the report
                            # back to the stimulus, weighting each step by how
                            # close it was to the error
                            acc = np.zeros((N, K), dtype=np.float32)
                            for k in range(len(hist)):
                                acc += (GAMMA ** (len(hist) - 1 - k)) * hist[k]
                            w += ETA_T * (acc / max(1, len(hist))) * g[:, None]
                            np.clip(w, -0.5, 0.5, out=w)
                        erru[:] = 0.99 * erru + 0.01 * np.abs(g)
                    if SPLIT and arm == "all+split" and t >= next_split:
                        split(st, alive, erru, tr, u, res, M, h, split_rng)
                        next_split += SPLIT_EVERY
            tr = GAMMA * tr + h[src_]
            gain = 1.0 + (RES_GAIN * res if use_res else 0.0)
            if use_res and use_typed:
                # facilitation is a property of E edges only
                gain = 1.0 + RES_GAIN * res * EMASK
            drive = (w * gain * h[src_]).sum(axis=1) / math.sqrt(K) + Pin @ o
            if use_assoc:
                # content-addressed recall: whatever co-fired recently pulls the
                # pattern back, so silence does not have to be survived passively
                drive = drive + (M * h[src_]).sum(axis=1) / math.sqrt(K)
            if use_typed:
                # one divisive inhibitory pool from the I population.  Divide by
                # the *active* count, not by the size of the I population: the
                # first version normalised by N_I=120 while only ~6 I nodes fire,
                # so the divisor was 1.004 and the mechanism was a no-op.
                pool = float(np.abs(h[NT == 1]).sum()) / max(1.0, TOP_A * FRAC_I)
                drive = drive / (1.0 + KAPPA * pool)
                pool_acc[0] += pool
                pool_acc[1] += 1
            if HNOISE:
                drive = drive + r.normal(0, HNOISE, size=N).astype(np.float32)
            h = sparsify(np.tanh(drive), TOP_A, SOFT)
            h[~alive] = 0.0
            st["h"] = h
            nz_acc[0] += float(np.count_nonzero(h))
            nz_acc[1] += 1
            if use_res:
                fired_e = (h[src_] != 0)
                res = np.where(fired_e, res + RES_ALPHA * (1.0 - res),
                               res * (1.0 - RES_BETA)).astype(np.float32)
            if use_umem:
                # activation intensity: a signed leaky copy with time constant
                # 1/beta.  This is the memory carrier; h itself decays in ~1 step.
                u = (1.0 - BETA) * u + BETA * h
                if UNOISE:
                    u = u + r.normal(0, UNOISE, size=N).astype(np.float32)
            if use_assoc:
                if use_typed:
                    # only the F population carries the association
                    mrow = (NT == 2).astype(np.float32)[:, None]
                    M = M * (1.0 - DECAY_F) + ETA_F * mrow * h[:, None] * h[src_]
                else:
                    M = M * (1.0 - DECAY_F) + ETA_F * h[:, None] * h[src_]
                np.clip(M, -1.0, 1.0, out=M)
            if "rtrace" in arm:
                hist.append(h[src_].copy())
            if step == d:                      # last blank: is the state still alive?
                keep[0] += float(np.abs(h).sum())
            t += 1
    return correct / episodes, keep[0] / max(1, episodes), \
        (pool_acc[0] / max(1, pool_acc[1])), (nz_acc[0] / max(1, nz_acc[1]))


t0 = time.time()
print("episodes/d %d  N %d  S %d  split %s" % (EPISODES, N, S, SPLIT))
# The contract in docs/status.md: the gate arm, the readout-only baselines, and
# one arm per mechanism.  A number without the readout baseline cannot separate
# "the encoding got sharper" from "the state is being held".
arms = ["frozen", "readout", "readout+umem", "all",
        "all+residue", "all+assoc", "all+umem"]
if TYPES:
    arms.insert(4, "all+typed")
    arms.append("all+typed+residue")
    arms.append("all+typed+assoc")
if SPLIT:
    arms.append("all+split")
if os.environ.get("OMNI_ARMS"):
    arms = [a for a in os.environ["OMNI_ARMS"].split(",") if a]
print("\n%-10s %s" % ("arm", "  ".join("d=%-3d" % d for d in D_LIST)))
for arm in arms:
    probe = {} if os.environ.get("OMNI_PROBE") == "1" else None
    out = [run(arm, d, EPISODES, probe if d == 8 else None) for d in D_LIST]
    accs = [a for a, _k, _p, _n in out]
    keep = [k for _a, k, _p, _n in out]
    pools = [p for _a, _k, p, _n in out]
    dens = [n for _a, _k, _p, n in out]
    print("%-14s %s  | state %s  | pool %s  | nz %s"
          % (arm, "  ".join("%.3f" % a for a in accs),
             " ".join("%.2f" % k for k in keep),
             " ".join("%.2f" % p for p in pools),
             " ".join("%.0f" % n for n in dens)))
    if probe and len(probe) > 1:
        # Is the state at report time actually different per symbol?  Between
        # symbol distance over within symbol spread.  A saturated fixed point
        # that ignores the stimulus reads ~0 here no matter how good accuracy is.
        keys = sorted(probe)
        means = {s: np.mean(probe[s], axis=0) for s in keys}
        within = np.mean([np.mean(np.linalg.norm(x - means[s]))
                          for s in keys for x in probe[s]])
        bd = [np.linalg.norm(means[a] - means[b])
              for i, a in enumerate(keys) for b in keys[i + 1:]]
        print("  %-12s state separability  between/within = %.2f   min pair %.2f"
              % ("", np.mean(bd) / max(within, 1e-9), min(bd)))
print("\nchance %.3f   %.1fs   splits %d"
      % (1.0 / S, time.time() - t0, n_split[0]))
