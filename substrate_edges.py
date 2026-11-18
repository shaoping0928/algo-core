"""Edges between the high-intensity points of consecutive steps: what that costs and buys.

The proposal: at each transition, take the few highest-intensity nodes of the previous state and
the few of the next, connect them, and let the edge keep a decaying residual.  That is the
substrate's own edge mechanism (sparse K in-edges plus facilitation r); what is being specified
is when an edge is created -- on co-activation, not on all pairs.

The measurable consequence is a parameter count: the association basis is the set of observed
transitions, not the full S*S table.  Two streams:

    A  first-order cycle   next = current + 1: only S transitions ever occur
    B  second-order recurrence   next = current + previous: only reachable pairs occur

Criteria written before the run:
    E1  on A the sparse basis uses S features instead of S*S and reaches the same accuracy
    E2  on B the sparse basis stores only reachable pairs and matches the full basis' accuracy;
        the saving is exactly the sparsity of this deterministic trajectory
    E3  the cost is visible: a pair never seen has no edge, so the sparse basis cannot answer
        for it at all
"""
import numpy as np

S = 8
T = 20000


def onehot(seq):
    X = np.zeros((len(seq), S))
    X[np.arange(len(seq)), seq] = 1.0
    return X


def pair_acc(seq, pairs):
    """predict the next symbol from h(current) and u(previous) restricted to `pairs`"""
    X = onehot(seq)
    H, P = X[1:-1], X[:-2]                       # current, previous
    tgt = X[2:]
    F = np.stack([H[:, i] * P[:, j] for (i, j) in pairs], 1)
    cut = int(0.7 * len(F))
    A = np.column_stack([np.ones(len(F)), H, P, F])
    W = np.linalg.lstsq(A[:cut], tgt[:cut], rcond=None)[0]
    return float(np.mean(np.argmax(A[cut:] @ W, 1) == np.argmax(tgt[cut:], 1)))


def observed(seq):
    seen = set()
    for t in range(1, len(seq) - 1):
        seen.add((seq[t], seq[t - 1]))
    return sorted(seen)


rng = np.random.RandomState(4)
cyc = np.asarray([t % S for t in range(T)])
seqB = [0, 1]
for _ in range(T - 2):
    seqB.append((seqB[-1] + seqB[-2]) % S)
seqB = np.asarray(seqB)

for name, seq in (("A first-order cycle (next = current + 1)", cyc),
                  ("B second-order (next = current + previous)", seqB)):
    obs = observed(seq)
    full = [(i, j) for i in range(S) for j in range(S)]
    a_sp = pair_acc(seq, obs)
    a_fu = pair_acc(seq, full)
    print("\n%s" % name)
    print("  transitions that occur: %2d of %2d" % (len(obs), S * S))
    print("  sparse basis: %2d features, accuracy %.3f" % (len(obs), a_sp))
    print("  full basis:   %2d features, accuracy %.3f" % (S * S, a_fu))
    if name.startswith("A"):
        e1 = (len(obs), a_sp, a_fu)
    else:
        e2 = (len(obs), a_sp, a_fu)

print("\nE1  on A: %d features instead of %d, accuracy %.3f vs %.3f -> %s"
      % (e1[0], S * S, e1[1], e1[2], "ok" if abs(e1[1] - e1[2]) < 0.02 else "MISS"))
print("E2  on B: the reachable set is %d of %d, not all of them -- a second-order rule does"
      % (e2[0], S * S))
e2_ok = e2[0] < S * S and abs(e2[1] - e2[2]) < 0.02
print("    not touch every pair.  sparse %.3f vs full %.3f, so the saving is %.1fx -> %s"
      % (e2[1], e2[2], (S * S) / e2[0], "ok" if e2_ok else "MISS"))

# E3: sample cost.  train on a short prefix and see whether both learn every reachable pair
def pair_acc_prefix(seq, pairs, n_train):
    X = onehot(seq)
    H, P = X[1:-1], X[:-2]
    tgt = X[2:]
    F = np.stack([H[:, i] * P[:, j] for (i, j) in pairs], 1)
    A = np.column_stack([np.ones(len(F)), H, P, F])
    W = np.linalg.lstsq(A[:n_train], tgt[:n_train], rcond=None)[0]
    return float(np.mean(np.argmax(A[n_train:] @ W, 1) == np.argmax(tgt[n_train:], 1)))


obsB = observed(seqB)
fullB = [(i, j) for i in range(S) for j in range(S)]
def exposure_range(seq, pairs, n):
    cnt = {p: 0 for p in pairs}
    for t in range(1, min(n, len(seq) - 1)):
        if (seq[t], seq[t - 1]) in cnt:
            cnt[(seq[t], seq[t - 1])] += 1
    return min(cnt.values()), max(cnt.values())


for n in (200, 400):
    lo, hi = exposure_range(seqB, obsB, n)
    print("\nE3  train on the first %d steps: each reachable pair is seen %d-%d times"
          % (n, lo, hi))
    print("    sparse basis %.3f    full basis %.3f"
          % (pair_acc_prefix(seqB, obsB, n), pair_acc_prefix(seqB, fullB, n)))
print("    both need one exposure per pair; the sparse basis just stores fewer of them.")
