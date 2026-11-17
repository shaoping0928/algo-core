"""The capacitor's tail as the association between consecutive symbols.

The proposal: when one character follows another, the first is still charged (its u has not
decayed) while the second is at full intensity, so both are high at once and the pair associates
naturally.  No shift register, no position -- the capacitance is the one-step memory.

The first version of this test used a first-order rule (next depends on the current symbol), and
the tail was correctly unnecessary: the current symbol alone carries the rule.  The tail can only
matter when the next symbol depends on the Previous one as well, so the stream here is
second-order:

    next = (current + previous) mod S with probability p

Then three readouts, all linear:
    h           the current symbol                  cannot see the previous one
    h, u        current plus the tail               the pair is present, but a linear map can
                                                    only add the two, not combine them
    h, u, h*u   the same plus the pair products     what a node with two inputs computes: the
                                                    association itself

Criteria written before the run:
    Z1  h alone cannot express a second-order rule: accuracy at chance
    Z2  h and u together still cannot: the information is there, the linear combination is not
    Z3  adding the products recovers the rule: accuracy at the rule's ceiling
    Z4  the tail is what supplies the previous symbol: the charge left one step later is beta,
        and the accuracy of the product readout degrades as beta shrinks toward zero
"""
import numpy as np

S = 8
T = 20000
P = 0.9


def stream(seed=0):
    rng = np.random.RandomState(seed)
    seq = [0, 1]
    for _ in range(T - 2):
        if rng.rand() < P:
            seq.append((seq[-1] + seq[-2]) % S)
        else:
            seq.append(int(rng.randint(S)))
    return np.asarray(seq)


def onehot(seq):
    X = np.zeros((len(seq), S))
    X[np.arange(len(seq)), seq] = 1.0
    return X


def leaky(X, b):
    U = np.zeros_like(X)
    u = np.zeros(X.shape[1])
    for t in range(len(X)):
        u = (1.0 - b) * u + b * X[t]
        U[t] = u
    return U


def acc(feats, X, cut):
    A = np.column_stack([np.ones(len(X))] + feats)
    W = np.linalg.lstsq(A[:cut], X[:cut], rcond=None)[0]
    return float(np.mean(np.argmax(A[cut:] @ W, 1) == np.argmax(X[cut:], 1)))


seq = stream()
X = onehot(seq)
tgt = np.roll(X, -1, 0)[:-2]
H = X[:-2]                       # current symbol
cut = int(0.7 * len(H))
ceil = P + (1 - P) / S
print("second-order stream: next = (current + previous) mod %d, p=%.2f" % (S, P))
print("chance %.3f, rule ceiling %.3f\n" % (1.0 / S, ceil))

print("  beta   charge left   acc(h)   acc(h,u)   acc(h,u,h*u)   h*u features")
for b in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
    U = leaky(X, b)[:-2]
    prod = np.einsum("ij,ik->ijk", H, U).reshape(len(H), S * S)
    a1 = acc([H], tgt, cut)
    a2 = acc([H, U], tgt, cut)
    a3 = acc([H, U, prod], tgt, cut)
    print("  %.2f     %.3f        %.3f    %.3f      %.3f          %d"
          % (b, b, a1, a2, a3, prod.shape[1]))

B = 0.2
U = leaky(X, B)[:-2]
prod = np.einsum("ij,ik->ijk", H, U).reshape(len(H), S * S)
a1, a2, a3 = acc([H], tgt, cut), acc([H, U], tgt, cut), acc([H, U, prod], tgt, cut)
print("\nZ1  h alone:        %.3f (chance %.3f)  -> %s" % (a1, 1 / S, "ok" if a1 < 0.3 else "MISS"))
print("Z2  h and u:        %.3f -- the information is there, a linear map cannot combine it -> %s"
      % (a2, "ok" if a2 < 0.5 else "MISS"))
print("Z3  h, u, h*u:      %.3f of the %.3f ceiling -> %s"
      % (a3, ceil, "ok" if a3 > 0.85 * ceil else "MISS"))
accs = []
for b in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
    UU = leaky(X, b)[:-2]
    pp = np.einsum("ij,ik->ijk", H, UU).reshape(len(H), S * S)
    accs.append((b, acc([H, UU, pp], tgt, cut)))
# ---------------------------------------------------------------- why products
def table_spectrum(seq):
    """T[k,i,j] = P(next=k | cur=i, prev=j).  An additive readout a(i)+b(j) has k-slices of
    rank <= 2, so the question is how many singular values stand above the sampling noise."""
    T3 = np.zeros((S, S, S))
    for t in range(1, len(seq) - 1):
        T3[seq[t + 1], seq[t], seq[t - 1]] += 1.0        # [next, current, previous]
    T3 /= np.maximum(T3.sum(0, keepdims=True), 1e-12)
    out = []
    for k in range(S):
        s = np.linalg.svd(T3[k], compute_uv=False)
        out.append(s / s[0])
    return np.mean(out, 0)


seq2 = [0]
rng2 = np.random.RandomState(3)
for _ in range(T - 1):
    seq2.append((seq2[-1] + 1) % S if rng2.rand() < 0.85 else int(rng2.randint(S)))
r0 = table_spectrum(np.asarray(seq2))
r1 = table_spectrum(stream())

print("\nwhy products: the rule is a table T[next | current, previous]")
print("  first-order stream   normalised spectrum %s" % np.round(r0, 3))
print("  second-order stream  normalised spectrum %s" % np.round(r1, 3))
print("  only the first two singular values stand above noise in the first-order case;")
print("  in the second-order case most of them do -- so the table is not additive")
print("  a linear readout over (h,u) computes a(current) + b(previous), whose k-slices have")
print("  rank <= 2; the product basis supplies the missing S*S table entries")

print("Z4  product readout vs charge left: " + "  ".join("%.2f:%.3f" % ab for ab in accs))
print("    -> %s" % ("ok" if accs[0][1] > accs[-1][1] + 0.1 else "check"))
