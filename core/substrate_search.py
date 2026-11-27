"""Search operators for the rule layer: propose a combination, then test it.

`substrate_rules` tests the candidates a caller supplies.  This module supplies them, so the
choice of feature is no longer made by hand:

    basis()            a candidate set built from primitive columns --
                       v, v^2, |v|, sign(v), v*|v|, v*w, v*|w|, |v|*w
    greedy_determine() forward selection of the smallest set that determines a target,
                       fitted on 70% of the rows and validated on the other 30%

Two properties matter and both are measured:

    a hit is exact      validation residual at machine precision, not "small"
    a miss is visible   with the wrong candidate set the residual sits at a definite value
                        (e.g. 4.97e-3 for a quadratic drag world without vy*|vy|), which says
                        "the basis is missing a term", not "the world is hard"
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["basis", "greedy_determine"]


def basis(d: Dict[str, np.ndarray], names: Sequence[str], nonlinear: bool = True,
          lags: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    """Candidate features from primitive columns.

    `lags` names extra columns (already computed by the caller) that should be offered as
    candidates without being expanded -- this is how a delayed quantity enters the search.
    """
    B: Dict[str, np.ndarray] = {n: np.asarray(d[n], dtype=float) for n in names}
    for n in lags:
        if n in d:
            B[n] = np.asarray(d[n], dtype=float)
    if not nonlinear:
        return B
    for n in names:
        B["%s^2" % n] = B[n] ** 2
        B["|%s|" % n] = np.abs(B[n])
        B["sign(%s)" % n] = np.sign(B[n])
        B["%s*|%s|" % (n, n)] = B[n] * np.abs(B[n])
    for a, b in combinations(names, 2):
        B["%s*%s" % (a, b)] = B[a] * B[b]
        B["%s*|%s|" % (a, b)] = B[a] * np.abs(B[b])
        B["|%s|*%s" % (a, b)] = np.abs(B[a]) * B[b]
    return B


def greedy_determine(
    d: Dict[str, np.ndarray],
    target: str,
    names: Sequence[str],
    nonlinear: bool = True,
    lags: Sequence[str] = (),
    k_max: int = 4,
    tol: float = 1e-9,
    holdout: float = 0.3,
) -> Tuple[List[str], np.ndarray, float, float]:
    """Forward selection of the smallest feature set that determines `target`.

    Returns (chosen features, coefficients including the constant, validation residual,
    scale).  A feature is never chosen twice; the fit sees 70% of the rows and the reported
    residual is on the other 30%.
    """
    t = np.asarray(d[target], dtype=float)
    scale = max(1.0, float(np.abs(t).max()))
    B = {n: f for n, f in basis(d, names, nonlinear, lags).items() if n != target}
    n = len(t)
    cut = int((1.0 - holdout) * n)
    ones = np.ones(n)
    chosen: List[str] = []
    coef = np.zeros(1)
    tr = float(np.abs(t[cut:] - t[:cut].mean()).max())
    for _ in range(k_max):
        best = None
        for name, f in B.items():
            if name in chosen:
                continue
            X = np.stack([ones] + [B[c] for c in chosen] + [f], 1)
            cf = np.linalg.lstsq(X[:cut], t[:cut], rcond=None)[0]
            res = float(np.abs(X[cut:] @ cf - t[cut:]).max())
            if best is None or res < best[1]:
                best = (name, res, cf)
        if best is None:
            break
        chosen.append(best[0])
        coef = best[2]
        tr = best[1]
        if tr <= tol * scale:
            break
    return chosen, coef, tr, scale
