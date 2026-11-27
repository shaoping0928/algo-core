"""Self-monitoring wired into the substrate, not bolted beside it.

Until now the monitor lived in experiment scripts: a loop measured signals, decided,
and called into a model.  That is a controller next to the system, not a system that
watches itself.  Here the monitor is a member of the substrate: the task loop just
calls `sys.step(x, y, ctx)` and never decides anything.  Every `every` steps the
substrate measures its own state, picks an action and applies it.

The visible consequence has to be that nothing in the task loop knows about the
controller, and the numbers match the external version (§106): conflict .366 -> 1.000.

    python live_core.py [trials]
"""
import math
import sys

import numpy as np

try:
    from substrate_rules import RuleBook
except Exception:          # the rule layer is optional; the substrate runs without it
    RuleBook = None

R, NC, LAM = 8, 8, 1.0
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def ridge(H, Y):
    return np.linalg.solve(H.T @ H + LAM * np.eye(H.shape[1]), H.T @ Y)


def eff_rank(H):
    s = np.linalg.svd(H, compute_uv=False)
    return float(s.sum() ** 2 / (s ** 2).sum())


_RULE_FNS = {}


def inv_fn_value(inv, row):
    """the target of an invariant on a row, as the caller defined it"""
    fn = _RULE_FNS.get(inv.name)
    return fn(row) if fn is not None else float('nan')


class Live:
    """Substrate with its own monitor.  The only public surface is step/score."""

    def __init__(self, rng, gain=0.2, every=1, monitor=True,
                 rule_specs=None, rule_thr=1e-6, rule_min=None, rule_budget=3):
        self.rng = rng
        self.gain = gain
        self.every = every
        self.monitor_on = monitor
        self.routes = []
        self.W1 = rng.normal(0, 1.0 / math.sqrt(2 * R), size=(2 * R, 64)).astype(np.float32)
        self.B1 = rng.normal(0, 1.0, size=64).astype(np.float32)
        self.t = 0
        self.actions = []
        self.fits = 0                # monitor overhead, counted
        self._last = None
        # the policy is three thresholds; a meta-level may rewrite them
        self.thr = dict(rank=4.0, gap=0.20, marg=0.02)
        # rule layer: a sixth signal (residual of the holding rules) and a sixth action
        # (scan for new invariants).  Off unless the caller supplies candidates.
        self.rules = RuleBook() if (RuleBook is not None and rule_specs) else None
        if self.rules is not None:
            for spec in rule_specs:
                self.rules.add_invariance(spec['name'], spec['fn'],
                                          spec['allowed'], spec['vary'])
                _RULE_FNS[spec['name']] = spec['fn']
        self.rule_rows = []
        self.rule_thr = float(rule_thr)
        # With the coverage guard in RuleBook, this stream needs about 1000 rows before
        # both true invariants have enough repeated groups to pass.
        self.rule_min = int(rule_min if rule_min is not None else 1000)
        self.rule_budget = int(rule_budget)
        self.scans = 0
        self.last_scan_rows = 0
        self.rule_err = float('inf')

    # ---- surface used by the task loop -------------------------------------
    def step(self, tr, te, row=None):
        self.t += 1
        s = self.measure(tr, te)          # always: a readout has to exist
        if self.rules is not None and row is not None:
            self.rules.observe(row)          # the store the scan reads
            self.rule_rows.append(row)       # the window the residual is measured on
            if len(self.rule_rows) >= self.rule_min:
                self.rule_err = (self._rule_residual()
                                 if self.rules.all_invariants_hold() else float('inf'))
        if self.monitor_on and self.t % self.every == 0:
            if self._act(s, tr):          # the monitor only decides to act
                s = self.measure(tr, te)  # report the state after acting
        if (self.rules is not None and self.monitor_on
                and len(self.rule_rows) >= self.rule_min
                and self.rule_err > self.rule_thr
                and len(self.rule_rows) - self.last_scan_rows >= self.rule_min
                and self.scans < self.rule_budget):
            self._scan()
        return s["held"]

    def _rule_residual(self):
        """How far the current rules are from the trace: 0 when they hold exactly."""
        if not self.rules.invariants:
            return float('inf')
        worst = 0.0
        for inv in self.rules.holding():
            for r in self.rule_rows[-self.rule_min:]:
                worst = max(worst, abs(inv.predict(r) - inv_fn_value(inv, r)))
        return worst

    def _scan(self):
        self.scans += 1
        self.rules.scan()
        self.last_scan_rows = len(self.rule_rows)
        self.rule_err = (self._rule_residual()
                         if self.rules.all_invariants_hold() else float('inf'))
        self.actions.append('scan')

    def rule_signal(self):
        return dict(held=len(self.rules.holding()) if self.rules else 0,
                    residual=self.rule_err, scans=self.scans)

    def score(self, te):
        Hte = self.design(te)
        ctx = np.array([r[2] for r in te])
        yte = np.array([r[1] for r in te])
        return float((self.predict(Hte, ctx) == yte).mean())

    # ---- internals ---------------------------------------------------------
    def design(self, rows):
        return np.stack([np.append(np.tanh(self.gain * (x @ self.W1 + self.B1)), 1.0)
                         for x, _y, _c in rows])

    def fit(self, tr):
        self.fits += 1
        Htr = self.design(tr)
        Y = np.eye(NC)[[r[1] for r in tr]]
        ctx = np.array([r[2] for r in tr])
        if not self.routes:
            self.routes = [(-1, ridge(Htr, Y))]
        elif len(self.routes) == 1 and self.routes[0][0] == -1:
            self.routes = [(-1, ridge(Htr, Y))]
        else:
            self.routes = [(c, ridge(Htr[ctx == c], Y[ctx == c]))
                           if (ctx == c).any() else (c, w) for c, w in self.routes]
        return Htr

    def predict(self, H, ctx, drop=None):
        routes = [r for i, r in enumerate(self.routes) if i != drop]
        out = np.zeros((len(H), NC), np.float32)
        for c in np.unique(ctx):
            sel = ctx == c
            ex = [w for (rc, w) in routes if rc in (c, -1)]
            if not ex:
                ex = [w for (_rc, w) in routes]
            out[sel] = np.mean([np.exp(H[sel] @ w) for w in ex], axis=0)
        return out.argmax(1)

    def measure(self, tr, te):
        Htr = self.fit(tr)
        Hte = self.design(te)
        ytr = np.array([r[1] for r in tr])
        yte = np.array([r[1] for r in te])
        ctr, cte = np.array([r[2] for r in tr]), np.array([r[2] for r in te])
        ptr, pte = self.predict(Htr, ctr), self.predict(Hte, cte)
        per = [float((pte[cte == c] == yte[cte == c]).mean()) if (cte == c).any() else 0.0
               for c in (0, 1)]
        marg = 0.0
        if len(self.routes) > 1:
            pd = self.predict(Hte, cte, drop=len(self.routes) - 1)
            marg = float((pte == yte).mean() - (pd == yte).mean())
        return dict(rank=eff_rank(Htr), train=float((ptr == ytr).mean()),
                    held=float((pte == yte).mean()), spread=float(np.ptp(per)),
                    routes=len(self.routes), marg=marg)

    def _act(self, s, tr):
        act = "none"
        if s["rank"] < self.thr["rank"]:
            self.gain = min(8.0, self.gain * 3.0)
            act = "gain"
        elif s["held"] < 0.60 and (s["train"] - s["held"]) > self.thr["gap"]:
            Htr = self.design(tr)
            Y = np.eye(NC)[[r[1] for r in tr]]
            ctx = np.array([r[2] for r in tr])
            self.routes = [(c, ridge(Htr[ctx == c], Y[ctx == c])) for c in (0, 1)]
            act = "route"
        elif s["routes"] > 1 and s["marg"] < self.thr["marg"]:
            self.routes = self.routes[:-1]
            act = "prune"
        self.actions.append(act)
        self._last = s
        return act != "none"


def make(rng, ctxs, keep=0.6):
    rows = []
    for c in ctxs:
        for a in range(R):
            for b in range(R):
                x = np.zeros(2 * R, np.float32)
                x[a] = 1.0
                x[R + b] = 1.0
                rows.append((x, a if c == 0 else b, c))
    rng.shuffle(rows)
    n = max(8, int(keep * len(rows)))
    return rows[:n], rows[n:]


def trial(seed, monitor):
    rng = np.random.RandomState(seed)
    sys_ = Live(rng, monitor=monitor)
    t1 = int(rng.randint(3, 6))
    t2 = t1 + int(rng.randint(3, 6))
    helds = []
    for t in range(14):
        ctxs = [0] if (t < t1 or t >= t2) else [0, 1]
        tr, te = make(rng, ctxs)
        helds.append(sys_.step(tr, te))
    return helds, t1, t2, sys_


print("monitor owned by the substrate; the task loop only calls step()")
for mode in (False, True):
    pre, mid, post, fits, nact = [], [], [], [], []
    for k in range(TRIALS):
        h, t1, t2, s = trial(21 + k, mode)
        pre.append(float(np.mean(h[:t1])))
        mid.append(float(np.mean(h[t1:t2])))
        post.append(float(np.mean(h[t2:])))
        fits.append(s.fits)
        nact.append(sum(1 for a in s.actions if a != "none"))
    print("  monitor=%-5s pre %.3f   conflict %.3f +- %.3f   post %.3f   "
          "fits %d   actions %d"
          % (mode, float(np.mean(pre)), float(np.mean(mid)), float(np.std(mid)),
             float(np.mean(post)), int(np.mean(fits)), int(np.mean(nact))))
