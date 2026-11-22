"""The latent world and the instruments around it, shared by the validation scripts.

    Flappy, with every flap taking effect `lag` frames later.  Everything here is the world
    plus the measuring instruments; the policies live in the scripts that use them.
"""
import os
from collections import deque

import numpy as np

for _k, _v in (("FL_G", "0.0016"), ("FL_FLAP", "0.020"), ("FL_SPEED", "0.010"),
               ("FL_GAP", "0.34"), ("FL_SPACING", "0.80")):
    os.environ.setdefault(_k, _v)

from flappy import BIRD_X, GAP, SPEED, Flappy as Env
from substrate_search import greedy_determine

CAP = 6000
BETAS = np.array([0.5, 0.2, 0.1, 0.05, 0.02])
SEEDS = list(range(100, 120))


class Latent(Env):
    lag = 0

    def __init__(self, seed, gravity=None, flap=None, drag=None, speed=None,
                 spacing=None):
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self._alive = 0.0
        super().__init__(
            seed, gravity=gravity, flap=flap, drag=drag, speed=speed,
            spacing=spacing,
        )
        self.max_t = CAP

    def reset(self):
        super().reset()
        self._q = deque([0] * (self.lag + 1), maxlen=self.lag + 1)
        self._alive = 0.0
        self.max_t = CAP

    def step(self, flap):
        self._q.append(int(flap))
        obs, dopamine, dead = super().step(self._q[0])
        # The flight distance in reward units: the integral of the world's
        # own dopamine feedback.  Exposed as an observable so the learner
        # can condition its predictions and scoring on how far the flight
        # has already gone - the world states a fact, the interpretation
        # stays with the learner.
        self._alive += dopamine
        return obs, dopamine, dead

    def pending(self):
        return list(self._q)[1:]              # a(t-lag) .. a(t-1)

    def obs(self):
        base = super().obs()
        return np.concatenate([base, [self._alive]]).astype(np.float32)


latent = {lag: type("Latent%d" % lag, (Latent,), {"lag": lag}) for lag in range(4)}


def recover(lag):
    """the transition, from data, with the true lagged column on offer"""
    W = latent[lag]
    rng = np.random.RandomState(5)
    rows = []
    for ep in range(400):
        env = W(3000 + ep)
        hist = deque(maxlen=max(lag, 1))
        while not env.dead:
            rec = dict(y=env.y, vy=env.vy, a=float(int(rng.rand() < 0.5)))
            h = list(hist)
            env.step(rec["a"])
            rec["y2"], rec["vy2"] = env.y, env.vy
            for k in range(1, lag + 1):
                rec["a_lag%d" % k] = float(h[-k]) if len(h) >= k else 0.0
            rows.append(rec)
            hist.append(rec["a"])
    d = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}
    ch, cf, res, _ = greedy_determine(d, "vy2", ["y", "vy", "a"], nonlinear=True,
                                      lags=["a_lag%d" % k for k in range(1, lag + 1)])
    coef = dict(zip(ch, cf[1:]))
    key = "a" if lag == 0 else "a_lag%d" % lag
    return dict(G0=float(cf[0]), CVY=float(coef.get("vy", 0.0)),
                F=float(coef.get(key, 0.0)), res=res,
                form="%.6f %s" % (cf[0], " ".join("%+.6f*%s" % (c, n)
                                                  for c, n in zip(cf[1:], ch))))


def make_policy(r, lag, hor, delta=0.0):
    """roll the rules out `hor` steps for both actions, take the closer end to the gap centre"""
    G0, CVY, F = r["G0"], r["CVY"], r["F"]

    def rollout(y, vy, pend, a_first):
        for j in range(1, hor + 1):
            imp = pend[j - 1] if j <= lag else (a_first if j == lag + 1 else 0)
            vy = G0 + CVY * vy + F * imp
            y = y + vy
        return y

    def target(env):
        best, gc = 1e9, env.pipes[0][1]
        for px, gy in env.pipes:
            dd = abs(px - BIRD_X - hor * SPEED)
            if dd < best:
                best, gc = dd, gy
        return gc

    def act(y, vy, gc, pend, prev):
        e0, e1 = rollout(y, vy, pend, 0), rollout(y, vy, pend, 1)
        benefit = abs(e0 - gc) - abs(e1 - gc)          # > 0 favours flapping
        if delta <= 0.0:
            return int(benefit > 0.0)
        if benefit > delta:
            return 1
        if benefit < -delta:
            return 0
        return prev                                    # inside the band: hold

    return act, target


class Decoder:
    """L linear readouts off the u bank, thresholded at the observed flap rate"""

    def __init__(self, ws, cs):
        self.ws, self.cs = ws, cs

    def hat(self, u):
        z = np.concatenate([[1.0], u])
        return [int(z @ w > c) for w, c in zip(self.ws, self.cs)][::-1]   # a(t-lag) .. a(t-1)


def fit_decoder(acts, lag):
    if lag == 0:
        return Decoder([], []), []
    u = np.zeros(len(BETAS))
    X, T = [], [[] for _ in range(lag)]
    for t, a in enumerate(acts):
        if t >= lag:
            X.append(u.copy())
            for j in range(lag):
                T[j].append(acts[t - 1 - j])      # lag j relative to the trace u(t-1)
        u = (1.0 - BETAS) * u + BETAS * a
    X = np.asarray(X)
    Z = np.concatenate([np.ones((len(X), 1)), X], 1)
    cut = int(0.7 * len(X))
    ws, cs, rep = [], [], []
    for j in range(lag):
        y = np.asarray(T[j], dtype=float)
        w = np.linalg.lstsq(Z[:cut], y[:cut], rcond=None)[0]
        rate = float(y[:cut].mean())
        c = float(np.quantile(Z[:cut] @ w, 1.0 - rate)) if 0.0 < rate < 1.0 else 0.5
        pred = Z[cut:] @ w > c
        truth = y[cut:] > 0.5
        acc = float(np.mean(pred == truth))
        base = float(max(truth.mean(), 1.0 - truth.mean()))
        tpr = float(pred[truth].mean()) if truth.any() else float("nan")
        tnr = float((~pred[~truth]).mean()) if (~truth).any() else float("nan")
        ws.append(w)
        cs.append(c)
        rep.append((lag - 1 - j, acc, base, 0.5 * (tpr + tnr), rate))
    return Decoder(ws, cs), rep


def run(lag, seed, pol, source, dec=None, eps=0.0, flap=True, cap=None, rng=None):
    """source: 'exact' | 'none' | 'decoded' | 'noisy'"""
    act, target = pol
    env = latent[lag](seed)
    if cap is not None:
        max_t = int(cap)
        if max_t < 1:
            raise ValueError("cap must be a positive integer")
        env.max_t = max_t
    u = np.zeros(len(BETAS))
    acts, wrong, total = [], 0, 0
    prev = 0
    while not env.dead:
        if source == "exact":
            pend = env.pending()
        elif source == "none":
            pend = [0] * lag
        elif source == "decoded":
            pend = dec.hat(u)
        else:
            pend = [int(b) ^ int(rng.rand() < eps) for b in env.pending()]
        if source in ("decoded", "noisy") and lag:
            wrong += sum(1 for x, y in zip(pend, env.pending()) if x != y)
            total += lag
        a = act(env.y, env.vy, target(env), pend, prev) if flap else 0
        prev = a
        acts.append(a)
        env.step(a)
        u = (1.0 - BETAS) * u + BETAS * a
    return env.t, acts, (wrong / total if total else 0.0)


class Schedule:
    """outstanding consequences with a due time, not a history of inputs"""

    def __init__(self, lag, mag):
        self.lag, self.mag = lag, mag
        self.due = {}

    def issue(self, t, a):
        if a:
            self.due[t + self.lag] = self.mag

    def drop_before(self, t):
        for k in [k for k in self.due if k < t]:
            del self.due[k]

    def upcoming(self, t):
        return [1 if (t + j) in self.due else 0 for j in range(self.lag)]

    def held(self):
        return len(self.due)
