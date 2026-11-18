"""Flappy Bird as a world: continuous state, binary action, survival as the score.

Observation (the things a human sees on screen):

    y            bird height, 0 bottom .. 1 top
    vy           vertical velocity (positive = rising)
    dx           horizontal distance to the pipe
    dy           gap centre minus bird height
    next_dy      same for the pipe after that

Action: flap (binary).  Reward: +1 per frame alive.  Death on the pipe edge or the
ground/ceiling.

Three policies are compared, all on the same seeds:

    zero      weights = 0 (flap never), the do-nothing floor
    scripted  flap when the bird is below the gap: the environment's upper bound
    learned   (1+lambda) evolution strategy on a linear policy over the observation

    python flappy.py [generations]
"""
import os
import sys
import time

import numpy as np

DEFAULT_GENS = 40
GRAVITY = float(os.environ.get("FL_G", "0.0022"))
FLAP = float(os.environ.get("FL_FLAP", "0.045"))
SPEED = float(os.environ.get("FL_SPEED", "0.012"))
GAP = float(os.environ.get("FL_GAP", "0.30"))
SPACING = float(os.environ.get("FL_SPACING", "0.6"))
DRAG = float(os.environ.get("FL_DRAG", "0.0"))    # quadratic drag: vy += -DRAG*vy*|vy|
KY = float(os.environ.get("FL_KY", "0.0"))        # gravity grows off-centre: G*(1+KY*(y-0.5)^2)
V0 = float(os.environ.get("FL_V0", "0.0"))        # flap weakens while falling: F*exp(vy/V0) if vy<0
BIRD_X = 0.22
MAXT = 1200
SEED = 21


class Flappy:
    def __init__(self, seed, max_t=None):
        self.rng = np.random.RandomState(seed)
        self.max_t = MAXT if max_t is None else int(max_t)
        if self.max_t < 1:
            raise ValueError("max_t must be a positive integer")
        self.reset()

    def reset(self):
        self.y = 0.5
        self.vy = 0.0
        self.pipes = [[1.0, self._gap()], [1.0 + SPACING, self._gap()]]
        self.t = 0
        self.dead = False
        self._feedback = {"dopamine": 0.0, "pain": 0.0}

    def _gap(self):
        return float(self.rng.uniform(GAP / 2 + 0.05, 1 - GAP / 2 - 0.05))

    def obs(self):
        px = self.pipes[0][0]
        p2 = self.pipes[1] if len(self.pipes) > 1 else self.pipes[0]
        return np.array([self.y - 0.5, self.vy * 10.0, px - BIRD_X,
                         self.pipes[0][1] - self.y,
                         (p2[0] - BIRD_X), p2[1] - self.y], np.float32)

    def step(self, flap):
        self.t += 1
        imp = FLAP
        if flap and V0 > 0.0 and self.vy < 0.0:
            imp *= float(np.exp(self.vy / V0))
        self.vy += -GRAVITY * (1.0 + KY * (self.y - 0.5) ** 2) \
            - DRAG * self.vy * abs(self.vy) + (imp if flap else 0.0)
        self.y += self.vy
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[0][0] < -0.2:
            self.pipes.pop(0)
            self.pipes.append([self.pipes[-1][0] + SPACING, self._gap()])
        # collision
        for px, gc in self.pipes:
            if abs(px - BIRD_X) < 0.06 and abs(self.y - gc) > GAP / 2:
                self.dead = True
        if self.y < 0.0 or self.y > 1.0:
            self.dead = True
        if self.t >= self.max_t:
            self.dead = True
        # World-provided valence channels; learners consume these as observations.
        self._feedback = {
            "dopamine": 0.0 if self.dead else 1.0,
            "pain": 1.0 if self.dead else 0.0,
        }
        return self.obs(), 1.0, self.dead

    def feedback(self):
        """Return the world-emitted valence channels from the most recent step."""
        return dict(self._feedback)


def run(policy_fn, seed, cap=None):
    env = Flappy(seed, max_t=cap)
    total = 0.0
    while not env.dead:
        o, r, d = env.step(policy_fn(env.obs()))
        total += r
        if d:
            break
    return total


def zero_policy(o):
    return 0


def scripted(o):
    # naive: flap whenever the gap is above the bird
    return 1 if o[3] > 0.02 else 0


def scripted_lookahead(o, horizon=14):
    """flap if the bird will be below the gap after `horizon` frames of falling"""
    vy, dy = o[1] / 10.0, o[3]      # dy = gap - y, positive means gap above
    # Exact discrete sum when gravity is constant: each step updates vy before y.
    drop = horizon * vy - GRAVITY * horizon * (horizon + 1) / 2.0
    return 1 if drop < dy - 0.02 else 0


def linear_policy(w):
    def f(o):
        x = np.concatenate([o, [1.0]])
        return int(x @ w > 0)
    return f


def fitness(w, seeds):
    return float(np.mean([run(linear_policy(w), s) for s in seeds]))


def es(gens=DEFAULT_GENS, lam=16, sigma=0.4, seeds=(1, 2, 3)):
    rng = np.random.RandomState(SEED)
    w = np.zeros(7, np.float32)
    best = fitness(w, seeds)
    curve = [(0, best)]
    for g in range(gens):
        cands = [w + rng.normal(0, sigma, size=7) for _ in range(lam)]
        scores = [fitness(c, seeds) for c in cands]
        i = int(np.argmax(scores))
        if scores[i] > best:
            w, best = cands[i], scores[i]
        curve.append((g + 1, best))
    return w, best, curve


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    gens = int(argv[0]) if argv else DEFAULT_GENS

    t0 = time.time()
    print("flappy: gravity %.4f  flap %.3f  gap %.2f  cap %d frames" % (GRAVITY, FLAP, GAP, MAXT))
    print("\npolicy        mean survival (20 unseen seeds)")
    for name, fn in (("zero", zero_policy), ("scripted", scripted),
                     ("lookahead", scripted_lookahead)):
        sc = [run(fn, s) for s in range(100, 120)]
        print("  %-10s %7.1f" % (name, float(np.mean(sc))))

    w, best, curve = es(gens=gens)
    sc = [run(linear_policy(w), s) for s in range(100, 120)]
    print("  %-10s %7.1f   (search fitness %.1f)" % ("learned", float(np.mean(sc)), best))
    print("\n  ES: fitness by generation %s" % " ".join(
        "%.0f" % c[1] for c in curve[::max(1, len(curve) // 8)]))
    print("  weights %s" % np.round(w, 3).tolist())
    print("%.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
