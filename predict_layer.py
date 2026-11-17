"""A prediction layer and a reality layer; the difference is the only teacher.

Separated from the experiment scripts so that more than one of them can learn with it:
`substrate_predict.py` uses it on supplied streams, `substrate_selfdelay.py` uses it inside a
closed loop on the agent's own actions.
"""
import numpy as np


class LayerPair:
    """d_hat = b + sum_k h_k * e(t-k); err = d - d_hat trains b and h, online"""

    def __init__(self, k_max=8, eta0=0.05, tau=2e4, learn=True):
        self.k_max = k_max
        self.eta0, self.tau = eta0, tau
        self.learn_on = learn
        self.b = 0.0
        self.h = np.zeros(k_max)
        self.hist = [0.0] * k_max          # e(t-1) .. e(t-k_max): capacity, not knowledge
        self.t = 0
        self.err = 0.0

    def predict(self):
        return self.b + float(np.dot(self.h, np.asarray(self.hist)))

    def observe(self, delta, event):
        """d(t) arrives, then the event issued at t is remembered for the future"""
        pred = self.predict()
        err = delta - pred
        self.t += 1
        if self.learn_on:
            eta = self.eta0 / (1.0 + self.t / self.tau)
            self.b += eta * err
            self.h += eta * err * np.asarray(self.hist)
        self.hist = [event] + self.hist[:-1]
        self.err = 0.999 * self.err + 0.001 * abs(err)
        return err

    def lag_profile(self):
        """(lag, gain) pairs, lag counted the way a world means it: 1 = last step"""
        return [(k + 1, float(self.h[k])) for k in range(self.k_max)]

    def identified(self, true_lag, min_gain=0.0, frac=0.25):
        """the true lag is the largest gain, it is substantial, and every other lag is small"""
        j = true_lag - 1
        if j < 0 or j >= self.k_max:
            return False
        h = np.abs(self.h)
        if h[j] <= max(min_gain, 1e-9):
            return False
        rest = float(np.max(np.delete(h, j)))
        return h[j] == np.max(h) and rest < frac * h[j]
