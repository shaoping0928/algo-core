"""Outward generation from a parent summary.

A parent keeps a few core keyword directions.  This module uses those
keywords as a prompt for the learned node system: it sets the dynamics
context to the parent's summary, starts from a representative stored state,
chooses actions with the supplied learned policy, and rolls the learned world
model forward.  The result is a new approximate flow, not a copy of the
stored children; its distance to the stored memories measures whether the
prompt actually steers generation toward the right regime.
"""
import numpy as np


class GenerativeExpander:
    def __init__(self, model, memory=None, policy=None, horizon=24,
                 phase_len=6, bound_scale=6.0):
        self.model = model
        self.memory = memory
        self.policy = policy
        self.horizon = max(int(horizon), 1)
        self.phase_len = max(int(phase_len), 1)
        self.bound_scale = float(bound_scale)
        self._bounds = None

    # ---- reference state / bounds ----------------------------------------
    def pick_start(self, context):
        if self.memory is None or not self.memory.frames:
            return np.zeros(self.model.obs_dim, dtype=float)
        context = np.asarray(context, dtype=float).reshape(-1)
        width = min(len(context), self.memory.context_dim)
        best = None
        for frame in self.memory.frames:
            candidate = frame.context[:width]
            distance = float(np.linalg.norm(candidate - context[:width]))
            if best is None or distance < best[0]:
                best = (distance, frame.observation)
        return np.asarray(best[1], dtype=float).copy()

    def bounds(self):
        if self._bounds is not None:
            return self._bounds
        if self.memory is None or not self.memory.frames:
            self._bounds = (None, None)
            return self._bounds
        observations = np.stack([
            frame.observation for frame in self.memory.frames
        ])
        mean = observations.mean(axis=0)
        spread = observations.std(axis=0) + 1e-9
        self._bounds = (
            mean - self.bound_scale * spread,
            mean + self.bound_scale * spread,
        )
        return self._bounds

    # ---- generation -------------------------------------------------------
    def _default_policy(self, obs, history):
        return int(np.sum(history) % 2)

    def expand(self, context, start_obs=None, horizon=None, policy=None):
        context = np.asarray(context, dtype=float).reshape(-1)
        width = (
            self.model.context.shape[0]
            if self.model.basis is not None else 0
        )
        if width:
            self.model.begin_episode(
                prior=context[:width], prior_precision=1e6
            )
        else:
            self.model.reset_context()
        observation = (
            self.pick_start(context) if start_obs is None
            else np.asarray(start_obs, dtype=float).copy()
        )
        history = np.zeros(self.model.action_dim, dtype=float)
        low, high = self.bounds()
        steps = self.horizon if horizon is None else max(int(horizon), 1)
        selected = self.policy if policy is None else policy
        if selected is None:
            selected = self._default_policy
        records = []
        for _ in range(steps):
            action = int(selected(observation.copy(), history.copy()))
            prediction = self.model.predict(
                observation.reshape(1, -1), history.reshape(1, -1)
            )[0]
            records.append(dict(
                observation=observation.copy(),
                action=action,
                prediction=np.asarray(prediction, dtype=float).copy(),
                context=self.model.context.copy(),
            ))
            if not np.isfinite(prediction).all():
                break
            if low is not None and (
                (prediction < low).any() or (prediction > high).any()
            ):
                break
            history = np.concatenate(
                [[float(action)], history[:-1]]
            )
            observation = np.asarray(prediction, dtype=float)
        return records

    # ---- summaries --------------------------------------------------------
    def summarize(self, records):
        phases = []
        for start in range(0, len(records), self.phase_len):
            block = records[start:start + self.phase_len]
            if not block:
                continue
            observations = np.stack([
                row["observation"] for row in block
            ])
            contexts = np.stack([row["context"] for row in block])
            phases.append(dict(
                start=start, length=len(block),
                observation=observations.mean(axis=0),
                context=contexts.mean(axis=0),
            ))
        return phases


def nearest_reference_distance(generated, reference):
    """Mean nearest-neighbour distance from generated to stored vectors."""
    generated = np.asarray(generated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(generated) == 0 or len(reference) == 0:
        return float("nan")
    differences = (
        generated[:, None, :] - reference[None, :, :]
    )
    distances = np.sqrt((differences ** 2).sum(axis=2))
    return float(distances.min(axis=1).mean())
