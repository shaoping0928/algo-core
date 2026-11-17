"""Learn -> sandbox sequence search -> feasibility check -> execute -> observe.

Candidates are action sequences, not a hand-written policy.  For each real state, the sandbox
rolls every binary sequence of length H through the learned delayed transition, rejects paths
that hit the world boundary or a pipe, scores feasible paths against the observed gap centre,
executes the first action, and searches again next step.

The transition, lag and impulse are learned from practice.  The horizon, binary candidate set and
gap-centre objective are scaffolding.  Sandbox predictions are not fed to the world model;
only real transitions update it.
"""
from itertools import product

import numpy as np

from latent_world import BIRD_X, GAP, SEEDS, SPEED, latent
from predict_layer import LayerPair

LAG = 3
H = 14
PRACTICE = 1000
SEQS = np.asarray(list(product((0, 1), repeat=H)), dtype=np.float64)


def practice(seed, steps=PRACTICE):
    lp = LayerPair(k_max=8)
    done = ep = 0
    while done < steps:
        env = latent[LAG](seed * 10000 + ep)
        rng = np.random.RandomState(seed * 10000 + ep)
        lp.hist = [0.0] * 8
        while not env.dead and done < steps:
            action = int(rng.rand() < 0.5)
            vy0 = env.vy
            env.step(action)
            lp.observe(env.vy - vy0, float(action))
            done += 1
        ep += 1
    j = int(np.argmax(np.abs(lp.h)))
    return lp, j + 1, float(lp.h[j])


def target_gap(env):
    # Select the pipe closest to reaching the bird during this short horizon.
    px_target = BIRD_X + H * SPEED
    _, gc = min(env.pipes, key=lambda p: abs(p[0] - px_target))
    return gc


def rollout_candidates(env, b, f, lag):
    n = len(SEQS)
    ys = np.full(n, env.y, dtype=float)
    vys = np.full(n, env.vy, dtype=float)
    pending = env.pending()
    feasible = np.ones(n, dtype=bool)
    cost = np.zeros(n)
    gc = target_gap(env)
    first_y = np.full(n, np.nan, dtype=float)

    # Check the entire predicted trajectory, not only its endpoint.
    for step in range(1, H + 1):
        if step <= lag:
            impulse = float(pending[step - 1])
        else:
            impulse = SEQS[:, step - lag - 1]
        vys = vys + b + f * impulse
        ys = ys + vys
        if step == 1:
            first_y[:] = ys
        feasible &= (ys >= 0.0) & (ys <= 1.0)
        for px, pipe_gc in env.pipes:
            pipe_x = px - step * SPEED
            if abs(pipe_x - BIRD_X) < 0.06:
                feasible &= np.abs(ys - pipe_gc) <= GAP / 2.0
        # Prefer paths that approach the observed target without leaving the feasible region.
        cost += (ys - gc) ** 2
    return feasible, cost, ys, first_y, gc


def run(seed, show=False):
    lp, lag, f = practice(seed)
    b = lp.b
    env = latent[LAG](seed)
    lp.hist = [0.0] * 8
    t = searches = feasible_total = no_feasible = 0
    max_model_error = 0.0
    selected = []
    search_memory = []

    while not env.dead and t < 6000:
        feasible, costs, endpoints, first_ys, gc = rollout_candidates(env, b, f, lag)
        searches += 1
        feasible_total += int(feasible.sum())
        if feasible.any():
            ids = np.flatnonzero(feasible)
            idx = int(ids[np.argmin(costs[ids])])
        else:
            # No candidate is feasible: retain the least-cost path, and count the failure.
            no_feasible += 1
            idx = int(np.argmin(costs))
        seq = SEQS[idx]
        action = int(seq[0])
        predicted_next_y = float(first_ys[idx])
        if show and len(selected) < 8:
            selected.append((t, "".join(str(int(x)) for x in seq[:10]),
                             float(endpoints[idx]), gc, bool(feasible[idx])))

        pending_before = tuple(int(x) for x in env.pending())
        applied = int(pending_before[0])  # next impulse: oldest action currently pending
        y0, vy0 = env.y, env.vy
        env.step(action)
        actual_next_y = float(env.y)
        observed_delta = env.vy - vy0
        max_model_error = max(max_model_error, abs(observed_delta - (b + f * applied)))
        # Search experience is stored separately from factual transition data.
        # It records the counterfactual plan and then the real one-step verification.
        search_memory.append({
            "t": t,
            "state": (float(y0), float(vy0)),
            "pending": pending_before,
            "candidate_count": int(len(SEQS)),
            "feasible_count": int(feasible.sum()),
            "chosen_sequence": tuple(int(x) for x in seq),
            "predicted_endpoint": float(endpoints[idx]),
            "target_gap": float(gc),
            "predicted_next_y": predicted_next_y,
            "actual_next_y": actual_next_y,
            "one_step_error": actual_next_y - predicted_next_y,
        })
        lp.observe(observed_delta, float(action))  # only real transitions update the model
        # Refresh the learned parameters from real data only.
        j = int(np.argmax(np.abs(lp.h)))
        lag, f, b = j + 1, float(lp.h[j]), float(lp.b)
        t += 1

    return dict(frames=env.t, lag=lag, f=f, b=b, searches=searches,
                feasible_fraction=feasible_total / max(searches * len(SEQS), 1),
                no_feasible=no_feasible, model_error=max_model_error, selected=selected,
                search_memory=search_memory)


print("sandbox: %d binary action sequences (H=%d); filter full paths; execute first action; re-search"
      % (len(SEQS), H))
print("practice=%d real steps/seed; sandbox predictions are not training samples\n" % PRACTICE)
print(" seed   frames   lag   learned F   feasible fraction   no-feasible steps   max model error")
rows = []
for seed in SEEDS[:8]:
    result = run(seed, show=(seed == SEEDS[0]))
    rows.append(result)
    print(" %4d   %6d   %3d   %.6f        %.3f              %5d             %.2e"
          % (seed, result["frames"], result["lag"], result["f"],
             result["feasible_fraction"], result["no_feasible"], result["model_error"]))

print("\nselected candidates from seed %d (prefix, predicted endpoint, target, feasible):" % SEEDS[0])
for t, prefix, endpoint, gc, ok in rows[0]["selected"]:
    print("  t=%3d  %s  y_pred=%.4f  gap=%.4f  %s" % (t, prefix, endpoint, gc, ok))

print("\nQ1  learned lag 3 and F=.020: %s (mean lag %.2f, F %.6f)"
      % ("ok" if all(r["lag"] == LAG and abs(r["f"] - .02) < .002 for r in rows) else "MISS",
         np.mean([r["lag"] for r in rows]), np.mean([r["f"] for r in rows])))
print("Q2  feasible candidates exist: %s (mean feasible fraction %.3f)"
      % ("ok" if np.mean([r["feasible_fraction"] for r in rows]) > 0 else "MISS",
         np.mean([r["feasible_fraction"] for r in rows])))
print("Q3  no-feasible steps: %d across %d seeds"
      % (sum(r["no_feasible"] for r in rows), len(rows)))
print("Q4  survival: mean %.1f, median %.1f frames; seed range %d-%d"
      % (np.mean([r["frames"] for r in rows]),
         np.median([r["frames"] for r in rows]),
         min(r["frames"] for r in rows), max(r["frames"] for r in rows)))
print("    reference only: the hand-derived exact-model rule scored 4887, but it is not a learned search policy")
print("Q5  one-step model/world discrepancy: %.2e"
      % max(r["model_error"] for r in rows))
mem = [item for r in rows for item in r["search_memory"]]
print("Q6  separate sandbox-memory records: %d; mean |predicted next y - real next y| %.2e"
      % (len(mem), np.mean([abs(item["one_step_error"]) for item in mem])))
print("    each record stores state, pending queue, candidate count, feasible count, selected plan,")
print("    predicted endpoint, target, and the real one-step verification; only that real transition")
print("    updates the world model.")
