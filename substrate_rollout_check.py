"""Does the planner's roll-out agree with the world?  Verify before any further claims.

Three timing bugs have been fixed in this area already, so the roll-out is checked directly:
take a fixed action sequence, run it open-loop in the world for 40 steps (before any pipe can
arrive), and compare the height with what the roll-out predicted from the same state and the same
pending queue.

    world      vy(t+1) = vy(t) + b + F*a(t-LAG)      (the env's own queue)
    roll-out   the same recurrence, fed by the same sequence

If they differ, every planner number in the last two experiments is about the bug, not the plan.
"""
import numpy as np

from latent_world import BIRD_X, latent, recover

LAG = 3
r = recover(LAG)
B_DRIFT, F = r["G0"], r["F"]


def rollout_seq(y, vy, pend, acts, lag, F=F, b=B_DRIFT):
    """predicted trajectory for an issued-action sequence `acts`, from the current state"""
    ys = [y]
    for j, a in enumerate(acts, start=1):
        imp = pend[j - 1] if j <= lag else acts[j - lag - 1]
        vy = vy + b + F * imp
        y = y + vy
        ys.append(y)
    return np.asarray(ys)


def world_seq(seed, acts, lag):
    env = latent[lag](seed)
    pend = env.pending()
    y0, vy0 = env.y, env.vy
    ys = [env.y]
    for a in acts:
        if env.dead:
            break
        env.step(a)
        ys.append(env.y)
    return np.asarray(ys), y0, vy0, pend


rng = np.random.RandomState(0)
print("  seed   steps   max |world - roll-out|      where")
worst = 0.0
for seed in (100, 101, 102, 103):
    acts = [int(rng.rand() < 0.15) for _ in range(40)]
    ys_w, y0, vy0, pend = world_seq(seed, acts, LAG)
    ys_r = rollout_seq(y0, vy0, pend, acts, LAG)
    n = min(len(ys_w), len(ys_r))
    d = float(np.max(np.abs(ys_w[:n] - ys_r[:n])))
    worst = max(worst, d)
    print("  %4d   %5d   %.2e                  %s"
          % (seed, n - 1, d, "match" if d < 1e-12 else "MISMATCH at step %d"
             % int(np.argmax(np.abs(ys_w[:n] - ys_r[:n])))))

print("\n  engine of the env at this lag:")
env = latent[LAG](100)
print("    pending at t=0: %s" % env.pending())
print("    the action issued now lands LAG steps later: a(t) acts at t+%d" % LAG)
print("\nverdict: roll-out %s" % ("matches the world to 1e-12" if worst < 1e-12 else "DOES NOT match"))
