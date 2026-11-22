"""Does a longer outcome-label window buy lead at 10-16 frames?

The hazard readout is trained on "pain within `pain_horizon` frames".  The
default window is lag+1 (4), while death is sealed about ten frames ahead, so
the planner is asked to price outcomes the readout was never trained on.  This
probe refits with different windows and reports the AUC of the composed readout
(and its channels) at several leads.

Read-only; the window is a training parameter, not a task quantity.
"""

import argparse

import numpy as np

from eval_protocol import parse_seed_spec
from decision_diag_probe import channel_table, episodes_of
from substrate_nodes import LAG, _curiosity_off, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100")
    parser.add_argument("--windows", default="4,10,16")
    parser.add_argument("--leads", default="1,2,4,8,12,16")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--precursor", action="store_true")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    windows = [int(value) for value in args.windows.split(",")]
    leads = [int(value) for value in args.leads.split(",")]

    for seed in seeds:
        for window in windows:
            world, agent = fit(
                seed=seed, steps=args.steps, epochs=1, max_nodes=1000,
                birth_threshold=0.45, curiosity_weight=0.5,
                memory_credit=False, dagger_rounds=1, dagger_episodes=2,
                dagger_cap=5000, dagger_frames=5000,
                dagger_horizon=args.horizon, step_stride=2,
                pain_horizon=None if window <= 0 else int(window),
                precursor_labels=bool(args.precursor),
            )
            episodes = episodes_of(agent.milestone_training_rows)
            table = channel_table(agent, episodes, leads)
            print("== seed %d: pain_horizon %d (readout window %d) ==" % (
                seed, window, int(getattr(agent, "pain_horizon", 0)),
            ))
            for name in ("composed_raw", "composed_iso", "prototype",
                         "feature", "memory"):
                values = table.get(name)
                if values is None:
                    continue
                print("  %-13s %s" % (
                    name,
                    "  ".join(
                        "h%d %s" % (
                            lead,
                            "n/a" if value is None else "%.3f" % value,
                        )
                        for lead, value in zip(leads, values)
                    ),
                ))


if __name__ == "__main__":
    main()
