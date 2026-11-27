"""Static kill test for the attraction term: score ties with different Phi.

Under a fixed-horizon exhaustive search every plan shares the start state,
so a PBRS attraction telescopes to an endpoint-only term.  It can only
change a decision when two candidates have (near-)equal scores and
different endpoint potential.  If no such pair exists, the term is inert
and no A/B is needed.

The probe runs the real planner with the candidate callback, recomputes the
endpoint curiosity potential ``Phi(s_H)`` from the same evidence counts the
objective uses, and reports tie statistics.  Two arms separate the
protocol effect: ``--epochs 1`` (current benchmark: features are installed
after the only learning pass, so evidence cells are empty and Phi is
constant) and ``--epochs 2`` (backfit pass: cells trained, Phi varies).

Usage::

    python n_tie_probe.py --epochs 2 --report n_tie_e2.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eval_protocol import parse_seed_spec
from latent_world import latent
from substrate_generic import history_features
from substrate_nodes import (
    LAG,
    _curiosity_off,
    _curiosity_potential,
    fit,
    policy_result,
)


def flatten(batches):
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }


def endpoint_potential(agent, sequences, final_states, prefix, horizon):
    """Phi at each candidate's predicted endpoint, from the same cells."""
    history = history_features(
        sequences, horizon, prefix, action_dim=agent.lag + 1
    )
    pending = history[:, :agent.lag][:, ::-1]
    encoded = (
        np.concatenate([final_states, pending], axis=1) / agent.scales
    )
    mask_table = agent._feature_mask_table(encoded[:, :agent.obs_dim])
    evidence = agent.evidence_batch(encoded, mask_table)
    return _curiosity_potential(evidence)


def tie_statistics(scores, phi, tolerance):
    """Tie statistics for one decision.

    ``exact`` uses machine ties; ``near`` uses ``tolerance``.  A flip is
    possible when the endpoint potential differs inside the tie set: the
    attraction prefers the higher Phi, so a different plan could win.
    """
    scores = np.asarray(scores, dtype=float)
    phi = np.asarray(phi, dtype=float)
    result = {}
    if len(scores) == 0:
        for label in ("exact", "near"):
            result[label] = dict(
                pairs=0, pairs_diff_phi=0, flip=False, phi_gap=0.0
            )
        return result
    order = np.argsort(scores)
    ordered_scores = scores[order]
    ordered_phi = phi[order]
    best = float(ordered_scores[0])
    for label, limit in (("exact", 1e-12), ("near", float(tolerance))):
        mask = ordered_scores <= best + limit
        tied = ordered_phi[mask]
        count = int(mask.sum())
        pairs = count * (count - 1) // 2
        different = 0
        if count > 1:
            spread = tied[:, None] - tied[None, :]
            different = int((np.abs(spread) > 1e-9).sum() // 2)
        flip = bool(
            count > 1 and abs(float(tied.max()) - float(tied[0])) > 1e-9
        )
        result[label] = dict(
            pairs=pairs,
            pairs_diff_phi=different,
            flip=flip,
            phi_gap=(
                float(tied.max() - tied.min()) if count else 0.0
            ),
        )
    return result


def run(args):
    world, agent = fit(
        seed=args.seed, steps=args.steps, epochs=args.epochs,
        max_nodes=args.max_nodes, birth_threshold=0.45,
        curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=args.rounds, dagger_episodes=2,
        dagger_cap=args.dagger_frames, dagger_frames=args.dagger_frames,
        dagger_horizon=args.horizon, step_stride=2,
    )
    decisions = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env = latent[LAG](int(eval_seed) * 10000)
            prefix = np.zeros(LAG, dtype=float)
            recent = [0.0] * LAG
            while not env.dead and env.t < args.cap:
                batches = []
                result = policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                if batches:
                    batch = flatten(batches)
                    feasible = (
                        np.asarray(batch["feasible"], dtype=bool)
                        & np.asarray(batch["finite"], dtype=bool)
                    )
                    scores = np.asarray(batch["scores"], dtype=float)[:, 0]
                    phi = endpoint_potential(
                        agent, np.asarray(batch["sequences"], dtype=float),
                        np.asarray(batch["final_states"], dtype=float),
                        prefix, args.horizon,
                    )
                    if feasible.any():
                        feasible_scores = scores[feasible]
                        feasible_phi = phi[feasible]
                        entry = dict(
                            candidates=int(len(scores)),
                            feasible=int(feasible.sum()),
                            phi_distinct=int(len(np.unique(
                                np.round(feasible_phi, 9)
                            ))),
                            phi_range=float(
                                feasible_phi.max() - feasible_phi.min()
                            ) if len(feasible_phi) else 0.0,
                        )
                        stats = tie_statistics(
                            feasible_scores, feasible_phi,
                            args.tie_tolerance,
                        )
                        for label in ("exact", "near"):
                            entry["%s_pairs" % label] = stats[label]["pairs"]
                            entry["%s_pairs_diff_phi" % label] = (
                                stats[label]["pairs_diff_phi"]
                            )
                            entry["%s_flip" % label] = stats[label]["flip"]
                            entry["%s_phi_gap" % label] = (
                                stats[label]["phi_gap"]
                            )
                        decisions.append(entry)
                action = int(result.actions[0])
                prefix = np.concatenate([[float(action)], prefix[:-1]])
                env.step(action)
                recent = [float(action)] + recent[:-1]
    total_pairs = sum(item["near_pairs"] for item in decisions)
    total_diff = sum(item["near_pairs_diff_phi"] for item in decisions)
    exact_pairs = sum(item["exact_pairs"] for item in decisions)
    exact_diff = sum(item["exact_pairs_diff_phi"] for item in decisions)
    return dict(
        seed=int(args.seed),
        epochs=int(args.epochs),
        arm="epochs%d" % args.epochs,
        decisions=len(decisions),
        candidates_mean=(
            float(np.mean([item["candidates"] for item in decisions]))
            if decisions else 0.0
        ),
        feasible_mean=(
            float(np.mean([item["feasible"] for item in decisions]))
            if decisions else 0.0
        ),
        phi_distinct_mean=(
            float(np.mean([item["phi_distinct"] for item in decisions]))
            if decisions else 0.0
        ),
        phi_range_mean=(
            float(np.mean([item["phi_range"] for item in decisions]))
            if decisions else 0.0
        ),
        exact_tie_pairs=exact_pairs,
        exact_tie_pairs_diff_phi=exact_diff,
        near_tie_pairs=total_pairs,
        near_tie_pairs_diff_phi=total_diff,
        near_tie_phi_gap_max=(
            float(max(item["near_phi_gap"] for item in decisions))
            if decisions else 0.0
        ),
        near_flip_decisions=sum(
            1 for item in decisions if item["near_flip"]
        ),
        exact_flip_decisions=sum(
            1 for item in decisions if item["exact_flip"]
        ),
        rows=decisions,
    )


def print_summary(report):
    print()
    print(
        "== %s: %d decisions, %.1f candidates, %.1f feasible =="
        % (report["arm"], report["decisions"],
           report["candidates_mean"], report["feasible_mean"])
    )
    print(
        "  Phi: distinct values/decision %.2f, mean range %.3f"
        % (report["phi_distinct_mean"], report["phi_range_mean"])
    )
    print(
        "  exact ties: %d pairs, %d with different Phi, flips %d"
        % (report["exact_tie_pairs"],
           report["exact_tie_pairs_diff_phi"],
           report["exact_flip_decisions"])
    )
    print(
        "  near ties (tol): %d pairs, %d with different Phi, flips %d, "
        "max Phi gap %.3f"
        % (report["near_tie_pairs"], report["near_tie_pairs_diff_phi"],
           report["near_flip_decisions"], report["near_tie_phi_gap_max"])
    )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--dagger-frames", type=int, default=5000)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--tie-tolerance", type=float, default=0.01)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    args.eval_seeds = parse_seed_spec(args.eval_seeds)
    report = run(args)
    print_summary(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
