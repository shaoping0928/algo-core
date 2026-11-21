"""How much can the risk column actually move a decision?

The wiring audits showed every channel can be made alive, yet survival
never moved.  This probe measures the missing link directly: at real
decision points, does the risk column change which first action wins, and
when it does, is it pointing at the plan that actually survives?

Reported per decision point:

* ``flips`` - the first action chosen with and without the risk column.
* ``oracle`` - the true survival of both plans (replayed in a copy of the
  world), so a flip can be scored right or wrong.
* ``gap`` - the risk spread across candidates relative to the total cost
  spread: the risk column's share of the decision gradient.
* ``branch`` - the risk of the best plan starting with action 1 versus
  the best starting with action 0: the contrast the planner can use at
  this state.

Nothing is learned; this is a read-only measurement on the default
planner.
"""

import argparse
import json

import numpy as np

from latent_world import latent
from planner_rollout_quality import flatten_batches, oracle_plan
from substrate_generic import history_features
from substrate_nodes import LAG, _curiosity_off, evaluate, fit, policy_result


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and sorted_values[end] == sorted_values[position]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    return ranks


def spearman(left, right):
    a = rankdata(left)
    b = rankdata(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else 0.0


def decision_points(frames, branches, tail):
    if frames < 2:
        return []
    points = set(
        int(value) for value in np.linspace(2, frames - 2, branches)
    )
    points.update(range(max(1, frames - tail), frames))
    return sorted(points)


def run(args):
    world, agent = fit(
        seed=args.seed, steps=1500, epochs=1, max_nodes=1000,
        birth_threshold=0.45, curiosity_weight=0.5, memory_credit=False,
        dagger_rounds=1, dagger_episodes=2, dagger_cap=5000,
        dagger_frames=5000, dagger_horizon=args.horizon, step_stride=2,
    )
    rows = []
    with _curiosity_off(agent):
        for eval_seed in args.eval_seeds:
            env_seed = int(eval_seed) * 10000
            frames, actions, _ = evaluate(
                args.seed, world, agent, cap=args.cap, horizon=args.horizon,
                lag=LAG, step_stride=2, env_seed=env_seed,
            )
            for point in decision_points(frames, args.branches, args.tail):
                env = latent[LAG](env_seed)
                for action in actions[:point]:
                    if env.dead:
                        break
                    env.step(int(action))
                if env.dead:
                    continue
                recent = list(reversed(actions[max(0, point - LAG):point]))
                prefix = np.zeros(LAG, dtype=float)
                prefix[:len(recent[:LAG])] = recent[:LAG]
                batches = []
                policy_result(
                    env, world, agent, prefix, args.horizon, lag=LAG,
                    step_stride=2, candidate_callback=batches.append,
                )
                if not batches:
                    continue
                trace = flatten_batches(batches)
                scores = np.asarray(trace["scores"], dtype=float)
                feasible = (
                    np.asarray(trace["feasible"], dtype=bool)
                    & np.asarray(trace["finite"], dtype=bool)
                )
                if feasible.sum() < 2:
                    continue
                sequences = np.asarray(trace["sequences"], dtype=float)
                first = sequences[:, 0].astype(int)
                total = scores[:, 0]
                risk = scores[:, 2]
                oracle = np.empty(len(sequences), dtype=int)
                for index, sequence in enumerate(sequences):
                    death, _state, _dead = oracle_plan(env, sequence)
                    oracle[index] = args.horizon + 1 if death < 0 else int(death)
                pool = np.flatnonzero(feasible)
                # First-step risk: the only step at which the plan's first
                # action is actually in force.  The summed column also
                # counts steps after a death the planner cannot see.
                history = history_features(
                    sequences, 1, prefix, action_dim=LAG + 1
                )
                pending = history[:, :LAG][:, ::-1]
                encoded = np.concatenate(
                    [np.asarray(trace["first_states"], dtype=float),
                     pending], axis=1,
                ) / agent.scales
                risk_first = np.asarray(
                    agent.pain_signal_batch(encoded), dtype=float
                )
                predicted_death = np.asarray(
                    trace["first_death"], dtype=int
                )
                best_total = pool[np.argmin(total[pool])]
                best_no_risk = pool[np.argmin((total - risk)[pool])]
                best_risk = pool[np.argmin(risk[pool])]
                a1 = pool[first[pool] == 1]
                a0 = pool[first[pool] == 0]
                branch = (
                    float(np.min(risk[a1]) - np.min(risk[a0]))
                    if len(a1) and len(a0) else 0.0
                )
                rows.append(dict(
                    eval_seed=int(eval_seed), point=int(point),
                    flip=bool(first[best_total] != first[best_no_risk]),
                    risk_only_flip=bool(first[best_total] != first[best_risk]),
                    chosen_survival=int(oracle[best_total]),
                    no_risk_survival=int(oracle[best_no_risk]),
                    risk_only_survival=int(oracle[best_risk]),
                    risk_share=float(
                        (risk[pool].max() - risk[pool].min())
                        / max(float(total[pool].max() - total[pool].min()),
                              1e-12)
                    ),
                    branch_gap=abs(branch),
                    risk_rho=spearman(risk[pool], oracle[pool]),
                    risk_rho_first=spearman(risk_first[pool], oracle[pool]),
                    tie_share=float(np.mean(
                        np.abs(total[pool] - total[pool].min()) <= 1e-9
                    )),
                    survival_span=int(
                        oracle[pool].max() - oracle[pool].min()
                    ),
                    predicted_death_share=float(np.mean(
                        predicted_death[pool] >= 0
                    )),
                ))
    return rows


def summarize(rows):
    if not rows:
        return {}
    flips = [row for row in rows if row["flip"]]
    right = sum(
        1 for row in flips
        if row["no_risk_survival"] > row["chosen_survival"]
    )
    wrong = sum(
        1 for row in flips
        if row["no_risk_survival"] < row["chosen_survival"]
    )
    risk_only_right = sum(
        1 for row in rows
        if row["risk_only_survival"] > row["chosen_survival"]
    )
    return dict(
        decisions=len(rows),
        flips=len(flips),
        flip_rate=len(flips) / len(rows),
        flip_right=right,
        flip_wrong=wrong,
        risk_only_right=risk_only_right,
        mean_risk_share=float(np.mean([row["risk_share"] for row in rows])),
        median_risk_share=float(np.median([row["risk_share"] for row in rows])),
        mean_branch_gap=float(np.mean([row["branch_gap"] for row in rows])),
        mean_risk_rho=float(np.mean([row["risk_rho"] for row in rows])),
        mean_risk_rho_first=float(np.mean(
            [row["risk_rho_first"] for row in rows]
        )),
        share_rho_informative=float(np.mean(
            [row["risk_rho"] <= -0.2 for row in rows]
        )),
        share_rho_wrong=float(np.mean(
            [row["risk_rho"] >= 0.2 for row in rows]
        )),
        mean_tie_share=float(np.mean([row["tie_share"] for row in rows])),
        mean_survival_span=float(np.mean(
            [row["survival_span"] for row in rows]
        )),
        predicted_death_share=float(np.mean(
            [row["predicted_death_share"] for row in rows]
        )),
        mean_survival=float(np.mean([row["chosen_survival"] for row in rows])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-seeds", default="120-124")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    from eval_protocol import parse_seed_spec
    args.eval_seeds = parse_seed_spec(args.eval_seeds)

    rows = run(args)
    summary = summarize(rows)
    print("== risk contrast: seed %d, %d decisions ==" % (
        args.seed, summary.get("decisions", 0)
    ))
    if summary:
        print("  risk flips the first action: %d (%.1f%%) | "
              "flip right %d / wrong %d" % (
                  summary["flips"], 100.0 * summary["flip_rate"],
                  summary["flip_right"], summary["flip_wrong"],
              ))
        print("  risk-only argmin beats the planner: %d" %
              summary["risk_only_right"])
        print("  risk share of the cost gradient: mean %.3f median %.3f" % (
            summary["mean_risk_share"], summary["median_risk_share"]
        ))
        print("  risk gap between action branches: mean %.4f" %
              summary["mean_branch_gap"])
        print("  mean rank corr(risk, oracle survival) %.3f | "
              "first-step %.3f" % (
                  summary["mean_risk_rho"], summary["mean_risk_rho_first"]
              ))
        print("  decisions where risk orders plans correctly "
              "(rho<=-0.2) %.0f%% | wrongly (rho>=0.2) %.0f%%" % (
                  100.0 * summary["share_rho_informative"],
                  100.0 * summary["share_rho_wrong"],
              ))
        print("  exact tie share %.2f | survival span across plans %.2f" % (
            summary["mean_tie_share"], summary["mean_survival_span"]
        ))
        print("  candidates with a predicted death %.0f%% (planner is "
              "death-blind when 0)" % (
                  100.0 * summary["predicted_death_share"]
              ))
        print("  mean chosen survival %.2f" % summary["mean_survival"])
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(summary=summary, rows=rows), handle,
                      indent=2, sort_keys=True)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
