"""Compare the query-graph world model with the shipped linear model.

The query model is an opt-in prediction path: same inputs, same target
dimensions, no planner change.  The probe checks that it matches the linear
model on held-out one-step rows, then compares both on the same eight-step
rollouts and reports the per-step error and prediction cost.
"""

import argparse
import time

import numpy as np

from query_world import fit_query_world
from substrate_generic import LinearObservationModel, collect
from substrate_nodes import LAG


def split(rows, modulus=5):
    train, test = [], []
    for row in rows:
        if int(row["episode"]) % int(modulus) == int(modulus) - 1:
            test.append(row)
        else:
            train.append(row)
    return train, test


def arrays(rows):
    return (
        np.asarray([row["obs"] for row in rows], dtype=float),
        np.asarray([row["history"] for row in rows], dtype=float),
        np.asarray([row["next_obs"] for row in rows], dtype=float),
    )


def r2(prediction, actual):
    residual = float(((prediction - actual) ** 2).sum())
    total = float(((actual - actual.mean(axis=0)) ** 2).sum())
    return 1.0 - residual / (total + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    rows, _ = collect(seed=args.seed, steps=args.steps, lag=LAG)
    train_rows, test_rows = split(rows)
    train_obs, train_hist, train_next = arrays(train_rows)
    test_obs, test_hist, test_next = arrays(test_rows)
    baseline = LinearObservationModel(
        train_obs.shape[1], train_hist.shape[1]
    ).fit(train_obs, train_hist, train_next)
    query = fit_query_world(
        train_obs, train_hist, train_next, max_terms=6,
    )

    baseline_prediction = baseline.predict(test_obs, test_hist)
    query_prediction = query.predict(test_obs, test_hist)
    print("== seed %d train %d test %d ==" % (
        args.seed, len(train_rows), len(test_rows),
    ))
    print("teacher-forced target  baseline RMSE  query RMSE  baseline R2  "
          "query R2")
    for target in range(test_next.shape[1]):
        baseline_rmse = float(np.sqrt(np.mean(
            (baseline_prediction[:, target] - test_next[:, target]) ** 2
        )))
        query_rmse = float(np.sqrt(np.mean(
            (query_prediction[:, target] - test_next[:, target]) ** 2
        )))
        print(
            "%-30d %13.6f %10.6f %12.6f %9.6f" % (
                target, baseline_rmse, query_rmse,
                r2(baseline_prediction[:, target], test_next[:, target]),
                r2(query_prediction[:, target], test_next[:, target]),
            )
        )
    print(
        "max |query - baseline| one-step: %.3e" % float(
            np.max(np.abs(query_prediction - baseline_prediction))
        )
    )

    scale = np.where(train_obs.std(axis=0) > 1e-8, train_obs.std(axis=0), 1.0)
    errors_baseline = [[] for _ in range(args.horizon)]
    errors_query = [[] for _ in range(args.horizon)]
    episodes = {}
    for row in test_rows:
        episodes.setdefault(int(row["episode"]), []).append(row)
    for sequence in episodes.values():
        for start in range(0, len(sequence) - args.horizon):
            state_baseline = np.asarray(sequence[start]["obs"], dtype=float)
            state_query = state_baseline.copy()
            for step in range(args.horizon):
                row = sequence[start + step]
                history = np.asarray(row["history"], dtype=float)
                actual = np.asarray(row["next_obs"], dtype=float)
                predicted_baseline = baseline.predict(
                    state_baseline.reshape(1, -1),
                    history.reshape(1, -1),
                )[0]
                predicted_query = query.predict(
                    state_query.reshape(1, -1),
                    history.reshape(1, -1),
                )[0]
                errors_baseline[step].append(float(np.sqrt(np.mean(
                    ((predicted_baseline - actual) / scale) ** 2
                ))))
                if np.all(np.isfinite(predicted_query)):
                    errors_query[step].append(float(np.sqrt(np.mean(
                        ((predicted_query - actual) / scale) ** 2
                    ))))
                state_baseline = predicted_baseline
                state_query = predicted_query
    print()
    print("rollout step  baseline RMSE  query RMSE  query rows")
    for step in range(args.horizon):
        print(
            "%12d %14.6f %11s %9d" % (
                step + 1, np.mean(errors_baseline[step]),
                ("%.6f" % np.mean(errors_query[step]))
                if errors_query[step] else "nan",
                len(errors_query[step]),
            )
        )

    start = time.perf_counter()
    for index in range(min(200, len(test_obs))):
        baseline.predict(
            test_obs[index:index + 1], test_hist[index:index + 1]
        )
    baseline_ms = (time.perf_counter() - start) * 1000.0 / min(
        200, len(test_obs)
    )
    start = time.perf_counter()
    for index in range(min(200, len(test_obs))):
        query.predict(
            test_obs[index:index + 1], test_hist[index:index + 1]
        )
    query_ms = (time.perf_counter() - start) * 1000.0 / min(
        200, len(test_obs)
    )
    print()
    print(
        "latency ms/row: baseline %.4f  query %.4f  ratio %.1fx" % (
            baseline_ms, query_ms, query_ms / max(baseline_ms, 1e-9),
        )
    )


if __name__ == "__main__":
    main()
