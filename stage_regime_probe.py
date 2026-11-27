"""Multi-seed episode-held-out probe for stage regime summarization and
controllable multi-step stage transition learning.

Evaluates strictly causal metrics (no future action-sequence oracle):
  1. Stage summarization & pure-drift stripping (D1/D2).
  2. Held-out long-horizon information content vs frame clock & continuous obs (D4).
  3. 1-step OOF-shrunk causal transitions (ALL and mid/late idx>=4, pooled vs action profile).
  4. Non-terminal action controllability (pure surviving next-regime delta-NLL and TV).
  5. Multi-step causal rollout (k=1, 2, 3, 5 stages) for terminal AUC, Brier skill,
     and elapsed frame prediction.

Probe-only; does not alter default training or planner paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from stage_regime import (
    CalibratedRegimeTransitionModel,
    StageRegimeSummarizer,
)
from substrate_generic import collect
from substrate_nodes import LAG
from substrate_value import binary_auc

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
DEFAULT_JOBS = min(8, os.cpu_count() or 1) if FREE_THREADED else 1


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _ridge_fit_predict(X_tr, y_tr, X_te, ridge=1.0):
    X_tr = np.asarray(X_tr, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z_tr = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sd])
    Z_te = np.column_stack([np.ones(len(X_te)), (X_te - mu) / sd])
    pen = float(ridge) * np.eye(Z_tr.shape[1])
    pen[0, 0] = 0.0
    w = np.linalg.solve(Z_tr.T @ Z_tr + pen, Z_tr.T @ y_tr)
    return Z_te @ w


def split_rows_by_episode(rows, folds=5, validation_fold=4):
    """Whole-episode split with zero episode leakage."""
    seen = []
    seen_set = set()
    for row in rows:
        ep = row.get("episode", 0)
        if ep not in seen_set:
            seen_set.add(ep)
            seen.append(ep)
    val_ids = {
        ep for idx, ep in enumerate(seen)
        if idx % int(folds) == int(validation_fold)
    }
    train_rows = [r for r in rows if r.get("episode", 0) not in val_ids]
    val_rows = [r for r in rows if r.get("episode", 0) in val_ids]
    return train_rows, val_rows


def score_one_step(model: CalibratedRegimeTransitionModel, episodes, min_idx=0):
    nlls, prior_nlls, briers, prior_briers = [], [], [], []
    hits, prior_hits, act_used = 0, 0, 0
    for stages in episodes:
        for idx, stage in enumerate(stages):
            if idx < int(min_idx):
                continue
            out = CalibratedRegimeTransitionModel.realized_outcome(stages, idx)
            if out[0] == "censored":
                continue
            hist = tuple(s.regime for s in stages[:idx + 1])
            pred = model.predict(hist, stage.action_profile)
            if pred["status"] != "supported":
                continue
            dist = pred["distribution"]
            pdist = model.prior(stage.action_profile)
            act_used += int(pred["action_conditioned"])
            p = max(dist.get(out, 0.0), 1e-6)
            pb = max(pdist.get(out, 0.0), 1e-6)
            nlls.append(-math.log(p))
            prior_nlls.append(-math.log(pb))
            briers.append(
                sum((v - (1.0 if k == out else 0.0)) ** 2 for k, v in dist.items())
                + (1.0 if out not in dist else 0.0)
            )
            prior_briers.append(
                sum((v - (1.0 if k == out else 0.0)) ** 2 for k, v in pdist.items())
                + (1.0 if out not in pdist else 0.0)
            )
            top = max(dist.items(), key=lambda kv: (kv[1], repr(kv[0])))[0]
            ptop = max(pdist.items(), key=lambda kv: (kv[1], repr(kv[0])))[0]
            hits += int(top == out)
            prior_hits += int(ptop == out)
    if not nlls:
        return dict(n=0)
    return dict(
        n=len(nlls),
        top1=round(hits / len(nlls), 4),
        base_top1=round(prior_hits / len(nlls), 4),
        nll_gain=round(float(np.mean(prior_nlls) - np.mean(nlls)), 4),
        brier_skill=round(
            1.0 - float(np.mean(briers)) / max(float(np.mean(prior_briers)), 1e-12),
            4,
        ),
        action_conditioned_rate=round(act_used / len(nlls), 4),
    )


def score_nonterminal_action_gain(m_pool, m_act, episodes):
    """Isolate surviving next-regime prediction gain from action profile."""
    nll_p, nll_a = [], []
    hit_p, hit_a = 0, 0
    for stages in episodes:
        for idx, stage in enumerate(stages):
            out = CalibratedRegimeTransitionModel.realized_outcome(stages, idx)
            if out[0] != "regime":
                continue
            hist = tuple(s.regime for s in stages[:idx + 1])
            dp = m_pool.predict(hist, None)["distribution"]
            da = m_act.predict(hist, stage.action_profile)["distribution"]
            sp = sum(v for k, v in dp.items() if k[0] == "regime")
            sa = sum(v for k, v in da.items() if k[0] == "regime")
            if sp <= 0.0 or sa <= 0.0:
                continue
            dp_r = {k: v / sp for k, v in dp.items() if k[0] == "regime"}
            da_r = {k: v / sa for k, v in da.items() if k[0] == "regime"}
            nll_p.append(-math.log(max(dp_r.get(out, 0.0), 1e-6)))
            nll_a.append(-math.log(max(da_r.get(out, 0.0), 1e-6)))
            hit_p += int(max(dp_r.items(), key=lambda kv: kv[1])[0] == out)
            hit_a += int(max(da_r.items(), key=lambda kv: kv[1])[0] == out)
    if not nll_p:
        return dict(n=0)
    n = len(nll_p)
    return dict(
        n=n,
        pool_top1=round(hit_p / n, 4),
        action_top1=round(hit_a / n, 4),
        delta_nll=round(float(np.mean(nll_p) - np.mean(nll_a)), 4),
    )


def score_multistep_rollout(model: CalibratedRegimeTransitionModel, episodes, horizons=(1, 2, 3, 5)):
    out = {}
    for k in horizons:
        term_probs, term_labels = [], []
        term_probs_mid, term_labels_mid = [], []
        elapsed_true, elapsed_pred = [], []
        surv_hits, surv_total = 0, 0
        for stages in episodes:
            for start in range(len(stages)):
                if stages[start].censored:
                    continue
                actual_term = False
                actual_cens = False
                el = 0
                for step in range(k):
                    pos = start + step
                    if pos >= len(stages):
                        actual_term = True
                        break
                    el += stages[pos].duration
                    if stages[pos].terminal:
                        actual_term = True
                        break
                    if stages[pos].censored:
                        actual_cens = True
                        break
                if actual_cens:
                    continue
                hist = tuple(s.regime for s in stages[:start + 1])
                roll = model.rollout(hist, steps=k, action_profiles=None)
                term_probs.append(roll["terminal_probability"])
                term_labels.append(1.0 if actual_term else 0.0)
                elapsed_true.append(float(el))
                elapsed_pred.append(float(roll["expected_elapsed_frames"]))
                if start >= 4:
                    term_probs_mid.append(roll["terminal_probability"])
                    term_labels_mid.append(1.0 if actual_term else 0.0)
                if not actual_term and start + k < len(stages) and roll["regime_distribution"]:
                    top_r = max(
                        roll["regime_distribution"].items(), key=lambda kv: kv[1]
                    )[0]
                    surv_total += 1
                    surv_hits += int(top_r == stages[start + k].regime)
        tl = np.asarray(term_labels, dtype=float)
        tp = np.asarray(term_probs, dtype=float)
        br = float(tl.mean()) if len(tl) else 0.0
        bs = (
            1.0 - float(np.mean((tp - tl) ** 2)) / max(br * (1.0 - br), 1e-12)
            if 0.0 < br < 1.0 else float("nan")
        )
        auc_all = binary_auc(term_probs, term_labels) if len(set(term_labels)) > 1 else float("nan")
        auc_mid = binary_auc(term_probs_mid, term_labels_mid) if len(set(term_labels_mid)) > 1 else float("nan")
        el_mae = float(np.mean(np.abs(np.asarray(elapsed_true) - np.asarray(elapsed_pred)))) if elapsed_true else float("nan")
        out[f"k={k}"] = dict(
            samples=len(term_labels),
            mean_frames=round(float(np.mean(elapsed_true)), 1) if elapsed_true else 0.0,
            elapsed_mae=round(el_mae, 2) if np.isfinite(el_mae) else None,
            term_base=round(br, 3),
            term_auc=round(float(auc_all), 4) if np.isfinite(auc_all) else None,
            term_auc_idx4=round(float(auc_mid), 4) if np.isfinite(auc_mid) else None,
            term_brier_skill=round(float(bs), 4) if np.isfinite(bs) else None,
            surv_top1=round(surv_hits / max(surv_total, 1), 4),
            surv_n=surv_total,
        )
    return out


def score_long_horizon_info(summarizer: StageRegimeSummarizer, train_eps, val_eps):
    def extract(eps):
        rows = []
        for stages in eps:
            if not stages or stages[-1].censored:
                continue
            tot = sum(s.duration for s in stages)
            el = 0
            for idx, s in enumerate(stages):
                rows.append(dict(
                    idx=idx, elapsed=float(el), regime=s.regime,
                    obs=s.entry_state[summarizer.keep_dims_],
                    rem=float(tot - el),
                ))
                el += s.duration
        return rows

    tr = extract(train_eps)
    va = extract(val_eps)
    if not tr or not va:
        return dict(n=0)
    y_tr = np.asarray([r["rem"] for r in tr])
    y_va = np.asarray([r["rem"] for r in va])
    gmean = float(y_tr.mean())
    p_clock = _ridge_fit_predict(
        [[r["elapsed"], r["elapsed"] ** 2, float(r["idx"])] for r in tr],
        y_tr,
        [[r["elapsed"], r["elapsed"] ** 2, float(r["idx"])] for r in va],
    )
    reg_vals = defaultdict(list)
    for r in tr:
        reg_vals[r["regime"]].append(r["rem"])
    p_reg = np.asarray([
        (sum(reg_vals[r["regime"]]) + 5.0 * gmean) / (len(reg_vals[r["regime"]]) + 5.0)
        for r in va
    ])
    X_tr = np.asarray([r["obs"] for r in tr])
    X_va = np.asarray([r["obs"] for r in va])
    p_obs = _ridge_fit_predict(X_tr, y_tr, X_va)
    aucs = {}
    for thresh in (20, 40):
        lbl = (y_va >= thresh).astype(float)
        if len(set(lbl.tolist())) < 2:
            continue
        aucs[f"ge_{thresh}"] = dict(
            pos=round(float(lbl.mean()), 3),
            clock_auc=round(float(binary_auc(p_clock, lbl)), 4),
            regime_auc=round(float(binary_auc(p_reg, lbl)), 4),
            obs_auc=round(float(binary_auc(p_obs, lbl)), 4),
        )
    return dict(
        n=len(va),
        r2_clock=round(_r2(y_va, p_clock), 4),
        r2_regime=round(_r2(y_va, p_reg), 4),
        r2_obs=round(_r2(y_va, p_obs), 4),
        aucs=aucs,
    )


def evaluate_seed(seed, steps=6500, coverage=0.1, n_regimes=16, min_dwell=6,
                  hysteresis=2, folds=5, validation_fold=4, lag=LAG):
    rows, _ = collect(int(seed), steps=int(steps), coverage=float(coverage), lag=int(lag))
    tr_rows, va_rows = split_rows_by_episode(rows, folds=folds, validation_fold=validation_fold)
    summarizer = StageRegimeSummarizer(
        n_regimes=n_regimes, min_dwell=min_dwell, hysteresis=hysteresis,
    ).fit_rows(tr_rows, lag=lag)
    tr_eps = summarizer.transform_rows(tr_rows, lag=lag)
    va_eps = summarizer.transform_rows(va_rows, lag=lag)

    m_pool = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2).fit(tr_eps, use_action=False)
    m_act = CalibratedRegimeTransitionModel(max_depth=2, min_episodes=2).fit(tr_eps, use_action=True)

    all_tr = [s for ep in tr_eps for s in ep]
    durs = np.asarray([s.duration for s in all_tr], dtype=float)
    profs = Counter(s.action_profile for s in all_tr)
    one_pool_all = score_one_step(m_pool, va_eps, min_idx=0)
    one_pool_4 = score_one_step(m_pool, va_eps, min_idx=4)
    one_act_all = score_one_step(m_act, va_eps, min_idx=0)
    one_act_4 = score_one_step(m_act, va_eps, min_idx=4)
    nonterm = score_nonterminal_action_gain(m_pool, m_act, va_eps)
    rollout = score_multistep_rollout(m_pool, va_eps, horizons=(1, 2, 3, 5))
    d4 = score_long_horizon_info(summarizer, tr_eps, va_eps)
    ctrl = m_act.controllability_report(n_regimes)

    gate_passed = bool(
        summarizer.drift_dims_ != ()
        and one_pool_all.get("nll_gain", -1.0) > 0.0
        and one_act_all.get("nll_gain", -1.0) > 0.0
        and nonterm.get("delta_nll", -1.0) > 0.0
        and rollout["k=2"].get("term_brier_skill", -1.0) > 0.0
        and d4.get("r2_regime", -1.0) > d4.get("r2_clock", 0.0)
        and ctrl["controllable_regimes"] >= max(2, n_regimes // 2)
    )
    return dict(
        seed=int(seed),
        summarizer=summarizer.snapshot(),
        train_episodes=len(tr_eps),
        val_episodes=len(va_eps),
        duration_mean=round(float(durs.mean()), 2),
        duration_p50=round(float(np.median(durs)), 1),
        duration_p90=round(float(np.quantile(durs, 0.90)), 1),
        unique_action_profiles=len(profs),
        singleton_profiles=sum(1 for c in profs.values() if c == 1),
        alpha_pool=m_pool.alpha_,
        alpha_act=m_act.alpha_,
        one_step_pool_all=one_pool_all,
        one_step_pool_idx4=one_pool_4,
        one_step_act_all=one_act_all,
        one_step_act_idx4=one_act_4,
        nonterminal_action_gain=nonterm,
        controllability=ctrl,
        multistep_rollout=rollout,
        long_horizon_info=d4,
        gate_passed=gate_passed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed causal audit for stage regime summarization & transitions."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--steps", type=int, default=6500)
    parser.add_argument("--coverage", type=float, default=0.1)
    parser.add_argument("--n-regimes", type=int, default=16)
    parser.add_argument("--min-dwell", type=int, default=6)
    parser.add_argument("--hysteresis", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    def _run_seed(seed):
        return evaluate_seed(
            seed, steps=args.steps, coverage=args.coverage,
            n_regimes=args.n_regimes, min_dwell=args.min_dwell,
            hysteresis=args.hysteresis,
        )

    jobs = max(1, int(args.jobs))
    if jobs > 1 and len(args.seeds) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(args.seeds))) as pool:
            results = list(pool.map(_run_seed, args.seeds))
    else:
        results = [_run_seed(seed) for seed in args.seeds]

    for seed, res in zip(args.seeds, results):
        print(
            f"seed {seed}: drift={res['summarizer']['drift_dims']} "
            f"dur_p50={res['duration_p50']} (p90={res['duration_p90']}) "
            f"profs={res['unique_action_profiles']} (singletons={res['singleton_profiles']}) "
            f"ctrl={res['controllability']['controllable_regimes']}/{args.n_regimes} (TV={res['controllability']['mean_max_tv']:.3f})"
        )
        print(
            f"  1-step pool(ALL/4+): nll_gain={res['one_step_pool_all']['nll_gain']}/{res['one_step_pool_idx4'].get('nll_gain')} "
            f"act(ALL/4+): nll_gain={res['one_step_act_all']['nll_gain']}/{res['one_step_act_idx4'].get('nll_gain')} "
            f"nonterm_act_dNLL={res['nonterminal_action_gain']['delta_nll']:+.4f} "
            f"({res['nonterminal_action_gain']['pool_top1']:.3f}->{res['nonterminal_action_gain']['action_top1']:.3f})"
        )
        r1 = res["multistep_rollout"]["k=1"]
        r2 = res["multistep_rollout"]["k=2"]
        r3 = res["multistep_rollout"]["k=3"]
        print(
            f"  rollout k=1/2/3: frames={r1['mean_frames']}/{r2['mean_frames']}/{r3['mean_frames']} "
            f"term_auc={r1['term_auc']}/{r2['term_auc']}/{r3['term_auc']} "
            f"brier_skill={r1['term_brier_skill']}/{r2['term_brier_skill']}/{r3['term_brier_skill']}"
        )
        d4 = res["long_horizon_info"]
        print(
            f"  D4 info: R2 clock={d4['r2_clock']} vs regime={d4['r2_regime']} (obs={d4['r2_obs']}) | "
            f"surv>=20 AUC clock={d4['aucs'].get('ge_20',{}).get('clock_auc')} vs regime={d4['aucs'].get('ge_20',{}).get('regime_auc')} | "
            f"gate={res['gate_passed']}"
        )

    passed = sum(1 for r in results if r["gate_passed"])
    print(f"\nSUMMARY: {passed}/{len(results)} seeds passed Phase 2+3 causal long-horizon gate.")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(dict(results=results, passed=passed, total=len(results)), handle, indent=2)


if __name__ == "__main__":
    main()
