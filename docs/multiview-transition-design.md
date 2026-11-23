# Multi-view state-transition mechanism

## Decision

Use one **action-aligned empirical transition measure** and query it at
different temporal views. Do not train an independent hazard/risk model for
each fixed window, and do not invent a stage identity or a task-specific
event name.

For the learned state representation `z`, an observed row contributes

```text
(z_t, a_acting, z_(t+1), feedback_t, terminal_t)
```

where `a_acting` is the oldest pending action (`history[lag]`), not the
newly-issued action (`history[0]`). The empirical kernel is

```text
K(z' , feedback, terminal | z, a_acting)
```

with edge support retained explicitly. All samples come from actual
transitions. An unobserved state/action pair returns `UNKNOWN`; it does not
fall back to “safe”, a global mean, or a counterfactual label.

## Temporal views are queries, not model copies

The same kernel can be composed along a supplied action schedule. At query
time, request any set of windows (for example `{1, 4, 8, 16}`) and receive:

- a distribution over successor state nodes at each window;
- cumulative feedback and terminal mass;
- support and UNKNOWN mass;
- optionally, first-passage mass into a caller-declared set of target nodes.

The first-passage CDF at `h` is `P(T_target <= h)`. Differences between
successive CDF values give duration mass. Thus frame, event, and longer views
are resolutions of the same transition object; there is no separate `span8`
or `span16` classifier to calibrate.

The target is structural (a set of learned state nodes/readout outcomes), not
a task vocabulary. Generic learned nodes and the world's observed feedback
remain the only semantics.

## Training and prediction contract

1. Fit the existing state-node representation on the real stream.
2. Replay the same real rows and record action-conditioned edge counts,
   feedback sufficient statistics, terminal frequency, and support.
3. Validate by holding out whole episodes. For each held-out start, give the
   model the **actually issued** action sequence and compare its predictions
   at several windows with the realized successor states, feedback, and
   terminal outcome.
4. In candidate search, query the same transition measure with each candidate
   schedule. Never train on unexecuted candidate outcomes. If any significant
   branch has no support, surface UNKNOWN and let the existing fine executor
   remain authoritative.
5. Only after held-out transition calibration and support pass should the
   decision layer consume the distribution. Its decision comparison is
   outcome-first and paired against the unchanged default; no global
   per-frame scalar penalty is introduced.

The current first implementation is in `multiview_transitions.py`. It builds
the empirical action-conditioned frame kernel during `fit()` and exposes
`predict_views()` for arbitrary windows. This is a **correctly aligned
baseline representation**, not yet the final long-view mechanism: preliminary
leave-episode-out runs show support collapsing as frame edges are composed
(UNKNOWN mass is already about 0.6 at h8 and >0.8 at h16 on seeds 100/101).
The default planner does not consume it.

Exact prototype run-length compression is not a sufficient fix either: in
the current streams the nearest prototype changes almost every frame (run
length median 1), so it would create mostly one-frame segments. The coarse
state therefore has to be learned from **predictive equivalence**, not
consecutive prototype identity or a manually named event.

## Required second step: predictive-state quotient + stopping-time measure

Build a hierarchy of state partitions by merging fine nodes whose
action-conditioned successor/feedback distributions are statistically
indistinguishable on held-out episodes (a predictive/bisimulation quotient).
Select merge depth by held-out likelihood/support, not a fixed node quota.
Then, from each real episode, record first-passage transitions between those
coarse predictive classes:

```text
(entry class, supported action exposure) ->
    (exit class, elapsed time, accumulated feedback, terminal/censoring)
```

The action input is the **ordered, actually acting delayed-action trace**;
do not reduce it to an unordered action histogram, because that would erase
timing. Prediction supplies the candidate's ordered schedule through the same
action-conditioned coarse kernel. The segment-level exit and duration are
trained from real trajectories, so a longer view does not require multiplying
an exact chain of rare fine state/action edges. Duration is an output of the
transition; it is not a selected horizon.

The same fitted stopping-time measure answers any view window by CDF query.
Its coarse destination classes retain state-transition semantics; terminal
and feedback are additional observed outcomes, not the only targets. If the
candidate's action exposure or entry class lacks support, return UNKNOWN and
leave control to the fine executor.

An exploratory quotient builder now exists in `predictive_state_quotient.py`.
It clusters fine nodes by their observed action-conditioned successor/feedback
profiles; `multiview_transition_probe.py --quotient-groups ...` compares
candidate merge capacities on held-out episodes. This is only a capacity
sweep: there is no accepted group count, no direct segment-duration learner,
and no planner consumption yet. Whole-episode fold-refit is the default so
the held-out episodes do not define the state vocabulary.

`hierarchical_transitions.py` adds the next candidate consumer: nested
quotients (fine → coarse → root) with per-branch statistical backoff. It keeps
a branch fine while its action edge is supported, ascends only that branch
when the edge is absent, and returns UNKNOWN if even the root has no evidence.
The pilot is still pending; a high support rate alone is not a pass if the
destination/feedback distribution collapses to a constant.

Validation order:

1. Check the quotient on held-out episodes: successor distribution,
   duration likelihood, feedback likelihood, and support at multiple
   windows. Compare with the unmerged frame kernel and a pooled base rate.
2. Verify the action-history alignment and report UNKNOWN by depth.
3. Only then shadow-score real planner candidates; no decision impact yet.
4. Run paired decision A/B only if the coarse transition readout is
   calibrated and supported across seeds.

## Concrete implementation defect found

`NodeRuntime.learn_episode()` already aligns action-conditioned learning with
`history[lag]`: that is the action which caused the current state transition.
The separate planner graph builder `_transition_table()` used `history[0]`
instead (the action just issued and still pending), then merged all actions
on a source/target pair and kept only the majority action. This breaks the
identity of the learned transition before value iteration sees it.

The legacy table is now corrected to use the acting action and retain
per-edge action counts. This is a high-value bug fix, but not evidence that
the graph can already replace the fine executor.

## Scope boundaries / gates

- No pipe/gap/collision terms, task constants, or task-labelled outcomes.
- No training from counterfactual rollouts.
- No fixed episode quotas or independently fitted horizon heads.
- State prototypes may index the kernel, but continuous world prediction
  remains available to the fine executor.
- The new kernel is opt-in at consumption time. First gate is held-out,
  multi-window transition prediction with explicit UNKNOWN/support. Second
  gate is a paired decision A/B (3 seeds, then 6 to confirm).
- If learned-node coverage or action support is inadequate, do not widen the
  kernel by assigning unsupported mass to a known outcome. Keep UNKNOWN.
