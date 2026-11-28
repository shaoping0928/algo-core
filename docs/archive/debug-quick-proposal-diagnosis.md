# Debug Session: quick-proposal-diagnosis
- **Status**: [OPEN]
- **progress_status**: analyzed
- **Issue**: Determine whether short-chunk search and failure backtracking can rapidly propose feasible decision strategies, and identify where target credit is lost.
- **PROJECT_ROOT**: `C:\workspace\code\algo_core`
- **Evidence Source**: HTTP Debug Server plus structured probe JSON
- **Evidence Directory**: `C:\Users\62733\AppData\Local\Temp\trae-debug\algo_core\quick-proposal-diagnosis`
- **Debug Server**: `http://127.0.0.1:7777/event`
- **Log File**: `C:\Users\62733\AppData\Local\Temp\trae-debug\algo_core\quick-proposal-diagnosis\trae-debug-log-quick-proposal-diagnosis.ndjson`

## Reproduction Steps
1. Train seed 100 for 1500 steps with 5 epochs, max_nodes 1000 and coverage 0.1.
2. Evaluate seeds 120-127 with horizon 8 and 12, beam width 4, hybrid stage demand.
3. Compare per-frame search with chunk size 3 and inspect actual preterminal transitions and chunk checkpoints.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence / Status |
|----|------------|------------|--------|-------------------|
| A | Global value for the alignment predicate conflicts with target-specific successor lift. | High | Low | **CONFIRMED** by the prior same-seed epochs=5 audit: feature_35 has weak positive global return evidence but mean target-specific lift contribution -0.243. |
| B | In preterminal windows, the requested successor is absent from all explored frontiers at H=8 and H=12. | High | Medium | **CONFIRMED as an observation**, not as proof of dynamical unreachability: 32/32 H=8 and 55/55 H=12 postmortem rows have `target_never_entered_frontier`. |
| C | Some target candidates reach the final beam, but the independent objective never selects the target endpoint. | High | Low | **CONFIRMED**: final-retained target decisions are 8/196 at H=8 frame, 8/196 at H=8 chunk=3, and 2/196 at H=12; selected target endpoints are 0/196 in all ranked arms. |
| D | Three-frame chunking increases candidate work without improving policy outcomes. | Medium | Low | **CONFIRMED for H=8**: per-frame and chunk=3 outcomes match exactly while mean candidates/decision rises 54 to 94. |
| E | Lagged action effects are mispredicted, invalidating short-rollout feasibility estimates. | Medium | Medium | **INCONCLUSIVE overall**: this run's actual-transition one-step residuals are near numerical precision, but they do not measure counterfactual multi-step rollout error and do not reconcile the earlier separate ~0.33 report. |

## Instrumentation
- Existing `stage_rank_gate_probe.py` postmortem fields are reused.
- Added only a tagged probe measurement, `#region debug-point E`, to calculate normalized one-step model error from actual recorded state, action history, and next observation.
- No policy, value, transition, or training logic was changed.

## Log Evidence
- Bootstrap POST accepted by `http://127.0.0.1:7777/event`; session env file identifies the active endpoint.
- The PowerShell health request could not run in non-interactive mode; the POST probe succeeded, so HTTP evidence transport is usable. Health status remains unverified.
- The structured reports are the active evidence source; the HTTP NDJSON file remains empty, so no claim is based on server-side log lines.
- `pre-h8-frame.json`: baseline 25.0 frames, ranked 34.875 (median 30), 0 passes; mean candidates/decision 54.0. Target entered/retained/final-retained/selected: 20/196, 20/196, 8/196, 0/196. All 32 preterminal rows report `target_never_entered_frontier`.
- `pre-h8-chunk3.json`: the same per-world ranked frames and pass counts as per-frame; mean candidates/decision rises to 94.0. Target entered/retained/final-retained/selected: 20/196, 17/196, 8/196, 0/196. All 32 preterminal rows report `target_never_entered_frontier`.
- `pre-h12-chunk3.json`: baseline 25.0 frames, ranked 34.875 (median 30), 0 passes; mean candidates/decision 182.0. Target entered/retained/final-retained/selected: 20/196, 17/196, 2/196, 0/196. All 55 preterminal rows report `target_never_entered_frontier`.
- H=8 residuals (32 rows; frame/chunk reports duplicate the same actual trajectories): mean `6.72e-8`, median `6.10e-8`, p95 `1.23e-7`, max `1.34e-7`. H=12 residuals (55 rows): mean `8.06e-8`, median `6.50e-8`, p95 `1.86e-7`, max `1.89e-7`. No large actual-transition residual cluster is visible near terminal frames or split by executed action.
- Residual scope: `world.predict` is the fitted `LinearObservationModel`; inputs are the recorded real `obs` and `history` (`[current_action, most-recent prior actions...]`), and the target is recorded `next_obs`. This validates only one-step predictions on actual transitions. It does not validate hypothetical branches, accumulated rollout error, or the separate prior probe's ~0.33 lag-window result.
- Supporting association evidence is recorded in `TODO.md`: after epochs=5, feature_35 has limited global value support, but only 24/66 active candidates have target-successor association support and mean per-feature lift is -0.243.

## Follow-up Prototype
- Added an episode-level strategy credit bank and a bounded top-3 successor proposal slate. The slate shares one chunked beam; actual stage matches and terminal outcomes update credit, while unresolved episode resets are censored. Credit changes only later rank priority.
- Environment episode IDs are now passed through provider reset so distinct held-out worlds are not collapsed into one credit sample.
- `pre-h8-chunk3-proposals.json`: 8 held-out worlds produced 56 actual intent outcomes (32 target matches, 20 different realized stages, 4 terminal misses); 16 intents proposed multiple successors, 72 total proposals. Credit adjusted 42 proposal rankings. Search remained 94 candidates/decision; ranked survival stayed 34.875 frames with 0 passes; selected target endpoint was 0/284 proposal slots. Proposal-rank evidence was concentrated on `tree 12`.
- Proposal generation, shared short-chunk screening, and real outcome credit are connected, but this smoke showed no decision improvement. The proposal credit therefore remains a rank hint, not a validated high-level policy-value estimate.
- Full regression suite after this prototype: 745 tests passed.
- Added an opt-in chunk-boundary `rank_group_quota`. With quota 1, the same held-out smoke retained the only positively supported proposal group (`tree 12`) in all 216/216 relevant frontiers; no other target had positive supported rank evidence. Candidate count, ranked survival (34.875 frames), passes (0), and selected target slots (0/284) were unchanged.
- Proposal slate width now uses the learned outcome-value margin and primary strategy's episode-match posterior: uncertain/unvalidated/mismatched primaries keep up to three proposals; a well-supported, high-confidence primary can collapse the slate.
- In the same 8-world smoke, 24/56 intents used slate width 2, 32/56 used width 3, and none collapsed to 1. The sampled contexts did not provide enough confidence to prune to one strategy; policy metrics remained unchanged.
- Follow-up regression suite after adaptive slate and group quota: 751 tests passed. This confirms the wiring and preservation mechanism, not a strategy-quality gain.

## Verification Conclusion
The original diagnostic identifies two separate losses: preterminal target stages usually never enter the explored candidate frontier, while in other decisions target candidates survive to the final beam but are never chosen by the independent objective. The follow-up prototype connects bounded successor proposals and real episode credit, but the held-out smoke did not improve survival or endpoint selection. Chunking still adds no H=8 policy benefit. The evidence does not prove whether absent targets are dynamically unreachable versus missing due to feature association, resolver support, or candidate generation; nor does it establish multi-step counterfactual model accuracy. The session remains open for further verification.

## Final Evidence Gate
- **Evidence Sources**: `C:\Users\62733\AppData\Local\Temp\trae-debug\algo_core\quick-proposal-diagnosis\pre-h8-frame.json`, `pre-h8-chunk3.json`, `pre-h12-chunk3.json`, `pre-h8-chunk3-proposals.json`; prior feature association measurements summarized in `TODO.md`. HTTP NDJSON exists but is empty.
- **progress_status**: analyzed
- **Hypothesis Conclusions**: A CONFIRMED; B CONFIRMED as frontier absence only (dynamic unreachability INCONCLUSIVE); C CONFIRMED; D CONFIRMED for H=8 outcomes/candidate counts; E INCONCLUSIVE for multi-step/counterfactual accuracy, with no evidence of large actual one-step residuals in these runs.
