# Multi-Dataset PSG Stage-First Transition Experiment v0.1

**Protocol frozen:** 2026-10-25

**Result and diagnostic closeout completed:** 2026-10-31

## Current stage

The frozen experiment is complete. The primary `STAGE-CONFIRM2` candidate failed the exploratory point gate and is stopped at v0.1. The result is descriptive mechanism evidence on reused datasets, not independent confirmation.

## Purpose

Test whether the project has been formulating the task incorrectly as isolated rare-event classification. A five-stage model will produce causal stage predictions from the same corrected multi-dataset PSG features, and deterministic REM-to-Wake transitions will be compared with the frozen direct detector.

## Evidence boundary

This is an exploratory mechanism comparison on datasets whose direct-event results are already known. It cannot serve as independent confirmation and cannot establish wearable or clinical performance.

## Pre-fit checkpoint

- Primary audit: 13/13 checks passed.
- Independent input validation: 10/10 checks passed.
- Feature-cache hashes: 196/196 matched.
- Known causal stage rows: 372,361 across all five stages and all three datasets. The original pre-result text incorrectly stated 367,697; the retained class-count table and score artifacts establish the corrected total.
- Unknown decision rows: 672 BOAS, 2,958 gx36m, and 2,063 ISRUC group 3.
- Corrected REM-to-Wake events: 482/482 have the four consecutive decisions required by `STAGE-CONFIRM2`.
- Model and score artifacts present at checkpoint: none.

The known-row total was initially transcribed incorrectly in this README and in the first runner invariant. The retained pre-fit class-count table has always contained the per-class values that sum to 372,361. Commit `c445818` corrected the invariant after the first execution exposed the mismatch; it did not change the data, model, predictions, event rule, or gate.

## Stage results

| Held-out dataset | Accuracy | Balanced accuracy | Macro F1 | Cohen kappa |
|---|---:|---:|---:|---:|
| BOAS | 0.6266 | 0.6931 | 0.5480 | 0.4822 |
| gx36m | 0.7686 | 0.6994 | 0.7059 | 0.6783 |
| ISRUC-3 | 0.4953 | 0.5054 | 0.4266 | 0.3503 |

All three fixed logistic fits reached the 1,000-iteration ceiling. Their probabilities were finite and normalized, but non-convergence fails the numerical condition and limits coefficient-level interpretation.

## Event results

| Held-out scope | Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|
| gx36m | `STAGE-HARD` | 0.0660 | 0.4941 | 0.1165 | 1.1960 |
| gx36m | `STAGE-CONFIRM2` | 0.0954 | 0.2941 | 0.1441 | 0.4772 |
| ISRUC-3 | `STAGE-HARD` | 0.0000 | 0.0000 | 0.0000 | 0.6013 |
| ISRUC-3 | `STAGE-CONFIRM2` | 0.0000 | 0.0000 | 0.0000 | 0.2621 |
| Pooled external | `STAGE-HARD` | 0.0622 | 0.3590 | 0.1061 | 1.1273 |
| Pooled external | `STAGE-CONFIRM2` | 0.0896 | 0.2137 | 0.1263 | 0.4523 |

The corrected direct `ADV3-CHANGE` detector had pooled external F1 0.2290, recall 0.2564, and 0.2048 false alarms/hour. `STAGE-CONFIRM2` reduced F1 by 0.1027 and increased false alarms by 0.2475/hour. Its participant-stratified bootstrap F1 difference was -0.1027 at the median with a 95% interval from -0.1859 to -0.0125. The interval is entirely below zero.

## Decision

`H-STAGE-FIRST-XDATA` is not supported. `STAGE-CONFIRM2` failed every performance condition and the numerical convergence condition. Moderate epoch-level stage scores did not translate into reliable rare-boundary localization, including zero true-positive alarms on ISRUC group 3 under either fixed derivation.

Do not tune another run length, lag, threshold, or smoothing rule on these held-out results. A separately frozen post-result diagnostic will quantify sequence fragmentation and timing displacement without creating another candidate.

## Failure diagnostic

The frozen post-result diagnostic found a consistent sequence-fragmentation mechanism:

| Dataset | Predicted/human all-stage transition ratio | Predicted/human REM-bout ratio | Human median REM bout | Predicted median REM bout |
|---|---:|---:|---:|---:|
| BOAS | 2.928 | 4.607 | 600 s | 30 s |
| gx36m | 2.290 | 7.520 | 1,050 s | 45 s |
| ISRUC-3 | 1.790 | 4.974 | 120 s | 30 s |

Only 41/365 BOAS, 20/85 gx36m, and 0/32 ISRUC references had the exact predicted sequence `REM, REM, W, W` at the four aligned causal decisions. For ISRUC, `STAGE-CONFIRM2` had no alarm within the descriptive +/-120-second window around any corrected reference. This is not merely a 15-second matching-window failure.

The diagnostic does not authorize temporal smoothing, a different confirmation length, or a lag correction. It shows that independent stage decisions fragment REM structure and fail to preserve boundary timing under dataset shift. A future sequence-level stage formulation would need a new protocol and fresh evaluation data.

## Validation

- In-run controls: 14/14 passed.
- Independent result validation: 18/18 passed.
- Failure-diagnostic controls: 7/7 passed.
- Independent failure-diagnostic validation: 13/13 passed.
- External artifacts: three model files and three score files retained outside Git with size and SHA-256 records.
- Model and score artifacts at the pre-fit checkpoint: none; all six were created only after protocol and input-audit commits.

See the [frozen protocol](../../docs/evaluation/multidataset_psg_stage_first_protocol_v0.1.md), [failure-diagnostic plan](../../docs/evaluation/multidataset_psg_stage_first_failure_diagnostic_plan_v0.1.md), and [decision record](../../docs/evaluation/multidataset_psg_stage_first_decision_2026-10-31.md).
