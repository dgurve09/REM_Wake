# Direct REM-to-Wake Feature Baseline v0.1

**Created:** 2026-08-22
**Protocol:** `docs/evaluation/direct_event_baseline_protocol_v0.1.md`
**Models:** DE-A boundary-pair logistic ablation; DE-B eight-epoch-context logistic primary

## Validation Result

| Model | Window AP | Window ROC AUC | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| DE-A | 0.3913 | 0.8599 | 0.97 | 0.0714 | 0.2973 | 0.1152 | 0.8909 |
| DE-B | 0.4544 | 0.9190 | 0.96 | 0.0648 | 0.4324 | 0.1127 | 1.4496 |

## Frozen Test Result

| Model | Role | Precision | Recall | Event F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|
| SF-A | fixed_descriptive_unknown_training_provenance | 0.3333 | 0.4237 | 0.3731 | 0.2609 |
| SF-C | transparent_stage_first_primary | 0.0438 | 0.3051 | 0.0766 | 1.9692 |
| DE-A | direct_boundary_pair_ablation | 0.0625 | 0.2203 | 0.0974 | 0.9703 |
| DE-B | direct_eight_epoch_context_primary | 0.0909 | 0.4237 | 0.1497 | 1.2571 |

## Decision

The prespecified directional direct-value criterion was met: **True**. The criterion requires DE-B to have both higher event F1 and lower false alarms per supported hour than SF-C.

This test partition was previously used for the planned stage-first comparator. The direct configuration and thresholds were frozen before direct test-feature access, but external confirmation remains necessary.

## Artifact Boundary

Fitted models and full-night continuous candidate-score artifacts remain outside Git. Git retains the protocol, fit and threshold records, event outputs, comparison, SHA-256 manifest, and a compact train/validation labeled-row score table required for exact reconstruction of the fitted-row metrics. No continuous test-score table is retained in Git.
