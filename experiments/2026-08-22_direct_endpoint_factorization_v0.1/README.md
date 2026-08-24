# Direct Endpoint Factorization v0.1

**Created:** 2026-08-22
**Status:** Sequential exploratory validation-only experiment
**Protocol:** `docs/evaluation/direct_endpoint_factorization_protocol_v0.1.md`
**Test access:** None

## Fit Record

| Head | Train positive | Train negative | Iterations | Convergence warnings |
|---|---:|---:|---:|---:|
| rem_before | 646 | 2097 | 99 | 0 |
| wake_after | 726 | 2017 | 107 | 0 |

## Endpoint Discrimination

| Head | Validation average precision | Validation ROC AUC |
|---|---:|---:|
| rem_before | 0.5174 | 0.8096 |
| wake_after | 0.7697 | 0.9003 |

## Validation Event Comparison

| Model | Threshold | Precision | Recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| DE-B | 0.96 | 0.0648 | 0.4324 | 0.1127 | 1.4496 |
| DE-D | 0.74 | 0.0971 | 0.4595 | 0.1604 | 0.9915 |

DE-D minus DE-B event F1 was +0.0477; false alarms per hour changed by -0.4581. The prespecified two-part success rule was met: **True**.

## Partial-Endpoint Errors

DE-B produced 151 validation false positives at REM-to-other or other-to-Wake boundaries. DE-D produced 110. This comparison is descriptive because the method was designed after the earlier test failure was known.

## Decision Boundary

This experiment does not alter the frozen Block 6 test result. Even if validation improves, DE-D remains a candidate for a new locked or external evaluation; it must not be applied to the current test partition for iterative selection.

Fitted endpoint models and the full-night validation candidate-score artifact remain outside Git. Their hashes are recorded in `external_artifact_manifest_v0.1.tsv`. Git retains a compact train/validation labeled-row endpoint-score table for exact endpoint and fit-row recomputation.
