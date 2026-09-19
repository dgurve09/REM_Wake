# First-Run Path-Check Failure

**Run date:** 2026-09-19

**Result-producing code commit:** `2db3c91`

**Run outcome:** Rejected before result acceptance

## Failure

Thirteen of fourteen in-run controls passed. The `validation_and_test_closed` control failed because its regular expression rejected any artifact path containing the word `validation`.

All 82 source feature paths include the historical directory name `block7_feature_generation_validation_v0.1`, even though those specific files are the frozen train features. The manifest contained no validation or test recording-feature segment, participant, score, or model artifact. Therefore, the failure was caused by an over-broad path-name predicate rather than unauthorized data access.

## Observed Output

The unaccepted first run produced the following primary train out-of-fold values:

| Model | Threshold | F1 | False alarms/hour |
|---|---:|---:|---:|
| LR-OOF | 0.98 | 0.0982 | 1.4679 |
| LC-1 | 0.99 | 0.1965 | 0.5306 |

These values were not accepted at this stage because one required control was failed.

## Correction

The predicate was narrowed to reject explicit `/validation/` and `/test/` path segments or a `test_` path component. The scientific method, inputs, folds, labels, models, thresholds, metrics, and decision rules were not changed. The full first-run outputs are retained in this folder, and the corrected implementation must produce a new reviewed output folder before interpretation.
