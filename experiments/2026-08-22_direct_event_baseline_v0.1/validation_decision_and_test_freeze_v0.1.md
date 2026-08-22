# Direct Baseline Validation Decision and Test Freeze v0.1

**Created:** 2026-08-22
**Marker:** `DIRECT_MODELS_AND_THRESHOLDS_FROZEN_FOR_SINGLE_TEST_EVALUATION`

The train/validation phase completed under the predeclared protocol. Test feature arrays were not loaded by this phase.

| Model | Window AP | Window ROC AUC | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| DE-A | 0.391280 | 0.859852 | 0.97 | 0.071429 | 0.297297 | 0.115183 | 0.890919 |
| DE-B | 0.454379 | 0.919006 | 0.96 | 0.064777 | 0.432432 | 0.112676 | 1.449563 |

DE-B improved validation event F1 over DE-A: **False**. This records H6.2 without changing DE-B's primary role.

## Frozen Test Decision

Apply both fitted models and their recorded validation thresholds once to test feature arrays. No model, feature, alarm, threshold, membership, tolerance, or selection rule may be changed after this file. Retain the outcome whether positive, negative, or inconclusive.
