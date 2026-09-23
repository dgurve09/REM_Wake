# Execution and Validation Record

**Execution date:** 2026-09-22

**Protocol commit:** `533cad3`

**Protocol correction commit:** `2eb0b7a`

**Result-producing code commit:** `b12f4f3`

**Partition boundary:** train only; validation and test were not accessed

## Pre-run Checks

Before result execution, all four frozen candidates passed output-shape and probability-range checks. The TCN retained the full eight-epoch input length. The corrected TCN used kernel size 3 and dilations 1, 2, and 4, giving a receptive field of 15 epochs.

## Execution

The nested run completed:

- 5 participant-grouped outer folds;
- 20 candidate-by-outer inner comparisons;
- 89 total model fits, each completing 60 fixed epochs;
- four deterministic duplicate-fit checks;
- 64 fixed thresholds for each candidate-by-outer comparison; and
- 202 external model, scaler, score, and membership artifacts recorded by size and SHA-256.

All 14 in-run integrity checks passed.

## Independent Reconstruction

The read-only validator was run twice after the first complete execution and once after the full deterministic rerun. Each invocation passed 12/12 checks. It independently reconstructed:

- inner threshold selections;
- five outer architecture rankings;
- fold-specific event predictions;
- aggregate and participant event metrics;
- paired participant bootstrap intervals; and
- all frozen decisions.

The full rerun reproduced the same five selected architectures, thresholds, aggregate event counts, metrics, and decisions. The final external manifest passed all 202 artifact hash checks.

## Result Boundary

The nested selected pipeline reached F1 0.1645 and 0.2812 false alarms/hour. The matched nested BLSTM-CRF reached F1 0.1604 and 0.2971 false alarms/hour. The F1 difference was +0.0041, with a paired participant 95% interval of -0.0532 to +0.0608. The false-alarm difference was -0.0159/hour, with interval -0.0621 to +0.0296/hour.

The predeclared +0.05 F1 advancement gate was not met. This experiment therefore stops at v0.1 and does not authorize replacement of `LC-1`, access to validation/test data, or post-result tuning on the outer-fold outcomes.
