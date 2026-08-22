# Direct Endpoint Factorization Protocol v0.1

**Created:** 2026-08-22  
**Project phase:** Block 6 method follow-up  
**Status:** Sequential exploratory improvement experiment  
**Authorized partitions:** Train and validation only  
**Test access authorized:** No

## 1. Known Evidence

The frozen DE-B test result and its post-result failure analysis are already known. DE-B improved over transparent stage-first SF-C but retained primary precision 0.0909 and 1.2571 false alarms per hour. Of its 250 test false positives, 108 occurred at human other-to-Wake pairs and 56 at human REM-to-other pairs.

Before this protocol, train/validation metadata and frozen train/validation feature-stage arrays were audited to determine whether these partial endpoints were absent from training. They were not:

| Partition | REM-to-other reviewed backgrounds | Other-to-Wake reviewed backgrounds | Contributing `pid` per category |
|---|---:|---:|---:|
| Train | 466 | 546 | 64/64 |
| Validation | 106 | 147 | 16/16 |

The full DE-B-supported train candidate sequence contains 12,132 REM-to-other and 10,460 other-to-Wake boundaries. The observed failure cannot be attributed simply to total absence of partial-endpoint examples.

This experiment was designed after the DE-B test failure was known. It is method development, not independent confirmation. The current test partition must not be loaded or evaluated.

## 2. Technological Question

Does explicitly factorizing the target into evidence for REM immediately before the boundary and Wake immediately after the boundary reduce partial-endpoint false alarms compared with one binary REM-to-Wake logistic classifier?

DE-B learns one linear decision surface for a conjunctive target. The proposed DE-D method tests whether two simple endpoint-specific surfaces and an explicit conjunction better represent the required event structure without adding a neural architecture.

## 3. Hypothesis

DE-D will exceed frozen DE-B validation event F1 0.112676 while reducing frozen DE-B validation false alarms from 1.449563 per supported hour under primary membership and +/-15 seconds.

The approach is supported only if both directions hold. Higher recall with equal or worse false-alarm burden is insufficient.

## 4. Frozen Candidate Rows and Inputs

Use exactly the same primary-quality labeled rows as DE-B:

- train: 180 REM-to-Wake positives and 2,563 reviewed backgrounds;
- validation: 37 REM-to-Wake positives and 620 reviewed backgrounds;
- no negative subsampling;
- no expanded-quality rows for fitting;
- no test rows.

Inputs are the frozen 80 DE-B features: two-channel log Welch bandpower over eight 30-second epochs from -120 through +90 seconds. No raw EDF is opened and no feature is regenerated.

For each labeled boundary at time `t`, human consensus stages define two training targets:

- `rem_before = 1` when the human stage at `t-30 s` is REM 4;
- `wake_after = 1` when the human stage at `t` is Wake 0.

Human stage codes are training labels only and are not model inputs.

## 5. Model and Score

DE-D contains two independent scikit-learn pipelines:

1. `StandardScaler` fitted on train rows only;
2. `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, tol=1e-4)`.

One pipeline predicts `P(rem_before)` and the other predicts `P(wake_after)`. The continuous event score is frozen as:

`P_direct_factorized = P(rem_before) * P(wake_after)`

The product is a deterministic conjunction score, not a calibrated joint probability claim. No head weighting, alternative combination rule, feature selection, hyperparameter search, or architecture comparison is permitted in v0.1.

## 6. Alarm and Threshold Rules

Score every supported validation boundary. Reuse the frozen direct alarm rule:

1. mark candidates at or above threshold;
2. group marked 30-second-adjacent candidates into contiguous runs;
3. emit the highest-score candidate per run, choosing the earlier time for an exact tie.

Search the fixed thresholds 0.01 through 0.99 in steps of 0.01. Select on validation primary membership and +/-15 seconds using:

1. maximum event F1;
2. lower false alarms per hour;
3. higher recall;
4. higher threshold.

This is the same validation procedure used for DE-B. Its selected performance is optimistic and is used only for a like-for-like sequential development comparison.

## 7. Reported Analyses

- endpoint-head train and validation average precision and ROC AUC;
- convergence warnings and iterations;
- complete 99-threshold validation curve;
- primary and expanded event precision, recall, F1, and false alarms per hour at +/-15 and +/-45 seconds;
- participant-bootstrap intervals with 2,000 resamples and seed `20260822`;
- partial-endpoint categories among primary +/-15-second validation false positives;
- direct comparison with the already frozen DE-B validation result.

## 8. Decision and Test Boundary

If DE-D improves both primary validation event F1 and false alarms per hour, retain it as a promising method for a future newly locked or external evaluation. Do not apply it to the current test partition.

If only one endpoint improves, retain the tradeoff as inconclusive. If neither improves, reject factorization v0.1 as insufficient. No post-result change to probability combination, threshold grid, feature context, or head configuration is allowed.

This experiment does not change the completed Block 6 primary result or the August 22 baseline-gate decision.
