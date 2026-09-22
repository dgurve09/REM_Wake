# Execution and Validation Record

**Protocol frozen:** 2026-09-19, commit `2f9d961`

**Initial implementation:** 2026-09-19, commit `2db3c91`

**Path-check correction:** 2026-09-19, commit `7f5df50`

**Accepted execution and validation:** 2026-09-22

## Chronology

The first result-producing run completed from `2db3c91` but was rejected because an over-broad artifact-path predicate treated the historical train-feature directory name `block7_feature_generation_validation_v0.1` as validation access. The full failed-run outputs and correction rationale are retained in `2026-09-19_lstm_crf_train_oof_v0.1_failed_path_check`.

Commit `7f5df50` narrowed only that path predicate. It did not change inputs, folds, labels, model configuration, optimization, thresholds, metrics, hypotheses, or decision rules.

The corrected implementation was accepted on 2026-09-22 after:

- all 14/14 in-run controls passed;
- the read-only independent reconstruction passed 10/10 checks twice;
- all 93 external artifacts matched their recorded size and SHA-256;
- both 99-point threshold curves and selected thresholds were reconstructed;
- event, recording, participant, bootstrap, and decision tables were reconstructed; and
- a full immutable rerun completed without changing any reviewed or external artifact.

## Data Boundary

Only the 82 train recordings from 64 `pid` groups were used. No validation or current-test feature, score, participant, model, or result artifact was accessed. Fold model states and full-night out-of-fold scores remain outside Git.

## Interpretation Boundary

The accepted result is a participant-grouped train-development screen. Threshold selection used the same pooled out-of-fold predictions reported in the result, so this is not independent confirmation. It does not revise any prior validation or test result and does not establish a deployable detector.
