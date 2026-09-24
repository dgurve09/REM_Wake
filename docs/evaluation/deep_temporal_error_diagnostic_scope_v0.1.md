# Deep Temporal Error Diagnostic Scope v0.1

**Created:** 2026-09-23

**Status:** Retrospective train-only diagnostic; not a model-selection protocol

## Purpose

The nested four-architecture experiment did not materially improve event F1. Its selected pipeline detected 32 of 180 primary train events and produced 177 false positives. Before proposing another model, this diagnostic separates four possible limitations:

1. boundary timing error caused by 30-second label resolution;
2. threshold transfer or score-calibration error;
3. weak score separation at true events; and
4. false alarms associated with nearby REM-to-Wake, Wake-to-REM, or lower-quality transitions.

## Inputs

Use only the frozen train outer-fold scores, thresholds, predictions, support, and label tables from `2026-09-22_deep_temporal_nested_cv_v0.1`. Do not fit a model and do not access validation or test scores, features, models, or results.

## Descriptive Analyses

- Re-evaluate the frozen predictions at tolerances from +/-15 to +/-135 seconds. These are timing diagnostics, not alternative performance endpoints.
- Calculate the best pooled and fold-specific thresholds on the already observed outer labels using the previously frozen 64-value threshold grid. These are optimistic, label-informed upper bounds and must not replace the frozen thresholds.
- For every primary event, summarize the selected-pipeline score at the nominal boundary and the maximum score within +/-30, +/-60, and +/-90 seconds.
- Compare detection recall for `primary_clean` and `primary_mad_flagged` events.
- Reconstruct the same quality-tier recall for the earlier LR-OOF and LC-1 outputs to test whether the association predates nested architecture selection.
- Classify primary false alarms by proximity to primary REM-to-Wake, quality-sensitivity REM-to-Wake, and Wake-to-REM transitions.

## Interpretation Rules

- If wider tolerance adds only a small number of detections, timing uncertainty alone is not the dominant limitation.
- If label-informed oracle thresholds substantially exceed frozen F1, threshold transfer is a contributor, but the oracle value is not achievable evidence.
- If even the oracle F1 remains low, further threshold tuning cannot solve the task with the current score representation.
- If false alarms cluster around reverse or lower-quality transitions, a documented physiological-context mechanism may be justified. Otherwise, generic architecture search is not justified.
- No result from this diagnostic authorizes validation/test access, model replacement, threshold replacement, or new-cohort confirmation.

## Required Outputs

- frozen tolerance curve;
- threshold upper-bound comparison;
- per-fold frozen-versus-oracle threshold table;
- event-level boundary-score diagnostic;
- event-quality recall table;
- false-alarm context table;
- source hashes and integrity checks; and
- a decision note identifying the most defensible next mechanism.
