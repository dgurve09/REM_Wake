# Multi-Dataset PSG Stage-First Failure Diagnostic Plan v0.1

**Frozen:** 2026-10-25, after primary results
**Status:** Descriptive post-result analysis; no model or event-rule selection

## Question

Why did five-stage classification produce moderate-to-good epoch metrics while deterministic REM-to-Wake event derivation underperformed the corrected direct detector?

## Fixed inputs

Use only the retained stage-score artifacts, corrected primary stage sequences, and frozen `STAGE-HARD` and `STAGE-CONFIRM2` alarms. Do not refit a model, alter a prediction, change the event matcher, or create another candidate.

## Fixed diagnostics

For each held-out dataset, report:

1. human and predicted all-stage transition counts per supported hour;
2. human and predicted REM-bout counts per supported hour;
3. median human and predicted REM-bout duration within contiguous known-stage segments;
4. the fraction of corrected REM-to-Wake references whose four aligned decisions have the exact predicted pattern `REM, REM, W, W`;
5. the fraction with a predicted `REM` to `W` change at the nominal scored-event midpoint;
6. for each frozen alarm rule, reference coverage by any alarm from 120 seconds before through 120 seconds after the reference; and
7. the median nearest-alarm signed offset for references having an alarm in that diagnostic window.

The four aligned decision times for a reference at `r` are `r-30`, `r-15`, `r`, and `r+15` seconds. This matches the stored reference convention: `r` is the midpoint of the first Wake epoch. The exact `STAGE-CONFIRM2` pattern therefore emits at `r+15`, the end of the frozen causal match interval.

## Interpretation boundary

- These diagnostics explain a known failure and are not confirmatory metrics.
- The +/-120-second window is descriptive and does not replace the frozen causal interval `[0, +15]` seconds.
- Do not choose a new lag, smoothing duration, confidence threshold, or run length from these results.
- A future temporal-stage experiment requires a new protocol and fresh evaluation data; it cannot be claimed as confirmed from these held-out outcomes.
