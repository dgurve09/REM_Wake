# Multi-Dataset PSG Stage-First Decision

**Protocol frozen:** 2026-10-25

**Decision completed:** 2026-10-31

**Evidence status:** Exploratory comparison on reused datasets

## Question

Does causal five-stage prediction followed by a fixed REM-to-Wake derivation improve external event performance over the corrected direct `ADV3-CHANGE` detector?

## Result

The primary `STAGE-CONFIRM2` candidate reached pooled external precision 0.0896, recall 0.2137, F1 0.1263, and 0.4523 false alarms/hour. Corrected direct `ADV3-CHANGE` reached precision 0.2069, recall 0.2564, F1 0.2290, and 0.2048 false alarms/hour on the same gx36m plus ISRUC scope.

The stage-first minus direct F1 difference was -0.1027. Its participant-stratified bootstrap median was -0.1027 with a 95% interval from -0.1859 to -0.0125. The false-alarm difference was +0.2475/hour, with a bootstrap interval from +0.1203 to +0.3551/hour.

`STAGE-CONFIRM2` failed every frozen performance condition. It also failed the numerical condition because all three logistic fits reached the fixed 1,000-iteration limit. Probabilities remained finite and normalized, so event diagnostics are retained, but coefficients and numerical optimality must not be overinterpreted.

## Failure mechanism

Epoch-level staging was not uniformly poor: macro F1 was 0.7059 on gx36m, 0.5480 on BOAS, and 0.4266 on ISRUC group 3. Event behavior nevertheless degraded because the predicted sequences contained too many short stage runs.

- Predicted all-stage transition rates were 1.790 to 2.928 times the human rates.
- Predicted REM-bout counts were 4.607 to 7.520 times the human counts.
- Median predicted REM bouts were 30 to 45 seconds, compared with human medians of 120 to 1,050 seconds.
- The exact four-decision `REM, REM, W, W` pattern occurred at 41/365 BOAS, 20/85 gx36m, and 0/32 ISRUC references.
- On ISRUC, neither fixed derivation produced a true positive; `STAGE-CONFIRM2` produced no alarm within +/-120 seconds of any corrected reference.

The failed event result is therefore consistent with stage-sequence fragmentation and external timing failure, not solely with the narrow primary matching interval.

## Decision

`H-STAGE-FIRST-XDATA` is not supported, and `STAGE-CONFIRM2` stops at v0.1. The corrected direct detector remains the stronger tested formulation, but its pooled external F1 of 0.2290 and recall of 0.2564 are still inadequate for reliable use.

Do not tune convergence limits, smoothing, lag, confidence thresholds, confirmation length, or event tolerance against these held-out outcomes. The result does not prove that all temporal stage models must fail. A future sequence-level method would require a new mechanistic hypothesis, a protocol frozen before fitting, and fresh locked or external evaluation data.

This PSG-only result does not establish wearable performance, sleep-paralysis detection, narcolepsy detection, diagnosis, or clinical utility.

## Verification

- Pre-fit audit: 13/13 checks.
- Independent input validation: 10/10 checks.
- Experiment controls: 14/14 checks.
- Independent result validation: 18/18 checks.
- Failure-diagnostic controls: 7/7 checks.
- Independent failure-diagnostic validation: 13/13 checks.
- Three model and three score artifacts are retained outside Git with SHA-256 records.
