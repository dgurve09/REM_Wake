# Causal Endpoint Factorization Decision

**Date:** 2026-10-15

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Partition:** Train only; validation and current test remained closed

**Decision:** Stop causal endpoint factorization at v0.1; carry the alarm-formation failure into the streaming gate

## Question tested

The prior causal detector learned 180 REM-to-Wake checkpoints against approximately 147,000 full-night negatives. A known-checkpoint direction classifier performed well, suggesting that state-direction evidence existed but rare checkpoint localization remained weak.

This experiment tested whether the full hypnogram could provide denser supervision. Separate heads learned whether REM occurred immediately before a candidate and Wake occurred immediately after it. Their probabilities were multiplied into a causal event score.

## Controlled change

The design retained the existing two wearable channels, 80 past-only bandpower features, 15-second decision grid, five participant folds, causal event timing, and 0.25 false-alarms/hour budget. It changed the fitting target from one rare event label to two common state labels.

The full grid supplied 24,772 REM-before positives and 21,076 Wake-after positives, compared with 235 exact REM-to-Wake pairs. No validation or current-test input was read.

## Result

All 50 state-head fits converged and passed the numerical gate. Outer out-of-fold state discrimination was measurable:

- REM-before average precision 0.2978 and ROC AUC 0.7692;
- Wake-after average precision 0.4672 and ROC AUC 0.8835.

This state information did not improve the event endpoint. CEF reached precision 0.0429, recall 0.0333, F1 0.0375, and 0.2130 false alarms/hour. The frozen CAUSAL-RW comparator reached F1 0.0517 and 0.2528 false alarms/hour. The F1 difference was -0.0142, failing the required +0.03 gate. Its participant interval, -0.0603 to +0.0267, also did not reach the gate.

## Alarm-formation failure

Every inner fold selected the minimum threshold. This placed 99.9583% of outer candidates above threshold. The rule then collapsed each contiguous above-threshold region to its single maximum:

- median run duration: 16,290 seconds;
- 111/142 runs lasted at least 30 minutes;
- 97/142 runs lasted at least two hours; and
- maximum run duration: 33,015 seconds.

The resulting alarms were dominated by partial endpoints: 71 other-to-Wake, 19 REM-to-other, 47 neither endpoint, and only five exact REM-to-Wake pairs.

The score still ranked exact REM-to-Wake pairs above background with ROC AUC 0.8132, but average precision was only 0.00448 at prevalence 0.00156. State occupancy is therefore detectable, while precise conjunction timing remains poorly ranked under the full-night imbalance.

## Interpretation

The experiment resolves the tested uncertainty negatively: increasing endpoint-label density did not improve participant-held-out causal event F1. The result also identifies a separate algorithmic limitation in the score-to-alarm layer. Thresholding before contiguous-run collapse allows a very low threshold to merge hours of data into one alarm and satisfy the false-alarm budget artificially.

This diagnostic does not authorize a post-result peak width, refractory interval, threshold grid, or combination-rule search. The frozen comparison and failure are retained unchanged.

## Next decision

Stop CEF v0.1. Do not open validation or current test and do not tune the probability product on these outer scores.

At the scheduled Block 11 streaming gate, treat long-run collapse as a no-go condition for the present alarm representation. If detector development continues later, predeclare a peak-first representation that identifies local score maxima before threshold selection and fixes its refractory rule independently of the present outer outcomes. Evaluation on these train participants would remain method development; confirmation still requires a new locked or external wearable cohort.
