# DE-D Threshold Robustness v0.1

**Created:** 2026-08-22
**Status:** Post-result exploratory validation analysis
**Plan:** `docs/evaluation/direct_endpoint_threshold_robustness_plan_v0.1.md`
**Test or model access:** None

## Threshold Perturbation

The two-part DE-B improvement rule held from threshold 0.67 through 0.88, covering 22 adjacent thresholds. The originally selected threshold was 0.74.

## Leave-One-Participant-Out Calibration

Fold-specific thresholds ranged from 0.74 to 0.74, with median 0.74.

Aggregated held-out performance was precision 0.0971, recall 0.4595, event F1 0.1604, and 0.9915 false alarms per hour.

Compared with frozen DE-B validation, the paired participant-bootstrap F1-difference interval was +0.0038 to +0.1008. The false-alarm-rate-difference interval was -0.8835 to -0.1423 per hour.

## Decision

The prespecified threshold-robustness rule was supported: **True**. This requires both a success interval of at least five adjacent thresholds and LOPO performance better than DE-B in both F1 and false alarms per hour.

This analysis uses saved validation probabilities and cannot establish independent test performance. No raw signal, feature array, fitted model, train row, or current-test artifact was accessed.
