# Causal Multiscale Change-Feature Decision

**Date:** 2026-10-25
**Scope:** BOAS train partition only

## Decision

Retain `ADV-PEAK` unchanged for a future new-cohort confirmation. Do not promote it as a reliable wearable detector and do not open the existing validation or current test partitions.

## Basis

Against the matched `BASE-PEAK` comparator, `ADV-PEAK` increased event F1 from 0.0236 to 0.1125 and reduced false alarms from 0.2464 to 0.1940/hour. The `+0.0889` F1 difference exceeded the frozen `+0.03` gate, all numerical and alarm-monotonicity controls passed, and the participant-bootstrap F1 interval was `+0.0433` to `+0.1393`.

The absolute limitation remains substantial: recall was 0.1000, 162/180 reference events were missed, and only 15/64 participant groups contributed a true positive. The false-alarm difference interval crossed the permitted point boundary even though the aggregate result improved. This is a supported feature-mechanism result, not a performance endpoint.

## What changed technically

The advanced candidate added 180 strictly causal values to the 80-value baseline. These measured 15-second spectral content, spectral entropy, a fixed aperiodic proxy, amplitude and line-length statistics, Hjorth parameters, inter-channel coupling, and immediate pre-decision change. Welch spectral estimation, Hjorth parameters, and spectral entropy follow the established methods cited in the [frozen protocol](causal_multiscale_change_feature_protocol_v0.1.md).

## Failure retained

The first post-audit execution stopped before fitting because a transient cache-access label made a reviewed support table differ across runs. The correction stored stable artifact provenance instead of invocation state. This was an implementation reproducibility failure; it did not expose labels or affect fitted results.

## Next boundary

1. Use the harmonized multi-dataset hypnograms to quantify REM-to-Wake prevalence, scorer dependence, and repeated-night variability under a separate frozen analysis.
2. Consider the local gx36m raw PSG only for external mechanism portability under a new protocol; it cannot confirm wearable performance.
3. Require a new paired wearable-PSG cohort before any wearable advancement decision.
4. Keep feature expansion stopped at v0.1 until one of those confirmation boundaries is available.
