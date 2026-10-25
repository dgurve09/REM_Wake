# Causal Multiscale Change-Feature Experiment v0.1

## Question

Can higher-resolution causal EEG descriptors improve full-night REM-to-Wake localization when the model, participant folds, alarm budget, and peak-first alarm rule are fixed?

This is a train-cohort mechanism experiment. Validation and the current test partition remained closed.

## Pre-result controls

- Protocol frozen in commit `7be8ae6` before implementation.
- Feature implementation and 13/13 pre-result audit frozen in commit `2f99ab3` before model fitting.
- The first model invocation stopped before fitting because a reviewed audit row changed from cache status `created` to `reused`. Commit `0ebcab6` made that operational field deterministic; the 13-control audit then reproduced.
- Inputs contained 82 recordings from 64 participant groups, 150,967 full-night causal decision points, and 180 REM-to-Wake positive rows.
- All features ended at the decision time. No validation or test inputs were accessed.

## Candidate representations

`BASE-PEAK` used the existing 80 causal band-power context values. `ADV-PEAK` added 180 values derived from four causal 15-second subwindows:

- five log band powers per channel;
- spectral entropy;
- log-log spectral slope and intercept as a fixed aperiodic proxy;
- RMS, line length, Hjorth mobility, and Hjorth complexity;
- inter-channel correlation and five band coherences; and
- signed and absolute changes between the final two subwindows.

Both candidates used the same natural-prevalence L2 logistic regression, participant-grouped nested folds, 0.25 false-alarms/hour selection budget, local-maximum rule, and 60-second refractory interval. The detailed definitions and method references are in the [frozen protocol](../../docs/evaluation/causal_multiscale_change_feature_protocol_v0.1.md).

## Primary result

| Candidate | Precision | Recall | F1 | False alarms/hour | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BASE-PEAK` | 0.0252 | 0.0222 | 0.0236 | 0.2464 | 4 | 155 | 176 |
| `ADV-PEAK` | 0.1286 | 0.1000 | 0.1125 | 0.1940 | 18 | 122 | 162 |

The F1 difference was `+0.0889`, exceeding the predeclared `+0.03` point gate. False alarms decreased by `0.0525/hour`, so the no-material-increase condition also passed. All 50 fits converged, probability controls passed, and alarm counts were monotonic over every inner threshold curve.

The participant bootstrap supported a positive F1 direction: 95% interval `+0.0433` to `+0.1393`. The false-alarm difference interval was `-0.2216` to `+0.0949/hour`, so the aggregate FAR improvement was not participant-conclusive.

## Interpretation

This is positive evidence for the specific mechanism tested: 15-second spectral, dynamical, coupling, and immediate-change descriptors contain causal information not captured by the older 30-second band-power context alone.

It is not evidence of reliable detection. `ADV-PEAK` detected 18/180 events, only 15/64 participants had a true positive, and F1 remained 0.1125. The peak-first mapping by itself did not improve the original feature set: `BASE-PEAK` F1 0.0236 was below the earlier contiguous-run `CAUSAL-RW` F1 0.0517. The previously reported enriched-feature F1 0.1796 used a different candidate formulation and alarm process, so it is context rather than a valid head-to-head comparator.

## Decision

Retain `ADV-PEAK` as a frozen candidate for new-cohort confirmation. Do not tune its feature set, window lengths, regularization, refractory interval, thresholds, or event tolerance on these outer results. Do not open the existing validation or test partitions.

The next scientific questions are whether the feature mechanism transfers to external PSG and whether transition prevalence and scorer uncertainty are stable across the available hypnogram collections. Neither question can establish wearable generalization without another paired wearable cohort.

## Verification

- Pre-result support audit: 13/13 passed.
- In-run controls: 10/10 passed.
- Independent reconstruction: 13/13 passed twice.
- External artifact verification: 134/134 hashes passed.
- Exact end-to-end rerun: all reviewed outputs and external artifacts reproduced.
