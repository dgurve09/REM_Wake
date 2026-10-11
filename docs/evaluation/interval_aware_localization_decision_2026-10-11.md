# Interval-Aware Localization Decision

**Date:** 2026-10-11  
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`  
**Partition:** Train only; validation and current test remained closed  
**Decision:** Stop the uniform interval-label mechanism at v0.1 and close Block 10 without post-result solver or label-width tuning

## Technical uncertainty

Sleep-stage annotations place a REM-to-Wake boundary on a 30-second epoch grid. It was uncertain whether treating the boundary as a small interval, rather than one exact timestamp, would improve event localization at a controlled false-alarm burden.

## Frozen comparison

The experiment constructed a genuine 15-second grid from raw wearable EEG. The `HARD15` comparator assigned unit positive weight to the nominal boundary. `INT15` distributed the same unit event weight across nominal -15, nominal, and nominal +15 seconds. Both candidates used the same 147,055 negatives, participant folds, fold-local feature pruning, scaler, model settings, inner threshold selection, and event evaluator.

The generated grid contained 150,967 supported boundaries from 82 recordings and 64 participant groups. All 180 primary events had complete support at the three interval positions. Fifty nested fits completed, and all stopped before the 2,000-iteration limit.

## Frozen result

At the primary +/-15-second tolerance and 0.25 false-alarms/hour selection budget:

| Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| HARD15 | 0.3333 | 0.0056 | 0.0109 | 0.0032 |
| INT15 | 0.0000 | 0.0000 | 0.0000 | 0.0048 |

`INT15 - HARD15` F1 was `-0.0109`, below the required `+0.03`. False alarms/hour increased by `+0.0016`, which was within the maximum `+0.05` increase but did not rescue the failed F1 condition. The paired participant-bootstrap F1-difference interval was `-0.0380` to `0.0000`. The frozen H-INT decision is `stop_v0.1`.

## Numerical failure mode

A post-run audit found that the iteration-count convergence control was insufficient. Across the 25 fits per candidate, fitted intercepts ranged from approximately `-1005` to `-749`. Of 150,967 outer scores, `HARD15` produced 150,763 exact zeros and six exact ones; `INT15` produced 150,746 exact zeros and 31 exact ones. Endpoint fractions were 99.87% for both candidates.

This saturation explains why most inner folds selected a threshold of exactly `1.0` or the no-alarm successor and why each candidate produced only three outer alarms at every tested alarm budget. The run therefore identifies a numerical failure of the selected SGD configuration. It does not provide a stable estimate of the biological value of interval-distributed labels.

## Verification

- All 11 required in-run controls passed; the convergence control also passed under its predeclared iteration-count definition.
- The independent validator passed 12/12 checks.
- It reconstructed event metrics, participant rows, matches, 2,000 paired bootstrap resamples, localization summaries, and the frozen decision.
- It rehashed 182 external feature, model, and score artifacts.
- Validation and current-test data remained closed.

## Decision and boundary

The uniform three-position interval-label mechanism stops at v0.1. The observed outer results will not be used to choose a replacement solver, new smoothing width, class weighting, alarm budget, or threshold grid. This preserves the predeclared stop rule and avoids converting a failed comparison into post-result tuning.

Block 10 is complete. The result package remains useful because it records the 15-second signal-derived feature grid, equal event-mass construction, a reproduced failure, and the newly identified endpoint-saturation failure mode. Any later formulation study must be separately predeclared, demonstrate non-degenerate score behavior before outcome inspection, and use a new confirmation boundary.
