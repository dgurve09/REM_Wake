# Causal Bidirectional REM/Wake Transition v0.1

**Date:** 2026-10-11

**Status:** Complete; independently validated; H-BI stopped at v0.1

**Partition:** Train only; validation and current test remain closed

## Purpose

Test whether a past-only wearable EEG detector benefits from treating both REM-to-Wake and Wake-to-REM boundaries as transition positives, while keeping transition direction as a separate diagnostic outcome.

## Frozen design

- decision at `nominal + 15 s`;
- eight 30-second feature windows ending at the decision time;
- `CAUSAL-RW` versus `CAUSAL-BI` with shared full-night negatives;
- participant-grouped nested threshold selection at 0.25 false alarms/hour; and
- separate participant-grouped REM-to-Wake versus Wake-to-REM direction classifier.

The complete pre-result specification is in `docs/evaluation/causal_bidirectional_transition_protocol_v0.1.md`.

## Pre-result support audit

The audit passed 8/8 controls before model fitting. It retained 150,967 causal decision points over 629.0292 supported hours and 146,987 shared negative fitting rows.

Of 242 quality-eligible train transitions, 241 had eight contiguous past windows at the fixed decision time:

- 180 REM-to-Wake events;
- 61 Wake-to-REM events; and
- one excluded Wake-to-REM event, transition 406, because a feature-grid gap prevented a complete causal context.

The exclusion and fixed resolution were recorded before model fitting in `docs/evaluation/causal_bidirectional_transition_protocol_amendment_2026-10-12.md`.

## Stage A result: full-night transition detection

At the inner-selected 0.25 false-alarms/hour budget, outer train out-of-fold REM-to-Wake results were:

| Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| `CAUSAL-RW` | 0.0536 | 0.0500 | 0.0517 | 0.2528 |
| `CAUSAL-BI` | 0.0462 | 0.0333 | 0.0387 | 0.1971 |

`CAUSAL-BI - CAUSAL-RW` F1 was -0.0130. False alarms/hour decreased by 0.0556, but the required F1 improvement was not achieved. The frozen H-BI point gate failed and the bidirectional formulation received `stop_v0.1`.

For all 241 transition events, `CAUSAL-RW` reached F1 0.0440 and `CAUSAL-BI` reached F1 0.0323. Neither model matched a Wake-to-REM event under the fixed causal timing rule.

## Stage B result: direction at known transition checkpoints

The separate oracle-event direction diagnostic produced:

| Metric | Result |
|---|---:|
| Macro F1 | 0.8290 |
| Balanced accuracy | 0.8272 |
| REM-to-Wake recall | 0.9167 |
| Wake-to-REM recall | 0.7377 |

The outer out-of-fold confusion counts were 165/180 correctly identified REM-to-Wake events and 45/61 correctly identified Wake-to-REM events.

This is evidence that the past-only wearable representation contains information about direction when the event checkpoint is supplied. It is not evidence that the full-night detector can first find that checkpoint.

Two limitations remain. Inner-selected Stage A thresholds varied from approximately `1.1e-7` to `0.0373` across folds, showing substantial participant-fold calibration variation despite numerically stable fits. The Stage B result may also reflect differences in recent REM versus Wake occupancy, not a uniquely identified transition waveform.

## Numerical controls and verification

- All 50 Stage A fits and five Stage B fits converged before 1,000 iterations.
- Stage A held-out exact-endpoint probability fractions were zero and all fits produced more than 100 unique scores.
- The independent validator passed 13/13 checks twice.
- It reconstructed alarms, event and participant metrics, causal matches, Stage B metrics, 2,000 participant resamples, and the frozen decision.
- It rehashed 57 external model and score artifacts.
- Validation and current-test partitions remained closed.

## Decision

Do not tune the bidirectional target, causal timing window, threshold rule, or loss on these outer results. Retain the Stage B direction result as a mechanism finding: endpoint localization, not direction separability, is the present bottleneck. A later causal detector study requires a new predeclared design and confirmation boundary.
