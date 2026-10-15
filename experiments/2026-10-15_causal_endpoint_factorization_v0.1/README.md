# Causal Endpoint Factorization v0.1

**Date:** 2026-10-15

**Protocol:** `docs/evaluation/causal_endpoint_factorization_protocol_v0.1.md`

**Partition:** Train only; validation and current test remained closed

## Question

Can abundant full-night supervision for `REM immediately before` and `Wake immediately after` a causal candidate improve REM-to-Wake localization over direct rare-event fitting?

Two participant-fold-specific logistic heads used the same 80 past-only wearable features. Their probabilities were multiplied to form the fixed `CEF` event score. Inner out-of-fold scores selected thresholds at 0.25 false alarms/hour; outer participants were scored once.

## Support and execution

- 82 recordings from 64 `pid` groups;
- 150,967 causal 15-second decisions;
- 24,772 REM-before and 21,076 Wake-after positive state labels;
- 235 exact REM-to-Wake endpoint pairs across the complete grid;
- 50 state-head fits, all converged before 1,000 iterations; and
- 51 external model/score artifacts recorded by SHA-256.

## State-head result

| Head | Positive fraction | Average precision | ROC AUC |
|---|---:|---:|---:|
| REM before | 0.1641 | 0.2978 | 0.7692 |
| Wake after | 0.1396 | 0.4672 | 0.8835 |

Both states were discriminable out of fold, with stronger evidence for Wake after the candidate than for REM before it.

## Event result

| Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| CAUSAL-RW | 0.0536 | 0.0500 | 0.0517 | 0.2528 |
| CEF | 0.0429 | 0.0333 | 0.0375 | 0.2130 |

`CEF - CAUSAL-RW` was -0.0142 F1 and -0.0397 false alarms/hour. The participant-bootstrap F1-difference interval was -0.0603 to +0.0267. The required +0.03 F1 gate failed, so the frozen decision is `stop_v0.1`.

## Alarm-formation diagnostic

All five inner folds selected the minimum threshold in the fixed grid. At those thresholds, 150,904/150,967 candidates were above threshold. Contiguous candidates merged into 142 runs with median duration 16,290 seconds; 97 runs lasted at least two hours.

The product score ranked exact REM-to-Wake pairs above chance (`ROC AUC 0.8132`) but retained low average precision (`0.00448` versus pair prevalence `0.00156`). Among 142 emitted alarms, five occurred at an exact REM-to-Wake pair, 19 at REM-to-other, 71 at other-to-Wake, and 47 at neither endpoint.

This shows two limitations. Abundant state supervision did not solve rare boundary localization, and the contiguous-run alarm rule became degenerate at the selected minimum threshold by reducing long above-threshold regions to a global maximum. The frozen event result remains unchanged, but it is not a basis for tuning the product or threshold on these outer scores.

## Verification and decision

The numerical gate passed. Independent validation passed 13/13 checks twice, reconstructing thresholds, alarms, event metrics, participant metrics, state-head metrics, bootstrap intervals, and the decision while verifying all external artifacts.

Stop causal endpoint factorization at v0.1. A later method-development experiment may test a predeclared peak-first alarm representation with a fixed refractory rule, but it must not be reported as confirmation on these reused train participants. Validation and current test remain closed.
