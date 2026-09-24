# Deep Temporal Error Diagnostic v0.1

**Work date:** 2026-09-23
**Source:** frozen nested train outer-fold results
**Model fitting:** No
**Validation accessed:** No
**Test accessed:** No

## Main Findings

Widening tolerance from +/-15 to +/-45 seconds increased true detections from 32 to 41 and F1 from 0.1645 to 0.2113. Most missed events therefore cannot be explained by one adjacent 30-second boundary bin.

A pooled outer-label oracle threshold reached F1 0.2177; separate outer-fold label oracles reached pooled F1 0.2324. These are optimistic post-result bounds, not usable estimates or replacement thresholds. Even the stronger bound remained below 0.25.

## Quality-Tier Recall

| Model | Membership tier | Tolerance (s) | Detected | Recall |
|---|---|---:|---:|---:|
| NESTED-SELECTED | primary_clean | 15 | 1/53 | 0.0189 |
| NESTED-SELECTED | primary_clean | 45 | 3/53 | 0.0566 |
| NESTED-SELECTED | primary_mad_flagged | 15 | 31/127 | 0.2441 |
| NESTED-SELECTED | primary_mad_flagged | 45 | 38/127 | 0.2992 |
| LR-OOF | primary_clean | 15 | 3/53 | 0.0566 |
| LR-OOF | primary_mad_flagged | 15 | 54/127 | 0.4252 |
| LR-OOF | primary_clean | 45 | 7/53 | 0.1321 |
| LR-OOF | primary_mad_flagged | 45 | 60/127 | 0.4724 |
| LC-1 | primary_clean | 15 | 9/53 | 0.1698 |
| LC-1 | primary_mad_flagged | 15 | 47/127 | 0.3701 |
| LC-1 | primary_clean | 45 | 13/53 | 0.2453 |
| LC-1 | primary_mad_flagged | 45 | 60/127 | 0.4724 |

## Primary False-Alarm Context

| Context | False alarms | Fraction |
|---|---:|---:|
| not_near_labeled_transition | 167 | 0.9435 |
| primary_r2w_16_to_45sec | 9 | 0.0508 |
| quality_only_r2w_within_45sec | 1 | 0.0056 |

## Decision

Threshold transfer contributes to the low frozen F1, but cannot explain most missed events. Timing uncertainty contributes a smaller component. The next model experiment should change the information or supervision mechanism rather than repeat architecture search on the same bandpower inputs. Candidate mechanisms are raw-waveform epoch embeddings and interval-aware boundary supervision. Any such experiment requires a separately committed protocol and a new evaluation boundary.

This retrospective diagnostic cannot authorize model or threshold replacement.

## Verification

All 9/9 integrity checks passed.
