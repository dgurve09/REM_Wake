# Block 8 Raw-Signal Noise Robustness v0.1

**Work date:** 2026-09-11
**Protocol commit:** `4055f2c`
**Result-producing code commit:** `028ef8a`
**Partition accessed:** Validation only
**Model fitting or threshold selection:** None
**Test data accessed:** No

## Primary Event Result

| Comparator | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| H2-CLEAN | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| H2-BOTH-20DB | 0.0924 | 0.2973 | 0.1410 | 0.6777 |
| H2-BOTH-10DB | 0.1250 | 0.1351 | 0.1299 | 0.2196 |
| H2-BOTH-0DB | 0.0000 | 0.0000 | 0.0000 | 0.0126 |
| H2-HB1-10DB | 0.0220 | 0.4324 | 0.0419 | 4.4616 |
| H2-HB2-10DB | 0.0667 | 0.1081 | 0.0825 | 0.3514 |

## Probability Fidelity

| Comparator | Spearman with clean | Mean absolute difference | Median shift | Threshold crossings |
|---|---:|---:|---:|---:|
| H2-BOTH-20DB | 0.8404 | 0.0705 | -0.0030 | 6.55% |
| H2-BOTH-10DB | 0.7276 | 0.1260 | -0.0094 | 7.06% |
| H2-BOTH-0DB | 0.4982 | 0.1640 | -0.0176 | 7.26% |
| H2-HB1-10DB | 0.7561 | 0.3090 | +0.2417 | 5.28% |
| H2-HB2-10DB | 0.6574 | 0.1649 | -0.0415 | 6.90% |

## Both-Channel Dose Response

| SNR dB | Event F1 | False alarms/hour | Spearman with clean | Mean absolute difference |
|---:|---:|---:|---:|---:|
| 20 | 0.1410 | 0.6777 | 0.8404 | 0.0705 |
| 10 | 0.1299 | 0.2196 | 0.7276 | 0.1260 |
| 0 | 0.0000 | 0.0126 | 0.4982 | 0.1640 |

## Paired Participant F1 Differences

| Comparison | Point difference | Paired-bootstrap 95% interval |
|---|---:|---:|
| H2-BOTH-20DB_minus_H2-CLEAN | +0.0287 | -0.0362 to +0.0869 |
| H2-BOTH-10DB_minus_H2-CLEAN | +0.0176 | -0.0924 to +0.1169 |
| H2-BOTH-0DB_minus_H2-CLEAN | -0.1123 | -0.1869 to -0.0410 |
| H2-HB1-10DB_minus_H2-CLEAN | -0.0704 | -0.1174 to -0.0265 |
| H2-HB2-10DB_minus_H2-CLEAN | -0.0298 | -0.1034 to +0.0310 |

## Frozen Decisions

| Hypothesis | Decision | Supported |
|---|---|---:|
| H8.3_mild_noise_tolerance | pass | True |
| H8.4_nested_probability_dose | supported | True |
| H8.5_channel_noise_asymmetry | supported | True |

## Boundary

All 17/17 in-run checks passed. This stationary Gaussian-noise experiment is a controlled validation stress test, not a model of natural wearable artefacts or evidence from a new cohort.
