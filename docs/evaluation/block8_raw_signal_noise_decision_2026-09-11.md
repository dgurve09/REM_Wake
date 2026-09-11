# Block 8 Raw-Signal Noise Robustness Decision

**Decision date:** 2026-09-11
**Protocol:** `block8_raw_signal_noise_protocol_v0.1.md`
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Partition:** Validation only, 20 recordings from 16 `pid` groups
**Frozen comparator:** `H2-D`, threshold 0.96
**Model fitting or threshold search:** None
**Current test access:** None

## Decision

The frozen detector passes the predeclared 20 dB mild-noise screen, shows the expected nested probability degradation as both-channel noise increases, and directionally reproduces the earlier `HB_1`/`HB_2` asymmetry. It is not robust to severe 0 dB both-channel noise or moderate isolated `HB_1` noise at the fixed operating threshold.

This is a validation stress-test result. It does not establish field robustness or justify adding noise at inference.

## Primary Result

| Comparator | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| `H2-CLEAN` | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| `H2-BOTH-20DB` | 0.0924 | 0.2973 | 0.1410 | 0.6777 |
| `H2-BOTH-10DB` | 0.1250 | 0.1351 | 0.1299 | 0.2196 |
| `H2-BOTH-0DB` | 0 | 0 | 0 | 0.0126 |
| `H2-HB1-10DB` | 0.0220 | 0.4324 | 0.0419 | 4.4616 |
| `H2-HB2-10DB` | 0.0667 | 0.1081 | 0.0825 | 0.3514 |

At 20 dB, F1 changed by +0.0287 and false alarms by -0.7781/hour. Both changes remain inside the predeclared mild-noise adverse bounds, so H8.3 passes. The paired participant interval for the F1 change crosses zero (`-0.0362` to `+0.0869`), while the false-alarm interval does not (`-1.5069` to `-0.3322`/hour). The observed F1 increase is therefore not evidence that noise is beneficial.

At 0 dB, no true event was detected. F1 changed by -0.1123 with paired interval `-0.1869` to `-0.0410`. The simultaneous false-alarm reduction reflects collapse below the fixed high threshold, not useful robustness.

## Probability Dose Response

| Both-channel SNR | Spearman with clean | Mean absolute probability difference | Median probability shift |
|---:|---:|---:|---:|
| 20 dB | 0.8404 | 0.0705 | -0.0030 |
| 10 dB | 0.7276 | 0.1260 | -0.0094 |
| 0 dB | 0.4982 | 0.1640 | -0.0176 |

Correlation decreased and mean absolute difference increased at every worsening SNR level, supporting H8.4. Event F1 was not monotonic: it remained near the weak clean value at 20 and 10 dB before collapsing at 0 dB. Continuous probabilities therefore reveal progressive degradation that a single thresholded F1 value obscures.

## Channel Asymmetry

At 10 dB, isolated `HB_1` noise reduced F1 by 0.0704 and increased false alarms by 3.0058/hour. Both paired intervals excluded zero: `-0.1174` to `-0.0265` for F1 and `+1.5100` to `+5.0621`/hour for false alarms. Its median probability shift was +0.2417.

Isolated `HB_2` noise reduced F1 by 0.0298, with interval `-0.1034` to `+0.0310`, and reduced false alarms by 1.1044/hour. Its median probability shift was -0.0415. The `HB_1` condition caused both the larger F1 loss and the larger mean absolute probability change, supporting H8.5 under the frozen rule.

The opposite score-shift directions show that the detector does not treat the two nominal wearable channels interchangeably. This agrees with the earlier feature-neutralization result, but it does not identify whether the mechanism comes from electrode placement, channel normalization, bandpower distributions, or fitted coefficients.

## Controls and Reproducibility

- All 160 recording-channel-condition calibrations met their SNR target; maximum absolute error was `8.89e-15` dB.
- The clean raw-to-feature path reproduced stored features within `4.67e-7`.
- Clean probabilities reproduced Block 7 within `1.12e-16`; all four clean event summaries reproduced within `1e-12`.
- All 17/17 result-producer checks passed.
- The independent validator reconstructed 100 noisy feature arrays from EDF, 114,738 probabilities, all event tables, all 2,000-resample paired intervals, and all decisions. It passed 12/12 checks twice.
- The immutable result rerun changed no reviewed or external artifact.
- The model, threshold, membership, temporal support, and matching rules were unchanged. No test artifact was read or hashed.

## Failure and Remaining Uncertainty

The frozen method failed under severe both-channel contamination and under moderate isolated `HB_1` contamination. The failure modes differed: both-channel noise progressively lowered rank agreement and largely suppressed alarms, whereas `HB_1`-only noise drove probabilities upward and produced excessive false alarms.

Stationary independent Gaussian noise is intentionally controlled but physically incomplete. Natural wearable degradation can be nonstationary, channel-correlated, amplitude-clipped, spectrally structured, or caused by impedance and reference changes. These results also reuse the validation cohort and cannot provide independent confirmation.

## Next Decision

Do not change the frozen threshold or treat noise as a performance intervention. Before any model fitting, predeclare one mechanism-specific robustness method using train data only, such as channel-quality masking or train-only noise augmentation, with the unchanged `H2-D` path as comparator. Keep validation use limited to the fixed decision rule and keep the current test partition closed. A new locked or external cohort remains necessary for confirmation.
