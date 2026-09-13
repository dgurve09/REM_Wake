# Block 8 Train-Only Noise-Augmentation Decision

**Decision date:** 2026-09-13
**Protocol:** `block8_noise_augmented_training_protocol_v0.1.md`
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Train partition:** 82 recordings from 64 `pid` groups
**Development validation:** 20 recordings from 16 `pid` groups
**Current test access:** None
**Decision:** Retain the mechanism finding; do not replace the core detector

## Question

Can fixed train-only Gaussian-noise exposure correct the frozen wearable logistic detector's isolated `HB_1` calibration failure without sacrificing clean performance, and does it produce a meaningful improvement toward the original detector objective?

## Fixed Method

`H2-NA` retained the existing signal preprocessing, five bandpower features per channel, eight-epoch context, balanced logistic regression, labels, background rows, and participant split. Every eligible train candidate was represented under clean, both-channel 20 dB, both-channel 10 dB, isolated `HB_1` 10 dB, and isolated `HB_2` 10 dB conditions.

Five participant-grouped train folds selected threshold 0.99 from 75,539 clean full-night out-of-fold scores. Train-OOF F1 was 0.1193 with 1.2169 false alarms/hour. The final model was then fitted on all 13,715 augmented rows and frozen before validation scoring.

## Primary Validation Result

| Condition | `H2-D` F1/FAR | `H2-NA` F1/FAR | F1 difference | FAR difference |
|---|---:|---:|---:|---:|
| Clean | 0.1123 / 1.4558 | 0.1024 / 1.2801 | -0.0099 | -0.1757 |
| Both 20 dB | 0.1410 / 0.6777 | 0.1270 / 0.8785 | -0.0140 | +0.2008 |
| Both 10 dB | 0.1299 / 0.2196 | 0.1064 / 0.3263 | -0.0235 | +0.1067 |
| Both 0 dB | 0 / 0.0126 | 0 / 0.0314 | 0 | +0.0188 |
| `HB_1` 10 dB | 0.0419 / 4.4616 | 0.0918 / 0.9413 | +0.0500 | -3.5204 |
| `HB_2` 10 dB | 0.0825 / 0.3514 | 0.1093 / 0.8534 | +0.0268 | +0.5020 |

FAR is false alarms per supported hour. All comparisons use primary membership and +/-15-second event matching. The two models retain their separately frozen thresholds: 0.96 for `H2-D` and train-OOF-selected 0.99 for `H2-NA`.

## Frozen Decisions

| Hypothesis | Result | Decision |
|---|---|---|
| H8.6 clean preservation | F1 -0.0099; FAR -0.1757/hour | Pass |
| H8.7 `HB_1` failure mitigation | F1 +0.0500; FAR -3.5204/hour | Pass |
| H8.8 degradation-gap reduction | F1 gap improved 0.0599; FAR gap reduced 3.3447/hour | Pass |
| H8.9 meaningful clean advancement | Required F1 gain at least 0.05; observed -0.0099 | Fail |

The method passes the targeted robustness gate but fails the core-detector advancement gate. It must not replace `H2-D` as the clean detector.

## Participant Uncertainty

For clean validation, the paired F1-difference interval was -0.0438 to +0.0132 and the FAR-difference interval was -0.4691 to +0.0576/hour. Neither clean change was resolved within the reused validation cohort.

Under isolated `HB_1` noise, the F1 improvement interval was -0.0129 to +0.1167, while the FAR reduction interval was -5.8546 to -1.6828/hour. The experiment therefore provides stronger evidence for correction of the false-alarm failure than for improved event detection.

Within `H2-NA`, isolated `HB_1` noise changed F1 by -0.0105 with interval -0.0454 to +0.0273 and FAR by -0.3389/hour with interval -0.6983 to -0.0549. The original large adverse `HB_1` degradation gap did not persist.

## Tradeoffs

The adaptation was not uniformly beneficial. Relative to `H2-D`, `H2-NA` increased false alarms under both-channel 20 dB, both-channel 10 dB, and isolated `HB_2` noise. Those FAR-difference intervals excluded zero. Severe 0 dB noise still produced no true detection.

This indicates that broad Gaussian-noise exposure redistributed calibration sensitivity rather than solving the underlying low-separability problem. The unchanged bandpower-logistic representation remains the main limitation.

## Reproducibility

- All 328 generated train feature arrays passed deterministic reconstruction.
- All 492 train SNR calibrations passed; maximum error was `1.43e-14` dB.
- Clean train features reproduced within `4.77e-7`.
- All six fits converged without warning.
- All 13/13 train-phase and 11/11 validation-phase checks passed.
- The independent validator rebuilt all perturbations, five fold models, the final model, threshold, 229,476 validation scores, event outputs, bootstrap intervals, and decisions; all 14/14 checks passed twice.
- Immutable train and validation reruns produced no artifact change.
- No test feature, score, model, or path was accessed.

## Knowledge Gained

Train-only noise exposure can suppress the specific high-score `HB_1` failure without changing the classifier family. That establishes a controllable calibration mechanism. It does not improve clean event detection, does not solve the severe-noise limit, and creates other degradation tradeoffs.

The result rejects the assumption that generic noise augmentation will materially improve the original REM-to-Wake detector. Further progress requires a representation hypothesis capable of improving clean participant-generalized event separability, not another variation of the current threshold or augmentation mix.

## Next Decision

Stop the generic Gaussian-noise augmentation path at v0.1. Preserve `H2-NA` as a robustness comparator only. Do not reopen the current test set.

The next model experiment should predeclare a temporal-representation comparison using train-only development and a new locked or external confirmation boundary. Before implementation, it must specify what information beyond eight-epoch bandpower concatenation is expected to resolve, a meaningful clean-performance gate, an alarm-burden gate, and a stop rule if richer temporal features remain weak.
