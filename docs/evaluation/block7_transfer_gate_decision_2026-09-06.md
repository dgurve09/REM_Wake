# Block 7 Transfer-Gate Decision

**Decision date:** 2026-09-06
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Validation freeze commit:** `4a17fbd`
**Test runner commit:** `c0d2e9e`
**Test role:** Fixed descriptive comparison; not independent confirmation
**Block decision:** Close Block 7 and proceed to a separately predeclared robustness investigation

## 1. Uncertainty Addressed

Block 7 tested how REM-to-Wake event performance changes between common six-channel PSG EEG, reduced frontal PSG EEG, and simultaneous two-channel wearable EEG when the feature method and classifier family are held fixed. It also tested whether strict PSG-to-wearable transfer failed in a way that opened one predefined feature-alignment method.

This was not a search for the best classifier. The unresolved issue was whether channel reduction and acquisition/electrode-location shift produced stable event-performance changes under participant grouping and fixed event matching.

## 2. Frozen Test Result

| Comparator | Threshold | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| `P6-D` | 0.99 | 0.1589 | 0.2881 | 0.2048 | 0.4525 |
| `P2-D` | 0.99 | 0.1235 | 0.1695 | 0.1429 | 0.3570 |
| `H2-D` | 0.96 | 0.0906 | 0.4237 | 0.1493 | 1.2621 |
| `P2-H2-Z` | 0.99 inherited | 0.0879 | 0.1356 | 0.1067 | 0.4173 |

All three fitted models matched the hashes frozen before test access. No model was refitted, recalibrated, selected, or changed. `P2-H2-A` remained excluded because its validation distribution gate had closed.

## 3. Validation-to-Test Assessment

### H7.1: added PSG channels

Validation favored `P2-D` at the point-estimate level and did not support H7.1. The descriptive test reversed that ordering: `P6-D - P2-D` F1 was +0.0620 with paired 95% interval +0.0019 to +0.1194. The corresponding false-alarm difference was +0.0955/hour with interval -0.0651 to +0.2510.

The correct conclusion is not that either montage is uniformly superior. The channel contribution changed across participant partitions, so its stability remains unresolved.

### H7.2: strict zero-shot transfer

Validation showed a direct `P2-D` versus zero-shot F1 advantage of +0.1202 with an interval excluding zero. The descriptive test difference remained positive at +0.0362, but its interval was -0.0434 to +0.1001. The false-alarm difference was also inconclusive.

Relative to direct wearable fitting, zero-shot transfer again had lower F1 but fewer false alarms. On test, `P2-H2-Z - H2-D` was -0.0426 for F1 with interval -0.0870 to +0.0085 and -0.8448/hour for false alarms with interval -1.2441 to -0.5081.

Strict transfer therefore did not show a stable, uniform degradation. It changed the sensitivity/alarm operating behavior, and the magnitude of its F1 deficit depended on the participant partition and comparator.

### H7.3: feature alignment

H7.3 was not evaluated. Only 11/80 mapped feature dimensions exceeded the predeclared distribution threshold on train data, below the required 16/80. Running alignment after observing the test result would violate the frozen protocol.

## 4. Fixed Sensitivity

Excluding the prespecified alignment-review recordings `sub-32` and `sub-50` retained `P6-D` as the highest-F1 comparator at 0.2112 and `P2-H2-Z` as the lowest at 0.1103. `P2-D` reached 0.1515 and `H2-D` 0.1379, reversing their close all-test ordering. The sensitivity does not remove the broader participant-instability concern.

## 5. Reproducibility Evidence

- 104 test feature arrays were generated outside Git and recorded by path, size, and SHA-256.
- Exact epoch, stage, context-center, and overlapping PSG feature parity passed for all 26 recordings.
- The result-producing run passed 13/13 checks.
- The independent validator rehashed 108 external artifacts and reproduced 95,460 probabilities, all event outputs, and four paired participant bootstraps.
- All 13/13 independent checks passed twice.
- A complete immutable rerun reproduced the stored outputs without change.

## 6. Decision and Remaining Uncertainty

Block 7 is complete. It generated three bounded findings:

1. additional PSG-channel value was not stable across the used participant partitions;
2. the direct-PSG versus zero-shot F1-loss magnitude was not stable; and
3. zero-shot versus direct wearable scoring repeatedly expressed a sensitivity-versus-false-alarm tradeoff rather than uniform loss.

Block 8 may now test robustness to single-channel loss, controlled signal degradation, and participant concentration under a new committed protocol. It must not tune a model or threshold against the current test partition. Any adaptation method requires a separately stated mechanism and hypothesis. A new locked or external cohort is required for confirmatory performance evaluation.
