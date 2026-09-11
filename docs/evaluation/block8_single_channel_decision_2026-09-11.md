# Block 8 Single-Channel Robustness Decision

**Decision date:** 2026-09-11
**Protocol commit:** `b55849f`
**Result-producing code commit:** `76fc5c4`
**Partition:** Validation only
**Model training:** None
**Test or raw-signal access:** None
**Decision:** Fail the single-channel feature-contribution robustness screen

## 1. Question

Does the frozen two-channel wearable detector retain its event behavior when either channel's 40 context dimensions are replaced by the corresponding train-fitted model means, and is the observed effect driven by a small number of validation participants?

The experiment holds the model, scaler, threshold, support, labels, and matching rule fixed. It tests classifier feature dependence, not physical electrode failure.

## 2. Primary Result

| Comparator | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| `H2-INTACT` | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| `H2-NO-HB1` | 0 | 0 | 0 | 0.5208 |
| `H2-NO-HB2` | 0.0549 | 0.2703 | 0.0913 | 1.0793 |

`HB_1` neutralization changed F1 by -0.1123 and false alarms by -0.9350/hour. The F1 decrease exceeded the predefined 0.03 material bound, so this ablation failed the robustness screen. The simultaneous false-alarm reduction does not make the loss beneficial; it shows that the fixed operating threshold moved into a low-sensitivity regime.

`HB_2` neutralization changed F1 by -0.0210 and false alarms by -0.3765/hour. Neither change crossed the predefined adverse bound, although the consistent loss of recall shows that `HB_2` still contributes to detections.

## 3. Participant Influence

For `HB_1` neutralization, all 16 leave-one-`pid`-out folds preserved both directions. F1 differences ranged from -0.1311 to -0.0916 and false-alarm differences from -1.0187 to -0.7686/hour.

For `HB_2` neutralization, all 16 folds also preserved both directions. F1 differences ranged from -0.0364 to -0.0105 and false-alarm differences from -0.4452 to -0.1781/hour.

The effect directions are therefore not explained by one influential validation participant. False alarms remain unevenly distributed: the four highest-burden participants contributed 51.7% of intact false alarms and 75.9% after `HB_1` neutralization.

## 4. Reproducibility

- Intact probabilities reproduced the Block 7 values within `1.12e-16`.
- All four intact membership/tolerance event summaries reproduced within `1e-12`.
- All 20 source feature arrays and 23 total external inputs/outputs were recorded by size and SHA-256.
- The result-producing run passed 15/15 checks and reproduced unchanged on rerun.
- The independent validator reconstructed 57,369 probabilities, event outputs, 32 leave-one-participant-out folds, and concentration tables.
- All 11/11 independent checks passed twice.

## 5. Knowledge Gained and Next Decision

The frozen detector is asymmetrically dependent on the two channel feature groups. `HB_1` supplies feature contributions required for any true detections at threshold 0.96, while `HB_2` contributes less strongly under the same controlled neutralization. Because the ablations also reduced false alarms, the result identifies coupled representation and operating-point sensitivity rather than a simple loss of signal information.

No channel-aware retraining or threshold change is authorized from this result. The next Block 8 experiment should test predefined raw-signal degradation levels before filtering and feature extraction, using validation only and the frozen model. That experiment must separate graceful degradation from threshold/calibration effects and must not inspect the current test partition.
