# Block 8 Single-Channel Robustness Protocol v0.1

**Created:** 2026-09-11
**Block:** 8, robustness and justified adaptation
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Partition authorized:** Validation only
**Frozen model:** Block 7 `H2-D`
**Model SHA-256:** `d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066`
**Frozen threshold:** `0.96`
**Current test partition authorized:** No
**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological Uncertainty

The direct two-channel wearable comparator achieved validation event F1 `0.1123` with `1.4558` false alarms per supported hour and descriptive test F1 `0.1493` with `1.2621` false alarms per hour. These aggregate results do not show whether the detector relies on both wearable channels or whether one channel can be lost without a material change in event behavior.

This matters because a forehead wearable may temporarily lose one electrode path. A two-channel model that depends strongly or unpredictably on one channel is not robust enough for later streaming work, even if its intact aggregate result appears acceptable.

## 2. Known Method and Limitation

The frozen `H2-D` pipeline uses eight 30-second epochs, five log-bandpower values per channel and epoch, a fitted `StandardScaler`, and logistic regression. Retraining a separate one-channel classifier would mix channel-loss robustness with a new model-fit question.

The present experiment instead neutralizes one channel at the fitted classifier input. For each of that channel's 40 context dimensions, the value is replaced by the corresponding train-fitted `StandardScaler.mean_`. The standardized value is therefore zero and the missing channel contributes no deviation from the training mean.

This is a controlled feature-contribution ablation. It does not reproduce amplifier saturation, impedance transients, correlated noise, or the physical behavior of an absent electrode. Raw-signal degradation requires a later protocol.

## 3. Fixed Comparators

| Comparator | Input treatment | Model | Threshold |
|---|---|---|---:|
| `H2-INTACT` | Frozen two-channel validation features | Frozen `H2-D` | 0.96 |
| `H2-NO-HB1` | Replace all `HB_1` context dimensions with their train-fitted model means | Frozen `H2-D` | 0.96 |
| `H2-NO-HB2` | Replace all `HB_2` context dimensions with their train-fitted model means | Frozen `H2-D` | 0.96 |

No coefficient, scaler, threshold, context window, candidate support, event-matching rule, or quality membership may change.

## 4. Hypotheses and Decision Bounds

### H8.1: single-channel robustness

For each channel ablation, compare the ablated result with `H2-INTACT` at primary membership and +/-15 seconds.

A channel loss is classified as **materially adverse** if either:

- event F1 decreases by at least `0.03`; or
- false alarms increase by at least `0.50` per supported hour.

These bounds reuse the material-change scale fixed for the Block 7 transfer gate. The detector passes this validation-only robustness screen only if neither channel ablation is materially adverse.

An apparent improvement after channel neutralization is not evidence that channel loss improves the detector. It indicates that the intact model's channel contribution or calibration is unstable and requires a separately designed investigation.

### H8.2: participant influence

For each ablation-minus-intact comparison, recompute aggregate F1 and false alarms per hour after leaving out each of the 16 validation `pid` groups in turn. Report the complete range and the number of folds matching the full-cohort direction.

An aggregate effect is classified as participant-stable only if all 16 leave-one-`pid`-out folds preserve its direction for both F1 and false alarms. Any direction reversal leaves participant stability unresolved.

This strict criterion is a robustness screen, not an inferential significance test.

## 5. Authorized Inputs

The result-producing script may read only:

- frozen validation `HB-2` feature arrays from `block7_transfer_validation_v0.1`;
- the frozen `H2-D` model after verifying its SHA-256;
- frozen validation participant membership and transition/background quality membership;
- committed Block 7 validation support and event-evaluation definitions; and
- train-fitted parameters contained in the frozen model solely to construct the predefined neutral value.

The script must not enumerate, load, score, summarize, or hash any current-test feature, score, model-result, event-output, or raw-signal artifact. It must not read raw EDF files.

## 6. Evaluation

### Primary

- event precision, recall, and F1 at +/-15 seconds;
- false alarms per supported hour;
- ablation-minus-intact differences; and
- 16 leave-one-`pid`-out influence values per comparison and metric.

### Fixed sensitivities

- primary membership at +/-45 seconds; and
- expanded quality membership at +/-15 and +/-45 seconds.

Continuous window average precision and ROC AUC are not authorized because the full-night candidate set is not a conventional negative-label set. No threshold curve will be produced.

## 7. Execution Order

1. Commit this protocol.
2. Commit the result-producing implementation and independent validator.
3. Verify the frozen model hash and validation-only assignment before loading features.
4. Reproduce the stored intact Block 7 validation probabilities and primary event metrics.
5. Apply both fixed channel neutralizations once.
6. Evaluate all three comparators on identical supported boundaries.
7. Compute the material-change and leave-one-participant-out results.
8. Retain the result regardless of direction.
9. Run the independent validator twice before the result decision is closed.

## 8. Required Records

Retain:

- model-hash and feature-schema verification;
- exact ablated dimension indices and feature names;
- validation support parity;
- continuous ablation scores outside Git with relative path, size, and SHA-256;
- compact event, participant, comparison, and leave-one-participant-out tables;
- in-run and independent integrity checks;
- the observed limitation and next decision; and
- confirmation that no test or raw-signal artifact was accessed.

## 9. Decision Boundary

This experiment can identify sensitivity of the frozen classifier to neutral loss of one channel and concentration of that sensitivity among validation participants. It cannot establish physical electrode-failure performance, clinical reliability, independent generalization, or the value of retraining.

If the robustness screen fails, a channel-aware training or signal-degradation method requires a new hypothesis and protocol. If it passes, the next Block 8 experiment may test controlled raw-signal degradation. Neither outcome authorizes revision against the current test partition.
