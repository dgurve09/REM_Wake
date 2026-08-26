# Block 7 Paired PSG-to-Wearable Transfer Protocol v0.1

**Created:** 2026-08-25
**Block:** 7, paired PSG-to-wearable transfer
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Split:** `splits/grouped_pid_split_v0.1/`
**Labels:** `labels/quality_analysis_membership_v0.1/`
**Protocol status:** Frozen before Block 7 feature extraction or model fitting
**Test features or Block 7 scores accessed:** No

## 1. Technological Question

How much REM-to-Wake event performance changes across a common six-channel PSG EEG montage, a reduced two-channel PSG montage, and the simultaneous two-channel wearable recording when the same simple direct-event method is used, and does a strict PSG-to-wearable zero-shot failure justify one predeclared training-only feature-alignment method?

The experiment measures event-specific modality transfer. It is not a sleep-staging experiment and is not a search for the best classifier.

## 2. Known Evidence and Remaining Uncertainty

The frozen wearable DE-B baseline uses eight 30-second epochs of two-channel log-bandpower context and logistic regression. It achieved test event F1 `0.1497` and `1.2571` false alarms per supported hour, but precision remained `0.0909`. That result establishes a simple wearable reference; it does not reveal how much performance is lost relative to simultaneous PSG or whether a PSG-trained detector transfers.

Wearable sleep analysis and EEG domain adaptation are established methods. The unresolved project question is narrower: the magnitude and structure of paired device and electrode-location shift for PSG-hypnogram-derived REM-to-Wake events under the frozen participant grouping and event metric.

The current project test partition has already been inspected in Blocks 5 and 6. Block 7 may use it once for a fixed paired descriptive comparison, but it cannot provide new independent confirmation.

## 3. Hypotheses

### H7.1 Channel reduction

`PSG-6` will outperform `PSG-2` under the same direct-event method if spatially distributed PSG EEG contributes transition information beyond the frontal pair.

**Not supported if:** the paired participant difference is negligible, inconsistent, or favors `PSG-2`.

### H7.2 Strict zero-shot transfer

A model fitted on `PSG-2` and applied without target-derived normalization or coefficient updates to mapped `HB-2` features will degrade relative to both direct `PSG-2` and direct `HB-2` fits because device and electrode-location differences alter the feature distribution.

**Not supported if:** zero-shot event F1 and false-alarm burden are comparable to direct wearable fitting under the predeclared validation gate.

### H7.3 Simple alignment

If the zero-shot gate fails and a feature-distribution mismatch is present, training-only paired robust location/scale alignment will recover part of the deficit without wearable labels or model-coefficient updates.

**Not supported if:** alignment does not improve both event F1 and false alarms per hour on validation, or its participant-level effect is unstable.

## 4. Fixed Modalities and Channel Mapping

| Comparator | Channels | Input features | Role |
|---|---|---:|---|
| `P6-D` | `PSG_F3/F4/C3/C4/O1/O2` | 240 | Common six-channel PSG EEG reference |
| `P2-D` | `PSG_F3/F4` | 80 | Direct reduced-PSG reference |
| `H2-D` | `HB_1/HB_2` | 80 | Direct wearable reference |
| `P2-H2-Z` | train on `PSG_F3/F4`; apply to `HB_1/HB_2` | 80 | Strict zero-shot transfer |
| `P2-H2-A` | `P2-H2-Z` after permitted robust feature alignment | 80 | Conditional adaptation comparator |

The zero-shot mapping is `PSG_F3 -> HB_1` and `PSG_F4 -> HB_2`. Both acquisitions report microvolts, 256 Hz, and M1 reference, but their electrode locations and hardware differ. Therefore `P2-H2-Z` estimates the combined real acquisition shift, not hardware shift alone.

`PSG-6` means the full common PSG EEG montage, not every clinical PSG sensor. Optional EOG, EMG, respiratory, pulse, oxygen-saturation, and motion channels are excluded because availability differs and the fixed EEG feature method is not appropriate for all sensor types.

## 5. Shared Labels, Split, and Support

- Use the frozen `pid`-grouped train/validation/test assignment with no participant crossing partitions.
- Use only primary REM-to-Wake labels and primary-eligible reviewed background rows from membership v0.1.
- Use the same candidate times, temporal-support mask, disconnection exclusions, event consolidation, and matching implementation for every comparator.
- Keep `sub-32`, `sub-39`, and `sub-50` in the primary analysis. A fixed sensitivity excluding `sub-32` and `sub-50` from the descriptive test summary may be reported separately.
- Do not alter membership from channel availability. A required-channel failure triggers protocol revision rather than selective exclusion.

## 6. Preprocessing and Features

The fixed signal pipeline is:

1. read required channels in microvolts;
2. continuous 0.3-35 Hz fourth-order Butterworth filtering with forward-backward second-order sections;
3. no rereferencing and no separate notch filter;
4. polyphase resampling from 256 to 128 Hz;
5. channel-wise robust center and scale estimated from a deterministic 1 Hz sample of train recordings only;
6. Welch mean power in delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, and beta 16-30 Hz;
7. base-10 log power for each channel-band pair; and
8. eight epochs from `t-120 s` through `t+90 s` at 30-second steps.

For `P6-D`, `P2-D`, and `H2-D`, channel scalers are fitted only on the corresponding modality's train recordings. `P6-D` and `P2-D` share the same fitted `F3/F4` channel scalers so their contrast changes channel count rather than preprocessing.

For strict `P2-H2-Z`, the frozen PSG `F3/F4` channel scalers and the fitted PSG model pipeline are applied to `HB_1/HB_2`; no wearable-derived center, scale, feature alignment, label, threshold, or coefficient is used.

Context cannot cross a recording edge, invalid scored epoch, disconnection, missing epoch, or non-30-second gap. Human stage codes define labels and support only; they are not input features.

## 7. Fixed Model and Threshold Rule

Every direct fit uses the DE-B method family:

- `StandardScaler` fitted on the model's train matrix;
- logistic regression with `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=500`, and `tol=1e-4`;
- no feature selection, architecture search, neural network, or hyperparameter sweep; and
- deterministic seed `20260825` for resampling and recorded software versions.

Validation thresholds are searched from `0.01` to `0.99` in steps of `0.01` and selected by maximum event F1, then minimum false alarms per hour, maximum recall, and highest threshold. `P2-H2-Z` and conditional `P2-H2-A` inherit the validation-selected `P2-D` threshold because target-specific threshold selection would itself be adaptation.

## 8. Zero-Shot Gate and Permitted Adaptation

The conditional adaptation comparator is permitted only if both validation conditions hold:

1. relative to `H2-D`, `P2-H2-Z` has event F1 at least `0.03` lower or false alarms at least `0.50` per supported hour higher; and
2. at least 20% of mapped feature dimensions have an absolute PSG-versus-wearable train median difference greater than `0.50` pooled robust scale units.

If the gate opens, `P2-H2-A` performs one fixed, label-free transformation. For each of the 80 mapped context features, compute median and `1.4826 * MAD` from paired PSG and wearable train rows. Transform the wearable feature to the PSG train location and scale, then apply the unchanged `P2-D` model and threshold. A zero MAD uses a scale of one and is reported. Wearable labels, validation outcomes, model coefficients, and test data cannot influence this transformation.

No adversarial adaptation, supervised fine-tuning, feature selection, or second adaptation method is authorized in v0.1. A negative alignment result is retained and closes the adaptation branch for this block.

## 9. Evaluation

### Primary metrics

- event precision, recall, and F1 at +/-15 seconds;
- false alarms per supported hour; and
- participant-level distribution and paired participant-cluster bootstrap differences.

### Sensitivity metrics

- the same event metrics at +/-45 seconds;
- expanded quality membership;
- per-recording results; and
- the fixed alignment-review sensitivity described in Section 5.

Window average precision and ROC AUC are diagnostic only. They cannot replace event-level evaluation.

### Paired contrasts

1. `P6-D - P2-D`: information associated with the four additional common PSG EEG channels;
2. `P2-D - H2-D`: acquisition and electrode-location gap under direct fitting;
3. `P2-H2-Z - H2-D`: cost of strict source-to-target transfer;
4. `P2-H2-A - P2-H2-Z`: recovery from the one permitted alignment method, if gated; and
5. `H2-D - frozen DE-B`: implementation-reproduction check, expected to be zero apart from explicitly documented Block 7 feature-generation changes.

## 10. Execution Order and Test Boundary

1. Commit this protocol and the channel-compatibility plan.
2. Run the channel audit and retain pass, revision, or no-go evidence.
3. Implement and validate feature generation using train recordings only.
4. Fit all direct models and evaluate train/validation.
5. Apply the zero-shot gate and, only if opened, run `P2-H2-A` exactly once on validation.
6. Freeze all model hashes, scalers, thresholds, feature hashes, comparator roles, and the adaptation decision.
7. Load the existing test partition once and apply every frozen comparator together.
8. Do not revise any method because of the descriptive test result.

A new locked partition or external cohort is required for later confirmatory performance evaluation.

## 11. Required Records

Retain:

- channel compatibility outputs and decision;
- participant and candidate construction counts by partition and comparator;
- channel-scaler and feature manifests;
- fit status, convergence warnings, thresholds, and the zero-shot gate table;
- event, participant, and paired-bootstrap summaries;
- failed or skipped comparator records;
- model, feature, and full-night score hashes with external paths; and
- software versions and random seeds.

Raw EDF, filtered signals, feature arrays, fitted models, and full-night candidate scores remain outside Git. Compact reviewed tables required for metric reconstruction may be retained.

## 12. References

1. Bitbrain Open Access Sleep dataset, OpenNeuro `ds005555` version `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
2. Welch PD. The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*. 1967;15:70-73. https://doi.org/10.1109/TAU.1967.1161901
3. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
4. Heremans ERM, Phan H, Borzee P, et al. From unsupervised to semi-supervised adversarial domain adaptation in electroencephalography-based sleep staging. *Journal of Neural Engineering*. 2022;19. https://doi.org/10.1088/1741-2552/ac6ca8
