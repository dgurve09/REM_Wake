# Block 8 Train-Only Noise-Augmentation Protocol v0.1

**Created:** 2026-09-12
**Block:** 8, robustness and justified adaptation
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Training partition:** 82 recordings from 64 `pid` groups
**Development evaluation partition:** 20 validation recordings from 16 `pid` groups
**Current test partition authorized:** No
**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological Uncertainty

The frozen direct wearable model has weak absolute performance and asymmetric degradation. On validation, clean `H2-D` produced event F1 0.1123 and 1.4558 false alarms per supported hour. At 10 dB, isolated `HB_1` noise reduced F1 to 0.0419 and increased false alarms to 4.4616/hour, while isolated `HB_2` noise reduced both F1 and alarm rate. Severe 0 dB both-channel noise eliminated all true detections.

The unresolved question is whether exposing the same simple classifier to fixed signal degradations during training can reduce its channel-specific calibration failure without sacrificing clean performance. This experiment does not test whether a more complex representation can solve the low absolute F1.

## 2. Known Method and Limitation

The comparator `H2-D` uses:

1. 0.3-35 Hz filtering and 256-to-128 Hz resampling;
2. train-fitted channel-wise robust signal normalization;
3. five Welch log-bandpower features per channel and 30-second epoch;
4. eight consecutive epochs, producing 80 input dimensions;
5. `StandardScaler` and balanced logistic regression; and
6. threshold 0.96 previously selected on validation.

This method was trained only on clean features. It therefore has no constraint encouraging stable behavior when either input channel is perturbed. Generic model complexity is not introduced because the observed failure is specifically linked to channel degradation and score calibration.

## 3. Fixed Adaptation

The sole new model is `H2-NA`, a noise-augmented version of the same `H2-D` pipeline. Its model class, feature schema, context, labels, background rows, class weighting, and optimizer remain unchanged.

For every train recording and channel, generate one standard-normal 256 Hz sequence using:

`SHA-256("20260912|subject|channel")`

Propagate the clean signal and unit noise separately through the frozen linear filter/resampling path. Calibrate noise over samples in valid scored epochs using the same RMS definition as the prior validation stress test. The same recording-channel noise sequence is reused across levels.

Each eligible labeled train candidate contributes one row from each condition:

| Training condition | Perturbation |
|---|---|
| `TRAIN-CLEAN` | No added noise |
| `TRAIN-BOTH-20DB` | 20 dB noise in both channels |
| `TRAIN-BOTH-10DB` | 10 dB noise in both channels |
| `TRAIN-HB1-10DB` | 10 dB noise in `HB_1` only |
| `TRAIN-HB2-10DB` | 10 dB noise in `HB_2` only |

There is no random condition sampling and no validation-driven choice of training SNR. The severe 0 dB condition is excluded from training because it destroyed event information in the fixed stress test; it remains an evaluation-only limit.

## 4. Model Configuration

`H2-NA` uses:

- `StandardScaler` fitted on the augmented train rows only;
- `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, tol=1e-4)`;
- random state `20260912`;
- no hyperparameter, feature, channel, or architecture search; and
- no validation rows during scaler fitting, classifier fitting, or threshold selection.

Every augmented copy of a labeled candidate remains in the same participant group. Positive and negative candidates are not resampled beyond the five fixed signal conditions.

## 5. Train-Only Threshold Selection

Select the `H2-NA` operating threshold using five-fold participant-grouped out-of-fold prediction on the 64 train `pid` groups:

1. use deterministic `GroupKFold(n_splits=5)`;
2. fit each fold model using all five feature conditions from the other train groups;
3. score the clean full-night feature sequence for held-out train groups;
4. concatenate the five held-out full-night score sets;
5. evaluate thresholds 0.01 through 0.99 in increments of 0.01 using primary membership and +/-15-second matching; and
6. select maximum event F1, then minimum false alarms/hour, then maximum recall, then maximum threshold.

After threshold selection, fit one final `H2-NA` model on all augmented train rows. The selected threshold and final model hash must be written before validation scoring. `H2-D` retains its existing model hash and threshold 0.96.

## 6. Frozen Validation Evaluation

Evaluate `H2-D` and `H2-NA` on the same 20 validation recordings under all six previously generated conditions:

- clean;
- both channels at 20 dB;
- both channels at 10 dB;
- both channels at 0 dB;
- `HB_1` only at 10 dB; and
- `HB_2` only at 10 dB.

The existing deterministic validation feature arrays are reused only after the model and train-only threshold are frozen. Both models use their own frozen thresholds. Alarm consolidation, event matching, quality membership, and supported time remain unchanged.

Primary reporting uses primary membership at +/-15 seconds. Primary and expanded membership at +/-45 seconds and expanded membership at +/-15 seconds are fixed sensitivities.

## 7. Hypotheses and Decision Rules

### H8.6: clean preservation

Relative to clean `H2-D`, clean `H2-NA` must:

- lose less than 0.03 event F1; and
- increase false alarms by less than 0.50/hour.

### H8.7: isolated `HB_1` failure mitigation

Under 10 dB isolated `HB_1` noise, `H2-NA` must simultaneously:

- improve F1 by at least 0.03 over `H2-D`; and
- reduce false alarms by at least 0.50/hour.

### H8.8: degradation-gap reduction

Compare each model's `HB_1` 10 dB result with its own clean result. Relative to the `H2-D` degradation gap, `H2-NA` must:

- reduce the F1 loss by at least 0.03; and
- reduce the false-alarm increase by at least 0.50/hour.

### H8.9: meaningful clean advancement

Because the absolute detector result remains weak, `H2-NA` counts as progress toward the original detector only if clean validation F1 improves by at least 0.05 and false alarms do not increase. This is an engineering advancement gate, not a clinical-performance threshold.

The adaptation advances as a robustness method only if H8.6-H8.8 all pass. It advances as a candidate core detector only if H8.9 also passes. A smaller clean F1 increase is retained but not described as meaningful advancement.

## 8. Participant Uncertainty

For every condition, compare `H2-NA` with `H2-D` using a paired participant-cluster bootstrap over the same 16 validation `pid` groups with 2,000 resamples and seed `20260912`. Report F1 and false-alarm differences with percentile 95% intervals.

Also report each model's paired `HB_1`-noise-minus-clean degradation interval. These intervals describe uncertainty within a reused development cohort; they do not provide independent confirmation.

## 9. Required Controls

Before accepting a result, verify:

1. exact 82-recording/64-`pid` train and 20-recording/16-`pid` validation membership;
2. no participant overlap between train and validation;
3. no current-test enumeration, path, score, feature, or model access;
4. exact clean train-feature reproduction within `1e-6`;
5. all train SNR calibrations within 0.02 dB;
6. one deterministic training noise basis per recording-channel;
7. identical labeled row identity and five-condition multiplicity;
8. no participant leakage across out-of-fold threshold-selection folds;
9. every train participant scored exactly once out of fold;
10. finite augmented features and probabilities;
11. final model fitted only after the out-of-fold threshold is frozen;
12. exact reuse of the existing `H2-D` model, threshold, validation stress features, support, and event rules;
13. complete paired participant support; and
14. external artifact size and SHA-256 verification.

## 10. Execution Order

1. Commit this protocol.
2. Commit the result-producing implementation and independent validator.
3. Generate all fixed train perturbations without validation access.
4. Reproduce clean train features and verify calibration controls.
5. Build participant-grouped out-of-fold train predictions and select one threshold.
6. Fit and hash the final augmented model.
7. Verify the threshold and model freeze record.
8. Open the previously frozen validation stress features and score both models once.
9. Compute event, participant, uncertainty, and hypothesis outputs.
10. Retain the result regardless of direction.
11. Run the independent validator twice.
12. Perform one immutable rerun.

## 11. Storage Boundary

Train noise feature arrays, the fitted model, full-night train/validation scores, and fold models remain outside Git. Compact calibration, row-construction, fold-membership, threshold-curve, event, bootstrap, hypothesis, hash, and validation tables are retained in Git.

## 12. Interpretation Boundary

This experiment can determine whether fixed train-only Gaussian-noise exposure corrects one observed calibration failure in the existing logistic pipeline. It cannot establish natural artefact robustness, clinical usefulness, independent generalization, or superiority of a learned temporal representation.

No outcome authorizes current-test reuse or retrospective adjustment. If the robustness gate fails, the noise-augmentation path stops at v0.1. If it passes but the meaningful advancement gate fails, the method may inform later quality-aware work but does not replace the core detector. A later temporal model requires a separate representation hypothesis and protocol.
