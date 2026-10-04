# Paired Enriched-Feature and Fold-Local Pruning Protocol v0.1

**Created:** 2026-10-04

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological uncertainty

The existing direct detectors summarize each 30-second epoch with five log-bandpower values from each of two channels. The paired Block 7 comparison found low event precision for both reduced PSG and wearable EEG, while later temporal models did not establish a material nested improvement. It remains uncertain whether the representation omits transition-relevant spectral shape, signal morphology, irregularity, and interchannel information, or whether the primary limitation is the derived target itself.

Adding many features without a fixed selection design could overfit 180 train REM-to-Wake events. This experiment therefore tests a bounded enriched representation and fold-local pruning while holding labels, participants, temporal context, model class, event evaluation, and advancement rules fixed.

## 2. Fixed modalities

Run the same comparison separately for:

- `P2`: reduced PSG channels `PSG_F3` and `PSG_F4`; and
- `H2`: wearable channels `HB_1` and `HB_2`.

Use only train recordings. The paired modalities use the same recordings, `pid` groups, candidate times, outer folds, inner folds, labels, and event-matching rules.

## 3. Frozen preprocessing

For each modality:

1. read the two required EDF channels at 256 Hz;
2. apply the existing fourth-order 0.3-35 Hz Butterworth bandpass;
3. resample to 128 Hz;
4. apply the frozen Block 7 train-derived channel median and robust scale; and
5. use the exact valid 30-second epoch onsets in the reviewed Block 7 feature artifact.

The five existing log-bandpowers must reproduce the reviewed Block 7 values within `1e-5` maximum absolute difference before enriched features are accepted.

## 4. Frozen per-epoch feature sets

### F0: Existing representation

For each channel, retain base-10 log mean Welch power in:

- delta, 0.5-4 Hz;
- theta, 4-8 Hz;
- alpha, 8-12 Hz;
- sigma, 12-16 Hz; and
- beta, 16-30 Hz.

This yields 10 features per epoch and 80 values across the fixed eight-epoch context.

### F1: Enriched representation

For each channel, calculate:

- the five F0 log-bandpowers;
- relative power in the same five bands using total 0.5-30 Hz power;
- normalized spectral entropy over 0.5-30 Hz;
- 95% spectral-edge frequency over 0.5-30 Hz;
- log RMS amplitude;
- log mean absolute line length;
- zero-crossing rate;
- Hjorth mobility; and
- Hjorth complexity.

For the two-channel pair, also calculate:

- Pearson correlation;
- mean magnitude-squared coherence in each of the five bands; and
- channel-1 minus channel-2 log-bandpower asymmetry in each band.

This yields 45 features per epoch and 360 values across the same eight epochs. Use Welch estimates with a Hann window, 256 samples per segment, 128-sample overlap, and 512-point FFT. Use machine epsilon only as a denominator and logarithm guard.

## 5. Fixed candidates

For each modality compare:

| Candidate | Input | Classifier |
|---|---|---|
| `F0-L2` | Existing 80-value context | Standard scaling plus fixed L2 logistic regression, `C=1` |
| `F1-L2` | All 360 enriched values | Standard scaling plus fixed L2 logistic regression, `C=1` |
| `F1-EN` | Fold-locally pruned enriched values | Standard scaling plus elastic-net logistic regression |

All classifiers use balanced class weights, a fixed intercept, seed `20261004`, tolerance `1e-4`, and at most 3,000 iterations. `F0-L2` and `F1-L2` use `lbfgs`. `F1-EN` uses `saga`, `l1_ratio=0.5`, and the fixed `C` grid `{0.01, 0.1, 1.0}`.

No neural architecture, augmentation, post-processing, label, context, or background-sampling change is included.

## 6. Fold-local pruning

Fit pruning separately inside every inner fitting subset and again on each outer fitting subset:

1. remove features with variance at or below `1e-12`;
2. inspect absolute Pearson correlation using fitting rows only;
3. traverse features in frozen schema order and remove a later feature when its absolute correlation with an already retained feature is at least `0.95`; and
4. fit the standard scaler and elastic-net classifier only on retained fitting columns.

No feature statistic, coefficient, or selection decision may use an inner-held-out or outer-held-out participant. Record retained counts and feature identities for every fit. Feature-selection frequency is descriptive and cannot be used to revise the current experiment.

## 7. Nested participant-grouped design

Reuse the exact five frozen train `pid` folds. For every outer fold and modality:

1. keep the outer participants closed;
2. rotate each of the remaining four fold identifiers as the inner-held-out fold;
3. fit each candidate on the other three folds;
4. score every supported full-night boundary for the inner-held-out participants;
5. concatenate the four inner score sets;
6. select a threshold for each candidate, and select `C` jointly with threshold for `F1-EN`;
7. refit the frozen outer candidate on all four outer-training folds; and
8. score the outer-held-out participants once.

After all five outer folds, concatenate each candidate's one-time outer predictions. Outer labels cannot influence features, pruning, regularization, thresholds, or any model choice.

## 8. Threshold and event evaluation

Use the existing nested threshold grid:

- 0.01 through 0.95 in increments of 0.05; and
- sigmoid-transformed logits 3.0 through 14.0 in increments of 0.25.

For a candidate and `C`, select maximum primary +/-15-second event F1, then minimum false alarms/hour, maximum recall, and maximum threshold. Consolidate contiguous above-threshold 30-second candidates into one alarm represented by the highest score, with the earliest time breaking exact ties.

Report primary-label event precision, recall, F1, false alarms/hour, and event counts at +/-15 seconds. Report fixed sensitivity results for primary +/-45 seconds, expanded +/-15 seconds, and expanded +/-45 seconds.

## 9. Hypotheses and decision rules

### H-F1: Enriched representation

Within a modality, `F1-L2` advances over `F0-L2` only if primary +/-15-second outer F1 improves by at least `0.05` without increasing false alarms/hour.

### H-EN: Fold-local pruning

Within a modality, `F1-EN` advances over `F0-L2` only if primary +/-15-second outer F1 improves by at least `0.05` without increasing false alarms/hour. Compare `F1-EN` with `F1-L2` secondarily to determine whether pruning improves the enriched representation.

Use paired `pid`-cluster bootstrap resampling with 2,000 resamples and seed `20261004`. Directional participant support requires the F1-difference interval entirely above zero and the false-alarm-difference interval entirely at or below zero. Material support additionally requires the F1 lower bound to be at least `+0.05`.

The P2-versus-H2 contrast is descriptive. It does not authorize choosing a modality after observing the outer results.

## 10. Required controls

1. Confirm exactly 82 train recordings and 64 train `pid` groups.
2. Confirm no validation or current-test feature, EDF, participant, score, model, or result is accessed.
3. Reuse the exact five frozen participant folds and verify disjointness at every fit boundary.
4. Confirm 2,743 retained labeled candidates, including 180 positives.
5. Confirm exact epoch, context-center, and candidate-time parity across P2 and H2.
6. Reproduce the stored F0 features within `1e-5` for every recording and modality.
7. Verify all enriched values are finite and all spectral and coherence values obey their mathematical bounds.
8. Record pruning decisions separately for every inner and outer fit.
9. Confirm every classifier converged within 3,000 iterations; retain any failure without changing the settings.
10. Select regularization and thresholds only from complete inner out-of-fold full-night scores.
11. Produce one outer score per supported boundary per candidate and one outer assignment per participant.
12. Run an independent read-only validator twice.
13. Keep enriched recording arrays, full-night scores, and fitted models outside Git; record sizes and SHA-256 values.

## 11. Stop rule and interpretation boundary

Do not add features, change the correlation threshold, expand the `C` grid, alter `l1_ratio`, or revise preprocessing after results are available. If neither enriched candidate passes its gate, stop hand-engineered feature expansion on the reused train cohort and proceed to the separately planned label-uncertainty experiment.

A passing train-only result remains developmental. It cannot establish independent performance, clinical utility, or deployment readiness and requires a new locked or external compatible cohort.

## 12. Method references

1. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*. 1967;15(2):70-73. https://doi.org/10.1109/TAU.1967.1161901
2. Hjorth B. EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*. 1970;29(3):306-310. https://doi.org/10.1016/0013-4694(70)90143-4
3. Inouye T, Shinosaki K, Sakamoto H, et al. Quantification of EEG irregularity by use of the entropy of the power spectrum. *Electroencephalography and Clinical Neurophysiology*. 1991;79(3):204-210. https://doi.org/10.1016/0013-4694(91)90138-T
4. Zou H, Hastie T. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*. 2005;67(2):301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
