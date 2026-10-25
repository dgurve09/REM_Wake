# Multi-Dataset PSG External-Generalization Protocol v0.1

**Date frozen:** 2026-10-25
**Precondition:** Eligibility audit commit `e67dcec`
**Model status:** No pooled PSG feature extraction, fitting, threshold selection, or event evaluation has been run

## 1. Purpose

Test whether REM-to-Wake boundary information generalizes across independent PSG datasets before returning to wearable transfer. This is an intermediate mechanism study. It does not replace the project's wearable endpoint and cannot establish wearable performance.

The local hypnogram collection is not treated as eleven interchangeable EEG datasets. All available hypnograms may contribute label-distribution analyses, but raw-signal training is limited to the three sources that passed the common-channel audit.

## 2. Eligible raw-signal cohort

| Dataset | Nights | Participant groups | Primary labels | Audited R-to-W events |
|---|---:|---:|---|---:|
| BOAS | 128 | 100 | Published human consensus | 365 |
| gx36m | 58 | 20 | Epoch majority of three experts | 124 |
| ISRUC group 3 | 10 | 10 | Two-expert agreement; disagreement is unknown | 32 |
| **Total** | **196** | **130** | | **521** |

Use three common right-sided channels: `F4`, `C4`, and `O2`. BOAS channels are referenced to M1; gx36m and ISRUC channels are referenced to A1. This near-match is not assumed to remove acquisition-domain shift.

The gx36m `SLP010_Day1` unscored 166-second tail is outside candidate support. No candidate may be created where its complete feature context or label interval exceeds reviewed signal and label support.

## 3. Dataset roles

### Signal experiment

Only BOAS PSG, gx36m, and ISRUC group 3 enter signal feature extraction and model fitting.

### Label-only analysis

Challenge 2018, DOD, the remaining ISRUC recordings, Sleep-EDF, HMC, CAP Sleep, iSLEEPS, Sleep-Accel, and DREAMS will be used separately for REM-to-Wake prevalence, timing, scorer, pathology-group, and repeated-night analyses where their metadata permit. They cannot be counted as EEG training samples from the files currently stored locally.

## 4. Label construction

Use 30-second stage sequences with the common labels `W`, `N1`, `N2`, `N3`, and `REM`.

- A primary event occurs when a reviewed `REM` epoch is immediately followed by a reviewed `W` epoch.
- Its nominal boundary is the start of the `W` epoch.
- For ISRUC, an epoch is reviewed only when both scorers agree.
- For gx36m, the primary stage is the three-scorer majority.
- Unknown, missing, or unscored epochs cannot form positives or confirmed negatives.
- Background candidates must be at least 135 seconds from any REM/Wake transition, unknown epoch boundary, or recording edge.

For gx36m and ISRUC, retain scorer-specific transition tables for sensitivity evaluation. Scorer-specific labels cannot alter fitting, feature selection, thresholds, or the primary majority/consensus result.

## 5. Signal preprocessing

Process each recording at its native sampling rate with a forward-only filter. For every candidate decision time, read only the preceding 240 seconds from the filtered three-channel EEG. The fixed steps are:

1. convert the physical signal to microvolts using the EDF calibration;
2. apply the fixed 0.3-35 Hz fourth-order Butterworth bandpass with `sosfilt` in chronological order and zero initial state;
3. retain the native 256 Hz BOAS, 500 Hz gx36m, and 200 Hz ISRUC sampling rates;
4. reject nonfinite or flat channels; and
5. extract all features from windows ending at the decision time.

The first supported decision is 240 seconds after recording start, which also removes the causal filter startup region from evaluated contexts. No filter or feature at a decision may read a later sample. This is causal with respect to information availability, although it remains an offline implementation rather than a streaming-runtime claim.

No test-dataset amplitude statistic may be used to normalize training data. Feature standardization is fitted on the training datasets only within each held-out-dataset run.

## 6. Fixed representations

### `PSD3-BASE`

For each channel, calculate five log Welch band powers from each of eight consecutive 30-second windows: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, and beta 16-30 Hz. Total: `3 x 5 x 8 = 120` values.

### `ADV3-CHANGE`

Retain the 120 baseline values and add four consecutive 15-second subwindows over the final minute. For each subwindow calculate:

- per channel: five log band powers, spectral entropy, log-log spectral slope and intercept over 2-30 Hz, log RMS, log line length, Hjorth mobility, and Hjorth complexity (`12 x 3 = 36` values);
- for each of the three channel pairs: correlation and five band coherences (`6 x 3 = 18` values); and
- signed and absolute change between the final two subwindows for all 54 subwindow measurements.

The advanced addition contains `4 x 54 + 2 x 54 = 324` values. The complete representation contains 444 values.

Welch settings are a four-second Hann window with 50% overlap at each native rate. Line length and zero-crossing quantities are expressed per second, and Hjorth mobility is expressed in Hz so sampling-rate differences do not redefine the features. The feature definition, not the observed performance, determines the schema.

## 7. Fixed model

Use natural-prevalence L2 logistic regression for both representations:

- `C = 0.1`;
- `solver = lbfgs`;
- no class weighting;
- maximum 1,000 iterations;
- tolerance `1e-6`; and
- training-fitted `StandardScaler`.

Do not add a neural network, resampling scheme, class weight, focal loss, feature pruning, calibration model, or dataset indicator in v0.1.

## 8. Dataset-held-out evaluation

Run three fixed leave-one-dataset-out evaluations:

1. train on gx36m plus ISRUC and hold out BOAS;
2. train on BOAS plus ISRUC and hold out gx36m; and
3. train on BOAS plus gx36m and hold out ISRUC.

All nights from a participant remain together. Within each two-dataset training pool, use five participant-grouped inner folds for threshold selection. Each inner fold must contain participants from both training datasets when mathematically possible.

The gx36m and ISRUC held-out results are the primary external tests because their signals have not been used in prior project model development. The BOAS-held-out result is descriptive because BOAS informed the representation and earlier experimental choices.

## 9. Alarm rule and metrics

Use the frozen peak-first procedure from the causal multiscale experiment:

- local maxima on the 15-second decision grid;
- descending-score greedy nonmaximum suppression;
- more than 60 seconds between retained peaks;
- threshold after peak selection; and
- one alarm per retained above-threshold peak.

Select the threshold from inner training scores at a maximum 0.25 false alarms/hour, maximizing F1, then recall, then minimizing false alarms/hour, then choosing the higher threshold.

An alarm matches one primary event only from `nominal + 15 s` through `nominal + 30 s`. Report TP, FP, FN, ignored alarms, precision, recall, F1, false alarms/hour, participant coverage, and threshold. Also report average precision and ROC AUC for the score-ranking diagnostic; these do not replace event metrics.

## 10. Hypothesis and advancement gate

`H-PSG-XDATA`: the advanced representation contains boundary information that generalizes beyond BOAS rather than only fitting BOAS acquisition and label characteristics.

Retain `ADV3-CHANGE` for later paired PSG-to-wearable transfer only if all conditions hold:

1. pooled gx36m plus ISRUC external event F1 is at least 0.15;
2. pooled external F1 exceeds `PSD3-BASE` by at least 0.03;
3. pooled external recall is at least 0.15;
4. pooled external false alarms do not exceed 0.30/hour;
5. neither external dataset has an advanced-minus-baseline F1 difference below -0.03;
6. every fit passes the numerical controls; and
7. scorer-sensitivity conclusions are reported without selecting a favorable scorer after results.

Use 2,000 participant-bootstrap resamples, stratified by held-out dataset, for the pooled external advanced-minus-baseline differences. The point gate determines v0.1 retention; intervals determine the strength and uncertainty of the evidence.

## 11. Stop rules

- Do not tune any feature, channel, window, label consensus, model hyperparameter, threshold objective, refractory interval, or event tolerance after held-out results are available.
- If the gate fails, stop this pooled formulation at v0.1 and retain the failure by dataset.
- If it passes, a final all-196-night PSG model may be fitted only as an unevaluated transfer source for a separately predeclared paired wearable study.
- Do not describe any PSG-only result as wearable validation, clinical detection, or diagnosis.

## 12. Method and dataset references

1. Welch PD. The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*. 1967;15(2):70-73. https://doi.org/10.1109/TAU.1967.1161901
2. Hjorth B. EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*. 1970;29(3):306-310. https://doi.org/10.1016/0013-4694(70)90143-4
3. Inouye T, Shinosaki K, Sakamoto H, et al. Quantification of EEG irregularity by use of the entropy of the power spectrum. *Electroencephalography and Clinical Neurophysiology*. 1991;79(3):204-210. https://doi.org/10.1016/0013-4694(91)90138-T
4. Donoghue T, Haller M, Peterson EJ, et al. Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*. 2020;23:1655-1665. https://doi.org/10.1038/s41593-020-00744-x
5. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
6. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
7. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
8. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
9. Bitbrain Open Access Sleep dataset, OpenNeuro `ds005555`, version `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
