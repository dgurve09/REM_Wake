# Causal Multiscale Change-Feature Protocol v0.1

**Created:** 2026-10-25

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Schedule and evidence boundary

This is a user-directed supplemental method-development experiment during Block 11. It does not replace the scheduled transition-measure and streaming go/no-go work. The BOAS train participants have already supported multiple development experiments, so a favorable result remains exploratory and cannot serve as independent confirmation.

The prior enriched-feature stop rule remains valid for the retrospective 30-second representation. This experiment addresses a different documented mechanism: high-resolution causal change across the putative boundary combined with a peak-first alarm representation. Validation and current test remain closed.

## 2. Known evidence and technical uncertainty

The existing causal REM-to-Wake model used 80 log-bandpower values from eight 30-second windows and reached train nested out-of-fold F1 0.0517. Causal endpoint factorization learned REM/Wake state information but reached F1 0.0375.

The endpoint-factorization audit found that all five inner folds selected the minimum threshold, 99.9583% of candidates exceeded it, and the threshold-then-collapse rule merged candidates into runs with median duration 16,290 seconds. That representation can turn hours of signal into one global-maximum alarm.

The earlier enriched experiment used 45 features per 30-second epoch and improved retrospective wearable F1 to 0.1796, but it did not isolate the 15 seconds before and after a causal candidate boundary. It remains uncertain whether boundary-centred multiscale changes contain information omitted by fixed 30-second summaries.

## 3. Dataset roles

The local harmonized corpus contains sleep-stage annotations from ten datasets. Those annotations are appropriate for Block 11 transition prevalence, scorer disagreement, and repeated-night stability analyses, but most datasets do not include locally available raw EEG.

Locally available raw signal suitable for new analysis is:

- BOAS: simultaneous PSG and two-channel wearable headband EEG; and
- `gx36m`: multi-night, three-scorer laboratory PSG without a wearable headband.

Only BOAS can answer the present wearable feature question. `gx36m` may later test whether a retained mechanism reproduces in external PSG under a separate protocol, but it cannot confirm wearable performance. Sleep-EDF remains unsuitable for direct application of the frozen BOAS channel model because its bipolar channel semantics differ.

## 4. Causal decision grid and target

Reuse the reviewed 15-second causal grid and eight 30-second base-feature windows ending at each decision time. A positive decision occurs at `nominal + 15 s`, after the complete 30-second uncertainty interval has elapsed. No sample after the decision time is permitted.

Primary positives are the 180 primary-quality REM-to-Wake events with complete causal support. Full-risk-set negatives must be at least 135 seconds from every labelled REM-to-Wake or Wake-to-REM boundary. Near-event candidates omitted from fitting remain in full-night scoring.

## 5. Fixed feature representations

### `BASE-PEAK`

Use the existing 80 causal values:

- two wearable channels, `HB_1` and `HB_2`;
- five log mean-power bands per channel: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, and beta 16-30 Hz; and
- eight contiguous 30-second windows ending at the decision time.

### `ADV-PEAK`

Retain the 80 base values and append 180 boundary-centred values derived from four 15-second subwindows with starts at `-60`, `-45`, `-30`, and `-15` seconds relative to the decision.

For each channel and subwindow, calculate 12 values:

1. five log mean Welch bandpowers using the fixed bands;
2. normalized spectral entropy over 0.5-30 Hz;
3. a log-log spectral slope and intercept over 2-30 Hz as a fixed aperiodic-shape proxy;
4. log RMS amplitude;
5. log mean absolute line length;
6. Hjorth mobility; and
7. Hjorth complexity.

For the channel pair and each subwindow, calculate Pearson correlation and mean magnitude-squared coherence in each of the five bands. Each subwindow therefore contributes 30 values and the four subwindows contribute 120.

The candidate boundary is 15 seconds before the decision. For all 30 subwindow metrics, append:

- signed change: final post-boundary subwindow minus final pre-boundary subwindow; and
- absolute change magnitude.

These 60 change values produce 180 advanced values and 260 total values for `ADV-PEAK`. No alternate window length, feature subset, feature selection, or interaction expansion is permitted in v0.1.

## 6. Preprocessing and feature controls

Reuse the frozen wearable preprocessing:

- fourth-order 0.3-35 Hz Butterworth bandpass;
- 256-to-128 Hz polyphase resampling;
- frozen train-derived robust channel scaling; and
- Welch Hann windows with 512 samples per segment, 256-sample overlap, and a 512-point FFT.

The 80 base values must reproduce the reviewed causal cache exactly. All advanced values must be finite. Spectral entropy, correlation, coherence, and Hjorth values must obey their mathematical bounds. Synthetic controls must verify band localization, finite constant-signal handling, and the expected ordering of flat versus 1/f spectral slopes before fitting.

## 7. Fixed model

Both candidates use identical L2 logistic regression:

- `StandardScaler` fitted on fitting participants only;
- `C=0.1`;
- `solver="lbfgs"`;
- no class weighting;
- `max_iter=1000`;
- `tol=1e-6`; and
- `random_state=20261025`.

The natural full-risk-set prevalence is retained. The experiment changes only the representation after both candidates receive the same causal target and peak-first alarm rule.

## 8. Peak-first alarm representation

Thresholding cannot create or merge peak candidates.

Within each contiguous 15-second recording segment:

1. identify a local maximum when its score is strictly greater than the preceding score and greater than or equal to the following score;
2. treat a missing segment neighbour as negative infinity;
3. this rule retains the earliest point of an exact plateau;
4. rank local maxima by descending score, breaking ties by earlier time; and
5. greedily retain a maximum only when it is more than 60 seconds from every already retained maximum.

Restore retained peaks to chronological order before thresholding. Each retained above-threshold peak is one alarm; no contiguous-run collapse is applied. The count of alarms must be monotonic non-increasing as the threshold increases.

## 9. Participant-grouped execution

Reuse the five frozen train `pid` folds. For every outer fold and candidate:

1. hold out all recordings from outer-fold participants;
2. rotate each remaining fold once as the inner-held-out fold;
3. fit on the other three folds;
4. concatenate inner full-night scores and convert them to peak candidates recording by recording;
5. select one threshold under the fixed 0.25 false-alarms/hour budget;
6. refit on all four outer-training folds; and
7. score every supported outer-fold decision once before applying the fixed peak rule.

Among thresholds satisfying the alarm budget, select maximum event F1, then maximum recall, minimum false alarms/hour, and higher threshold. Outer labels cannot influence scaling, fitting, peak construction, or threshold selection.

## 10. Evaluation and hypothesis gate

Use the prior causal matching interval: an alarm matches a primary event only from `nominal + 15 s` through `nominal + 30 s`. Matching is one-to-one. Corresponding windows around excluded or non-target transitions are ignored before false alarms are counted.

Report precision, recall, F1, false alarms/hour, TP, FP, FN, ignored alarms, threshold, peak counts, alarm monotonicity, convergence, score ranges, and participant-level results. Use 2,000 paired participant-bootstrap resamples with seed `20261025`.

`ADV-PEAK` is retained only for a new confirmation cohort if, relative to `BASE-PEAK`:

1. event F1 improves by at least +0.03;
2. false alarms/hour increase by no more than 0.05;
3. every fit passes the numerical gate; and
4. peak/alarm monotonicity passes in every inner curve.

The `BASE-PEAK` comparison with the older `CAUSAL-RW` result is descriptive because the alarm representation was designed after the prior run-collapse result was known.

## 11. Stop rule

Do not change the subwindow lengths, feature list, aperiodic proxy, model, regularization, class weighting, local-maximum rule, refractory interval, threshold objective, timing, negative exclusion, or alarm budget after results are available.

If the gate fails, stop advanced causal feature expansion at v0.1. A favorable train-only result does not authorize validation/test access, a deployable detector claim, or a wearable generalization claim. External PSG can test mechanism portability only under a separate protocol; independent wearable confirmation requires a new paired wearable cohort.

## 12. Method references

1. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*. 1967;15(2):70-73. https://doi.org/10.1109/TAU.1967.1161901
2. Hjorth B. EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*. 1970;29(3):306-310. https://doi.org/10.1016/0013-4694(70)90143-4
3. Inouye T, Shinosaki K, Sakamoto H, et al. Quantification of EEG irregularity by use of the entropy of the power spectrum. *Electroencephalography and Clinical Neurophysiology*. 1991;79(3):204-210. https://doi.org/10.1016/0013-4694(91)90138-T
4. Donoghue T, Haller M, Peterson EJ, et al. Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*. 2020;23:1655-1665. https://doi.org/10.1038/s41593-020-00744-x
5. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
6. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
7. BOAS dataset, OpenNeuro `ds005555`, snapshot `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
