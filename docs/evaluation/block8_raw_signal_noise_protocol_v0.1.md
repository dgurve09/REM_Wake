# Block 8 Raw-Signal Noise Robustness Protocol v0.1

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

The first Block 8 experiment showed that neutralizing all `HB_1` feature contributions eliminated true detections at the fixed operating threshold, while neutralizing `HB_2` had a smaller effect. That controlled ablation did not establish how the detector behaves when real-valued signals are progressively contaminated before preprocessing.

The remaining uncertainty is whether the frozen wearable detector degrades gradually under controlled broadband noise, whether severe noise changes event behavior primarily through score calibration or candidate ranking, and whether the prior `HB_1`/`HB_2` asymmetry is also present under a moderate signal-level perturbation.

## 2. Known Method and Limitation

The frozen `H2-D` pipeline applies:

1. 0.3-35 Hz fourth-order Butterworth filtering with forward-backward second-order sections;
2. polyphase resampling from 256 to 128 Hz;
3. fixed channel-wise robust normalization fitted on train recordings;
4. five Welch log-bandpower features per channel and 30-second epoch;
5. eight-epoch context; and
6. frozen logistic scoring at threshold 0.96.

Adding independent white Gaussian noise is a standard controlled stress test, but it is not a physiological artefact model. It does not reproduce electrode pops, motion, sweat, impedance drift, saturation, shared-reference noise, or nonstationary interference. The selected signal-to-noise ratios are fixed engineering stress levels, not estimates of BOAS acquisition prevalence.

## 3. Deterministic Noise Construction

For each validation recording and channel, generate one standard-normal raw-rate sequence using a seed derived from:

`SHA-256("20260911|subject|channel")`

The same channel-specific sequence is reused across signal-to-noise levels so severity comparisons are nested rather than confounded by different noise realizations. `HB_1` and `HB_2` use independent sequences.

Noise is defined at 256 Hz before preprocessing. Because filtering and resampling are linear, process the clean raw signal and unit noise separately through the frozen filter/resampling path, then scale and add their processed values. This is mathematically equivalent to filtering and resampling the corresponding raw-domain sum and avoids repeating the same linear operations for every level.

For each channel, compute clean and unit-noise root-mean-square values over samples belonging to valid scored 30-second epochs. Set the scale so the processed valid-sample ratio satisfies:

`SNR_dB = 20 * log10(clean_RMS / noise_RMS)`

The achieved SNR must be within 0.02 dB of its target. The valid-epoch sample mask, preprocessing support, and feature schema must remain identical across all conditions.

## 4. Fixed Comparators

| Comparator | Perturbation | Target SNR |
|---|---|---:|
| `H2-CLEAN` | No added noise | Not applicable |
| `H2-BOTH-20DB` | Independent noise in both channels | 20 dB per channel |
| `H2-BOTH-10DB` | Independent noise in both channels | 10 dB per channel |
| `H2-BOTH-0DB` | Independent noise in both channels | 0 dB per channel |
| `H2-HB1-10DB` | Noise in `HB_1`; `HB_2` unchanged | 10 dB in `HB_1` |
| `H2-HB2-10DB` | Noise in `HB_2`; `HB_1` unchanged | 10 dB in `HB_2` |

All six comparators use the same model, threshold, labels, temporal support, alarm consolidation, and event matching.

## 5. Hypotheses and Decision Rules

### H8.3: mild-noise tolerance

`H2-BOTH-20DB` passes the mild-noise screen only if, relative to `H2-CLEAN`:

- event F1 decreases by less than 0.03; and
- false alarms increase by less than 0.50 per supported hour.

The bounds reuse the material-change scale fixed earlier in Block 8. An improvement in either metric does not establish benefit from noise.

### H8.4: nested dose response

Across `H2-CLEAN`, `H2-BOTH-20DB`, `H2-BOTH-10DB`, and `H2-BOTH-0DB`:

- Spearman correlation between clean and degraded continuous probabilities should not increase as SNR worsens; and
- mean absolute probability difference from clean should not decrease as SNR worsens.

Event F1 and false alarms per hour are reported at every level. Their monotonicity is descriptive because a fixed high threshold may reduce rather than increase alarms as scores move.

The probability-dose hypothesis is supported only if both correlation and mean-absolute-difference orderings hold without tolerance reversal. Values within `1e-12` are treated as equal.

### H8.5: channel-specific degradation asymmetry

At 10 dB, compare `H2-HB1-10DB` with `H2-HB2-10DB`. The prior feature-neutralization asymmetry is directionally reproduced if `HB_1` degradation causes:

- an equal or larger F1 loss from clean; and
- an equal or larger mean absolute probability difference from clean.

If the two criteria disagree, the signal-level asymmetry is inconclusive.

### Participant uncertainty

For every degraded-minus-clean event comparison, perform a paired participant-cluster bootstrap over the same 16 validation `pid` groups using 2,000 resamples and seed `20260911`. Report F1 and false-alarm differences with percentile 95% intervals. These intervals describe participant uncertainty within the reused validation cohort; they are not independent confirmation.

## 6. Authorized Inputs and Outputs

The result-producing script may read:

- the 20 frozen validation headband EDF recordings;
- frozen validation event/stage annotations needed to define valid preprocessing support;
- train-fitted wearable channel scalers;
- the frozen `H2-D` model after SHA-256 verification;
- frozen validation participant and quality membership; and
- the prior validation `H2-D` features, probabilities, support, and event metrics solely for exact clean-path reproduction.

The script must not enumerate, load, score, summarize, or hash current-test artifacts. It must not fit a scaler or model, search a threshold, select a noise level, exclude a participant, or change quality membership.

Full degraded feature arrays and continuous scores remain outside Git. Compact noise-calibration, event, participant, paired-bootstrap, dose-response, integrity, and decision tables are retained in Git.

## 7. Evaluation

### Primary event result

- primary membership at +/-15 seconds;
- precision, recall, F1, and false alarms per supported hour; and
- degraded-minus-clean paired participant intervals.

### Fixed sensitivities

- primary membership at +/-45 seconds; and
- expanded membership at +/-15 and +/-45 seconds.

### Continuous-score diagnostics

- Spearman correlation with clean probabilities;
- mean and maximum absolute probability difference from clean;
- median probability shift; and
- fraction of candidate probabilities crossing the fixed 0.96 threshold in either direction.

No threshold curve, ROC AUC, average precision, recalibration, or best-noise selection is authorized.

## 8. Required Controls

Before accepting a result, verify:

1. exact 20-recording and 16-`pid` validation membership;
2. exact frozen model and source-feature hashes;
3. clean feature reproduction within `1e-6` before float32 storage;
4. clean probability reproduction within `1e-12`;
5. clean event-metric reproduction within `1e-12`;
6. identical epoch onsets, stages, context centers, and support across conditions;
7. achieved SNR within 0.02 dB for every perturbed recording-channel pair;
8. exactly one deterministic noise basis per recording-channel reused across levels;
9. finite features and probabilities;
10. complete paired participant support; and
11. absence of test paths or artifacts from every manifest and reviewed table.

## 9. Execution Order

1. Commit this protocol.
2. Commit the result-producing implementation and independent validator.
3. Verify membership, source hashes, and clean-path reproduction.
4. Generate all five fixed degraded conditions without inspecting event outcomes between conditions.
5. Score all six comparators together with the unchanged model and threshold.
6. Compute event, score-fidelity, dose-response, channel-asymmetry, and paired participant results.
7. Retain the outcome regardless of direction.
8. Run the independent validator twice.
9. Perform one immutable rerun before closing the decision.

## 10. Decision Boundary

This experiment can show how one frozen detector responds to controlled stationary broadband noise on the BOAS validation recordings. It cannot establish robustness to physical electrode faults, natural home artefacts, new participants, clinical cohorts, or alternative devices.

No result authorizes post-test model revision. A later channel-aware or noise-aware training experiment requires a new protocol that specifies its mechanism, training-only augmentation, comparator, and confirmation boundary.
