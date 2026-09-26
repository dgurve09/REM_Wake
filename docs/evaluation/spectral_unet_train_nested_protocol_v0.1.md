# Train-Only Spectral U-Net Protocol v0.1

**Created:** 2026-09-25

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological uncertainty

The existing nested temporal comparison found primary +/-15-second event F1 of 0.1645 for the selected pipeline and 0.1604 for the matched BLSTM-CRF. Changing the epoch-level temporal encoder did not produce the required 0.05 improvement. A later error analysis found that widening the tolerance and selecting optimistic post-hoc thresholds still left F1 below 0.25. These results stop further architecture swapping on the same ten 30-second bandpower summaries.

The remaining bounded uncertainty is whether those summaries discard short-timescale spectral changes around a REM-to-Wake boundary that a multiscale convolutional representation can use. The experiment tests that representation question. It does not test sleep-stage classification and does not reopen the prior epoch-feature architecture search.

## 2. Known method and project-specific limitation

U-Net combines a contracting path, an expanding path, and skip connections for multiscale localization [1]. U-Time adapted that structure to physiological time-series segmentation and sleep staging [2]. U-Sleep subsequently demonstrated a fully convolutional high-frequency sleep representation across heterogeneous datasets [3].

Those papers establish the architecture family, not the present endpoint. They do not determine whether simultaneously recorded two-channel wearable EEG contains enough short-timescale information to detect an event-specific REM-to-Wake boundary derived from the PSG hypnogram. The labels here also have a nominal 30-second scoring uncertainty, so sample-exact localization is not asserted.

## 3. Fixed input representation

Use only `HB_1` and `HB_2` from the train recordings.

1. Apply the frozen Block 7 preprocessing: 0.3-35 Hz fourth-order Butterworth band-pass, zero-phase filtering, resampling from 256 to 128 Hz, and the frozen train-derived robust channel scaling.
2. Divide each retained 30-second scored epoch into fifteen non-overlapping 2-second bins.
3. In each bin, estimate Welch power with a 256-sample Hann window and no overlap.
4. Average power within delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, and beta 16-30 Hz for each channel; apply `log10(max(power, machine epsilon))`.
5. Concatenate the same eight contiguous epochs used previously, producing one `10 x 120` input for each supported candidate time.

This representation changes temporal resolution while holding channels, frequency bands, 240-second context, labels, support, and endpoint fixed. It does not retain phase or waveform morphology, so a negative result cannot rule out all raw-waveform methods.

## 4. Fixed 1D U-Net

The model receives `10 x 120` feature sequences.

- Encoder widths: 8, 16, and 32 channels.
- Bottleneck width: 64 channels.
- Each block: two length-five 1D convolutions with same padding, group normalization, and GELU.
- Three length-two max-pooling operations.
- Decoder: three length-two transposed convolutions, concatenated encoder skip connections, and the same two-convolution block at each scale.
- Output: one length-one convolution producing 120 logits.
- Candidate logit: mean of the 16 output logits whose 2-second bin centers span -15 to +15 seconds around the nominal boundary.
- Candidate probability: sigmoid of the candidate logit.

The three pooling levels give the bottleneck a full-context theoretical receptive field. No width, depth, kernel, pooling, normalization, aggregation interval, or activation is changed after results are observed.

## 5. Fixed optimization

- Binary cross-entropy with positive candidate weight equal to `negative_count / positive_count` in the fitting subset.
- Adam, learning rate 0.001, weight decay 0.0001.
- Batch size 128, 30 fixed epochs, gradient-norm clipping at 5.
- No early stopping, scheduler, augmentation, pretrained weights, or hyperparameter search.
- Per-feature standardization fitted only on candidate sequences from the current fitting participants.
- Deterministic CPU algorithms, one PyTorch thread, and seed `20260925 + 100 * outer_fold + 10 * inner_fold`; inner fold 9 is reserved for the outer-final fit.

## 6. Nested participant-grouped evaluation

Reuse the five frozen train `pid` folds. For each outer fold:

1. keep the outer participants closed;
2. use each of the other four fold identifiers once as an inner held-out fold;
3. fit the fixed U-Net on the remaining three folds;
4. score every supported full-night context in the inner held-out recordings;
5. concatenate the four inner out-of-fold score sets;
6. select one threshold using only those inner scores;
7. refit the unchanged U-Net on all four outer-training folds; and
8. apply the frozen threshold once to the closed outer fold.

After all outer folds, concatenate the one-time outer predictions. Outer scores do not alter preprocessing, architecture, optimization, threshold grid, or decision rules.

## 7. Threshold and event evaluation

Use the exact threshold grid from the nested temporal comparison: 0.01 through 0.95 in increments of 0.05 plus `sigmoid(x)` for logits 3.0 through 14.0 in increments of 0.25. Select maximum primary +/-15-second event F1, then minimum false alarms/hour, maximum recall, and maximum threshold.

Contiguous above-threshold 30-second candidates form one alarm; the maximum score represents the run and exact ties use the earliest time. The primary endpoint is event F1 for primary labels at +/-15 seconds. Fixed sensitivity analyses are primary +/-45 seconds, expanded +/-15 seconds, and expanded +/-45 seconds.

## 8. Hypothesis and stop rule

**H-UN1:** The nested spectral U-Net will improve primary +/-15-second event F1 by at least 0.05 relative to the existing nested BLSTM-CRF without increasing false alarms per supported hour.

Use a paired `pid`-cluster bootstrap with 2,000 resamples and seed 20260925. Report point differences and 95% percentile intervals for F1 and false alarms/hour.

The advancement gate passes only when the point F1 difference is at least +0.05 and the point false-alarm difference is at most zero. Directional participant support additionally requires the F1 interval entirely above zero and the false-alarm interval entirely at or below zero. If the point gate fails, stop this spectral U-Net configuration at v0.1. No architecture repair or threshold revision is permitted against the outer results.

## 9. Required controls

1. Confirm exact train membership of 82 recordings and 64 participants.
2. Confirm no validation or current-test file, participant, feature, score, or result is accessed.
3. Reuse the exact five frozen outer folds and verify participant disjointness at every fit boundary.
4. Verify raw EDF sampling rate, channel names, finite samples, epoch alignment, and contiguous eight-epoch support.
5. Verify generated 2-second features against an independently calculated synthetic sinusoid test and a direct recomputation sample.
6. Fit preprocessing standardizers only within each fitting subset.
7. Verify U-Net input, skip, output, probability, gradient, and receptive-field behavior with synthetic tests.
8. Complete exactly 30 epochs with finite losses, gradients, parameters, and probabilities.
9. Select thresholds only from complete inner out-of-fold full-night scores.
10. Produce one outer score per supported candidate and one outer assignment per participant.
11. Verify support is identical to the nested BLSTM-CRF comparison.
12. Record external artifact hashes and sizes; keep raw signals, generated tensors, scores, and model states outside Git.
13. Run an independent read-only validator twice and confirm immutable rerun agreement.

## 10. Interpretation boundary

This is a bounded, train-only representation experiment motivated by results already observed on the same cohort. Nested outer evaluation limits direct threshold fitting but does not make the result independent confirmation. It cannot establish clinical performance, prospective alarm utility, narcolepsy or sleep-paralysis detection, or deployment readiness. A positive result would only justify freezing this representation for a separately authorized confirmation cohort.

## References

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention*. 2015:234-241. https://doi.org/10.1007/978-3-319-24574-4_28
2. Perslev M, Jensen M, Darkner S, Jennum PJ, Igel C. U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging. *Advances in Neural Information Processing Systems*. 2019;32. https://proceedings.neurips.cc/paper/2019/hash/57bafb2c2dfeefba931bb03a835b1fa9-Abstract.html
3. Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C. U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*. 2021;4:72. https://doi.org/10.1038/s41746-021-00440-5
