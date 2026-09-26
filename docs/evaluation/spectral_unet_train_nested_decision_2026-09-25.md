# Train-Only Spectral U-Net Decision

**Decision date:** 2026-09-25

**Protocol commit:** `03d0e0c`

**Result-producing code commit:** `90477d8`

**Validator path correction:** `62744b1`

**Accepted result commit:** `86b1c48`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Stop spectral U-Net v0.1; do not tune or advance it on the current cohort

## Question

Does two-second wearable EEG spectral structure contain REM-to-Wake boundary information that was discarded by the earlier 30-second bandpower summaries, and can a fixed multiscale 1D U-Net use that information without increasing full-night false alarms?

## Fixed comparison

The experiment retained the same two wearable EEG channels, five frequency bands, eight-epoch 240-second context, 2,743 reviewed candidates, 180 positive events, 64 participant groups, five outer folds, full-night support, event consolidation, and event evaluator used by the prior temporal work.

The representation changed from one ten-value summary per 30-second epoch to fifteen ten-value spectral bins per epoch. Each input therefore had shape `10 x 120`. A fixed 67,697-parameter 1D U-Net used three encoder levels, a full-context bottleneck, three decoder levels, skip connections, and a fixed +/-15-second central aggregation interval.

Each outer fold used four rotating inner participant folds for threshold selection. The unchanged model was then refit on all outer-training participants and applied once to the closed outer participants. The comparator was the existing nested BLSTM-CRF evaluated on the same outer participants and support.

## Primary result

| Pipeline | True positive | False positive | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
| Nested BLSTM-CRF | 32 | 187 | 0.1461 | 0.1778 | 0.1604 | 0.2971 |
| Spectral U-Net | 48 | 463 | 0.0939 | 0.2667 | 0.1389 | 0.7355 |

The U-Net recovered 16 additional primary events but added 276 false positives. Its F1 difference was `-0.0215`, against a required improvement of `+0.05`. Its false-alarm difference was `+0.4384`/hour, while the gate required no increase.

The participant-cluster bootstrap intervals were:

- F1 difference: `-0.0774` to `+0.0444`; and
- false-alarm difference: `+0.2919` to `+0.6192`/hour.

The F1 interval includes zero and excludes the required +0.05 material gain. The entire false-alarm interval is adverse. The advancement hypothesis and directional-support rule both fail.

## Sensitivity result

At +/-45 seconds, the U-Net reached 59/180 detections, F1 0.1718, and 0.7117 false alarms/hour. The matched BLSTM-CRF reached 46/180 detections, F1 0.2317, and 0.2716 false alarms/hour. Wider tolerance therefore did not reverse the conclusion.

For expanded label membership at +/-15 seconds, U-Net F1 was 0.1575 versus 0.1762 for the comparator. At +/-45 seconds, U-Net F1 was 0.1976 versus 0.2467. The negative decision is not specific to the primary +/-15-second endpoint.

## What was learned

Higher temporal resolution increased sensitivity, which is evidence that the two-second representation retained information not expressed by the coarse epoch summaries. It did not produce usable event separation. Inner-selected thresholds varied from 0.91 to 0.9890, and the outer detector emitted 522 alarms for 180 primary reference events.

This failure is informative: the present limitation is not simply that the earlier temporal models lacked multiscale depth. A U-Net-like multiscale representation shifted the operating point toward more detections but substantially worsened precision and alarm burden. The result does not justify concluding that all raw-waveform methods must fail because this experiment retained bandpower and discarded waveform phase and morphology.

## Verification and execution failures

- All 8/8 synthetic representation and model checks passed.
- All 25/25 fits completed the fixed 30 epochs.
- The independent validator passed 15/15 reconstruction and integrity checks twice.
- A direct raw-EDF recomputation matched the cached first epoch.
- A separate full rerun regenerated all 82 feature caches and all 25 fits.
- The rerun reproduced 82/82 feature arrays with maximum absolute difference 0.0, 25/25 model artifacts byte-for-byte, 26/26 score artifacts byte-for-byte, and 18/18 reviewed output files byte-for-byte.

The first validator call failed because of a read-only cache-path error; only the path was corrected. A cache-reuse check later rejected a pandas decimal-text round trip even though every parsed probability was exactly equal. It was replaced by the stronger isolated full rerun. Both failures are retained in dedicated failure notes.

## Next decision

Stop this spectral U-Net at v0.1. Do not widen, deepen, change the loss, alter the threshold grid, or select another pooling interval against these outer results.

The scheduled project step remains Block 9 external-PSG compatibility, which is not wearable confirmation. If train-only wearable model development resumes, the already drafted quality-balanced normalization experiment is the next bounded mechanism because it directly tests the large clean-versus-MAD-flagged dependence. It must remain a separate prespecified experiment. No result here authorizes validation, current-test access, clinical claims, or application deployment.

## References

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention*. 2015:234-241. https://doi.org/10.1007/978-3-319-24574-4_28
2. Perslev M, Jensen M, Darkner S, Jennum PJ, Igel C. U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging. *Advances in Neural Information Processing Systems*. 2019;32. https://proceedings.neurips.cc/paper/2019/hash/57bafb2c2dfeefba931bb03a835b1fa9-Abstract.html
3. Perslev M, Darkner S, Kempfner L, et al. U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*. 2021;4:72. https://doi.org/10.1038/s41746-021-00440-5
