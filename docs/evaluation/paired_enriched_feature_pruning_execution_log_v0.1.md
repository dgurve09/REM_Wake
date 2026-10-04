# Paired Enriched-Feature and Pruning Execution Log v0.1

## 2026-10-04 single-record feature check

The first feature check reproduced all ten reduced-PSG F0 features exactly for `sub-3`. The wearable pass then stopped because `scipy.signal.coherence` returned nonfinite values in a zero-power spectral segment.

The frozen protocol already required machine epsilon as a denominator guard. The implementation was corrected before the result-producing code commit to calculate magnitude-squared coherence from the cross-spectrum and the two auto-spectra, using the specified epsilon guard and clipping only numerical excursions outside `[0, 1]`.

The repeated `sub-3` check then produced 45 finite enriched features across all 997 epochs, reproduced the reviewed wearable F0 values exactly, and passed every feature-bound check. No label, participant, fold, feature family, pruning rule, model setting, threshold rule, or advancement gate changed.
