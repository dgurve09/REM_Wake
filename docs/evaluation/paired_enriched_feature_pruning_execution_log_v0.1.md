# Paired Enriched-Feature and Pruning Execution Log v0.1

## 2026-10-04 single-record feature check

The first feature check reproduced all ten reduced-PSG F0 features exactly for `sub-3`. The wearable pass then stopped because `scipy.signal.coherence` returned nonfinite values in a zero-power spectral segment.

The frozen protocol already required machine epsilon as a denominator guard. The implementation was corrected before the result-producing code commit to calculate magnitude-squared coherence from the cross-spectrum and the two auto-spectra, using the specified epsilon guard and clipping only numerical excursions outside `[0, 1]`.

The repeated `sub-3` check then produced 45 finite enriched features across all 997 epochs, reproduced the reviewed wearable F0 values exactly, and passed every feature-bound check. No label, participant, fold, feature family, pruning rule, model setting, threshold rule, or advancement gate changed.

## 2026-10-04 full nested run

The full run completed 230 predeclared fits and wrote the corresponding models and full-night scores outside Git. Finalization initially stopped at the convergence check: 24 elastic-net inner-fold fits at `C=1.0` reached the predeclared 3,000-iteration limit. All outer-final fits converged.

The iteration limit, solver, tolerance, candidate grid, and selection rules were retained without post-result adjustment. Finalization was corrected to preserve the failed convergence check while allowing the completed results and immutable artifact hashes to be written. The correction reuses the exact saved model and score artifacts rather than refitting them.
