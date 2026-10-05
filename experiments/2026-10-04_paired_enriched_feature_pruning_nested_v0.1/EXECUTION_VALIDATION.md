# Execution and Validation Record

**Primary execution:** 2026-10-04

**Initial result code commit:** `086c43c`

**Result-finalization commit:** `e71a042`

**Authorized data:** 82 train recordings from 64 participant groups

**Validation accessed:** No

**Current test accessed:** No

## Primary run

- Generated 45 enriched features per epoch for both reduced PSG and wearable EEG.
- Reproduced all existing F0 features with maximum absolute difference `9.54e-7`.
- Constructed 360 enriched and 80 F0 context values per candidate.
- Completed 200 inner fits and 30 outer-final fits.
- Produced one participant-held-out outer score for every supported full-night boundary.
- Selected regularization and event thresholds from inner out-of-fold scores only.
- Evaluated 180 primary train events across 64 participant groups.

## Preserved failures

The first wearable feature check stopped when library coherence returned nonfinite values for a zero-power spectral segment. The frozen protocol already required a machine-epsilon denominator guard. The implementation was corrected before the result code commit to calculate magnitude-squared coherence from the cross-spectrum and auto-spectra with that guard.

The completed nested run then stopped during finalization because 24 elastic-net inner fits at `C=1.0` reached the 3,000-iteration limit. All outer-final fits converged. Model settings were not changed. Commit `e71a042` changed only finalization so the failed convergence check and completed outputs could be retained; it read the exact existing model and score files rather than refitting.

## Independent validation

The read-only validator was run twice against the same result package. Both passes completed 10/10 checks:

- 624 external artifact hashes;
- enriched-feature reconstruction and F0 parity;
- feature schema and candidate accounting;
- 230-fit accounting, including 24 retained inner-fit convergence failures;
- 30 nested selections;
- event metrics and participant rows;
- 12 paired bootstrap interval rows;
- four advancement decisions; and
- 1,600 feature-stability rows.

The second validation pass reproduced the first report byte-for-byte.
