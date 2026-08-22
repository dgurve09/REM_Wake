# Direct Endpoint Factorization Participant Analysis Plan v0.1

**Created:** 2026-08-22  
**Status:** Post-result exploratory analysis  
**Known aggregate validation result:** DE-D met the two-part DE-B comparison rule  
**Test access authorized:** No

## 1. Question

Is the aggregate validation improvement from factorized DE-D broadly supported across the 16 validation participants, or is it explained by a small subset?

## 2. Frozen Inputs

- DE-B validation participant event counts at primary membership and +/-15 seconds;
- DE-D validation participant event counts at the same membership and tolerance;
- no raw signal, feature array, fitted model, continuous score, train row, or test artifact.

## 3. Analyses

1. Merge participants one-to-one by `pid` and verify all 16 are present.
2. Record per-participant differences in true positives, false positives, false negatives, F1, and false alarms per hour.
3. Count participants with increased, unchanged, or decreased true positives and false positives.
4. Use a paired participant bootstrap with 2,000 resamples and seed `20260822`. Each resample draws the same `pid` values for both methods, recomputes aggregate metrics separately, and reports DE-D minus DE-B event F1 and false alarms per hour.

## 4. Interpretation

Intervals excluding zero in the favorable direction would support consistency within this small validation cohort. Intervals crossing zero would make the aggregate result participant-unstable. Neither outcome is independent confirmation because DE-D was designed after the prior DE-B test result and threshold selection used this validation set.
