# Full-Dataset Alignment Drift Audit Plan v0.1

**Created:** 2026-08-23
**Status:** Retrospective validation audit
**Input:** Saved July 4 pulse-alignment window table
**Model or label changes authorized:** No

## 1. Reason for the Audit

The full-dataset alignment experiment sampled pulse lag at 10%, 30%, 50%, 70%, and 90% of each recording and reported how many individual lags were within +/-2 seconds. It did not explicitly quantify whether lag changed systematically across the night. The original claim about possible offset or drift was therefore broader than the reported analysis.

## 2. Fixed Analysis

Use only rows where both pulse channels were available and the absolute correlation met the original usability threshold of 0.20. Analyze a recording when at least three of its five windows are usable.

For each eligible recording:

1. regress signed pulse lag in seconds on window-center time in hours;
2. report slope in seconds per hour;
3. multiply the slope by the observed usable time span to obtain projected lag change;
4. report the observed first-to-last lag change and lag range; and
5. flag an absolute projected change greater than 2 seconds for review.

The 2-second review threshold reuses the original near-zero-lag tolerance. It is a screening threshold, not a calibrated device-failure criterion.

## 3. Interpretation Boundary

The pulse cross-correlation peak is an alignment proxy, not a direct clock measurement. A slope can reflect waveform or sensor differences, artifacts, a weak peak, or actual timing change. Results will therefore be used to identify whether drift-like behavior is widespread and to mark recordings for review, not to assert clock drift or to revise frozen labels automatically.
