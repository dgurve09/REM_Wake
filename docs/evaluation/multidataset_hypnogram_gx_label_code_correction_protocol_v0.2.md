# Multi-Dataset Hypnogram gx36m Label-Code Correction Protocol v0.2

**Date frozen:** 2026-10-25
**Precondition:** v0.1 result commit `368c910`
**Result status at freeze:** The parsing defect is identified; no corrected v0.2 label-analysis output has been generated

## 1. Correction question

Do the v0.1 cross-source, scorer, and repeated-night decisions remain unchanged after gx36m labels outside the documented stage vocabulary are represented as unknown rather than indexed as a valid stage?

The gx36m source README defines numeric codes `0..4` as Wake, N1, N2, N3, and REM. The local scorer files also contain raw code `-1`, which is outside that documented vocabulary. The v0.1 loader evaluated `stage_names[-1]`; NumPy therefore returned the last entry, `REM`. This is an implementation defect, not a defensible label assumption.

## 2. Permitted change

Change only gx36m numeric-stage parsing:

- map `0..4` to `W`, `N1`, `N2`, `N3`, and `REM`;
- map every other value, including `-1`, to `UNKNOWN`;
- require at least two known votes for the primary gx36m consensus; and
- exclude unknown epochs from scorer-pair comparisons and transition formation under the existing rules.

All ten other sources, participant identifiers, sleep-period support, strict adjacent REM-to-Wake definition, scorer-pair metrics, repeated-night metrics, full-record sensitivity, thresholds, and hypothesis decisions remain fixed.

## 3. Pre-result audit

Before regenerating results, record and independently verify:

1. the raw gx36m code vocabulary and frequency;
2. the number of affected files, recordings, and scorer epochs;
3. the number of primary consensus epochs changed by safe parsing;
4. old and corrected unknown-epoch and REM-to-Wake counts; and
5. absence of any v0.2 result table.

Commit the audit before running the corrected analysis. Preserve every v0.1 result and validator unchanged.

## 4. Interpretation

Report all v0.2-minus-v0.1 changes, including changes that do not alter the three high-level decisions. Do not infer a physiological meaning for code `-1`; it is treated only as unresolved because the source documentation does not define it as one of the five stages.

This remains a label-only analysis. It cannot validate EEG detection, wearable transfer, clinical diagnosis, sleep-paralysis detection, or narcolepsy detection.

## 5. Reference

Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
