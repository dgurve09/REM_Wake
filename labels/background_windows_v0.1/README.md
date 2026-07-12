# Background Windows v0.1

**Created:** 2026-07-09
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Transition labels:** `labels/transition_labels_v0.1/`
**Model training performed:** No

## 1. Purpose

This artifact defines deterministic non-transition background-window rules for later wearable EEG preprocessing.

The uncertainty addressed here is whether negative/background windows can be selected without overlapping the 30-second uncertainty interval around REM/Wake transition labels.

## 2. Method

- Use the same 240-second extraction window as the transition-label artifact.
- Treat every adjacent PSG epoch boundary as a possible background center.
- Exclude direct REM-to-Wake and Wake-to-REM centers.
- Exclude any candidate whose 240-second window intersects a REM/Wake boundary uncertainty interval.
- The exclusion radius is 135 seconds: 120 seconds half-window plus 15 seconds label uncertainty.
- Exclude edge windows, missing labels, non-30-second epochs, and PSG disconnection windows.
- Keep two background tiers:
  - `strict_same_stage_window`: all epochs in the window have the same PSG stage.
  - `nontarget_window_no_remwake_nearby`: the window may contain other stage changes but no REM/Wake boundary within the exclusion radius.

## 3. Result

| Item | Value |
|---|---:|
| Recordings checked | 128 |
| Potential boundary centers | 119967 |
| Eligible background centers | 115275 |
| Strict same-stage backgrounds | 83688 |
| Non-target backgrounds with no nearby REM/Wake boundary | 31587 |
| Review candidate rows written | 4302 |
| REM/Wake exclusion radius, seconds | 135 |

## 4. Outputs

| File | Purpose |
|---|---|
| `background_window_pool_summary_v0.1.tsv` | Overall pool and exclusion counts |
| `recording_background_summary_v0.1.tsv` | Per-recording eligible background counts |
| `background_stage_pair_summary_v0.1.tsv` | Counts by background tier and center-stage pair |
| `background_review_windows_v0.1.tsv` | Deterministic review-sized candidate table; not a final training set |

## 5. Limitations

- The full eligible pool is summarized but not written as a large table.
- The review table is for preprocessing inspection and split-policy design, not model training.
- Background windows are still derived from 30-second PSG epochs.
- Signal amplitude or artifact quality is handled separately by the quality-flag artifact.

## 6. Decision

Use these rules as background-window specification `v0.1` for the label/preprocessing gate. Do not create final train/validation/test splits until quality flags and background sampling policy are reviewed together.
