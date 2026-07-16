# Minimal Wearable EEG Preprocessing Validation v0.2

**Work date:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/preprocessing/minimal_wearable_eeg_preprocessing_spec_v0.2.md`
**Partition processed:** Train only
**Model training performed:** No

## 1. Result

- Train `pid` values processed: 64
- Train recordings processed: 82
- Retained transition windows processed: 302
- Retained background review windows processed: 2761
- Recording checks passed: 82 of 82
- Window checks passed: 3063 of 3063
- Synthetic frequency checks passed: 3 of 3
- Validation/test recordings read: 0

## 2. Train-Only Robust Scaling

| Channel | 1 Hz samples | Median, uV | MAD, uV | Robust scale, uV |
|---|---:|---:|---:|---:|
| HB_1 | 2,298,070 | 0.079269 | 8.643662 | 12.815094 |
| HB_2 | 2,298,070 | 0.114734 | 9.404625 | 13.943297 |

## 3. Interpretation

The continuous-recording filter, 256-to-128 Hz resampling, window mapping, and train-only robust scaling are mechanically reproducible for the retained train candidates if all checks above pass. This validation establishes preprocessing integrity only; it does not establish that the choices improve REM-to-Wake detection.

Targeted-review windows were processed to test pipeline stability but remain separately flagged. Their use in the primary analysis is not decided here.

## 4. Decision

Pass the mechanical preprocessing validation for the declared input artifact.

## 5. Outputs

| File | Purpose |
|---|---|
| `filter_response_v0.2.tsv` | Measured single-pass and forward-backward filter response |
| `synthetic_frequency_checks_v0.2.tsv` | Predeclared retain/attenuate checks |
| `train_robust_scaler_v0.2.tsv` | Scaling fitted from train recordings only |
| `train_recording_preprocessing_checks_v0.2.tsv` | Continuous-recording integrity checks |
| `train_window_preprocessing_checks_v0.2.tsv` | Compact raw, filtered, and normalized window summaries |
| `train_window_preprocessing_summary_v0.2.tsv` | Counts by window and quality class |

## 6. Decision Boundary

Validation and test signals remain untouched. Model training remains blocked until targeted-review treatment is frozen and the label/preprocessing gate decision is recorded.
