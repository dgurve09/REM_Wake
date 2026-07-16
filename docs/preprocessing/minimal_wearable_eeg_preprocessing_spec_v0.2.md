# Minimal Wearable EEG Preprocessing Specification v0.2

**Created:** 2026-07-15
**Inherits pipeline from:** `minimal_wearable_eeg_preprocessing_spec_v0.1.md`
**Quality artifact:** `labels/signal_quality_flags_v0.3/`
**Split:** `splits/grouped_pid_split_v0.1/`
**Model training performed:** No

## 1. Trigger

Preprocessing validation v0.1 processed 3,065 retained train windows but failed the exact-length check for `T0120` and `T0196`. Both had finite data and valid normalization summaries, but their 240-second extraction intervals extended beyond available signal and returned only 54,016 and 44,032 input samples respectively.

The failure identified a missing absolute-coverage rule in quality v0.2. Signal-quality v0.3 now requires exactly 61,440 input samples and excludes incomplete windows rather than padding them.

## 2. Unchanged Pipeline

- continuous train-recording filtering from 0.3 to 35 Hz;
- fourth-order Butterworth second-order sections with forward-backward filtering;
- no rereferencing and no separate notch filter;
- polyphase 256-to-128 Hz resampling;
- train-only robust scaler fitted from a deterministic 1 Hz sample;
- no filtered arrays stored in Git;
- no validation/test signals read;
- no model training.

## 3. Revised Input Expectation

- 302 retained train transition windows after coverage-aware exclusions;
- 2,761 retained train background review windows;
- 3,063 total windows;
- every input window exactly 61,440 samples per channel;
- every output window exactly 30,720 samples per channel.

## 4. Success Rule

Preprocessing v0.2 passes only if all 82 train recordings, all 3,063 retained windows, and all three synthetic frequency checks pass. The previously failed v0.1 artifact remains preserved as evidence of the insufficient coverage rule.
