# Minimal Wearable EEG Preprocessing Specification v0.1

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Split:** `splits/grouped_pid_split_v0.1/`
**Quality artifact:** `labels/signal_quality_flags_v0.2/`
**Model training performed:** No

## 1. Purpose

This specification defines the smallest reproducible preprocessing pipeline needed to prepare the two BOAS headband EEG channels for later baseline work. The immediate test is limited to the frozen train partition. Validation and test signals must not be inspected during preprocessing development.

## 2. Inputs

- Headband channels `HB_1` and `HB_2` as recorded.
- Original sampling frequency: 256 Hz.
- Reviewed 240-second transition and background windows.
- Train `pid` assignments only.
- All quality outcomes except `exclude_critical` are processed for pipeline stability testing. Their eventual primary-versus-sensitivity use remains separate.

## 3. Pipeline

1. Read both channels from each complete train recording and convert MNE output from volts to microvolts.
2. Require finite continuous data. Do not silently interpolate missing values.
3. Apply a fourth-order Butterworth band-pass from 0.3 to 35 Hz using second-order sections and forward-backward filtering.
4. Do not add a 50/60 Hz notch because the 35 Hz low-pass already excludes line frequency from the retained band.
5. Do not rereference `HB_1` and `HB_2`. With only two recorded channels, average rereferencing would make them algebraically dependent and remove a degree of freedom.
6. Resample continuously from 256 to 128 Hz using polyphase anti-alias filtering.
7. Extract windows after continuous filtering and resampling, preserving the existing 240-second boundaries.
8. Fit a separate robust center and scale for each channel using only filtered train recordings. Use a deterministic 1 Hz sample from each continuous 128 Hz train signal. Scale is `1.4826 * MAD`.
9. Do not store filtered signal arrays in Git. Store only scaler parameters, filter/synthetic checks, and compact window/recording summaries.

## 4. Validation Checks

### Synthetic frequency test

Using 120-second unit-amplitude sinusoids:

- 10 Hz RMS gain after filtering/resampling must be at least 0.90;
- 0.05 Hz RMS gain must be at most 0.10;
- 50 Hz RMS gain must be at most 0.10.

### Recording checks

- both required channels present;
- input sampling frequency exactly 256 Hz;
- all continuous samples finite;
- resampled duration differs by no more than one output sample.

### Window checks

- input length exactly 61,440 samples per channel;
- output length exactly 30,720 samples per channel;
- raw and filtered samples finite;
- normalized summary values finite;
- window `pid` belongs to the train partition.

## 5. Decisions Not Made Here

- No model input epoch length is selected.
- No data augmentation is applied.
- No targeted-review window is visually adjudicated.
- No validation or test signal is processed.
- No model is trained and no performance is estimated.
- The zero-phase filter is an offline preprocessing choice. A future streaming demonstration must test a causal implementation separately and must not assume identical behavior.

## 6. Rationale and References

The filter is intentionally simple and its response is measured rather than assumed. Forward-backward filtering avoids phase displacement for offline analysis but is noncausal. Filter design and reporting follow the practical concerns described by Widmann, Schroger, and Maess.

1. Widmann A, Schroger E, Maess B. Digital filter design for electrophysiological data - a practical approach. *Journal of Neuroscience Methods*. 2015;250:34-46. https://doi.org/10.1016/j.jneumeth.2014.08.002
2. Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*. 2013;7:267. https://doi.org/10.3389/fnins.2013.00267
