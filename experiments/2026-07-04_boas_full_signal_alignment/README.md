# BOAS Full Signal Alignment Validation

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 alignment validation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**EDF scope:** 128 paired PSG/headband recordings
**Model training performed:** No

## 1. Purpose

This validation tests whether PSG-derived REM/Wake transition labels can be mapped onto BOAS headband EEG across the downloaded EDF dataset.

The technical uncertainty is whether header agreement generalizes to sample-level transition-window alignment across recordings, and whether signal-level proxies reveal evidence of timing offset or drift that would invalidate PSG-to-headband label mapping.

## 2. Method

The validation used four checks:

1. EDF timeline agreement for each PSG/headband pair.
2. Transition-window sample-index agreement for all E0 REM/Wake candidates.
3. `HB_PULSE` versus `PSG_PULSE` lag estimates in five 300-second windows across each recording where both pulse channels are present.
4. `HB_1` versus `PSG_F3` 1-second RMS-envelope lag estimates around each REM/Wake candidate as an exploratory shared-activity proxy.

Pulse and EEG-envelope lag estimates are supporting proxies. The exact sample-index mapping remains the primary alignment check.

## 3. Main Results

| Check | Result |
|---|---:|
| PSG/headband EDF pairs checked | 128 |
| EDF pairs with matching timeline fields | 128 |
| REM/Wake transition windows checked | 476 |
| Transition windows with matching sample indices | 476 |
| Subjects with pulse channels available | 101 |
| Pulse windows checked | 505 |
| Pulse windows usable for lag estimation | 383 |
| Usable pulse windows with lag within +/-2 seconds | 377 |
| Subjects where all usable pulse windows were within +/-2 seconds | 90 |
| EEG-envelope transition windows checked | 476 |
| EEG-envelope windows usable for lag estimation | 465 |
| Usable EEG-envelope windows with lag within +/-2 seconds | 349 |
| Subjects where all usable EEG-envelope windows were within +/-2 seconds | 49 |

## 4. Interpretation

The full EDF timeline and transition-window sample-index checks passed for the downloaded dataset. This supports using PSG `stage_hum` transition labels on the headband sample timeline under the current deterministic label rule.

The pulse and EEG-envelope proxies provide additional evidence but are not treated as ground truth. Pulse channels are not available in every recording, and pulse waveform differences can shift cross-correlation peaks. EEG-envelope correlations depend on montage, sensor location, artifacts, and sleep physiology, so inconsistent envelope lag does not automatically prove synchronization failure.

## 5. Limitations

- The labels still have 30-second hypnogram uncertainty.
- Signal proxies test broad timing consistency, not exact physiological transition onset.
- Some recordings lack pulse channels.
- Future preprocessing should still retain recording-level quality flags.

## 6. Decision

Proceed to versioned deterministic transition-label table generation and minimal preprocessing.

Model training remains blocked until the label/preprocessing gate reviews the derived label table, split policy, and signal-quality flags.
