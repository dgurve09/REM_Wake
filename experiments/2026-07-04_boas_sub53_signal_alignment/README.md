# BOAS sub-53 Signal Alignment Pilot

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 alignment validation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Recording:** `sub-53`
**Model training performed:** No
**Scope:** One already-downloaded paired PSG/headband recording

## 1. Purpose

This pilot tests whether PSG-derived transition labels can be mapped onto the simultaneously recorded headband signal at the sample-window level for `sub-53`.

The uncertainty is whether sidecar/header agreement is enough for label mapping, or whether signal-level checks reveal offset, drift, or window extraction problems that would make PSG-derived REM/Wake boundaries unreliable for wearable EEG analysis.

## 2. Method

The validation used three checks:

1. EDF timeline check: compare PSG and headband start time, sampling rate, sample count, and duration.
2. Transition-window sample check: extract 240-second windows around each `sub-53` REM/Wake candidate and verify identical PSG/headband sample indices.
3. Signal proxy checks:
   - compare `HB_PULSE` with `PSG_PULSE` in five 300-second windows across the night as a physiological drift proxy;
   - compare headband EEG and PSG EEG 1-second RMS envelopes around transition windows as an exploratory artifact/physiology proxy.

## 3. Main Results

| Check | Result |
|---|---:|
| EDF timeline fields matched | True |
| Transition windows checked | 6 |
| Transition windows with matching PSG/headband sample indices | 6 |
| Pulse windows checked | 5 |
| Pulse windows usable for lag estimation | 4 |
| Usable pulse windows with lag within +/-2 seconds | 4 |
| Median usable pulse lag, seconds | 0.938 |
| Usable pulse lag range, seconds | -1.219 to 1.938 |
| Median usable pulse absolute correlation | 0.358 |
| EEG-envelope comparisons checked | 24 |
| EEG-envelope comparisons usable for lag estimation | 24 |
| Usable EEG-envelope comparisons with lag within +/-2 seconds | 18 |
| `HB_1` versus `PSG_F3` windows with lag within +/-1 second | 6 |
| `HB_1` versus `PSG_F3` median absolute correlation | 0.852 |

## 4. Interpretation

For `sub-53`, the EDF timeline and transition-window sample checks passed. The six REM/Wake candidate windows use matching PSG and headband sample indices, and the event-table boundary samples match the calculated sample positions.

The pulse comparison is a supporting physiological drift proxy. Four of five pulse windows were usable, and all usable pulse lags were within +/-2 seconds. This does not prove perfect synchronization because the PSG and headband pulse sensors may have different filtering, placement, and waveform morphology.

The EEG-envelope comparison is exploratory. The strongest cross-device proxy was `HB_1` versus `PSG_F3`, where all six transition windows peaked within +/-1 second and the median absolute correlation was 0.852. Other cross-montage pairs were less consistent, so low correlation or shifted peaks in those pairs should not be treated as proof of misalignment.

## 5. Limitations

- Only `sub-53` EDF files are currently local.
- This does not validate synchronization across all 128 BOAS recordings.
- This does not prove exact physiological REM-to-Wake timing inside a 30-second hypnogram epoch.
- Pulse and EEG-envelope lag estimates are proxies, not ground truth event markers.
- A full-dataset validation requires acquiring additional EDFs or selecting a representative EDF subset.

## 6. Decision

Proceed with `sub-53` as a sample-aligned pilot recording for transition-label table development and minimal preprocessing tests.

Do not generalize this signal-level alignment result to all BOAS recordings yet. The next alignment step should validate a representative subset of EDF pairs before full model work.
