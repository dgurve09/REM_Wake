# Block 9 Sleep-EDF Compatibility Decision

**Decision date:** 2026-10-04

**Protocol commit:** `a3c81fe`

**Result-producing code commit:** `28dfd1a`

**Validator correction commit:** `c11f133`

**Dataset:** Sleep-EDF Database Expanded, PhysioNet version 1.0.0

**Dataset DOI:** https://doi.org/10.13026/C2X676

**Model fitting performed:** No

**Decision:** Complete no-go for direct external evaluation of the frozen BOAS reduced-PSG model

## Question

Can Sleep-EDF Expanded support a scientifically interpretable external evaluation of the existing BOAS `F3-M1`/`F4-M1` reduced-PSG method for REM-to-Wake boundary detection?

## Authorized inspection

The audit used the official record list, checksum annex, SC/ST subject spreadsheets, and two pilot pairs selected from the manifest before signal inspection:

- `SC4001E0-PSG.edf` with `SC4001EC-Hypnogram.edf`; and
- `ST7011J0-PSG.edf` with `ST7011JP-Hypnogram.edf`.

All seven locally accessed files matched the official SHA-256 annex. Raw pilot files remain outside Git.

## Manifest and grouping result

| Study | Recordings | Participants | One night | Two nights |
|---|---:|---:|---:|---:|
| Sleep cassette | 153 | 78 | 3 | 75 |
| Sleep telemetry | 44 | 22 | 0 | 22 |
| Total | 197 | 100 | 3 | 97 |

Every official PSG path paired uniquely with one checksum-listed hypnogram. The filename rules preserve participant grouping and night identity, preventing repeated-night leakage.

## Pilot signal and label result

Both pilot PSGs contained `EEG Fpz-Cz` and `EEG Pz-Oz` at 100 Hz in microvolts. Both hypnograms contained explicit R&K Wake and REM labels. The two pilots contained eight contiguous REM-to-Wake transitions on the 30-second grid: three in the SC pilot and five in the ST pilot.

All annotation durations were multiples of 30 seconds and annotations did not overlap. One excluded `Sleep stage ?` interval in the SC pilot began at the PSG end and extended beyond signal support. The frozen protocol required complete annotation support, so the timing criterion failed even though the scored R-to-W examples were usable.

## Decisive channel incompatibility

The frozen BOAS reduced-PSG model expects ten ordered features from a left/right frontal pair, `F3-M1` and `F4-M1`. Sleep-EDF provides anterior-posterior bipolar derivations, `Fpz-Cz` and `Pz-Oz`.

Feeding these two Sleep-EDF channels into the ten input positions would change:

- electrode locations;
- reference/derivation geometry;
- left/right feature meaning; and
- the physiological interpretation of any performance difference.

The input dimensions happen to match, but the scientific variables do not. Renaming the channels or treating them as interchangeable would not constitute external generalization of the frozen method.

## Decision

Two required criteria failed independently:

1. complete annotation support; and
2. preservation of the frozen BOAS feature semantics.

Therefore:

- do not download the complete 8.1 GB dataset for Block 9;
- do not apply or refit the frozen BOAS reduced-PSG model on Sleep-EDF;
- do not describe Sleep-EDF as wearable confirmation; and
- close Block 9 with `complete_no_go`.

This is an informative compatibility result. It prevents an executable but uninterpretable comparison from being reported as external validation.

## Verification

- 7/7 downloaded files matched official SHA-256 values.
- 197/197 PSG paths paired uniquely with hypnograms.
- Participant grouping reconstructed 197 recordings from 100 participants.
- Both pilots independently reproduced the stored EEG-channel and 100 Hz claims.
- Eight pilot R-to-W transitions were reconstructed directly from EDF+ annotations.
- The independent validator passed 8/8 checks twice.
- The initial validator contradiction and a shell-condition error are retained in the execution log.

## Next work

Begin Block 10 using BOAS: temporal localization under explicit 30-second label uncertainty. Predeclare the hard-label comparator, interval-aware target, temporal context, participant-grouped evaluation, F1/false-alarm gate, and stop rule before producing model results.

## References

1. PhysioNet. Sleep-EDF Database Expanded, version 1.0.0. https://physionet.org/content/sleep-edfx/1.0.0/
2. Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering*. 2000;47(9):1185-1194. https://doi.org/10.1109/10.867928
