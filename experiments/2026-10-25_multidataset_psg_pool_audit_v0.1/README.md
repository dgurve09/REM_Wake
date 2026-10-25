# Multi-Dataset PSG Pool Audit v0.1

## Purpose

This audit determines which locally available datasets can contribute raw EEG to a common-channel REM-to-Wake experiment. It is an availability and compatibility audit, not a model result. No model was fitted.

## Finding

The ten-dataset harmonized collection does not contain ten local raw-EEG datasets. Most stored EDF or MAT files are hypnogram or label containers. Three sources currently provide eligible local overnight PSG with a common right-sided EEG montage:

| Dataset | Recordings | Participant groups | REM-to-Wake events | Common channels | Source sampling rate |
|---|---:|---:|---:|---|---:|
| BOAS | 128 | 100 | 365 | F4, C4, O2 referenced to M1 | 256 Hz |
| gx36m | 58 | 20 | 124 | F4, C4, O2 referenced to A1 | 500 Hz |
| ISRUC group 3 | 10 | 10 | 32 | F4, C4, O2 referenced to A1 | 200 Hz |
| **Total** | **196** | **130** | **521** | | |

The common pool contains 190,497 scored 30-second epochs. The A1/M1 references are anatomically close but are not assumed identical; dataset-held-out evaluation must retain this acquisition-domain difference.

## Label rules

- BOAS uses the publisher's human-consensus hypnogram.
- gx36m uses an epoch-wise majority across three expert scorers.
- ISRUC group 3 retains a stage only when its two expert scorers agree; disagreements become `UNKNOWN`.

Mean per-record disagreement was 0.2865 for gx36m and 0.1187 for ISRUC group 3. BOAS does not distribute the individual human scorings needed to calculate the same quantity locally. These differences prevent treating all derived boundaries as equally certain.

## Retained audit failure

The first automated audit failed its label-support check because `SLP010_Day1` contains 166 seconds of trailing EEG after the final scored epoch. Labels did not extend beyond the signal. The corrected rule requires every scored epoch to lie within EEG support and records trailing unscored duration separately; the unscored tail will be excluded from candidates.

## Dataset roles

All eleven inventoried sources remain useful, but not for the same task:

- BOAS, gx36m, and ISRUC group 3 can enter the common-channel PSG signal experiment.
- DOD, the remaining ISRUC scorings, and gx36m scorer variants support scorer-uncertainty analysis.
- Challenge 2018, Sleep-EDF, HMC, CAP, iSLEEPS, Sleep-Accel, and DREAMS can support transition-distribution analyses from their local hypnograms; they cannot add raw EEG training samples from the files currently stored locally.
- The single `boas_eeg` pilot EDF is a duplicate BOAS recording and is not an additional sample.

## Verification

- Eligibility audit: 13/13 checks passed.
- Independent validation: 12/12 checks passed twice.
- External gx36m and ISRUC source containers: 30/30 SHA-256 values verified.
- BOAS raw EDF identity is covered by the prior full-dataset integrity audit.

## Sources

1. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
2. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
3. Bitbrain Open Access Sleep dataset, OpenNeuro `ds005555`, version `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
