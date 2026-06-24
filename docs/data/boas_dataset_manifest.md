# BOAS Dataset Manifest

**Audit date:** 2026-06-22  
**Audit scope:** Metadata only  
**Raw signal data acquired:** No  
**Status:** Suitable for a limited paired-recording pilot, subject to the checks listed below

## 1. Dataset Identity

| Field | Verified value |
|---|---|
| Dataset | Bitbrain Open Access Sleep (BOAS) dataset |
| OpenNeuro accession | `ds005555` |
| Latest snapshot reported by the OpenNeuro API | `1.1.1` |
| Snapshot identifier | `ds005555:1.1.1` |
| Snapshot creation date | 2025-05-22 |
| DOI | https://doi.org/10.18112/openneuro.ds005555.v1.1.1 |
| Official dataset page | https://openneuro.org/datasets/ds005555/versions/1.1.1 |
| License | CC0 |
| BIDS version | 1.8.0 |
| Recording folders | 128 (`sub-1` through `sub-128`) |
| Participant grouping field | `pid` in `participants.tsv` |

The previously recorded version `1.2.1` could not be verified: its DOI returned 404, and the OpenNeuro API reported `1.1.1` as the latest available snapshot on the audit date. Project documents have been corrected to use the verified snapshot.

## 2. Source Verification

The audit used two official representations of the same snapshot:

1. the OpenNeuro GraphQL API, queried on 2026-06-22; and
2. the OpenNeuroDatasets GitHub mirror at tag `1.1.1`, commit `0225bb258566172fa97a4f75dc2c2689243df2a2`.

The OpenNeuro API reported three snapshots: `1.0.0`, `1.1.0`, and `1.1.1`. The metadata-only mirror was inspected without retrieving annexed EDF contents.

## 3. File Inventory and Storage

| Item | Count or size |
|---|---:|
| Files in recursive snapshot listing | 1,418 |
| Annexed EDF files | 256 |
| PSG EDF files | 128 |
| Headband EDF files | 128 |
| OpenNeuro snapshot size | 35,921,406,226 bytes (35.92 GB; 33.45 GiB) |
| EDF data size | 35,913,652,480 bytes |
| PSG EDF size | 22,473,353,472 bytes |
| Headband EDF size | 13,440,299,008 bytes |
| Combined EDF size per recording | 106,977,024 to 395,656,448 bytes |

Raw data are stored outside Git. The limited `sub-53` pilot acquisition was completed on 2026-06-24 using `scripts/download_boas_sub53_pilot.py`, with the default storage location set to `<repo-parent>/REM_W_data/boas_ds005555_v1.1.1`. The full dataset has not been downloaded.

## 4. Recording Structure

Each recording folder contains a paired PSG and headband dataset with the pattern:

```text
sub-XX/
|-- sub-XX_scans.tsv
`-- eeg/
    |-- sub-XX_task-Sleep_acq-headband_channels.tsv
    |-- sub-XX_task-Sleep_acq-headband_eeg.edf
    |-- sub-XX_task-Sleep_acq-headband_eeg.json
    |-- sub-XX_task-Sleep_acq-headband_events.json
    |-- sub-XX_task-Sleep_acq-headband_events.tsv
    |-- sub-XX_task-Sleep_acq-psg_channels.tsv
    |-- sub-XX_task-Sleep_acq-psg_eeg.edf
    |-- sub-XX_task-Sleep_acq-psg_eeg.json
    |-- sub-XX_task-Sleep_acq-psg_events.json
    `-- sub-XX_task-Sleep_acq-psg_events.tsv
```

All 128 paired sidecars report 256 Hz sampling. Within each recording, the PSG and headband sidecars report equal recording durations. Durations range from 12,290 to 33,598 seconds (3.41 to 9.33 hours). Signal-level clock alignment remains unverified until EDF data are inspected.

## 5. Channel Coverage

| Channel group | Recordings with group |
|---|---:|
| PSG EEG | 128 |
| PSG EOG | 128 |
| PSG EMG | 128 |
| PSG breathing belts | 62 |
| PSG thermistor | 62 |
| PSG nasal cannula | 19 |
| PSG PPG | 104 |
| Headband EEG | 128 |
| Headband IMU | 94 |
| Headband pulse | 112 |

The primary wearable channels are `HB_1` and `HB_2`, approximately located at AF7 and AF8. When present, the headband also contains three accelerometer channels, three gyroscope channels, and `HB_PULSE` in the same EDF file.

## 6. Labels

- Every PSG event file contains `stage_hum`.
- Human stages use 30-second rows and the codes `0` Wake, `1` N1, `2` N2, `3` N3, `4` REM, and `8` PSG disconnection.
- Three scorers annotated the PSG data, with majority consensus and a fourth scorer resolving three-way disagreement, according to the official README.
- `stage_ai` is also present but will not be used as reference ground truth.
- Human labels reside in the PSG event files and must be aligned to the simultaneous headband signals.

The complete transition inventory is intentionally deferred to the E0 feasibility audit. This metadata audit only checked label presence, coding, duration, and one pilot candidate.

## 7. Participant-Identifier Discrepancy

The official README states that the 128 recordings represent 108 unique individuals. However, `participants.tsv` in snapshot `1.1.1` contains 100 unique `pid` values: 80 occur once, 12 occur twice, and 8 occur three times.

This discrepancy is unresolved. Until it is clarified, all grouping will use the explicit `pid` values in the versioned participant table, and publications will report both the README statement and the observed table count. No train, validation, or test split will be created before this issue is documented in the split specification.

## 8. Recommended Pilot Recording

**Recording:** `sub-53`  
**Unique participant field:** `pid = 88`  
**Other recordings with the same `pid`:** `sub-50`, `sub-54`  
**Duration:** 20,008 seconds (5.56 hours)  
**Sampling frequency:** 256 Hz for PSG and headband

Reasons for selection:

- it contains paired PSG and headband EDF files;
- it contains both headband EEG channels, all six IMU channels, and headband pulse;
- it contains PSG EEG, EOG, EMG, respiratory, and PPG channels;
- its unfiltered human-stage sequence contains four direct REM-to-Wake adjacencies and two Wake-to-REM adjacencies, which is sufficient for pipeline inspection but is not a feasibility result;
- it is the smallest complete-multimodal recording found with at least one direct REM-to-Wake adjacency.

Pilot EDF sizes:

- headband: 92,199,424 bytes;
- PSG: 143,421,184 bytes;
- combined: 235,620,608 bytes (approximately 224.7 MiB).

The pilot should retrieve the two EDF files, their channel and JSON sidecars, both event files, `sub-53_scans.tsv`, and the root participant and channel metadata.

## 9. Checks Required During the Pilot

1. Confirm that the existing Python environment can read both EDF files before installing a package.
2. Verify channel names, units, sample counts, and sampling frequencies from the actual EDF headers.
3. Verify PSG and headband start times and sample-level duration agreement.
4. Confirm how PSG `stage_hum` onsets map to the headband timeline.
5. Inspect disconnections, missing values, clipping, and channel dropout around the selected transitions.
6. Record exact downloaded file paths, byte sizes, and SHA-256 or official annex keys.
7. Keep `sub-50`, `sub-53`, and `sub-54` in one participant group in all later splits.

## 10. Decision

**Proceed to a limited `sub-53` pilot acquisition.** The metadata support paired PSG-headband analysis and confirm that human consensus labels are available. Full acquisition remains blocked until EDF readability, timing alignment, label mapping, and data-integrity checks pass.

## 11. Pilot Acquisition and Header Inspection Update

**Update date:** 2026-06-24

The limited `sub-53` pilot files were acquired outside Git and inspected with the existing `SMRI` Python environment. The pilot acquisition included 16 files and totalled 235,678,065 bytes locally.

The two EDF files matched the official git-annex byte sizes and SHA-256 hashes:

| Acquisition | Bytes | SHA-256 |
|---|---:|---|
| headband | 92,199,424 | `f7c2756bb5d1563d38fd1859f8c057b3d657d5a4485135c360b4b9e88f7822f0` |
| PSG | 143,421,184 | `f17ea0541b6f94f8decef502c2b660ae756a630553dc23d2016f4be840de9705` |

Header-level inspection confirmed that both EDF files are readable and share:

- 256 Hz sampling;
- 5,122,048 samples;
- 20,008-second duration;
- recorded start time `2023-03-24 01:28:56 UTC`.

The PSG event file contains `stage_hum` and `stage_ai`; the headband event file contains `stage_ai` only. This means human-derived transition labels must be derived from the PSG event table and mapped onto the headband timeline.

For `sub-53`, direct adjacent PSG `stage_hum` epochs contain four REM-to-Wake and two Wake-to-REM candidates. Event tables cover 19,980 seconds, leaving a 28-second unstaged tail relative to EDF duration.

Detailed output is stored in `experiments/2026-06-24_boas_sub53_pilot/`.

## 12. Official Sources

- OpenNeuro snapshot: https://openneuro.org/datasets/ds005555/versions/1.1.1
- Dataset DOI: https://doi.org/10.18112/openneuro.ds005555.v1.1.1
- Official metadata mirror: https://github.com/OpenNeuroDatasets/ds005555/tree/1.1.1
- OpenNeuro API documentation: https://github.com/OpenNeuroOrg/openneuro/blob/master/docs/api.md
