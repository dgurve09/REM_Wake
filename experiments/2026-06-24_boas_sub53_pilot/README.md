# BOAS Sub-53 Pilot Inspection

**Date:** 2026-06-24
**Project phase:** Block 2 setup and pilot verification
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Recording:** `sub-53`
**Raw data in Git:** No
**Model training performed:** No

## 1. Purpose

This pilot checks whether one paired BOAS PSG/headband recording can be read and mapped to a stage-derived REM/Wake transition target before larger acquisition or model development.

The work addresses four setup questions:

1. Can the required pilot files be acquired without downloading the full dataset?
2. Do the downloaded EDF files match the official git-annex byte sizes and SHA-256 hashes?
3. Are the PSG and headband EDF headers compatible at the level needed for a first alignment check?
4. Does the pilot contain direct REM-to-Wake and Wake-to-REM candidates from human PSG labels?

## 2. Acquisition

The acquisition script is:

```text
scripts/download_boas_sub53_pilot.py
```

It downloads only the pilot files required for inspection. The default storage location is outside the repository:

```text
<repo-parent>/REM_W_data/boas_ds005555_v1.1.1
```

Sixteen files were acquired for the pilot, including two EDF files, event tables, channel tables, JSON sidecars, scans metadata, root participant metadata, README, CHANGES, and dataset description. The local pilot download size was 235,678,065 bytes.

## 3. EDF Header Findings

The inspection script is:

```text
scripts/inspect_boas_sub53_pilot.py
```

It generated the committed summary tables in this folder.

| Acquisition | EDF bytes | SHA-256 verified | Channels | Sampling rate | Samples | Duration | Start time |
|---|---:|---|---:|---:|---:|---:|---|
| headband | 92,199,424 | Yes | 9 | 256 Hz | 5,122,048 | 20,008 s | 2023-03-24 01:28:56 UTC |
| PSG | 143,421,184 | Yes | 14 | 256 Hz | 5,122,048 | 20,008 s | 2023-03-24 01:28:56 UTC |

Header-level conclusion: the paired PSG and headband EDF files are readable with the existing Python environment and match at sampling rate, sample count, duration, and recorded start time.

This does not yet prove sample-level physiological alignment. That remains a later check using signal content and transition windows.

## 4. Channel Findings

Headband channels:

```text
HB_1, HB_2, HB_IMU_1, HB_IMU_2, HB_IMU_3, HB_IMU_4, HB_IMU_5, HB_IMU_6, HB_PULSE
```

PSG channels:

```text
PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2, PSG_EOG, PSG_EMG,
PSG_THER, PSG_THOR, PSG_ABD, PSG_PULSE, PSG_BEAT, PSG_SPO2
```

The wearable EEG channels for later analysis are `HB_1` and `HB_2`. IMU and pulse channels are available but should be treated as secondary unless the feasibility work shows they are needed.

## 5. Event-Label Findings

Both event tables contain 666 rows of 30-second epochs. Event coverage ends at 19,980 seconds, leaving a 28-second unstaged tail relative to the 20,008-second EDF duration.

| Event file | Stage column available | Stage counts |
|---|---|---|
| headband events | `stage_ai` only | -2=56; 0=113; 1=18; 2=315; 4=164 |
| PSG events | `stage_hum` and `stage_ai` | 0=200; 1=24; 2=368; 4=65; 8=9 |

Important correction: the human reference labels for transition derivation are present in the PSG events file, not in the headband events file. The headband event table contains algorithmic labels only. Therefore the project should derive transition targets from PSG `stage_hum` and map those boundaries onto the headband signal timeline.

## 6. Stage-Derived Transition Candidates

Using direct adjacent PSG `stage_hum` epochs:

| Transition type | Count |
|---|---:|
| REM-to-Wake | 4 |
| Wake-to-REM | 2 |

The candidate boundaries are saved in:

```text
sub53_stage_hum_transition_candidates.tsv
```

Each boundary uses the onset of the second epoch as the nominal transition time and records a 30-second label-resolution uncertainty interval as nominal boundary time plus or minus 15 seconds.

## 7. Decision

Proceed with the next limited pilot step.

The dataset passes the first acquisition and header-readability check for `sub-53`. The next work should focus on signal-level validation:

- confirm PSG/headband alignment beyond matching header start times;
- inspect wearable signal quality around the six pilot transition candidates;
- decide how to handle the 28-second unstaged tail;
- define the first versioned transition-label specification before full event counting.

## 8. Files Produced

| File | Purpose |
|---|---|
| `edf_header_summary.tsv` | EDF readability, channels, sample counts, duration, hashes |
| `event_table_summary.tsv` | Event-table structure and stage-count summary |
| `sidecar_summary.tsv` | Technical sidecar fields used for the header check |
| `sub53_stage_hum_transition_candidates.tsv` | Pilot transition-boundary candidates derived from PSG `stage_hum` |
