# BOAS Full EDF Acquisition

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 signal-alignment preparation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Raw EDF storage:** outside Git
**Status:** Acquisition run
**Model training performed:** No

## 1. Purpose

This acquisition prepares the full BOAS PSG/headband EDF set for representative and full-dataset PSG-to-headband signal-alignment validation.

The technical uncertainty being prepared for is whether PSG-derived `stage_hum` transition labels remain temporally valid when mapped to wearable headband EEG across BOAS recordings, beyond the already completed `sub-53` pilot.

## 2. Acquisition Summary

| Item | Value |
|---|---:|
| EDF files in scope | 256 |
| Files complete locally | 256 |
| Files with partial download data | 0 |
| Files with no complete local EDF | 0 |
| Expected total EDF bytes | 35,913,652,480 |
| Complete local EDF bytes | 35,913,652,480 |

## 3. Verification Method

Each EDF is checked against the remote object byte size reported by the OpenNeuro S3 endpoint. The script is resumable: incomplete downloads are retained as `.part` files outside Git and continued on the next run.

## 4. Outputs

| File | Purpose |
|---|---|
| `edf_acquisition_manifest.tsv` | Per-EDF expected size, local size, partial size, and acquisition status |

## 5. Boundary

This step acquires raw signal files only. It does not train a model and does not by itself validate signal alignment. The subsequent full signal-alignment validation is recorded in `experiments/2026-07-04_boas_full_signal_alignment/`.
