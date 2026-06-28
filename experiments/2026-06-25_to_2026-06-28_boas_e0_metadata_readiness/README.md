# BOAS E0 Metadata Readiness

**Work period:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Project phase:** Block 2 closeout, E0 readiness
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Raw EDF files downloaded:** No
**Full E0 transition inventory performed:** No
**Model training performed:** No

## 1. Purpose

This readiness check prepares the metadata/event inputs required for the scheduled E0 feasibility audit beginning on 2026-06-29. It verifies file availability and schema consistency without counting REM-to-Wake events or making the feasibility decision.

## 2. File Readiness

- Metadata/event files checked: 1157
- Missing files: 0
- Total downloaded metadata/event size: 7,750,181 bytes
- EDF signal files are intentionally excluded from this acquisition.

## 3. Participant Readiness

- Recording rows in `participants.tsv`: 128
- Unique `pid` values: 100
- Repeated-participant grouping is available through `pid`.

Participant repeat summary:

| Recordings per pid | Number of pid values |
|---:|---:|
| 1 | 80 |
| 2 | 12 |
| 3 | 8 |

## 4. Event Schema Readiness

- PSG event files checked: 128
- Headband event files checked: 128
- All PSG event files contain `stage_hum`: True
- Headband event files containing `stage_hum`: 0
- Event duration values observed: 30
- Sampling frequencies observed in sidecars: 256.0
- Unlabeled tail range across event files: 0.0 to 29.0 seconds

Interpretation: the metadata/event inputs are ready for the scheduled E0 event inventory. Human-derived labels should come from PSG `stage_hum`, and headband event files should not be treated as human ground truth.

## 5. Outputs

| File | Purpose |
|---|---|
| `metadata_file_inventory.tsv` | File presence and byte size for E0 metadata inputs |
| `participant_pid_summary.tsv` | Number of recordings per participant identifier |
| `event_schema_summary.tsv` | Per-recording event schema, coverage, and label-column checks |

## 6. Boundary

This check does not count REM-to-Wake events, does not estimate participant-level feasibility, and does not train or evaluate any model. Those decisions remain assigned to the E0 feasibility audit.
