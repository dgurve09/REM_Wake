# BOAS E0 Transition Inventory

**Work period:** 2026-06-29 to 2026-07-05
**Run date:** 2026-07-01
**Project phase:** Block 3 E0 feasibility audit
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Label specification:** `docs/labels/transition_label_spec_v0.1.md`
**Model training performed:** No

## 1. Purpose

This inventory tests whether BOAS contains enough direct REM/Wake transition events and participant-level spread to justify later preprocessing and model work.

The uncertainty being tested is not model performance. The uncertainty is whether the available human PSG labels contain enough usable REM-to-Wake boundaries, with adequate participant grouping, to support the planned REM-to-Wake wearable EEG project.

## 2. Method

- Read all 128 PSG event tables from the local BOAS metadata/event snapshot.
- Use `stage_hum` as the only reference label source.
- Count direct adjacent REM-to-Wake events: `stage_hum[t] = 4` and `stage_hum[t + 1] = 0`.
- Count direct adjacent Wake-to-REM events separately as secondary information.
- Record the nominal boundary as the onset of the second epoch.
- Record a 30-second uncertainty interval as `boundary_onset_sec - 15` to `boundary_onset_sec + 15`.
- Exclude unlabeled EDF tails from transition generation and report the tail duration.
- Record missing labels, non-30-second epochs, PSG disconnection epochs, timing gaps, and PSG/headband sidecar mismatches.

## 3. Main Count Summary

| Item | Value |
|---|---:|
| PSG recordings checked | 128 |
| Unique `pid` values checked | 100 |
| REM-to-Wake candidates | 365 |
| Wake-to-REM candidates | 111 |
| Recordings with at least one REM-to-Wake candidate | 112 |
| Recordings with at least one Wake-to-REM candidate | 57 |
| `pid` values with at least one REM-to-Wake candidate | 88 |
| `pid` values with at least one Wake-to-REM candidate | 46 |

## 4. Label-Quality Summary

| Item | Value |
|---|---:|
| Recordings with missing `stage_hum` epochs | 0 |
| Recordings with non-30-second epochs | 0 |
| Recordings with PSG disconnection epochs | 37 |
| Candidate windows containing PSG disconnection epochs | 0 |
| Unlabeled tail minimum, seconds | 0.0 |
| Unlabeled tail maximum, seconds | 29.0 |

## 5. Outputs

| File | Purpose |
|---|---|
| `recording_transition_inventory.tsv` | Transition counts and quality fields by recording |
| `participant_transition_inventory.tsv` | Transition counts grouped by BOAS `pid` |
| `candidate_transition_events.tsv` | Row-level REM/Wake transition candidates and uncertainty intervals |
| `label_quality_summary.tsv` | Label, duration, timing, and sidecar consistency checks |
| `unlabeled_tail_summary.tsv` | Unlabeled recording tail per PSG event table |
| `psg_disconnection_summary.tsv` | PSG disconnection counts and timing by recording |
| `e0_decision_report.md` | Manual review of the inventory against E0 feasibility questions |

## 6. Interpretation Boundary

This is an event inventory and label-quality audit. It is not a model result, not a classifier-performance estimate, and not evidence of clinical utility.

The companion decision report reviews these counts against the E0 feasibility criteria before any model training.
