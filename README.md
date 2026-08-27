# REM-to-Wake Wearable EEG Research Project

This repository investigates event-specific REM-to-Wake boundary detection from simultaneously recorded PSG and two-channel wearable forehead EEG. Sleep staging is a comparator and label source, not the project target.

## Current Status

As of 2026-08-26, Blocks 3-6 are complete and Block 7 is in progress.

- BOAS snapshot `1.1.1` is frozen at 128 paired recordings and 100 participant-table `pid` groups.
- The conservative primary set contains 276 REM-to-Wake events across 72 groups; the expanded quality-sensitivity set contains 348 across 88 groups.
- The transparent stage-first SF-C baseline obtained test event F1 0.0766 and 1.9692 false alarms/hour.
- The direct DE-B baseline obtained test event F1 0.1497 and 1.2571 false alarms/hour, but precision remained 0.0909.
- Validation-only factorized DE-D improved validation F1 from 0.1127 to 0.1604 and reduced false alarms from 1.4496 to 0.9915/hour. It has not been evaluated on a new locked cohort.
- All 256 local EDF files match the official snapshot SHA-256 annex keys.
- The Block 7 channel gate passed for all 128 pairs: the complete common PSG EEG input is `F3/F4/C3/C4/O1/O2`, the reduced PSG input is `F3/F4`, and the wearable input is `HB_1/HB_2`.

The Block 7 paired-transfer protocol was committed before the channel audit and before feature extraction or fitting. The current test partition is no longer an independent confirmatory set; its Block 7 role is limited by the written entry conditions below.

## Start Here

- [Project proposal](Proposal.md)
- [Overall timeline](docs/planning/overall_project_timeline.md)
- [Project working rules](PROJECT_RULES.md)
- [End-to-end audit, 2026-08-23](docs/audit/project_audit_2026-08-23.md)
- [Historical experiment first-commit index](docs/audit/experiment_commit_index_2026-08-23.md)
- [Reference DOI audit](experiments/2026-08-23_reference_doi_audit_v0.1/README.md)
- [Block 7 entry conditions](docs/evaluation/block7_entry_conditions_v0.1.md)
- [Block 7 paired-transfer protocol](docs/evaluation/block7_paired_transfer_protocol_v0.1.md)
- [Block 7 feature-generation validation plan](docs/evaluation/block7_feature_generation_validation_plan_v0.1.md)
- [Current weekly record](docs/weekly/2026-08-24_to_2026-08-30.md)
- [BOAS dataset manifest](docs/data/boas_dataset_manifest.md)
- [Label/preprocessing gate](docs/feasibility/label_preprocessing_gate_closeout_2026-07-18.md)
- [Block 6 baseline decision](docs/evaluation/block6_baseline_gate_decision_2026-08-22.md)

## Repository Layout

| Path | Purpose |
|---|---|
| `docs/` | Protocols, decisions, planning, audits, and weekly records |
| `experiments/` | Immutable dated result packages and failure records |
| `labels/` | Versioned derived labels, background windows, quality flags, and membership |
| `splits/` | Frozen participant-grouped assignments and balance summaries |
| `scripts/` | Small reproducible acquisition, validation, analysis, and audit scripts |

Raw EDF files, fitted models, full feature arrays, and full-night candidate-score artifacts are stored outside Git and referenced by manifests. Compact reviewed tabular predictions and labeled-row scores may be retained when they are needed to recompute published metrics; they contain public BOAS identifiers rather than direct personal identifiers.

## Research Boundary

BOAS does not contain sleep-paralysis episodes, narcolepsy diagnoses, treatment outcomes, or exact physiological transition times. This project can establish technical feasibility and failure modes for wearable REM-to-Wake measurement. It cannot establish diagnosis, clinical utility, prevention, or intervention effectiveness.
