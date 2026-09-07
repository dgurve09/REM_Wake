# REM-to-Wake Wearable EEG Research Project

This repository investigates event-specific REM-to-Wake boundary detection from simultaneously recorded PSG and two-channel wearable forehead EEG. Sleep staging is a comparator and label source, not the project target.

## Current Status

As of 2026-09-06, Blocks 3-6 are complete and Block 7 is in progress.

- BOAS snapshot `1.1.1` is frozen at 128 paired recordings and 100 participant-table `pid` groups.
- The conservative primary set contains 276 REM-to-Wake events across 72 groups; the expanded quality-sensitivity set contains 348 across 88 groups.
- The transparent stage-first SF-C baseline obtained test event F1 0.0766 and 1.9692 false alarms/hour.
- The direct DE-B baseline obtained test event F1 0.1497 and 1.2571 false alarms/hour, but precision remained 0.0909.
- Validation-only factorized DE-D improved validation F1 from 0.1127 to 0.1604 and reduced false alarms from 1.4496 to 0.9915/hour. It has not been evaluated on a new locked cohort.
- All 256 local EDF files match the official snapshot SHA-256 annex keys.
- The Block 7 channel gate passed for all 128 pairs: the complete common PSG EEG input is `F3/F4/C3/C4/O1/O2`, the reduced PSG input is `F3/F4`, and the wearable input is `HB_1/HB_2`.
- The train-only Block 7 feature gate passed for 82 recordings across 64 `pid` groups. All 164 signal-path checks, five synthetic spectral checks, and 82 per-recording checks passed; PSG-6/PSG-2 overlap was exact and wearable features reproduced the frozen reference within `4.77e-7`.
- An independent validator rehashed and reopened all 246 external feature arrays and passed 15/15 membership, schema, parity, and provenance checks.
- The Block 7 validation comparison is frozen. Reduced PSG (`P2-D`) produced the strongest validation result, with event F1 0.1887 and 0.3702 false alarms/hour; six-channel PSG reached F1 0.1631 and 1.1107 false alarms/hour; direct wearable reached F1 0.1123 and 1.4558 false alarms/hour.
- Strict PSG-to-wearable zero-shot transfer reached validation F1 0.0685 and 0.6526 false alarms/hour. Its F1 loss versus direct reduced PSG was 0.1202 with paired 95% interval 0.0336 to 0.2001, while its false-alarm difference remained inconclusive.
- The conditional alignment was not run: the performance condition opened, but only 11/80 feature dimensions exceeded the fixed distribution-shift threshold, below the required 20%.

The Block 7 paired-transfer protocol was committed before the channel audit and before feature extraction or fitting. Train/validation modeling is complete and independently validated, but the one-time frozen descriptive test comparison remains pending, so Block 7 is not complete. The current test partition is no longer an independent confirmatory set; its Block 7 role is limited by the written entry conditions below.

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
- [Block 7 feature-gate decision](docs/evaluation/block7_feature_gate_decision_2026-09-06.md)
- [Block 7 transfer-validation result](experiments/2026-09-06_block7_transfer_validation_v0.1/README.md)
- [Block 7 zero-shot hypothesis analysis](experiments/2026-09-06_block7_zero_shot_hypothesis_analysis_v0.1/README.md)
- [Block 7 validation freeze and test entry](docs/evaluation/block7_validation_freeze_and_test_entry_v0.1.md)
- [Current weekly record](docs/weekly/2026-08-31_to_2026-09-06.md)
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
