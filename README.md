# REM-to-Wake Wearable EEG Research Project

This repository investigates event-specific REM-to-Wake boundary detection from simultaneously recorded PSG and two-channel wearable forehead EEG. Sleep staging is a comparator and label source, not the project target.

## Current Status

As of 2026-09-13, Blocks 3-7 are complete and Block 8 robustness work is in progress.

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
- The frozen descriptive test comparison was executed once on 26 recordings from 20 `pid` groups without refitting. Six-channel PSG produced the highest test F1 at 0.2048, followed by direct wearable at 0.1493, reduced PSG at 0.1429, and strict zero-shot transfer at 0.1067.
- The validation ordering did not fully reproduce. `P6-D - P2-D` test F1 was +0.0620 with paired 95% interval +0.0019 to +0.1194, reversing the validation point direction. The direct reduced-PSG versus zero-shot F1 difference remained positive at +0.0362 but its interval crossed zero.
- Zero-shot transfer again had lower F1 but substantially fewer false alarms than direct wearable: the paired test differences were -0.0426 F1 and -0.8448 false alarms/hour, with only the false-alarm interval excluding zero.
- All 13/13 in-run checks and 13/13 independent output checks passed. The validator rehashed 108 external artifacts and reproduced 95,460 probabilities, event outputs, and paired contrasts twice.
- The first Block 8 experiment tested single-channel feature-contribution robustness using the frozen direct wearable model and validation partition only. Neutralizing `HB_1` reduced F1 from 0.1123 to 0 and false alarms from 1.4558 to 0.5208/hour, failing the predefined material-change screen. Neutralizing `HB_2` reduced F1 to 0.0913 and false alarms to 1.0793/hour.
- Both ablation directions persisted across all 16 leave-one-`pid`-out folds. This identifies asymmetric channel dependence in the frozen classifier; it does not simulate physical electrode failure.
- The second Block 8 experiment added deterministic raw-rate Gaussian noise before feature extraction. Both-channel probability agreement degraded monotonically from 20 to 0 dB; at 0 dB the detector found no true event. At 10 dB, isolated `HB_1` noise reduced F1 to 0.0419 and increased false alarms to 4.4616/hour, while isolated `HB_2` noise produced F1 0.0825 and 0.3514 false alarms/hour.
- The raw-signal experiment passed 17/17 in-run checks and 12/12 independent checks twice. The validator regenerated 100 degraded feature arrays from EDF and reproduced 114,738 probabilities, event outputs, participant intervals, and frozen decisions. These controlled white-noise findings do not represent natural field artefacts.
- A predeclared train-only augmentation then exposed the unchanged bandpower-logistic pipeline to clean and four fixed noise conditions. Its grouped train-OOF threshold was 0.99. It reduced the isolated `HB_1` false-alarm result from 4.4616 to 0.9413/hour, but clean F1 decreased from 0.1123 to 0.1024 and other noise conditions showed false-alarm tradeoffs.
- The augmentation passed its targeted `HB_1` robustness gates but failed the required +0.05 clean-F1 advancement gate. It is retained as a robustness comparator and does not replace the core detector. All 13 train, 11 validation, and 14 independent checks passed; the full independent reconstruction passed twice.
- A frozen validation event-overlap analysis found 13/37 primary references detected by both direct PSG and wearable, seven by PSG only, three by wearable only, and 14 missed by all three direct models. The diagnostic union recall was 0.6216 but is not an ensemble result because its false-alarm burden was not evaluated.
- Widening event tolerance from +/-15 to +/-45 seconds increased direct-model union recall by 0.1081, with a participant interval of 0.0238 to 0.2400. This supports explicit boundary-uncertainty work; it does not make the present detector reliable. All 9/9 in-run and 10/10 independent checks passed without current-test access.

The Block 7 paired-transfer protocol was committed before the channel audit and before feature extraction or fitting. The fixed descriptive test now closes Block 7, but the test partition was already used in earlier blocks and does not provide independent confirmation. Its results cannot be used to revise the frozen models or thresholds. The observed validation-to-test reversal makes participant and channel robustness the next uncertainty rather than justifying immediate adaptation or a more complex classifier.

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
- [Block 7 descriptive test result](experiments/2026-09-06_block7_descriptive_test_v0.1/README.md)
- [Block 7 transfer-gate decision](docs/evaluation/block7_transfer_gate_decision_2026-09-06.md)
- [Block 8 single-channel protocol](docs/evaluation/block8_single_channel_robustness_protocol_v0.1.md)
- [Block 8 single-channel result](experiments/2026-09-11_block8_single_channel_robustness_v0.1/README.md)
- [Block 8 single-channel decision](docs/evaluation/block8_single_channel_decision_2026-09-11.md)
- [Block 8 raw-signal noise protocol](docs/evaluation/block8_raw_signal_noise_protocol_v0.1.md)
- [Block 8 raw-signal noise result](experiments/2026-09-11_block8_raw_signal_noise_v0.1/README.md)
- [Block 8 raw-signal noise decision](docs/evaluation/block8_raw_signal_noise_decision_2026-09-11.md)
- [Block 8 train-only augmentation protocol](docs/evaluation/block8_noise_augmented_training_protocol_v0.1.md)
- [Block 8 train-only augmentation result](experiments/2026-09-12_block8_noise_augmented_training_v0.1/README.md)
- [Block 8 train-only augmentation decision](docs/evaluation/block8_noise_augmented_training_decision_2026-09-13.md)
- [Block 8 cross-modality overlap protocol](docs/evaluation/block8_cross_modality_event_overlap_protocol_v0.1.md)
- [Block 8 cross-modality overlap result](experiments/2026-09-13_block8_cross_modality_event_overlap_v0.1/README.md)
- [Block 8 cross-modality overlap decision](docs/evaluation/block8_cross_modality_event_overlap_decision_2026-09-13.md)
- [Current weekly record](docs/weekly/2026-09-07_to_2026-09-13.md)
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
