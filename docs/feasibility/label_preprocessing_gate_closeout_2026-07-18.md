# Label and Preprocessing Gate Closeout

**Closeout date:** 2026-07-18
**Project block:** Block 4, deterministic labels and minimal preprocessing
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Model training performed:** No

## 1. Decision

**Pass the label/preprocessing gate with fixed primary and quality-sensitivity tiers.**

The label table, 30-second uncertainty representation, coverage-aware signal-quality rules, participant-grouped split, and minimal offline preprocessing are reproducible. The remaining uncertainty around targeted amplitude, jump, and endpoint flags is preserved as a prespecified expanded quality-sensitivity analysis rather than resolved through subjective removal or unrestricted inclusion.

This decision closes Block 4. It does not establish detector performance or clinical utility. Model work remains scheduled for Block 5, beginning July 27, 2026.

## 2. Frozen Artifacts

| Component | Frozen version | Role |
|---|---|---|
| Transition labels | `transition_labels_v0.1` | 365 primary REM-to-Wake and 111 secondary Wake-to-REM rows with nominal and +/-15-second boundary fields |
| Background review pool | `background_windows_v0.1` | Deterministic review pool separated from the final sampled negative set |
| Signal quality | `signal_quality_flags_v0.3` | Structural, amplitude, alignment, and exact 61,440-sample coverage decisions |
| Analysis membership | `quality_analysis_membership_v0.1` | Primary, expanded quality-sensitivity, and critical-exclusion membership |
| Participant split | `grouped_pid_split_v0.1` | Frozen 64/16/20 `pid` train/validation/test assignment |
| Minimal preprocessing | `minimal_wearable_eeg_preprocessing_v0.2` | Train-validated offline 0.3-35 Hz filtering, 256-to-128 Hz resampling, and train-only robust scaling |

The LF-normalized split assignment SHA-256 is `52450EDA07795D198E2722D4D804E71D0E17A8A4B62BA5AF93AE811B211D83A7`. LF normalization makes the integrity check independent of operating-system line endings.

## 3. Gate Evidence

| Check | Result |
|---|---:|
| Transition rows preserved | 476 of 476 |
| Background review rows preserved | 4,302 of 4,302 |
| Membership outputs reproduced with identical SHA-256 hashes | 7 of 7 |
| Cross-artifact integrity checks passed | 71 of 71 |
| Frozen source/result files recorded in hash manifest | 19 |
| Paired PSG/headband transition sample mappings previously validated | 476 of 476 |
| Primary REM-to-Wake critical exclusions | 17 |
| Primary REM-to-Wake targeted-review rows | 72 |
| Primary-analysis REM-to-Wake rows | 276 across 72 `pid` groups |
| Expanded quality-analysis REM-to-Wake rows | 348 across 88 `pid` groups |
| Primary-analysis background review rows | 3,993 across 100 `pid` groups |
| Expanded quality-analysis background review rows | 4,282 across 100 `pid` groups |
| Participant leakage across partitions | 0 |
| Preprocessing v0.2 train recordings passed | 82 of 82 |
| Preprocessing v0.2 retained train windows passed | 3,063 of 3,063 |
| Synthetic frequency checks passed | 3 of 3 |
| Validation/test EDF files opened during membership work | 0 |

The independent integrity audit in `experiments/2026-07-18_block4_artifact_integrity_v0.1/` linked every label, quality, membership, split, and preprocessing row by identity and invariant fields. It rebuilt the expected noncritical train set and matched all 3,063 preprocessing rows and all 82 train recordings exactly, with zero validation/test rows. All three audit outputs were identical on deterministic rerun.

Primary REM-to-Wake membership by partition is 180 events across 47 positive `pid` groups in train, 37 across 10 in validation, and 59 across 15 in test. Expanded membership is 227/51/70 events across 56/14/18 positive groups.

## 4. Tested Approaches and Resolution

### Nonspecific 10-MAD rule

The initial review rule was insufficient because it flagged heavy-tailed raw EEG across all participant groups and did not distinguish artifact from plausible physiology. It remains a visible flag but does not exclude a window from the primary analysis.

### Equal-length coverage rule

Preprocessing v0.1 showed that matching PSG/headband window lengths did not guarantee a complete 240-second input. Two retained train transitions were equally short in both devices. Quality v0.3 resolved this by requiring exactly 61,440 input samples and excluding incomplete windows without padding.

### Targeted-review treatment

Including every noncritical window in the primary analysis would ignore unresolved artifact indicators. Removing targeted windows from all analyses would treat uncalibrated screening thresholds as proven failure labels. The selected policy uses clean and 10-MAD-only windows in the primary analysis, retains targeted windows in an expanded quality-sensitivity analysis, and excludes critical windows from both.

## 5. Remaining Limitations

- The primary validation set has only 37 events across 10 positive participant groups, so metric uncertainty and participant-level dispersion must be reported.
- Sixteen participant groups have primary events only in the targeted-review tier: nine train, four validation, and three test.
- Targeted flags are not independently calibrated device-failure labels.
- The 30-second scoring interval supports an uncertainty interval, not exact physiological onset time.
- Preprocessing v0.2 is an offline, zero-phase pipeline and cannot be treated as a streaming implementation.
- The background artifact is an eligibility pool, not the final balanced or sampled negative set.

## 6. Conditions for Reopening the Gate

A new version is required if label adjacency, uncertainty width, quality thresholds, participant assignment, channel set, filter, sampling rate, or scaling changes. Later model performance cannot be used to modify v0.1 membership or the test partition. Any revision must retain the failed or superseded artifact and state the technical reason independent of test performance.

## 7. Next Work

Block 5 must predeclare the stage-first comparator, final training-only negative sampling, event matching, metrics, and threshold-selection rule before fitting a model. The locked test partition must remain unopened until the baseline protocol and validation decisions are complete.
