# Transition Labels v0.1

**Created:** 2026-07-04
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Alignment evidence:** `experiments/2026-07-04_boas_full_signal_alignment/`
**Model training performed:** No

## 1. Purpose

This artifact converts the E0 REM/Wake transition inventory into a versioned deterministic label table for later wearable EEG preprocessing.

The uncertainty addressed here is whether PSG-derived REM/Wake transition labels can be represented reproducibly with explicit 30-second label uncertainty and sample-level headband mapping.

## 2. Method

- Start from `candidate_transition_events.tsv` created during the E0 inventory.
- Preserve the nominal boundary and the `+/-15` second uncertainty interval from label specification `v0.1`.
- Add zero-based PSG/headband sample indices using the full signal-alignment validation.
- Keep primary REM-to-Wake labels separate from secondary Wake-to-REM labels.
- Add quality flags from event-label checks and PSG/headband alignment checks.
- Do not create train, validation, or test splits in this artifact.

## 3. Result

| Item | Value |
|---|---:|
| Total transition-label rows | 476 |
| Primary REM-to-Wake rows | 365 |
| Secondary Wake-to-REM rows | 111 |
| Unique recordings | 112 |
| Unique `pid` values | 88 |
| Rows marked include | 476 |
| Rows marked review | 0 |
| Rows with sample-alignment pass | 476 |

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_labels_v0.1.tsv` | Versioned deterministic REM/Wake transition-label table |
| `transition_label_summary_v0.1.tsv` | Count summary by transition type |
| `pid_transition_distribution_v0.1.tsv` | Participant-level label distribution for grouped split planning |
| `grouped_split_policy_draft_v0.1.md` | Draft rules for later leakage-safe train/validation/test splitting |

## 5. Limitations

- The labels are derived from 30-second sleep-stage epochs, not exact physiological transition onsets.
- BOAS is not a dedicated narcolepsy or sleep-paralysis cohort.
- This table does not include negative/background windows yet.
- This table does not define train, validation, or test splits.
- Model training remains blocked until the label/preprocessing gate.

## 6. Decision

Use `transition_labels_v0.1.tsv` as the first reviewed positive-event label artifact for deterministic preprocessing and split-policy design.

Next work should define background-window rules and recording/window-level signal-quality flags before model work.
