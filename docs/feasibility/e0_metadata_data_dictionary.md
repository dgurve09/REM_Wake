# E0 Metadata Data Dictionary

**Prepared during:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Scope:** BOAS metadata/event readiness and planned E0 inventory fields

## 1. Source Files

| Source | Role |
|---|---|
| `participants.tsv` | Participant grouping and demographics available in BOAS |
| `sub-XX_task-Sleep_acq-psg_events.tsv` | Human reference sleep-stage labels through `stage_hum` |
| `sub-XX_task-Sleep_acq-headband_events.tsv` | Algorithmic headband labels through `stage_ai`; not reference ground truth |
| `sub-XX_task-Sleep_acq-psg_eeg.json` | PSG duration and sampling metadata |
| `sub-XX_task-Sleep_acq-headband_eeg.json` | Headband duration and sampling metadata |
| channel TSV files | Channel availability and names |
| scan TSV files | Recording file list per subject |

## 2. Participant Fields

| Field | Meaning | Use |
|---|---|---|
| `participant_id` | BOAS recording folder identifier | Recording-level join key |
| `pid` | Participant grouping identifier | Required grouping key for later splits |
| `age` | Age value from BOAS metadata | Descriptive only at E0 |
| `sex` | Sex value from BOAS metadata | Descriptive only at E0 |
| `bmi` | BMI value from BOAS metadata | Descriptive only at E0 |

## 3. Event Table Fields

| Field | Meaning | E0 use |
|---|---|---|
| `onset` | Epoch onset in seconds | Boundary timing |
| `duration` | Epoch duration in seconds | Must be 30 seconds for primary rule |
| `begsample` | Start sample index from source event table | Consistency check |
| `endsample` | End sample index from source event table | Consistency check |
| `offset` | Source offset field | Preserve, not primary |
| `stage_hum` | Human PSG consensus stage | Reference label source |
| `stage_ai` | Algorithmic stage estimate | Not reference ground truth |

## 4. Derived E0 Fields

| Field | Meaning |
|---|---|
| `recording` | BOAS recording folder, such as `sub-53` |
| `pid` | Participant grouping identifier |
| `transition_type` | `REM_to_Wake` or `Wake_to_REM` |
| `from_stage_code` | Stage code at epoch `t` |
| `to_stage_code` | Stage code at epoch `t + 1` |
| `from_epoch_onset_sec` | Onset of source epoch |
| `boundary_onset_sec` | Onset of target epoch |
| `uncertainty_start_sec` | `boundary_onset_sec - 15` |
| `uncertainty_end_sec` | `boundary_onset_sec + 15` |
| `contains_psg_disconnection` | Whether disconnection is present in the candidate context |
| `label_quality_flag` | Rule-based label-quality flag |

## 5. Readiness Fields

| Field | Meaning |
|---|---|
| `has_stage_hum` | Whether an event table contains human PSG labels |
| `has_stage_ai` | Whether an event table contains algorithmic labels |
| `coverage_end_sec` | Last scored epoch end time |
| `recording_duration_sec` | Duration reported by EEG sidecar |
| `unlabeled_tail_sec` | `recording_duration_sec - coverage_end_sec` |
| `sampling_frequency_hz` | Sampling frequency reported by EEG sidecar |
| `stage_hum_disconnection_epochs` | Count of `stage_hum = 8` epochs |

## 6. Interpretation Rules

- `stage_hum` is the only human reference label source.
- `stage_ai` can be used for descriptive comparison only.
- `pid` is required for later participant-grouped evaluation.
- Unlabeled tails are excluded from event generation.
- Non-30-second epochs require a label-quality flag before event generation.
