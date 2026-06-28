# Transition-Label Specification v0.1

**Work period:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Status:** Pilot specification for E0 feasibility work
**Dataset context:** BOAS `ds005555`, snapshot `1.1.1`
**Model training covered:** No

## 1. Project Target

The project target is REM-to-Wake boundary detection from wearable EEG, not sleep-stage classification.

Sleep-stage labels are used as source annotations for deriving event labels and as a later comparator. The direct target is the transition boundary.

## 2. Reference Label Source

Use PSG event files as the human reference source:

```text
sub-XX/eeg/sub-XX_task-Sleep_acq-psg_events.tsv
```

Use the `stage_hum` column for human-derived transition labels.

Do not use headband `stage_ai` as reference ground truth. The `sub-53` pilot confirmed that the headband event file contains `stage_ai` only, while the PSG event file contains `stage_hum`.

## 3. Stage Codes

| Code | Meaning |
|---:|---|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |
| 8 | PSG disconnection |

## 4. Primary Transition Rule

The primary event is a direct adjacent REM-to-Wake transition:

```text
stage_hum[t] = 4 and stage_hum[t + 1] = 0
```

The nominal boundary time is the onset of the second epoch:

```text
boundary_onset_sec = onset[t + 1]
```

## 5. Secondary Pilot Rule

Wake-to-REM transitions may be retained as secondary pilot events and quality-control examples:

```text
stage_hum[t] = 0 and stage_hum[t + 1] = 4
```

These are not the primary project target but are useful for checking bidirectional REM/Wake boundary handling.

## 6. Label Uncertainty

BOAS stage labels are 30-second epoch labels. A derived transition boundary therefore has coarse-label uncertainty.

For v0.1, record the nominal boundary and the uncertainty interval:

```text
uncertainty_start_sec = boundary_onset_sec - 15
uncertainty_end_sec = boundary_onset_sec + 15
```

This interval does not claim the true physiological transition occurred exactly at the nominal boundary. It records the label resolution limitation explicitly.

## 7. Exclusion and Flagging Rules

Do not create a primary transition label when either adjacent epoch has:

- missing `stage_hum`;
- non-30-second duration;
- `stage_hum = 8` PSG disconnection.

Flag, but do not automatically discard, candidate windows that contain PSG disconnection epochs elsewhere inside a larger inspection window. The E0 audit should count these separately.

## 8. Unlabeled Tail Rule

If the EDF duration is longer than the final scored event epoch, the unlabeled tail is excluded from transition-label generation.

For `sub-53`, both EDF files are 20,008 seconds long, while the event tables cover 666 epochs:

```text
666 epochs * 30 seconds = 19,980 seconds
```

The remaining 28 seconds are not assigned any derived transition label in v0.1. The tail may be reported as unlabeled recording time but should not be used as a positive or negative transition example.

During E0, record the unlabeled-tail length for every recording.

## 9. Mapping to Wearable Signals

Derived labels come from PSG `stage_hum` but are mapped to the simultaneously recorded headband EDF timeline.

The `sub-53` pilot confirmed header-level agreement between PSG and headband EDFs:

- same start time;
- same sampling rate;
- same sample count;
- same duration.

This is sufficient for pilot mapping, but the E0 work should still record timing mismatches if they appear in other recordings.

## 10. Participant Grouping

All later split decisions must group by BOAS `pid`, not by recording folder alone.

For the pilot participant:

```text
sub-50, sub-53, and sub-54 share pid = 88
```

No train, validation, or test split should separate recordings with the same `pid`.

## 11. Version Notes

Version `v0.1` is intentionally conservative:

- direct adjacent REM-to-Wake is the primary target;
- Wake-to-REM is retained only as secondary pilot information;
- coarse-label uncertainty is explicit;
- unlabeled tails are excluded from label generation;
- full-dataset counts are deferred to the scheduled E0 feasibility audit.
