# E0 Feasibility Audit Protocol

**Prepared during:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Scheduled audit period:** 2026-06-29 to 2026-07-12
**Project phase:** Block 3 preparation
**Status:** Protocol executed; E0 feasibility closeout completed on 2026-07-11 and recorded in `e0_feasibility_closeout_2026-07-11.md`

## 1. Purpose

E0 will determine whether BOAS contains enough technically usable REM/Wake transition events, distributed across enough independent participants, to justify later preprocessing and model work.

The audit is a feasibility gate. It is not a model-training stage and it should not be treated as a performance result.

## 2. Inputs

Use BOAS `ds005555`, snapshot `1.1.1`.

Required inputs:

- `participants.tsv`;
- PSG event tables;
- PSG event JSON sidecars;
- headband event tables;
- headband event JSON sidecars;
- PSG/headband EEG sidecars;
- channel tables;
- scan metadata.

EDF signal files are not required for the first event-count inventory. EDFs are required only for signal-quality checks or later preprocessing.

## 3. Reference Label Rule

Use `docs/labels/transition_label_spec_v0.1.md`.

Primary event:

```text
stage_hum[t] = 4 and stage_hum[t + 1] = 0
```

Nominal boundary:

```text
boundary_onset_sec = onset[t + 1]
```

Secondary pilot event:

```text
stage_hum[t] = 0 and stage_hum[t + 1] = 4
```

Secondary events should be counted separately and should not be merged into the primary feasibility count.

## 4. Required E0 Outputs

E0 must produce:

1. recording-level transition inventory;
2. participant-level transition inventory using `pid`;
3. label-quality summary;
4. unlabeled-tail summary;
5. PSG disconnection summary;
6. participant-repeat summary;
7. feasibility decision report.

## 5. Minimum Tables

| Table | Purpose |
|---|---|
| `recording_transition_inventory.tsv` | Direct REM-to-Wake and Wake-to-REM counts per recording |
| `participant_transition_inventory.tsv` | Event counts grouped by `pid` |
| `label_quality_summary.tsv` | Missing labels, non-30-second epochs, PSG disconnections, unlabeled tails |
| `candidate_transition_events.tsv` | Candidate event table with nominal boundary and uncertainty interval |
| `e0_decision_report.md` | Proceed, narrow, redesign, or stop decision |

## 6. Feasibility Questions

E0 must answer:

1. Are direct REM-to-Wake events present in enough recordings to proceed?
2. Are events distributed across enough unique `pid` values to support grouped evaluation?
3. Are repeated recordings from the same `pid` common enough to affect splitting?
4. How often do PSG disconnections, missing labels, or abnormal epoch durations affect candidate windows?
5. Does the unlabeled tail pattern create any systematic issue for label generation?
6. Is Wake-to-REM useful as secondary information, or should the project narrow to REM-to-Wake only?

## 7. Decision Options

| Decision | Meaning |
|---|---|
| Proceed | Enough primary events and participant spread to continue as planned |
| Narrow | Continue only with REM-to-Wake or a stricter subset |
| Redesign | Change target to transition-risk, boundary analysis, or descriptive feasibility |
| Stop | BOAS is insufficient for this technical question |

## 8. Guardrails

- Do not train a model during E0.
- Do not report classifier performance from E0.
- Do not split recordings from the same `pid` across train, validation, and test in later work.
- Do not treat headband `stage_ai` as human ground truth.
- Do not invent exact transition times beyond the 30-second label resolution.
- Do not use unlabeled EDF tails as positive or negative examples.

## 9. Completion Criteria

E0 is complete only when:

- all expected PSG event files have been checked;
- transition candidates have been derived from PSG `stage_hum`;
- events are summarized by recording and by `pid`;
- label-quality flags are summarized;
- the proceed/narrow/redesign/stop decision is recorded with limitations.
