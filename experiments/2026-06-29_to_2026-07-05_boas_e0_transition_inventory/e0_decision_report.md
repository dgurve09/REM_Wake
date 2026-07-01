# E0 Feasibility Decision Report

**Work period:** 2026-06-29 to 2026-07-05
**Decision date:** 2026-07-01
**Project phase:** Block 3 E0 feasibility audit
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Label specification:** `docs/labels/transition_label_spec_v0.1.md`
**Model training performed:** No

## 1. Decision

**Proceed to deterministic transition-label generation and minimal preprocessing.**

This is a proceed decision for the next data-engineering and label-validation phase only. It is not a decision to train models immediately.

The BOAS event tables contain enough direct REM-to-Wake candidates and enough participant-level spread to justify building the versioned transition-label table, validating PSG-to-headband timing assumptions, and preparing the first minimal preprocessing workflow.

## 2. Evidence Reviewed

| Evidence item | Result |
|---|---:|
| PSG recordings checked | 128 |
| Unique `pid` values checked | 100 |
| REM-to-Wake candidates | 365 |
| Wake-to-REM candidates | 111 |
| Recordings with at least one REM-to-Wake candidate | 112 |
| `pid` values with at least one REM-to-Wake candidate | 88 |
| Recordings with at least one Wake-to-REM candidate | 57 |
| `pid` values with at least one Wake-to-REM candidate | 46 |
| Recordings with missing `stage_hum` epochs | 0 |
| Recordings with non-30-second epochs | 0 |
| Recordings with PSG/headband duration mismatch in sidecars | 0 |
| Recordings with PSG/headband sampling-frequency mismatch in sidecars | 0 |
| Recordings with PSG disconnection epochs | 37 |
| Candidate 240-second windows containing PSG disconnection epochs | 0 |
| Unlabeled tail range | 0 to 29 seconds |

The `sub-53` pilot result is consistent with the full inventory: four REM-to-Wake candidates and two Wake-to-REM candidates were found for `sub-53`.

## 3. E0 Questions

### Are direct REM-to-Wake events present in enough recordings to proceed?

Yes at the event-table level. The inventory found 365 direct adjacent REM-to-Wake candidates across 112 of 128 PSG recordings.

### Are events distributed across enough unique participants?

Yes at the `pid` level. Primary REM-to-Wake candidates are present in 88 of 100 unique `pid` values. This is enough to justify grouped label-table construction and later grouped evaluation design.

### Are repeated recordings common enough to affect splitting?

Yes. The participant table contains 80 `pid` values with one recording, 12 with two recordings, and 8 with three recordings. Any later train, validation, or test split must group by `pid`.

### How often do label-quality issues affect candidate windows?

No missing `stage_hum` epochs or non-30-second epochs were observed. PSG disconnection epochs occur in 37 recordings, but none of the derived REM/Wake candidate windows contained PSG disconnection epochs within the 240-second inspection window used for this count audit.

This does not replace EDF-level signal-quality inspection. It only means the PSG event labels do not show disconnection flags near the derived REM/Wake candidates under the current window rule.

### Does the unlabeled tail pattern create a systematic issue?

No for transition-label generation under v0.1. Unlabeled tails range from 0 to 29 seconds, which is less than one 30-second epoch. These tails should remain excluded from positive and negative label generation.

### Is Wake-to-REM useful as secondary information?

Yes as secondary quality-control and bidirectional boundary information. The inventory found 111 Wake-to-REM candidates across 57 recordings and 46 `pid` values. These should remain separate from the primary REM-to-Wake target.

## 4. Limitations

- This decision is based on event tables and sidecar metadata, not full EDF signal inspection.
- The label resolution remains 30 seconds; the nominal boundary is not treated as an exact physiological transition time.
- Direct adjacent REM-to-Wake labels do not capture gradual arousals, micro-awakenings, or events not represented by adjacent hypnogram epochs.
- PSG disconnection flags in event tables do not cover all possible signal-quality problems.
- BOAS is not a dedicated narcolepsy or sleep-paralysis cohort, so later outputs must not claim clinical utility from this dataset alone.
- The participant-count discrepancy remains: the BOAS README describes 108 individuals, while `participants.tsv` contains 100 unique `pid` values.

## 5. Next Decision

Proceed to Block 4 work:

1. create a versioned deterministic transition-label table from `candidate_transition_events.tsv`;
2. preserve the 30-second uncertainty interval for each boundary;
3. validate PSG-to-headband timing assumptions beyond sidecar duration agreement;
4. define a grouped split policy using `pid`;
5. begin minimal preprocessing only after the label table and timing checks are reviewed.

No model training should start until the label/preprocessing gate is complete.
