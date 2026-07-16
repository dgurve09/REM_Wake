# Signal Quality Flags v0.3

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/signal_quality_flag_spec_v0.3.md`
**Model training performed:** No

## 1. Trigger

Minimal preprocessing v0.1 found two retained train windows with incomplete 240-second signal coverage. The earlier rule verified equal PSG/headband lengths but did not verify the required absolute length.

## 2. Result

| Artifact | Include | Include with 10-MAD sensitivity flag | Targeted review | Critical exclusion |
|---|---:|---:|---:|---:|
| Primary REM-to-Wake | 76 | 200 | 72 | 17 |
| Secondary Wake-to-REM | 23 | 69 | 19 | 0 |
| Background review | 2702 | 1291 | 289 | 20 |

- Transition windows with incomplete 240-second coverage: 8
- Newly excluded windows relative to v0.2: 2
- Primary REM-to-Wake windows retained: 348
- `pid` values retaining at least one primary REM-to-Wake window: 88

## 3. Failure and Resolution

The previous check was insufficient because equal device-window lengths can still be equally short. Version 0.3 requires exactly 61,440 input samples for every 240-second window. Incomplete windows are excluded rather than padded or redefined.

Six incomplete transition windows were already critically excluded by amplitude/flatline evidence. Two additional train transitions are newly excluded. The frozen participant assignment is unchanged because inspected train participants must not move into validation or test.

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_window_quality_flags_v0.3.tsv` | Transition decisions with explicit coverage evidence |
| `background_window_quality_flags_v0.3.tsv` | Background decisions with explicit coverage evidence |
| `recording_signal_quality_summary_v0.3.tsv` | Recording-level counts and coverage failures |
| `quality_flag_summary_v0.3.tsv` | Counts by artifact and current decision |

## 5. Decision

Use v0.3 for preprocessing and later model input construction. Preprocessing v0.2 subsequently passed the train-only mechanical checks. Model training remains blocked pending targeted-review policy and the final label/preprocessing gate.
