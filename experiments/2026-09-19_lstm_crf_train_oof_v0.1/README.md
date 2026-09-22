# Train-Only LSTM-CRF Temporal Representation v0.1

**Work date:** 2026-09-19
**Protocol commit:** `2f9d961`
**Result-producing code commit:** `7f5df50`
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Evaluation role:** Train-only participant-grouped developmental screen
**Validation accessed:** No
**Test accessed:** No

## Primary Train-OOF Result

| Model | Threshold | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| LR-OOF | 0.98 | 0.0581 | 0.3167 | 0.0982 | 1.4679 |
| LC-1 | 0.99 | 0.1436 | 0.3111 | 0.1965 | 0.5306 |

## Frozen Decision

LC-1 minus LR-OOF event F1: `+0.0983` (required at least `+0.05`).
LC-1 minus LR-OOF false alarms/hour: `-0.9373` (required no increase).
Decision: **eligible for new confirmation**.

## Boundary

All 14/14 in-run checks passed. Thresholds were selected on pooled out-of-fold predictions from the same 64 train participants. This result is developmental and is not independent confirmation.
