# Nested Train-Only Deep Temporal Exploration v0.1

**Work date:** 2026-09-22
**Protocol commit:** `533cad3`
**Result-producing code commit:** `b12f4f3`
**Validation accessed:** No
**Test accessed:** No

## Inner Selections

| Outer fold | Candidate | Threshold | Inner F1 | Inner FAR/hour |
|---:|---|---:|---:|---:|
| 1 | BLSTM-2H | 0.98201379 | 0.2180 | 0.2151 |
| 2 | BLSTM-CRF | 0.99330715 | 0.2167 | 0.4135 |
| 3 | BGRU-CRF | 0.99142251 | 0.2021 | 0.3971 |
| 4 | BGRU-CRF | 0.99908895 | 0.1983 | 0.1466 |
| 5 | TCN-CRF | 0.99142251 | 0.2367 | 0.5786 |

Inner-ranked candidate for future confirmation: **BLSTM-2H**.

## Primary Nested Outer Result

| Pipeline | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| NESTED-BLSTM-CRF | 0.1461 | 0.1778 | 0.1604 | 0.2971 |
| NESTED-SELECTED | 0.1531 | 0.1778 | 0.1645 | 0.2812 |

## Frozen Decision

Selected minus baseline F1: `+0.0041`.
Selected minus baseline false alarms/hour: `-0.0159`.
Decision: **stop v0.1**.

## Boundary

All 14/14 in-run checks passed. This is nested development on the reused train cohort, not independent confirmation.
