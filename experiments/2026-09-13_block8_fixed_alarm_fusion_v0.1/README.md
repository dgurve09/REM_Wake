# Block 8 Fixed Alarm Fusion v0.1

**Work date:** 2026-09-13
**Protocol commit:** `36b002c`
**Partition:** Reused development validation only
**Model fitting or threshold selection:** None
**Test data accessed:** No

## Primary +/-15-Second Results

| Method | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| `P6-D` | 0.0969 | 0.5135 | 0.1631 | 1.1107 |
| `H2-D` | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| `P6-H2-OR` | 0.0604 | 0.6216 | 0.1100 | 2.2465 |
| `P6-P2-H2-OR` | 0.0594 | 0.6216 | 0.1085 | 2.2842 |
| `P6-P2-H2-2OF3` | 0.1485 | 0.4054 | 0.2174 | 0.5397 |

## Frozen Decisions

- `H8.14_two_modality_or_advancement`: fail
- `H8.15_consensus_advancement`: pass
- `H8.16_consensus_boundary_sensitivity`: fail
- `H8.17_all_direct_or_alarm_penalty`: pass
- `overall_fixed_fusion_advance`: advance

## Boundary

The fusion rules operate on already frozen alarms. They are diagnostic combinations, not independently validated detectors. A result can explain whether complementary reference recovery survives false-alarm accounting, but it cannot authorize deployment or test access.

All 10/10 in-run checks passed.
