# Block 7 Paired-Transfer Validation v0.1

**Work date:** 2026-09-06
**Execution-plan commit:** `a10dd71`
**Feature-gate result commit:** `1d914ab`
**Result-producing code commit:** `37eacf2`
**Partitions accessed:** Train and validation only
**Test data accessed:** No

## Primary Validation Result

| Comparator | Labeled-window AP | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
| H2-D | 0.4548 | 0.96 | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| P2-D | 0.5493 | 0.99 | 0.1449 | 0.2703 | 0.1887 | 0.3702 |
| P2-H2-Z | not applicable | 0.99 | 0.0459 | 0.1351 | 0.0685 | 0.6526 |
| P6-D | 0.5364 | 0.99 | 0.0969 | 0.5135 | 0.1631 | 1.1107 |

The direct comparators selected their own validation thresholds. Strict zero-shot and conditional alignment, when executed, inherited the P2-D threshold.

## Adaptation Gate

- Zero-shot F1 deficit relative to H2-D: `0.0438`.
- Zero-shot false-alarm excess per hour: `-0.8032`.
- Shifted dimensions: `11/80`.
- Performance condition open: **True**.
- Distribution condition open: **False**.
- Adaptation action: **skip_P2-H2-A**.

## Paired Participant Contrasts

| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |
|---|---|---:|---:|
| P6-D_minus_P2-D | event_f1_difference | -0.0256 | -0.1164 to +0.0808 |
| P6-D_minus_P2-D | false_alarms_per_hour_difference | +0.7405 | +0.2402 to +1.4109 |
| P2-D_minus_H2-D | event_f1_difference | +0.0764 | -0.0144 to +0.1671 |
| P2-D_minus_H2-D | false_alarms_per_hour_difference | -1.0856 | -1.6901 to -0.6782 |
| P2-H2-Z_minus_H2-D | event_f1_difference | -0.0438 | -0.1014 to +0.0193 |
| P2-H2-Z_minus_H2-D | false_alarms_per_hour_difference | -0.8032 | -1.0402 to -0.5956 |

## Boundary

All 17/17 in-run checks passed. This is validation-only evidence. It does not authorize method revision from the later test result and does not provide independent confirmation.
