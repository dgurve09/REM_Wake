# Block 7 Frozen Descriptive Test v0.1

**Work date:** 2026-09-06
**Validation freeze commit:** `4a17fbd`
**Result-producing code commit:** `c0d2e9e`
**Partition accessed:** Frozen test only
**Interpretation:** Descriptive; the same test participants were used in earlier project blocks

## Primary Descriptive Result

| Comparator | Threshold | References | Alarms | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2-D | 0.96 | 59 | 282 | 0.0906 | 0.4237 | 0.1493 | 1.2621 |
| P2-D | 0.99 | 59 | 87 | 0.1235 | 0.1695 | 0.1429 | 0.3570 |
| P2-H2-Z | 0.99 | 59 | 96 | 0.0879 | 0.1356 | 0.1067 | 0.4173 |
| P6-D | 0.99 | 59 | 113 | 0.1589 | 0.2881 | 0.2048 | 0.4525 |

All models and thresholds were frozen before this test execution. No model was fitted, recalibrated, selected, or revised after test access. The predeclared feature-alignment branch remained closed and was not evaluated.

## Paired Participant Contrasts

| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |
|---|---|---:|---:|
| P6-D_minus_P2-D | event_f1_difference | +0.0620 | +0.0019 to +0.1194 |
| P6-D_minus_P2-D | false_alarms_per_hour_difference | +0.0955 | -0.0651 to +0.2510 |
| P2-D_minus_H2-D | event_f1_difference | -0.0064 | -0.0896 to +0.0640 |
| P2-D_minus_H2-D | false_alarms_per_hour_difference | -0.9051 | -1.5538 to -0.3421 |
| P2-D_minus_P2-H2-Z | event_f1_difference | +0.0362 | -0.0434 to +0.1001 |
| P2-D_minus_P2-H2-Z | false_alarms_per_hour_difference | -0.0603 | -0.3360 to +0.2141 |
| P2-H2-Z_minus_H2-D | event_f1_difference | -0.0426 | -0.0870 to +0.0085 |
| P2-H2-Z_minus_H2-D | false_alarms_per_hour_difference | -0.8448 | -1.2441 to -0.5081 |

## Validation-Pattern Check

| Question | Comparison | Test F1 difference [95% interval] | Test false-alarm difference [95% interval] |
|---|---|---:|---:|
| H7.1_channel_reduction | P6-D_minus_P2-D | +0.0620 [+0.0019, +0.1194] | +0.0955 [-0.0651, +0.2510] |
| H7.2_source_to_zero_shot | P2-D_minus_P2-H2-Z | +0.0362 [-0.0434, +0.1001] | -0.0603 [-0.3360, +0.2141] |
| H7.2_zero_shot_to_direct_wearable | P2-H2-Z_minus_H2-D | -0.0426 [-0.0870, +0.0085] | -0.8448 [-1.2441, -0.5081] |

The table compares the frozen validation interpretation with the descriptive test direction; it is not an independent confirmatory test. The fixed `sub-32`/`sub-50` exclusion is retained separately in the full metric table and does not replace the all-test result.

## Boundary

All 13/13 in-run checks passed. These results close the fixed Block 7 comparison but cannot authorize post-test tuning. Any later performance claim requires a newly locked or external cohort.
