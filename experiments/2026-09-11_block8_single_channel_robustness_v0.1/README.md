# Block 8 Single-Channel Robustness v0.1

**Work date:** 2026-09-11
**Protocol commit:** `b55849f`
**Result-producing code commit:** `76fc5c4`
**Partition accessed:** Validation only
**Test or raw-signal artifacts accessed:** No
**Screen decision:** `fail_material_single_channel_effect`

## Primary Result

| Comparator | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| H2-INTACT | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| H2-NO-HB1 | 0.0000 | 0.0000 | 0.0000 | 0.5208 |
| H2-NO-HB2 | 0.0549 | 0.2703 | 0.0913 | 1.0793 |

## Ablation Effects

| Comparison | F1 difference | False-alarm difference/hour | Materially adverse | Any apparent improvement |
|---|---:|---:|---:|---:|
| H2-NO-HB1_minus_H2-INTACT | -0.1123 | -0.9350 | True | True |
| H2-NO-HB2_minus_H2-INTACT | -0.0210 | -0.3765 | False | True |

A materially adverse change was predefined as an F1 decrease of at least 0.03 or a false-alarm increase of at least 0.50/hour. An apparent improvement after neutralization is treated as channel-contribution or calibration instability, not as evidence that losing a channel is beneficial.

## Leave-One-Participant-Out Influence

| Comparison | F1 direction matches | FAR direction matches | F1 difference range | FAR difference range | Participant-stable |
|---|---:|---:|---:|---:|---:|
| H2-NO-HB1_minus_H2-INTACT | 16/16 | 16/16 | -0.1311 to -0.0916 | -1.0187 to -0.7686 | True |
| H2-NO-HB2_minus_H2-INTACT | 16/16 | 16/16 | -0.0364 to -0.0105 | -0.4452 to -0.1781 | True |

## False-Alarm Concentration

| Comparator | Participants with false alarms | Top-four participant share |
|---|---:|---:|
| H2-INTACT | 16/16 | 51.72% |
| H2-NO-HB1 | 14/16 | 75.90% |
| H2-NO-HB2 | 16/16 | 45.93% |

## Interpretation Boundary

All 15/15 in-run checks passed. This validation-only feature-contribution ablation does not simulate physical electrode failure and cannot support post-test model revision. A raw-signal degradation experiment requires a separate protocol.
