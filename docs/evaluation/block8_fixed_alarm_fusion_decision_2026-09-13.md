# Block 8 Fixed Alarm-Fusion Decision

**Decision date:** 2026-09-13
**Protocol:** `block8_fixed_alarm_fusion_protocol_v0.1.md`
**Protocol commit:** `36b002c`
**Result-producing code commit:** `3321e72`
**Partition:** Reused development validation
**Model fitting or threshold selection:** None
**Current test access:** None

## Question

Does the complementary reference recovery observed across frozen PSG and wearable models translate into better event performance after the combined false-alarm burden is counted?

## Primary Results

| Method | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| `P6-D` control | 0.0969 | 0.5135 | 0.1631 | 1.1107 |
| `H2-D` control | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| `P6-H2-OR` | 0.0604 | 0.6216 | 0.1100 | 2.2465 |
| `P6-P2-H2-OR` | 0.0594 | 0.6216 | 0.1085 | 2.2842 |
| `P6-P2-H2-2OF3` | 0.1485 | 0.4054 | 0.2174 | 0.5397 |

Simple OR fusion raised recall but accumulated false alarms, causing lower F1 than the six-channel PSG control. The 2-of-3 consensus instead retained 15/37 primary references, reduced the alarm count from 210 to 110, and reduced false positives from 177 to 86 relative to `P6-D`.

## Frozen Decisions

| Gate | Result | Decision |
|---|---|---|
| H8.14 two-modality OR advancement | F1 difference -0.0530; FAR difference +1.1358/hour | Fail |
| H8.15 consensus advancement | F1 difference +0.0543; FAR difference -0.5710/hour | Pass |
| H8.16 consensus boundary sensitivity | +/-45 versus +/-15 F1 difference +0.0363; required at least +0.05 | Fail |
| H8.17 all-direct OR alarm penalty | FAR difference +1.1735/hour; gate at least +0.50 | Pass |
| Overall fixed-fusion decision | H8.15 passed | Advance as a mechanism comparator only |

## Participant Uncertainty

For `P6-H2-OR`, the F1-difference interval was -0.1072 to -0.0244 and the FAR-difference interval was +0.7536 to +1.6837/hour. Both directions were consistent: OR fusion was worse than `P6-D` under the fixed rule.

For the all-direct OR, the F1-difference interval was -0.1086 to -0.0257 and the FAR-difference interval was +0.7828 to +1.7355/hour. Adding `P2-D` did not recover another primary reference beyond the `P6-D` plus `H2-D` union, but it added alarms.

For 2-of-3 consensus, the F1-difference interval was -0.0192 to +0.1323, so the apparent F1 gain was not stable across participants. The FAR-difference interval was -1.1759 to -0.1346/hour and excluded zero, providing stronger evidence for false-alarm suppression than for improved event detection.

The consensus F1 increase from +/-15 to +/-45 seconds was +0.0363 with interval +0.0059 to +0.0754. Its direction was consistent, but it failed the predeclared +0.05 materiality gate.

## Contributor Analysis

The 110 retained consensus alarms comprised:

- 41 clusters supported by `P6-D` and `H2-D`;
- 31 supported by all three direct models;
- 21 supported by `P2-D` and `H2-D`; and
- 17 supported by `P6-D` and `P2-D`.

Of the 15 primary true detections, nine were supported by all three models, five by `P6-D` and `H2-D`, and one by `P6-D` and `P2-D`. No `P2-D` plus `H2-D` cluster without `P6-D` produced a primary true detection at +/-15 seconds.

This contributor analysis is descriptive. It was derived from the fixed retained-cluster identities and did not select a new rule.

## Integrity

- All 12 frozen direct-model metric rows were reproduced.
- Every source alarm entered each applicable clustering exactly once.
- All clusters respected the fixed 30-second maximum span.
- All 2-of-3 retained alarms had at least two distinct contributors.
- All 10/10 in-run and 10/10 independent checks passed.
- An immutable analysis rerun reproduced every reviewed output.
- No raw signal, feature array, model object, full probability table, or current-test artifact was accessed.

## Knowledge Gained

Complementary detections do not justify simple alarm union. The false alarms are sufficiently non-overlapping that OR fusion materially worsens the event result. Agreement across at least two models filters many of those alarms, showing that cross-view consistency contains useful specificity information.

The consensus result does not solve the wearable problem. It requires simultaneously recorded laboratory PSG, uses a reused validation cohort, retains low precision 0.1485 and recall 0.4054, and has an F1-gain interval crossing zero. Its value is mechanistic: false alarms are less consistent across signal views than a subset of true detections.

## Decision

Stop fixed OR and consensus fusion development at v0.1. Preserve 2-of-3 consensus as a laboratory mechanism comparator, not as the project detector. Do not evaluate it on the current test partition.

The next deployable-method uncertainty remains whether wearable-only temporal evidence and interval-aware boundary handling can reproduce the specificity benefit without PSG at inference. That requires a separately committed train-only protocol, a meaningful clean F1 and false-alarm gate, and a new locked or external confirmation boundary.
