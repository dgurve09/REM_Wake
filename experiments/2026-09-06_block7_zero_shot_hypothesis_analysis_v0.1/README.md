# Block 7 Zero-Shot Hypothesis Analysis v0.1

**Work date:** 2026-09-06
**Analysis status:** Post-result completion analysis
**Plan commit:** `03be56b`
**Input result commit:** `15dc9b4`
**Analysis code commit:** `30918ae`
**Input partition:** Validation only
**Raw signal, feature, model, or test access:** No

## Paired Results

| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |
|---|---|---:|---:|
| P2-D_minus_P2-H2-Z | event_f1_difference | +0.1202 | +0.0336 to +0.2001 |
| P2-D_minus_P2-H2-Z | false_alarms_per_hour_difference | -0.2824 | -0.7972 to +0.0506 |
| H2-D_minus_P2-H2-Z | event_f1_difference | +0.0438 | -0.0193 to +0.1014 |
| H2-D_minus_P2-H2-Z | false_alarms_per_hour_difference | +0.8032 | +0.5956 to +1.0402 |
| P2-D_minus_H2-D | event_f1_difference | +0.0764 | -0.0144 to +0.1671 |
| P2-D_minus_H2-D | false_alarms_per_hour_difference | -1.0856 | -1.6901 to -0.6782 |

## Interpretation

Relative to direct P2-D, strict zero-shot transfer changed event F1 by -0.1202 and false alarms per hour by +0.2824. This is the direct source-to-target transfer cost under one unchanged model and threshold.

At least one paired source-versus-zero-shot interval crosses zero, so the complete two-metric source-transfer loss remains inconclusive.

Relative to direct H2-D, zero-shot transfer changed event F1 by -0.0438 and false alarms per hour by -0.8032. The lower F1 occurred with fewer, not more, false alarms, so this comparison is a recall/alarm tradeoff rather than uniform degradation.

Relative to direct wearable fitting, the evidence is mixed: zero-shot F1 is lower but alarm burden is also lower. The hypothesis of uniformly worse wearable event performance is therefore not supported as stated on this validation partition.

All 7/7 analysis checks passed. These are post-result validation-only contrasts and do not provide independent confirmation.
