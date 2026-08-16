# Stage-First Comparator Closeout v0.1

**Created:** 2026-08-15
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Primary event definition:** primary quality membership, +/-15-second matching

## Frozen Test Comparison

| Comparator | Stage macro F1 | Supported hours | Event precision | Event recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
| SF-A | 0.7248 | 191.625 | 0.3333 | 0.4237 | 0.3731 | 0.2609 |
| SF-B | 0.4382 | 200.975 | 0.0192 | 0.1356 | 0.0337 | 2.0301 |
| SF-C | 0.4979 | 199.575 | 0.0438 | 0.3051 | 0.0766 | 1.9692 |

## Answer to the Block 5 Question

The transparent stage-first result is inadequate as a REM-to-Wake alarm. SF-C obtains test stage macro F1 0.4979, but its event precision is only 0.0438 and it produces 1.9692 false alarms/hour. Five-epoch context improves over the epoch-only ablation, but does not resolve the event-specific failure.

SF-A is substantially stronger, with test event F1 0.3731, but it has unknown training provenance and cannot be assumed independent of BOAS. It remains a fixed descriptive reference, not the primary held-out estimate and not a model-selection target.

Valid prediction coverage differs slightly across comparators: SF-A, SF-B, and SF-C contribute 191.625, 200.975, and 199.575 supported test hours, respectively. Event metrics and false alarms/hour therefore use comparator-specific exposure and should not be interpreted as an identical-coverage ranking.

## Remaining Uncertainty

Block 5 does not establish that direct boundary detection will work. It establishes a measured failure mode for a reproducible stage-first approach: modest stage errors create many spurious adjacent REM-to-Wake boundaries. Block 6 must test whether a direct event objective can reduce false alarms while preserving useful recall under the same split, labels, quality tiers, and uncertainty tolerances.

An exploratory failure analysis using frozen result tables found a specific mechanism. On test, SF-C produced 2.1784 times the human all-stage transition rate, 799 predicted REM bouts versus 156 human bouts, and a median REM-bout duration of 60 seconds versus 600 seconds. All 20 test participants produced at least one false positive. These post-result diagnostics support sequence fragmentation as the main stage-first limitation but do not replace the primary metrics.

A second exploratory paired diagnostic found SF-C stage macro F1 improved for all 20 test participants, but event F1 improved for 10 and was unchanged for 10. The paired-bootstrap event-F1 difference was +0.0431 with a 95% interval of +0.0199 to +0.0676; the false-alarm-rate difference interval crossed zero. Six of seven additional SF-C matches admitted at +/-45 seconds were one epoch early, and 59.20% of predicted REM bouts lasted only 30 or 60 seconds. These findings explain context's partial benefit without establishing an adequate alarm rate.

## Decision

Proceed to the prespecified simple direct feature baseline. Do not add a CNN until the direct feature result identifies a specific limitation requiring a richer representation. Preserve the SF-C 500-iteration convergence warning when comparing methods.
