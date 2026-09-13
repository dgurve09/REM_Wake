# Block 8 Cross-Modality Event-Overlap Decision

**Decision date:** 2026-09-13
**Protocol:** `block8_cross_modality_event_overlap_protocol_v0.1.md`
**Protocol commit:** `a7ac70f`
**Result-producing code commit:** `29d0969`
**Partition:** Reused development validation
**Model fitting or threshold selection:** None
**Current test access:** None

## Question

Do the frozen direct PSG and wearable models miss the same REM-to-Wake reference events, or do the input modalities recover complementary event subsets?

## Primary Result

The primary analysis contained 37 eligible validation references at +/-15 seconds.

| Outcome | Events | Fraction |
|---|---:|---:|
| Detected by direct PSG and direct wearable | 13 | 0.3514 |
| Detected by direct PSG only | 7 | 0.1892 |
| Detected by direct wearable only | 3 | 0.0811 |
| Missed by all three direct models | 14 | 0.3784 |

Six-channel PSG detected 19/37 references, two-channel PSG detected 10/37, and direct wearable detected 16/37. The direct PSG union detected 20/37. The diagnostic union of all three direct models detected 23/37, or 0.6216 recall, which was 0.1081 above the best individual direct model.

The diagnostic union is not a deployable ensemble result. Its false-alarm burden was not evaluated, and combining all alarms would be expected to increase that burden.

## Frozen Decisions

| Gate | Observed | Decision |
|---|---:|---|
| H8.10 shared direct failure | Shared-miss fraction 0.3784; gate at least 0.50 | Fail |
| H8.11 wearable-specific recovery gap | PSG-only fraction 0.1892; gate at least 0.15 | Pass |
| H8.12 direct-model complementarity | Union recall gain 0.1081; gate at least 0.10 | Pass |
| H8.13 boundary-tolerance sensitivity | Union recall gain at +/-45 seconds 0.1081; gate at least 0.10 | Pass |

## Participant Uncertainty

The participant-grouped 95% interval for the shared-miss fraction was 0.2258 to 0.5510. Although the point estimate failed H8.10, the interval crosses the 0.50 gate, so a majority shared limitation cannot be excluded.

The PSG-only fraction interval was 0.1111 to 0.2963. The direct-union gain over the best individual model had interval 0 to 0.1892. These point estimates passed their gates, but their participant uncertainty does not establish a stable improvement.

The increase in union recall from +/-15 to +/-45 seconds had interval 0.0238 to 0.2400. This was the only new contrast with an interval clearly above zero and provides the strongest evidence that coarse boundary timing materially changes recoverability.

## Additional Findings

- Six-channel PSG and direct wearable shared 12 detected references, while six-channel PSG uniquely detected seven and wearable uniquely detected four relative to that pair.
- Two-channel PSG added only one primary detection not already found by six-channel PSG.
- Every primary event detected by the zero-shot model was also detected by the direct wearable model; zero-shot transfer added no unique primary event at +/-15 seconds.
- At +/-45 seconds, the direct-model union detected 27/37 references and the shared-miss count fell from 14 to 10.

## Integrity and Correction

- All 9/9 in-run checks passed.
- All 10/10 independent reconstruction checks passed.
- The first successful runs emitted a warning caused by capturing parentheses in the test-path regular expression. The expression still returned the intended Boolean result and did not affect any source, event indicator, metric, interval, or decision.
- Commit `179cae7` replaced the capturing groups with noncapturing groups. Warning-free immutable reruns reproduced every reviewed output.
- No raw signal, feature array, model object, full probability table, or current-test artifact was accessed.

## Knowledge Gained

Low F1 is not explained solely by all modalities missing an identical majority of events. PSG recovers a material subset that direct wearable misses, but wearable also recovers a smaller complementary subset. This supports a wearable-information and representation problem rather than a simple threshold problem.

Boundary tolerance has a measurable effect on event recovery. The exact stage-derived boundary is therefore a material uncertainty, not just an evaluation detail. The result does not show that a wider tolerance makes the detector useful because precision and alarm burden remain limiting.

## Decision

Do not combine the existing models or revise their thresholds. Preserve the overlap result as a mechanism diagnostic.

Continue the planned external-data compatibility audit, but give the subsequent interval-aware boundary experiment high priority. Any temporal representation experiment must remain participant-grouped, use train-only development, retain the frozen wearable comparator, require a meaningful clean F1 gain without increased false alarms, and keep the current test partition closed.
