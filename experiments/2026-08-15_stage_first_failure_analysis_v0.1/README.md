# Stage-First Failure Analysis v0.1

**Created:** 2026-08-15
**Plan:** `docs/evaluation/stage_first_failure_analysis_plan_v0.1.md`
**Status:** Exploratory diagnostic after the primary result
**Raw signals, external features, or models opened:** No
**Frozen input tables hashed:** 18

## SF-C Test Reference-Side Mechanisms

| Mechanism | Reference events | Fraction |
|---|---:|---:|
| both_target_endpoints_not_predicted | 9 | 0.1525 |
| detected_rem_to_wake | 18 | 0.3051 |
| following_wake_not_predicted | 4 | 0.0678 |
| preceding_rem_not_predicted | 28 | 0.4746 |

Only 18 of 59 primary reference boundaries were represented as an exact predicted 4-to-0 pair. No primary test reference lacked prediction coverage. Of the 41 missed references, 28 missed only the preceding REM endpoint, 4 missed only the following Wake endpoint, and 9 missed both.

At the preceding true REM epoch, SF-C predicted REM for 22 of 59 references and predicted Wake for 27. At the following true Wake epoch, it predicted Wake for 46 of 59 references. Failure to retain REM immediately before the boundary is therefore the dominant reference-side mechanism.

## SF-C Test False-Positive Human Stage Pairs

| Human pair category | False positives | Fraction |
|---|---:|---:|
| human_other_to_wake | 158 | 0.4020 |
| human_rem_to_other | 148 | 0.3766 |
| no_human_stage_change | 66 | 0.1679 |
| other_human_stage_transition | 21 | 0.0534 |

These categories show whether a predicted 4-to-0 boundary occurred across a true non-target transition or without any human stage change. They describe the source of false alarms; they do not redefine the primary event score.

Human REM-to-other and other-to-Wake pairs account for 306 of 393 false positives (0.7786). A further 66 predictions occurred without any human stage change.

## SF-C Test Stage Classes

| Stage | Precision | Recall | F1 | Epochs |
|---|---:|---:|---:|---:|
| REM | 0.5139 | 0.4401 | 0.4742 | 3,817 |
| Wake | 0.6370 | 0.6388 | 0.6379 | 4,635 |
| N1 | 0.1734 | 0.6417 | 0.2730 | 854 |
| N3 | 0.2324 | 0.6587 | 0.3436 | 999 |
| N2 | 0.8966 | 0.6605 | 0.7607 | 13,686 |

The weakest class by recall was REM (0.4401). Boundary detection requires both REM and Wake endpoints to be correct in sequence, so class-wise errors compound at the event level.

## Participant Dispersion

SF-C detected at least one event in 10 of 15 test participants with a primary reference. 5 reference-positive participants had no true positive. All 20 test participants produced at least one false positive. The highest-FP 20% of participants contributed 0.4504 of all false positives. Median participant false alarms/hour was 1.7456, and the maximum was 4.3344.

## Sensitivity Interpretation

Moving from +/-15 to +/-45 seconds changed SF-C test event F1 by +0.0300 and recall by +0.1186. Expanding quality membership at +/-15 seconds changed event F1 by +0.0102. The persistent false-alarm rate indicates that coarse timing and the conservative quality tier are not the sole causes of poor performance.

## Sequence Fragmentation

Across the same supported SF-C test epochs, the human hypnogram contained 1979 all-stage transitions (9.9161/hour), while SF-C produced 4311 (21.6009/hour), a 2.1784-fold rate. SF-C produced 799 REM bouts versus 156 human REM bouts. Median REM-bout duration was 60.0 seconds for SF-C and 600.0 seconds for the human sequence.

This directly supports a fragmentation mechanism: independently classified epochs create too many short stage runs and therefore too many opportunities for spurious 4-to-0 boundaries.

## Technical Interpretation

The failure is not simply low average stage accuracy. Stage-first event derivation requires a particular two-epoch sequence, so endpoint errors create false negatives while isolated REM-to-Wake prediction flips create false positives. Participant dispersion and true-pair categories indicate whether the problem is broad or concentrated, but no subgroup is used to revise the frozen primary result.

## Decision

Retain the Block 5 conclusion: temporal context improves the transparent stage model but does not produce an adequate event detector. These diagnostics may motivate the already-planned direct-event hypothesis, but no direct model, threshold, or later-phase experiment is included here.
