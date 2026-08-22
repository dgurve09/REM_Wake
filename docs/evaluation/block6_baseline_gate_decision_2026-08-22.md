# Block 6 Baseline Gate Decision

**Decision date:** 2026-08-22  
**Target gate date:** 2026-08-23  
**Decision:** Continue direct-event research with limitations; close the simple baseline; defer a CNN

## 1. Question Resolved

Block 6 tested whether direct REM-to-Wake boundary classification from wearable EEG features adds event-level value over a transparent stage-first method under the frozen participant split, quality tiers, and label uncertainty.

## 2. Evidence

At primary membership and +/-15 seconds:

| Method | Precision | Recall | Event F1 | False alarms/hour | Interpretation |
|---|---:|---:|---:|---:|---|
| SF-A | 0.3333 | 0.4237 | 0.3731 | 0.2609 | Fixed descriptive comparator; training provenance unknown |
| SF-C | 0.0438 | 0.3051 | 0.0766 | 1.9692 | Transparent stage-first primary |
| DE-A | 0.0625 | 0.2203 | 0.0974 | 0.9703 | Direct boundary-pair ablation |
| DE-B | 0.0909 | 0.4237 | 0.1497 | 1.2571 | Direct eight-epoch primary |

DE-B met the predeclared directional criterion relative to SF-C: event F1 increased by 0.0731 and false alarms decreased by 0.7121 per hour. In the paired participant bootstrap, the 95% interval was +0.0110 to +0.1659 for event-F1 difference and -1.2709 to -0.2044 for false alarms per hour.

These intervals quantify participant variation in this fixed comparison. They do not convert the result into independent confirmation because the project test partition had already been inspected for the planned stage-first comparison.

## 3. Hypothesis Decisions

**H6.1 direct-boundary value:** Directionally supported relative to SF-C. Direct labeling reduced stage-sequence error propagation but did not produce adequate precision.

**H6.2 broader context:** Not supported on the predeclared validation endpoint. DE-B validation event F1 was 0.1127 versus 0.1152 for DE-A, while false alarms increased. DE-B performed better on test, but that cannot reverse the validation hypothesis decision.

**H6.3 remaining representation limit:** Unresolved. Fixed log-bandpower context is insufficient for low-false-alarm detection, but the present evidence does not isolate neural architecture as the cause.

## 4. Failure Mechanism

DE-B produced 250 primary false positives. Of these, 108 occurred at human other-to-Wake pairs and 56 at human REM-to-other pairs. Partial-endpoint confusion therefore accounted for 65.6% of false alarms. Only 6.8% occurred within 135 seconds of a human REM/Wake boundary, so the background exclusion zone is not the dominant observed source.

False positives occurred in 18 of 20 test participants. The top four participants contributed 58.8%, indicating a broad failure with moderate concentration rather than a single-recording explanation.

## 5. CNN Decision

Do not add a CNN within Block 6. A CNN could test whether time-frequency morphology beyond fixed bandpower improves endpoint discrimination, but that is now a post-test hypothesis and requires a new protocol and a new confirmatory lock. Low performance alone is not evidence that architecture complexity is the correct solution.

## 6. Limitations

- DE-B precision 0.0909 and 1.2571 false alarms per hour are not application-ready.
- Only nine of 20 test participants had a true-positive DE-B event; 18 had at least one false positive.
- SF-A remains substantially stronger descriptively, but its unknown training provenance prevents an independent participant-held-out interpretation.
- The test partition is useful for the planned baseline comparison but no longer supports iterative model development.
- BOAS alone cannot establish clinical performance for sleep paralysis, narcolepsy, or other end applications.

## 7. Next Scheduled Work

Block 7 begins on 2026-08-24. Its first task is a predeclared paired device-shift assessment comparing full PSG, reduced PSG, and wearable inputs, with zero-shot/no-adaptation transfer evaluated before fine-tuning. No Block 7 experiment is attributed to 2026-08-22.

## 8. Validation-Only Method Follow-up

After this gate decision, the partial-endpoint failure mechanism was used to define a separate sequential experiment without reopening test data. An audit found that partial endpoints were already present in reviewed training backgrounds across all train participants, so the new test addressed target structure rather than merely adding more negatives.

DE-D used two logistic heads on the same 80 context features: REM-before and Wake-after. Their probabilities were multiplied as a fixed conjunction. Relative to DE-B on validation, DE-D increased F1 from 0.1127 to 0.1604 and reduced false alarms from 1.4496 to 0.9915 per hour. Partial-endpoint false positives decreased from 151 to 110.

In a post-result paired bootstrap across the 16 validation participants, the DE-D minus DE-B event-F1 interval was +0.0038 to +0.1008 and the false-alarm-rate interval was -0.8835 to -0.1423 per hour. False positives decreased for 11 participants, were unchanged for one, and increased for four.

This strengthens the decision to defer a CNN because a simpler explicit event structure improved validation behavior. It does not replace the frozen DE-B test result. DE-D must remain unevaluated on the current test partition and requires a new locked or external evaluation before any generalization claim.

### Threshold robustness

A further validation-only analysis used saved DE-D probabilities without loading models or features. The two-part improvement over DE-B held across 22 adjacent thresholds from 0.67 through 0.88. In leave-one-participant-out calibration, every fold selected threshold 0.74 using the other 15 participants. The aggregated held-out metrics remained F1 0.1604 and 0.9915 false alarms per hour.

This reduces concern that the validation improvement is an artifact of a single narrow threshold or one calibration participant. It still does not supply new-cohort confirmation.
