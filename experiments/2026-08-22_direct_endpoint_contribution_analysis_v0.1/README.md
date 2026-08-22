# Direct Endpoint Contribution Analysis v0.1

**Created:** 2026-08-22  
**Status:** Completed post-result validation diagnostic  
**Protocol:** `docs/evaluation/direct_endpoint_contribution_plan_v0.1.md`  
**Repository base:** `ab76720`  
**Test access:** None

## Selected Validation Results

| Comparator | Threshold | Precision | Recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| DE-D-rem-only | 0.89 | 0.0357 | 0.2162 | 0.0613 | 1.3554 |
| DE-D-wake-only | 0.85 | 0.0319 | 0.6757 | 0.0609 | 4.7629 |
| DE-D-product | 0.74 | 0.0971 | 0.4595 | 0.1604 | 0.9915 |

## Decision

Decision: **both_endpoint_contribution_supported**.

The product had higher F1 than both heads: **True**. It had a lower false-alarm rate than both heads: **True**.

## False-Positive Mechanism

REM-only scoring produced 73 REM-to-other false alarms, compared with 35 for the product. Wake-only scoring produced 464 other-to-Wake false alarms, compared with 75 for the product. Both changes followed the predeclared mechanism expectation that a single endpoint admits the corresponding partial transition.

These are diagnostic counts at independently selected validation thresholds, not paired alarm removals. Their magnitude cannot be attributed solely to the conjunction rule.

The product control reproduced the previously saved DE-D threshold and event metrics exactly. The analysis did not load fitted models, raw EEG, train score rows, current-test scores, features, predictions, or metrics. The project-wide transition-membership table was read only to select validation references, as allowed by the plan.

The in-run checks passed 7/7 and the independent validator passed 11/11 across 26 hashed inputs.

## Interpretation Boundary

The result describes mechanism on the already used validation partition. It does not update the frozen test result or establish new-cohort performance. DE-D still requires a new locked or external evaluation.
