# Direct Endpoint Contribution Analysis Plan v0.1

**Created:** 2026-08-22  
**Project phase:** Block 6 post-result diagnostic  
**Status:** Pre-analysis plan  
**Authorized partition:** Validation only  
**Test access authorized:** No

## 1. Known Evidence

DE-D combines separate REM-before and Wake-after logistic-head scores by multiplication. At its selected validation threshold of 0.74, the product score obtained event F1 0.160377 and 0.991476 false alarms per supported hour. The REM-before head has validation average precision 0.517417 and ROC AUC 0.809638; the Wake-after head has validation average precision 0.769724 and ROC AUC 0.900255.

The event behavior of either head alone has not been evaluated. It is therefore unresolved whether the DE-D improvement requires evidence from both endpoints or is mainly carried by the easier Wake-after target.

## 2. Technological Question

Does explicit conjunction of REM-before and Wake-after evidence add validation event-level value beyond either endpoint head alone?

This is a mechanism analysis of an already completed validation experiment. It is not a new model fit and cannot provide independent confirmation.

## 3. Fixed Comparators

Use the saved continuous validation candidate scores from DE-D without loading fitted models or regenerating features:

- `DE-D-rem-only`: `P(rem_before)`;
- `DE-D-wake-only`: `P(wake_after)`;
- `DE-D-product`: `P(rem_before) * P(wake_after)`.

No score weighting, alternative combination rule, calibration model, feature change, or additional endpoint model is permitted.

## 4. Threshold and Event Rules

For each comparator, search thresholds 0.01 through 0.99 in steps of 0.01. Select the threshold using the existing validation-primary rule:

1. maximum event F1 at +/-15 seconds;
2. lower false alarms per supported hour;
3. higher recall;
4. higher threshold.

Above-threshold adjacent 30-second candidates are collapsed to one alarm by retaining the highest score, with the earlier time retained for an exact tie. Eligible references are matched one-to-one before quality-excluded references.

The product control must reproduce the saved DE-D threshold and primary event counts exactly before any head-only result is interpreted.

## 5. Hypothesis and Decision Rule

Evidence that both endpoints contribute requires the product comparator to have:

- higher selected validation event F1 than both single-head comparators; and
- fewer false alarms per supported hour than both single-head comparators.

If only one condition holds, the contribution is inconclusive. If a single head equals or exceeds the product on both measures, the explicit-conjunction explanation is not supported and the dominant head must be reported.

This strict rule is diagnostic. The independently optimized validation thresholds are optimistic and must not be used to claim generalization.

## 6. Failure-Category Analysis

At each selected threshold, categorize primary +/-15-second false positives using the human stages immediately before and after the alarm:

- REM-to-other;
- other-to-Wake;
- exact REM-to-Wake left unmatched;
- no stage change;
- another stage transition.

The expected mechanism is that REM-only scoring will admit more REM-to-other alarms, Wake-only scoring will admit more other-to-Wake alarms, and the product will suppress both. This expectation is secondary to the fixed event-level decision rule.

## 7. Access and Reproducibility Boundary

Permitted inputs are the hashed DE-D validation candidate-score file, saved validation support, frozen transition membership and boundary-time tables, and the onset/stage fields for validation recordings. The script must reject non-validation score rows.

It must not load or evaluate:

- the current test candidate scores, features, predictions, or metrics;
- fitted model files;
- train labeled rows;
- raw EDF signals;
- Block 7 PSG or transfer inputs.

The frozen project-wide transition-membership table may be read only to select validation references; test rows must be discarded before event evaluation and must not appear in any output.

Continuous candidate scores remain outside Git. Repository outputs are limited to threshold curves, selected metrics, false-positive category summaries, integrity checks, and a concise result record.

## 8. Interpretation Boundary

This analysis can identify whether the observed validation behavior is consistent with a two-endpoint conjunction mechanism. It cannot establish performance on a new participant cohort, clinical utility, or superiority over the provenance-limited SF-A output. DE-D remains unavailable for iterative evaluation on the current test partition.
