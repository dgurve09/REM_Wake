# Direct Endpoint Threshold Robustness Plan v0.1

**Created:** 2026-08-22  
**Status:** Post-result exploratory validation analysis  
**Known result:** DE-D selected threshold 0.74 and met the aggregate DE-B comparison rule  
**Test or model access authorized:** No

## 1. Purpose

DE-D improved aggregate validation event F1 and false alarms per hour, but the threshold was selected and evaluated on the same validation participants. This analysis tests whether the result depends narrowly on threshold 0.74 and whether threshold calibration transfers across participants.

## 2. Frozen Inputs

- saved DE-D validation continuous candidate scores;
- saved DE-D validation support table and complete 99-threshold curve;
- frozen transition membership and boundary-time tables;
- saved DE-B validation event and participant metrics for comparison.

No raw EDF, feature array, fitted model, train row, current-test artifact, refitting, or new prediction is permitted.

## 3. Threshold Perturbation Analysis

Using the already saved DE-D validation threshold curve, identify every threshold satisfying both:

- event F1 greater than frozen DE-B validation F1 0.112676;
- false alarms per hour lower than frozen DE-B validation rate 1.449563.

Report all contiguous threshold intervals in 0.01 steps. The aggregate improvement is considered locally stable if the interval containing selected threshold 0.74 contains at least five adjacent thresholds, representing a span of at least 0.04 from its lowest to highest value.

This rule is fixed before interval results are calculated.

## 4. Leave-One-Participant-Out Calibration

The validation partition contains 16 `pid` groups. For each held-out `pid`:

1. remove all recordings belonging to that participant;
2. use the remaining 15 participants as the calibration subset;
3. evaluate the fixed thresholds 0.01 through 0.99 using primary membership and +/-15 seconds;
4. select by maximum event F1, then lower false alarms per hour, higher recall, and higher threshold;
5. apply that threshold to the held-out participant's saved DE-D candidate scores;
6. retain all held-out alarms without using the held-out labels for threshold selection.

After all folds, combine each participant's held-out alarms and evaluate the full validation cohort under primary and expanded membership at +/-15 and +/-45 seconds.

## 5. Paired Comparison

For the primary +/-15-second result, merge LOPO DE-D and frozen DE-B participant event counts by `pid`. Use a paired participant bootstrap with 2,000 resamples and seed `20260822`. Each resample draws the same participants for both methods and reports DE-D minus DE-B event F1 and false alarms per hour.

## 6. Decision Rule

Threshold robustness is supported only if:

1. the selected-threshold success interval contains at least five adjacent thresholds; and
2. LOPO DE-D has both higher aggregate event F1 and lower false alarms per hour than frozen DE-B validation.

Bootstrap intervals describe participant uncertainty but are not an additional pass/fail condition because only 16 validation participants are available.

If either condition fails, retain DE-D's original validation improvement but classify its threshold robustness as insufficient or inconclusive. Under no outcome may this analysis open the current test partition or revise the frozen DE-D model and score.
