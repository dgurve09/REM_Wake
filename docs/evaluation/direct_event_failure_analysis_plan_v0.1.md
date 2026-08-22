# Direct Event Failure Analysis Plan v0.1

**Created:** 2026-08-22  
**Status:** Post-result exploratory analysis  
**Primary model:** DE-B  
**Model or threshold changes authorized:** No

## 1. Known Result

This plan was written after the frozen direct test result was known. DE-B obtained primary test event F1 0.1497, precision 0.0909, recall 0.4237, and 1.2571 false alarms per supported hour at +/-15 seconds. It improved directionally over SF-C but retained 250 false-positive alarms. The analysis below is explanatory and cannot revise that result.

## 2. Questions

1. Do false-positive alarms occur preferentially at a human REM endpoint, a Wake onset, another transition, or no human stage change?
2. Do false positives cluster within the +/-135-second REM/Wake exclusion region that was intentionally absent from reviewed background training rows?
3. Is the false-positive burden confined to a small number of participants or distributed broadly?
4. Are the additional matches at +/-45 seconds predominantly one epoch early or late?
5. What tradeoff did eight-epoch context make relative to the two-epoch ablation?

## 3. Frozen Inputs

- saved DE-B test alarms and event matches;
- saved DE-A and DE-B test participant and aggregate metrics;
- frozen test continuous candidate scores stored outside Git;
- frozen stage-first recording feature arrays, using their retained human stage code only for post hoc categorization;
- transition membership v0.1 and signal-quality v0.3 boundary times.

No raw EDF, fitted model, refitting, threshold search, or new test prediction is permitted.

## 4. Definitions

A primary +/-15-second false positive is a saved DE-B alarm that was matched to neither an eligible primary reference nor a quality-ignored reference under the frozen evaluator.

The human stage pair is the human-scored stage at `t-30 s` and `t`. It is categorized as:

- `human_REM_to_Wake`;
- `human_REM_to_other`;
- `human_other_to_Wake`;
- `human_no_stage_change`;
- `human_other_transition`.

Distance is measured from each false alarm to the closest human-derived REM/Wake boundary of either direction in the same recording. Prespecified bins are exact, 30-45 seconds, 60-135 seconds, and beyond 135 seconds. If a recording contains no REM/Wake reference, the distance is undefined and is retained as `no_remwake_reference_in_recording`. The 135-second boundary distinguishes candidates represented by the background exclusion rule from those outside it.

Participant concentration reports the share of false positives contributed by the four highest-burden test `pid` values, representing 20% of the 20 test participants.

For timing sensitivity, signed error is prediction time minus reference time. The +/-45-second matches are summarized as early, exact, or late, and the additional matches beyond the primary tolerance are reported separately.

## 5. Interpretation Rule

Concentration near REM/Wake endpoints would identify background coverage near physiological transitions as a limitation of the current training construction. Enrichment at REM-to-other or other-to-Wake pairs would indicate incomplete direct discrimination of the two required endpoints. Broad participant dispersion would argue against fixing the result by excluding a few recordings.

These findings may motivate a future experiment, but they do not authorize post-test changes within v0.1. A later CNN or negative-construction experiment requires a new predeclared hypothesis and validation process.
