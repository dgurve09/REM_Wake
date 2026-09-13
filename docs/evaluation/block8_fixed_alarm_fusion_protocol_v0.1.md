# Block 8 Fixed Alarm-Fusion Feasibility Protocol v0.1

**Protocol date:** 2026-09-13
**Project block:** Block 8 signal and channel robustness
**Analysis role:** Fixed diagnostic evaluation of frozen validation alarms
**Model fitting:** None
**Threshold selection:** None
**Current test access:** Prohibited

## 1. Question

The cross-modality overlap analysis found complementary reference-event detections, but its union recall did not include the false alarms produced by combining models. This experiment asks whether fixed, non-optimized alarm fusion converts complementary detection into higher event F1 or merely accumulates alarm burden.

## 2. Frozen Inputs

Only these reviewed Block 7 and label-provenance files may be opened:

- `validation_predicted_events_v0.1.tsv`;
- `validation_event_metrics_v0.1.tsv`;
- `validation_support_v0.1.tsv`;
- `transition_analysis_membership_v0.1.tsv`; and
- `transition_window_quality_flags_v0.3.tsv`.

Input paths, sizes, and SHA-256 values will be recorded. Raw EDF, feature arrays, model objects, full candidate-score tables, and all current-test artifacts remain outside the analysis.

## 3. Frozen Source Alarms

The source alarms and thresholds are unchanged:

| Comparator | Input | Threshold |
|---|---|---:|
| `P6-D` | Six-channel PSG EEG | 0.99 |
| `P2-D` | Two-channel PSG EEG | 0.99 |
| `H2-D` | Two-channel wearable EEG | 0.96 |

The strict zero-shot model is excluded because it added no unique primary +/-15-second reference detection beyond `H2-D` in the preceding predeclared overlap analysis.

## 4. Fixed Alarm Clustering

Fusion will operate on the already consolidated event alarms, not on probabilities. For each recording:

1. pool the selected source-model alarms and sort by time;
2. form a cluster beginning at the earliest unused alarm;
3. include subsequent alarms while the total cluster span remains at most 30 seconds;
4. use the observed alarm time nearest the cluster median as the cluster time, breaking an exact tie toward the earlier alarm; and
5. record the number and identity of distinct contributing comparators.

This maximum-span rule prevents chained alarms from merging events more than one epoch apart. No clustering interval will be searched.

## 5. Fixed Fusion Rules

| Fusion | Rule |
|---|---|
| `P6-H2-OR` | Retain every cluster formed from `P6-D` and `H2-D` alarms |
| `P6-P2-H2-OR` | Retain every cluster formed from all three direct models |
| `P6-P2-H2-2OF3` | Form all-direct clusters but retain only clusters supported by at least two distinct direct models |

The unchanged `P6-D`, `P2-D`, and `H2-D` results will be reproduced as controls.

## 6. Evaluation

The primary endpoint is primary-membership event performance at +/-15 seconds. Expanded membership and +/-45 seconds are fixed sensitivity analyses. The existing one-to-one event matcher, ignored-quality handling, recording support, and definitions of precision, recall, F1, and false alarms per supported hour remain unchanged.

The analysis will save:

- every fused alarm with its source membership;
- event matches, recording and participant counts, and aggregate metrics;
- fixed comparisons with `P6-D`;
- 5,000 participant-grouped bootstrap contrasts using seed `20260913`; and
- the hypothesis decisions below.

## 7. Predeclared Hypotheses

| Gate | Pass condition | Purpose |
|---|---|---|
| H8.14 two-modality OR advancement | `P6-H2-OR` F1 is at least 0.05 above `P6-D` and FAR is no more than 0.25/hour higher | Test whether the main complementary pair provides a meaningful net event benefit |
| H8.15 consensus advancement | `P6-P2-H2-2OF3` F1 is not below `P6-D` and FAR is not above `P6-D` | Test whether cross-model agreement controls alarms without losing net event performance |
| H8.16 consensus boundary sensitivity | The 2-of-3 F1 at +/-45 seconds is at least 0.05 above its +/-15-second F1 | Test whether boundary timing materially changes the fixed consensus result |
| H8.17 all-direct OR alarm penalty | `P6-P2-H2-OR` FAR is at least 0.50/hour above `P6-D` | Test the expected cost hidden by the earlier recall-only union |

A fusion method advances only if H8.14 or H8.15 passes. H8.16 or H8.17 may explain failure but cannot authorize a detector.

## 8. Participant Uncertainty

Participant bootstrap samples will draw the 16 validation `pid` groups with replacement. All recordings, reference events, predictions, ignored-quality matches, and supported hours belonging to a selected `pid` will receive the same multiplicity. Intervals will be reported for F1 and FAR differences for each fusion versus `P6-D`, plus the 2-of-3 +/-45 versus +/-15 F1 difference.

Point-estimate gates remain the frozen decision rule. Intervals will determine whether the direction is stable enough to guide the next experiment.

## 9. Integrity Checks

The run must verify:

- exactly 20 validation recordings and 16 validation `pid` groups;
- identical temporal support across source comparators;
- exact reproduction of all 12 frozen direct-model metric rows;
- every source alarm enters the applicable OR clustering exactly once;
- each fused cluster spans at most 30 seconds;
- 2-of-3 alarms have at least two distinct contributors;
- exact event accounting for every fusion, membership, and tolerance;
- participant-grouped bootstrap reconstruction;
- no fitting, probability access, or threshold search; and
- no current-test path or row in any source or output.

An independent validator must reconstruct the source hashes, fused alarms, event outputs, comparisons, bootstrap intervals, and decisions.

## 10. Decision Boundary

This is a reused-validation mechanism test, not an independently confirmatory model evaluation. A passing fusion gate cannot establish clinical utility or justify reopening the current test partition. A failed fusion result will show that complementary hits do not overcome the alarm burden of the current representations and will strengthen the case for interval-aware labels or a new temporal representation rather than further threshold or ensemble tuning.
