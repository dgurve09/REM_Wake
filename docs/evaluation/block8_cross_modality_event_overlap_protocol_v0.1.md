# Block 8 Cross-Modality Event-Overlap Protocol v0.1

**Protocol date:** 2026-09-13
**Project block:** Block 8 signal and channel robustness
**Analysis role:** Mechanistic analysis of frozen development-validation outputs
**Model fitting:** None
**Threshold selection:** None
**Current test access:** Prohibited

## 1. Question

The frozen direct models all have low event F1, including the PSG inputs. This analysis asks whether they fail on the same REM-to-Wake reference events or recover different subsets. The result will distinguish a shared representation or label-timing limitation from a wearable-specific loss and will determine which uncertainty should be tested next.

## 2. Frozen Inputs

The analysis will use only the reviewed Block 7 development-validation outputs and existing label provenance:

- `validation_event_matches_v0.1.tsv`;
- `validation_event_metrics_v0.1.tsv`;
- `validation_support_v0.1.tsv`;
- `transition_analysis_membership_v0.1.tsv`; and
- `transition_window_quality_flags_v0.3.tsv`.

Input paths, byte counts, and SHA-256 values will be recorded before analysis. No raw EDF, feature array, model, full probability table, or test result will be opened.

## 3. Frozen Comparators

The primary comparison uses the three separately fitted direct models with their already selected validation thresholds:

| Comparator | Input | Threshold |
|---|---|---:|
| `P6-D` | Six-channel PSG EEG | 0.99 |
| `P2-D` | Two-channel PSG EEG | 0.99 |
| `H2-D` | Two-channel wearable EEG | 0.96 |

`P2-H2-Z`, the strict PSG-to-wearable zero-shot model, will be retained as a separate descriptive column but will not define the direct-model decision gates.

## 4. Reference and Detection Rules

The primary set is the frozen development-validation REM-to-Wake membership marked `primary_analysis_eligible`. The primary tolerance is +/-15 seconds. The same reference identity is considered detected by a comparator only when the reviewed match table contains an `eligible` match for the same subject and nominal boundary.

The analysis will construct one row per primary reference event with binary detection indicators for all four comparators. It will then assign each reference to one mutually exclusive direct-modality category:

- `both_psg_and_wearable`: detected by `P6-D` or `P2-D` and by `H2-D`;
- `psg_only`: detected by `P6-D` or `P2-D` and missed by `H2-D`;
- `wearable_only`: detected by `H2-D` and missed by both direct PSG models; or
- `shared_miss`: missed by all three direct models.

Expanded-quality membership and +/-45-second matching are fixed sensitivity analyses. They cannot change a primary decision.

## 5. Fixed Summaries

The analysis will report:

1. detected reference counts and recall for each comparator;
2. pairwise both-hit, first-only, second-only, and both-miss counts;
3. Jaccard overlap among detected reference sets;
4. the four direct-modality categories;
5. direct-model union recall and its gain over the best individual direct model;
6. the change in direct-model union recall from +/-15 to +/-45 seconds; and
7. participant-grouped bootstrap intervals for the primary proportions and recall contrasts.

The participant bootstrap will resample the 16 validation `pid` groups with replacement for 5,000 iterations using seed `20260913`. Duplicate sampled groups will contribute duplicate copies of all their reference events.

## 6. Predeclared Decision Gates

| Gate | Pass condition | Interpretation if passed |
|---|---|---|
| H8.10 shared direct failure | At least 50% of primary references are `shared_miss` | The current representation or event definition has a shared limitation not explained by wearable input alone |
| H8.11 wearable-specific recovery gap | At least 15% of primary references are `psg_only` | PSG contains materially recoverable cases that the direct wearable path misses |
| H8.12 direct-model complementarity | Direct-model union recall exceeds the best individual direct-model recall by at least 0.10 | The modalities or channel sets contain complementary event evidence |
| H8.13 boundary-tolerance sensitivity | Direct-model union recall at +/-45 seconds exceeds its +/-15-second value by at least 0.10 | Coarse boundary timing materially contributes to missed-event accounting |

These are mechanism screens, not performance-acceptance gates. No combination rule will be optimized, and union recall will not be presented as a deployable ensemble because it ignores the union false-alarm burden.

## 7. Integrity Checks

The run must verify:

- exactly 20 validation recordings and 16 validation `pid` groups;
- exactly 37 primary eligible references at +/-15 seconds;
- one unique row per reference identity;
- reviewed match counts reproduce each comparator's frozen true-positive count;
- category counts sum to the reference count;
- direct-union recall is not below any component recall;
- all bootstrap draws preserve participant grouping;
- no test path or test-partition row enters any output; and
- rerunning the analysis does not alter reviewed outputs.

An independent validator must reconstruct all event indicators, categories, summaries, intervals, decisions, and hashes from the frozen inputs.

## 8. Decision Boundary

This analysis cannot rescue the current detector, revise a threshold, authorize a new model, or establish clinical usefulness. It will select the next uncertainty:

- prioritize temporal or label-uncertainty work if shared failure or boundary sensitivity dominates;
- prioritize wearable representation or additional sensing if the PSG-only gap dominates; or
- justify a separately predeclared fusion hypothesis only if complementary recovery is material and its alarm burden can be evaluated.

The current test partition remains closed. Any subsequent model experiment requires a new protocol and a meaningful clean-performance and false-alarm gate.
