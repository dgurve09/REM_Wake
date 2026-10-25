# Multi-Dataset Hypnogram Transition-Uncertainty Analysis v0.1

## Current stage

The frozen transition, scorer, repeated-night, group, and decision analyses are complete.

## Intended input

- Ten harmonized public hypnogram sources represented by 2,381 stored scorer records.
- One primary consensus or provided human hypnogram per 1,935 nights.
- Fifty-eight additional gx36m nights with three scorers.
- Expected complete primary pool: 1,993 nights from eleven sources.

## Input checkpoint

- The primary audit passed 18/18 checks.
- The independent validator passed 11/11 checks twice.
- All 2,556 metadata and source-stage files passed size and SHA-256 verification.
- The primary pool contains 1,822 participant groups.
- Two Sleep-EDF Telemetry files contain timestamp gaps in late Wake padding. One missing epoch in `ST7121` and 65 missing epochs in `ST7221` were represented explicitly as `UNKNOWN`, preserving time and preventing transitions across missing support.

The first audit execution stopped on an invalid NumPy vocabulary expression. The corrected execution then exposed 28 nonzero Sleep-EDF Telemetry start offsets, which are valid constant grid origins. After allowing a constant origin, two real timestamp gaps remained and were regularized as described above. These corrections occurred before any transition result was calculated.

The first independent input-validation execution stopped because blank `night` fields in the reviewed TSV reload as missing values while the reconstructed table retains empty strings. A second attempt showed that numeric night identifiers such as `1` and `1.0` also serialize equivalently. Normalizing the display-only night field corrected the comparison without changing an input or audit result.

## Transition result

The 1,993 primary nights contained 4,046 strict adjacent REM-to-Wake events. Aggregate sleep-period event rates ranged from 0.1625/hour in CAP Sleep to 0.4533/hour in ISRUC, a 2.789-fold ratio. Event-positive-night fractions ranged from 0.6579 to 0.9125, a difference of 0.2546. Both predeclared heterogeneity conditions passed.

These are dataset-specific annotation summaries, not pooled clinical prevalence estimates. Cohorts, scoring systems, recording support, and populations differ.

## Scorer uncertainty

The analysis reconstructed 1,100 same-night scorer pairs:

| Source | Median stage disagreement | Median exact event F1 | Median +/-30 s event F1 |
|---|---:|---:|---:|
| DOD | 0.1902 | 0.5000 | 0.5455 |
| gx36m | 0.1759 | 0 | 0.3095 |
| ISRUC | 0.1778 | 0.6667 | 0.6667 |

Every source failed at least one fixed scorer-dependence threshold. Thirty-second tolerance did not remove the uncertainty.

## Repeated nights

The analysis covered 145 linked participants with repeated nights. Discordant event presence occurred in 15.0% of repeated BOAS participants, 55.0% of gx36m participants, 25.0% of ISRUC subgroup-2 participants, and 24.7% of repeated Sleep-EDF participants. Median within-participant rate ranges were 0.1403-0.2078 events/hour across sources. The repeated-night decision was material under both fixed measures.

## gx36m consensus defect discovered

The strict 2-of-3 gx36m rule produced 1,398 `UNKNOWN` epochs where no stage received two votes and 85 REM-to-Wake events. The earlier PSG v0.1 majority function used an unconstrained `argmax`, which assigned every three-way tie to the first stage (`W`) and reported 124 gx36m events. That behavior contradicts the written majority rule and affects the prior PSG labels and performance result.

The completed PSG v0.1 artifacts remain preserved, but their scientific decision must be treated as invalidated pending an exact v0.2 rerun with strict 2-of-3 consensus. No feature, model, threshold, or event rule may change in that correction.

## Verification

- 13/13 in-run checks passed.
- 13/13 independent result checks passed twice.
- The validator reconstructed all 4,046 events, 1,100 scorer pairs, 145 repeated participants, source/group summaries, sensitivity results, and decisions.

## Decision

All three frozen uncertainty screens were material. A single source, scorer, or night cannot be treated as an exact and transportable REM-to-Wake event prior. Future model evaluation must retain source-specific reporting, scorer sensitivity where available, participant grouping across nights, and explicit unknown labels for unresolved consensus.

## Boundary

This is a label-only analysis. It reads no EEG signal, features, model, probability, or classifier output. The result will characterize the derived REM-to-Wake event prior and its annotation and repeated-night uncertainty; it cannot validate wearable detection.

See the [frozen protocol](../../docs/evaluation/multidataset_hypnogram_transition_uncertainty_protocol_v0.1.md).
