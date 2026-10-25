# Multi-Dataset Hypnogram Transition-Uncertainty Decision

> **Superseded gx36m parsing:** The v0.1 gx36m loader interpreted raw code `-1`
> as `REM` through negative array indexing. Because `-1` is outside the source's
> documented `0..4` stage vocabulary, corrected v0.2 results replace the gx36m
> portions of this decision. The original files remain preserved.

**Decision date:** 2026-10-25  
**Sources:** Eleven public hypnogram sources  
**Primary nights:** 1,993  
**Participant groups:** 1,822  
**Model or EEG use:** None

## Question

Are REM-to-Wake event frequency and timing sufficiently stable across datasets, scorers, and repeated nights to treat one derived hypnogram boundary set as an exact and transportable target?

## Method

The analysis reduced 2,381 harmonized scorer records to one predefined primary hypnogram per night and added 58 gx36m nights. DOD used a strict 3-of-5 epoch majority, ISRUC required agreement from both experts, and gx36m required 2-of-3 support. Unresolved epochs were unknown. Rates used sleep-period support to prevent dataset-specific Wake padding from dominating denominators.

No EEG, feature, probability, model, or classifier output was read.

## Cross-source result

Across 1,993 nights, 4,046 strict adjacent REM-to-Wake events were identified. Aggregate event rates ranged from 0.1625 to 0.4533 per sleep-period hour. The maximum-to-minimum ratio was 2.789, exceeding the fixed 2.0 condition. Event-positive-night fractions ranged from 0.6579 to 0.9125, a 0.2546 difference that exceeded the fixed 0.20 condition.

The cross-source heterogeneity decision is `material_heterogeneity`. This describes target-prior transportability; it does not establish biological or clinical group differences.

## Scorer result

The 1,100 scorer-pair records showed median known-epoch stage disagreement of 0.1902 in DOD, 0.1759 in gx36m, and 0.1778 in ISRUC. Median exact REM-to-Wake event F1 was 0.5000, 0, and 0.6667, respectively. Allowing +/-30 seconds increased these medians only to 0.5455, 0.3095, and 0.6667.

The scorer decision is `material_scorer_dependence`. Stage agreement that appears acceptable in aggregate does not imply stable derived event boundaries.

## Repeated-night result

Among 145 linked repeated participants, discordant event presence affected 15.0% in BOAS, 55.0% in gx36m, 25.0% in ISRUC subgroup 2, and 24.7% in Sleep-EDF. The largest median within-participant event-rate range was 0.2078/hour.

The repeated-night decision is `material_repeated_night_dependence`. The analysis cannot separate biological night variability from scorer and acquisition effects.

## Correction triggered by this analysis

Strict gx36m consensus marked 1,398 three-way-tie epochs unknown and yielded 85 events. The completed PSG external-generalization v0.1 code instead used an unrestricted `argmax`; a three-way tie therefore became `W`. That implementation yielded 124 gx36m events and does not implement the written 2-of-3 majority rule.

The PSG v0.1 performance decision is invalidated as evidence for advancement. Its artifacts remain immutable. A corrected v0.2 rerun must change only gx36m consensus, preserve the same signal features, folds, model, thresholds, alarm rule, metrics, bootstrap, and gate, and report whether the earlier conclusion changes.

## Project decision

1. Preserve source-specific metrics instead of estimating one pooled clinical event prevalence.
2. Retain scorer-specific sensitivity for every multi-scored evaluation.
3. Keep all nights from a participant grouped and report repeated-night dependence.
4. Represent unresolved consensus as unknown; never break ties by label order.
5. Complete the corrected PSG v0.2 rerun before planning paired wearable transfer.
6. Do not interpret these label statistics as EEG, wearable, diagnostic, or clinical validation.

## Verification

The input audit passed 18/18 checks, the independent input validator passed 11/11 checks twice, the analysis passed 13/13 run controls, and the independent result reconstruction passed 13/13 checks twice. The source manifest contains 2,556 verified hashes.

## References

1. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9(1):81-87. https://doi.org/10.5664/jcsm.2350
2. Guillot A, Sauvet F, During EH, Thorey V. Dreem Open Datasets: Multi-Scored Sleep Datasets to Compare Human and Automated Sleep Staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2020;28(9):1955-1965. https://doi.org/10.1109/TNSRE.2020.3011181
3. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
4. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
