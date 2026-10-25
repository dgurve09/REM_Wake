# Corrected Multi-Dataset Hypnogram Transition-Uncertainty Decision v0.2

**Decision date:** 2026-10-25
**Sources:** Eleven public hypnogram sources
**Primary nights:** 1,993
**Participant groups:** 1,822
**Model or EEG use:** None

## Correction outcome

The gx36m files contain 818 scorer labels with numeric code `-1`, outside the source README's documented `0..4` stage vocabulary. Safe parsing changed 81 primary consensus epochs from a falsely assigned known stage to `UNKNOWN`. Corrected gx36m primary unknown epochs increased from 1,398 to 1,479.

The strict adjacent gx36m REM-to-Wake count remained 85, and the eleven-source total remained 4,046. This stability is specific to event count: gx36m REM duration decreased by 0.675 hours and known REM-exit opportunities decreased from 234 to 233.

## Cross-source result

The corrected source event rates remain 0.1625 to 0.4533 events per sleep-period hour. The maximum-to-minimum ratio remains 2.789, and the event-positive-night range remains 0.2546. Both exceed the frozen heterogeneity conditions.

The decision remains `material_heterogeneity`. These are dataset-specific annotation summaries and must not be interpreted as pooled clinical prevalence.

## Scorer result

The 1,100 scorer pairs remain available, but corrected gx36m parsing reduced informative event pairs from 152 to 149. gx36m median known-epoch stage disagreement changed from 0.1759 to 0.1684; exact event F1 remained 0; and tolerant +/-30-second event F1 changed from 0.3095 to 0.3333. DOD and ISRUC scorer results were unchanged.

The decision remains `material_scorer_dependence`. Excluding unresolved labels changes estimates but does not make the derived boundary stable across scorers.

## Repeated-night result

Primary gx36m REM-to-Wake events and repeated-night measures were unchanged. Across the four repeated-night sources, the maximum discordant event-presence fraction remains 0.55 and the maximum median within-participant rate range remains 0.2078/hour.

The decision remains `material_repeated_night_dependence`. The analysis cannot separate physiological night variability from annotation and acquisition effects.

## Consequence for PSG modeling

The corrected label analysis confirms 85 strict gx36m primary events, rather than the 124 used in the superseded PSG v0.1 experiment. The PSG correction must also represent all 818 out-of-vocabulary scorer labels as unknown in scorer-sensitivity evaluation. Because gx36m enters training or evaluation in every outer run, all fits must be repeated under the frozen v0.2 correction protocol.

## Verification

- Pre-analysis correction audit: 12/12 checks passed.
- Independent input validation: 10/10 checks passed.
- Corrected analysis controls: 15/15 checks passed.
- Independent result reconstruction: 13/13 checks passed.
- v0.1-to-v0.2 comparison: 7/7 checks passed.
- Non-gx36m dataset, scorer, and repeated-night rows were unchanged.

## Scientific boundary

This correction supports the need to model annotation uncertainty explicitly. It does not demonstrate EEG detection, wearable transfer, diagnostic performance, or clinical usefulness for sleep paralysis or narcolepsy.

## References

1. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
2. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9(1):81-87. https://doi.org/10.5664/jcsm.2350
3. Guillot A, Sauvet F, During EH, Thorey V. Dreem Open Datasets: Multi-Scored Sleep Datasets to Compare Human and Automated Sleep Staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2020;28(9):1955-1965. https://doi.org/10.1109/TNSRE.2020.3011181
